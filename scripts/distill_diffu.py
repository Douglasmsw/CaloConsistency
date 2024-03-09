import numpy as np
import os
import yaml
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py as h5
import torch.optim as optim
import torch.utils.data as torchdata

from utils import *
from CaloDiffu import *
from models import *
import sys

if __name__ == '__main__':
    print("CONSISTENCY DISTILLATION FROM DIFFU")

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/wclustre/cms_mlsim/denoise/CaloChallenge/', help='Folder containing data and MC files')
    parser.add_argument('--model', default='Diffu', help='Diffusion model to train')
    parser.add_argument('--model_loc', default='test', help='Location of model')
    parser.add_argument('-c', '--config', default='configs/test.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=int, default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float, default=0.85, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False, help='Load pretrained weights to continue the training')
    parser.add_argument('--seed', type=int, default=1234,help='Pytorch seed')
    parser.add_argument('--reset_training', action='store_true', default=False,help='Retrain')
    parser.add_argument('--binning_file', type=str, default=None)
    parser.add_argument('--save_folder_prepend', type=str, default=None, help='Optional text to append to training folder to separate outputs of training runs with the same config file')
    parser.add_argument('--num_sample', default=18, type=int, help='number of times to exeucute ODE solver step in sampling')
    parser.add_argument('--mu', default=0.99, type=float, help='keep weight for teacher model ema param update')
    parser.add_argument('--sigma2', default=0.5, type=float, help='data sigma^2 value for c_skip and c_out weight calculations')
    parser.add_argument('--ode_solver', type=str, default='paper_euler', help="determine ODE solver used, (paper_euler, paper_heun, back_euler, back_direct, or back_heun")
    parser.add_argument('--rho', default=7., type=float, help='adjusts aggressiveness of n-to-time-step function')
    parser.add_argument('--n_save', default=1, type=int, help='epoch interval to save model at, higher intervals speeds up training')
    parser.add_argument('--lr', default=4.e-4, type=float, help='optimizer learning rate')
    parser.add_argument('--no_sched', action='store_true', default=False, help='turns off learning rate scheduler')
    parser.add_argument('--sched_factor', default=0.1, type=float, help='factor by which to decay learning rate every sched_patience epochs')
    parser.add_argument('--sched_patience', default=15, type=int, help='frequency at which to decay learning rate by sched_factor')
    parser.add_argument('--time_embed', default=None, type=str, help='pre-embedding time step rescaling method (identity, sin, scaled, sigma, log)')
    parser.add_argument('--noise_sched', default='ddpm', type=str, help='type of noise schedule to follow in foward pass (ddpm, std)')
    
    flags = parser.parse_args()

    def trim_file_path(cwd:str, num_back:int):
        '''
        '''
        split_path = cwd.split("/")
        trimmed_split_path = split_path[:-num_back]
        trimmed_path = "/".join(trimmed_split_path)

        return trimmed_path

    cwd = __file__
    calo_challenge_dir = trim_file_path(cwd=cwd, num_back=3)
    sys.path.append(calo_challenge_dir)
    from scripts.utils import *
    from CaloChallenge.code.XMLHandler import *

    print("TRAINING OPTIONS")
    dataset_config = LoadJson(flags.config)
    print(dataset_config, flush = True)

    torch.manual_seed(flags.seed)

    cold_diffu = dataset_config.get('COLD_DIFFU', False)
    cold_noise_scale = dataset_config.get('COLD_NOISE', 1.0)

    nholdout  = dataset_config.get('HOLDOUT', 0)

    batch_size = dataset_config['BATCH']
    num_epochs = dataset_config['MAXEPOCH']
    early_stop = dataset_config['EARLYSTOP']
    training_obj = dataset_config.get('TRAINING_OBJ', 'noise_pred')
    loss_type = dataset_config.get("LOSS_TYPE", "l2")
    dataset_num = dataset_config.get('DATASET_NUM', 2)
    shower_embed = dataset_config.get('SHOWER_EMBED', '')
    orig_shape = ('orig' in shower_embed)
    energy_loss_scale = dataset_config.get('ENERGY_LOSS_SCALE', 0.0)

    if flags.ode_solver not in {'paper_euler', 'paper_heun', 'back_euler', 'back_direct', 'back_heun'}: 
        raise ValueError(f"\nInvalid ODE solver name input: '{flags.ode_solver}'; must be one of 'paper euler', 'paper heun', 'back euler', or 'back heun'\n")

    data = []
    energies = []

    for i, dataset in enumerate(dataset_config['FILES']):
        data_,e_ = DataLoader(
            os.path.join(flags.data_folder, dataset),
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            nevts = flags.nevts,
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],

            nholdout = nholdout if (i == len(dataset_config['FILES']) -1 ) else 0,
            dataset_num  = dataset_num,
            orig_shape = orig_shape,
        )


        if(i ==0): 
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))
        
    avg_showers = std_showers = E_bins = None
    if(cold_diffu):
        f_avg_shower = h5.File(dataset_config["AVG_SHOWER_LOC"])
        #Already pre-processed
        avg_showers = torch.from_numpy(f_avg_shower["avg_showers"][()].astype(np.float32)).to(device = device)
        std_showers = torch.from_numpy(f_avg_shower["std_showers"][()].astype(np.float32)).to(device = device)
        E_bins = torch.from_numpy(f_avg_shower["E_bins"][()].astype(np.float32)).to(device = device)

    NN_embed = None
    if('NN' in shower_embed):
        if(dataset_num == 1):
            if flags.binning_file is None:
                flags.binning_file = "../CaloChallenge/code/binning_dataset_1_photons.xml"
            bins = XMLHandler("photon", flags.binning_file)
        else: 
            if flags.binning_file is None:
                flags.binning_file = "../CaloChallenge/code/binning_dataset_1_pions.xml"
            bins = XMLHandler("pion", flags.binning_file)

        NN_embed = NNConverter(bins = bins).to(device = device)
        

    dshape = dataset_config['SHAPE_PAD']
    energies = np.reshape(energies,(-1))    
    if(not orig_shape): data = np.reshape(data,dshape)
    else: data = np.reshape(data, (len(data), -1))

    num_data = data.shape[0]
    print("Data Shape " + str(data.shape))
    data_size = data.shape[0]
    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)
    del data

    torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
    nTrain = int(round(flags.frac * num_data))
    nVal = num_data - nTrain
    train_dataset, val_dataset = torch.utils.data.random_split(torch_dataset, [nTrain, nVal])

    loader_train = torchdata.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    loader_val = torchdata.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    del torch_data_tensor, torch_E_tensor, train_dataset, val_dataset

    if flags.save_folder_prepend is None:
        student_checkpoint_folder = '../models/{}_CDstudent/'.format(dataset_config['CHECKPOINT_NAME'])
        teacher_checkpoint_folder = '../models/{}_CDteacher/'.format(dataset_config['CHECKPOINT_NAME'])
    else:
        student_checkpoint_folder = '../models/{}/{}_CDstudent/'.format(flags.save_folder_prepend, dataset_config['CHECKPOINT_NAME'])
        teacher_checkpoint_folder = '../models/{}/{}_CDteacher/'.format(flags.save_folder_prepend, dataset_config['CHECKPOINT_NAME'])

    if not os.path.exists(student_checkpoint_folder):
        os.makedirs(student_checkpoint_folder)
    if not os.path.exists(teacher_checkpoint_folder):
        os.makedirs(teacher_checkpoint_folder)

    # store flag dictionary as config file for documentation
    flag_config = vars(flags)
    student_flag_config_path = f"{student_checkpoint_folder}flag_config.yml"
    teacher_flag_config_path = f"{teacher_checkpoint_folder}flag_config.yml"

    with open(student_flag_config_path, 'w+') as f:
        yaml.dump(flag_config, f, default_flow_style=False)

    with open(teacher_flag_config_path, 'w+') as f:
        yaml.dump(flag_config, f, default_flow_style=False)

    # prep checkpoint paths
    student_checkpoint = dict()
    teacher_checkpoint = dict()
    student_checkpoint_path = os.path.join(student_checkpoint_folder, "checkpoint.pth")
    teacher_checkpoint_path = os.path.join(teacher_checkpoint_folder, "checkpoint.pth")

    if(flags.load and os.path.exists(student_checkpoint_path)): 
        print("Loading training checkpoint from %s" % student_checkpoint_path, flush = True)
        student_checkpoint = torch.load(student_checkpoint_path, map_location = device)
        print(student_checkpoint.keys())
    
    if(flags.load and os.path.exists(teacher_checkpoint_path)): 
        print("Loading training checkpoint from %s" % teacher_checkpoint_path, flush = True)
        teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location = device)
        print(teacher_checkpoint.keys())


    if(flags.model == "Diffu"):
        shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]

        #initialize student model
        student_model = CaloDiffu(shape, config=dataset_config, training_obj = training_obj, NN_embed = NN_embed, nsteps = dataset_config['NSTEPS'],
                                cold_diffu = cold_diffu, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins,
                                num_sample = flags.num_sample, sigma2 = flags.sigma2, ode_solver = flags.ode_solver,
                                rho = flags.rho, time_embed = flags.time_embed, noise_sched = flags.noise_sched).to(device = device)
        #initialize teacher model as copy of student (shares initial parameters)
        student_params_dict = student_model.state_dict()
        teacher_model = CaloDiffu(shape, config=dataset_config, training_obj = training_obj, NN_embed = NN_embed, nsteps = dataset_config['NSTEPS'],
                                cold_diffu = cold_diffu, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins,
                                num_sample = flags.num_sample, sigma2 = flags.sigma2, ode_solver = flags.ode_solver,
                                rho = flags.rho, time_embed = flags.time_embed, noise_sched = flags.noise_sched).to(device = device)
        teacher_model.load_state_dict(student_params_dict)
        teacher_model._N = None

        #initialize trained diffusion model            
        trained_model = CaloDiffu(shape, config=dataset_config, training_obj = training_obj, NN_embed = NN_embed, nsteps = dataset_config['NSTEPS'],
                                cold_diffu = cold_diffu, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins).to(device = device)
        # load in pre-trained weights
        trained_model.load_state_dict(torch.load(flags.model_loc)['model_state_dict'])

        #sometimes save only weights, sometimes save other info
        if('model_state_dict' in student_checkpoint.keys()): student_model.load_state_dict(student_checkpoint['model_state_dict'])
        elif(len(student_checkpoint.keys()) > 1): student_model.load_state_dict(student_checkpoint)

        if('model_state_dict' in teacher_checkpoint.keys()): teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
        elif(len(teacher_checkpoint.keys()) > 1): teacher_model.load_state_dict(teacher_checkpoint)


    else:
        print("Model %s not supported!" % flags.model)
        exit(1)

    '''
    os.system('cp CaloDiffu.py {}'.format(checkpoint_folder)) # bkp of model def
    os.system('cp models.py {}'.format(checkpoint_folder)) # bkp of model def
    os.system('cp {} {}'.format(flags.config,checkpoint_folder)) # bkp of config file
    '''

    early_stopper = EarlyStopper(patience = dataset_config['EARLYSTOP'], mode = 'diff', min_delta = 1e-5)
    if('early_stop_dict' in student_checkpoint.keys() and not flags.reset_training): early_stopper.__dict__ = student_checkpoint['early_stop_dict']
    print(early_stopper.__dict__)
    

    criterion = nn.MSELoss().to(device = device)

    optimizer = optim.Adam(student_model.parameters(), lr = flags.lr)#float(dataset_config["LR"]))
    if('optimizer_state_dict' in student_checkpoint.keys() and not flags.reset_training): optimizer.load_state_dict(student_checkpoint['optimizer_state_dict'])
    # optional learning rate scheduler
    if flags.no_sched is False: 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = flags.sched_factor, patience = flags.sched_patience, verbose = True) 
        if('scheduler_state_dict' in student_checkpoint.keys() and not flags.reset_training): scheduler.load_state_dict(student_checkpoint['scheduler_state_dict'])

    training_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    start_epoch = 0
    min_validation_loss = 99999.
    if('train_loss_hist' in student_checkpoint.keys() and not flags.reset_training): 
        training_losses = student_checkpoint['train_loss_hist']
        val_losses = student_checkpoint['val_loss_hist']
        start_epoch = student_checkpoint['epoch'] + 1

    #training loop
    for epoch in range(start_epoch, num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        for i, param in enumerate(student_model.parameters()):
            break
        train_loss = 0

        student_model.train()
        teacher_model.train()
        for i, (E,data) in tqdm(enumerate(loader_train, 0), unit="batch", total=len(loader_train)):
            student_model.zero_grad()
            teacher_model.zero_grad()
            optimizer.zero_grad()

            data = data.to(device = device)
            E = E.to(device = device)

            noise = torch.randn_like(data)
            
            if(cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
                noise = student_model.gen_cold_image(E, cold_noise_scale, noise)

            batch_loss = student_model.compute_distillation_loss(data, E, noise = noise, n = None, 
                                                                    teacher_model = teacher_model,
                                                                    trained_model = trained_model,
                                                                    loss_type = loss_type, 
                                                                    energy_loss_scale = energy_loss_scale)

            batch_loss.backward()

            optimizer.step()
            teacher_model = ema_param_update(student_model, teacher_model, mu=flags.mu)

            train_loss+=batch_loss.item()

            del data, E, noise, batch_loss

        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        val_loss = 0
        student_model.eval()
        teacher_model.eval()
        for i, (vE, vdata) in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):
            vdata = vdata.to(device=device)
            vE = vE.to(device = device)

            noise = torch.randn_like(vdata)
            if(cold_diffu): noise = student_model.gen_cold_image(vE, cold_noise_scale, noise)

            with torch.no_grad():
                batch_loss = student_model.compute_distillation_loss(vdata, vE, noise = noise, n = None, 
                                                                        teacher_model = teacher_model,
                                                                        trained_model = trained_model,
                                                                        loss_type = loss_type, 
                                                                        energy_loss_scale = energy_loss_scale)

            val_loss+=batch_loss.item()
            del vdata,vE, noise, batch_loss

        val_loss = val_loss/len(loader_val)
        val_losses[epoch] = val_loss
        print("val_loss: "+ str(val_loss), flush = True)

        if flags.no_sched is False: scheduler.step(torch.tensor([train_loss]))

        if(val_loss < min_validation_loss):
            torch.save(student_model.state_dict(), os.path.join(student_checkpoint_folder, 'best_val.pth'))
            min_validation_loss = val_loss

        '''
        if(early_stopper.early_stop(val_loss - train_loss)):
            print("Early stopping!")
            break
        '''
        # save the model every n_save steps
        if (epoch % flags.n_save == 0) or (epoch == num_epochs-1):
            student_model.eval()
            print("SAVING")
            
            #save full training state so can be resumed
            if flags.no_sched is False:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss_hist': training_losses,
                    'val_loss_hist': val_losses,
                    'early_stop_dict': early_stopper.__dict__,
                    }, student_checkpoint_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss_hist': training_losses,
                    'val_loss_hist': val_losses,
                    'early_stop_dict': early_stopper.__dict__,
                    }, student_checkpoint_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': teacher_model.state_dict(),
                }, teacher_checkpoint_path)

            with open(student_checkpoint_folder + "/training_losses.txt","w") as tfileout:
                tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
            with open(student_checkpoint_folder + "/validation_losses.txt","w") as vfileout:
                vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")


    print("Saving to %s" % student_checkpoint_folder, flush=True)
    torch.save(student_model.state_dict(), os.path.join(student_checkpoint_folder, 'final.pth'))

    print("Saving to %s" % teacher_checkpoint_folder, flush=True)
    torch.save(teacher_model.state_dict(), os.path.join(teacher_checkpoint_folder, 'final.pth'))

    with open(student_checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(student_checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")

