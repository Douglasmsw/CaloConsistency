import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time, sys, copy
import utils
import torch
import torch.utils.data as torchdata
from CaloDiffu import *
# import h5py

if(torch.cuda.is_available()): device = torch.device('cuda')
else: device = torch.device('cpu')

plt_exts = ["png", "pdf"]
#plt_ext = "pdf"
rank = 0
size = 1

utils.SetStyle()


parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/wclustre/cms_mlsim/denoise/CaloChallenge/', help='Folder containing data and MC files')
parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
parser.add_argument('--generated', '-g', default='', help='Generated showers')
parser.add_argument('--model_loc', default='test', help='Location of model')
parser.add_argument('--config', default='config_dataset2.json', help='Training parameters')
parser.add_argument('--nevts', type=int,default=-1, help='Number of events to load')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for generation')
parser.add_argument('--model', default='Diffu', help='Diffusion model to load. Options are: Diffu, AE, Consist, all')
parser.add_argument('--plot_label', default='', help='Add to plot')
parser.add_argument('--sample', action='store_true', default=False,help='Sample from learned model')
parser.add_argument('--sample_steps', default = -1, type = int, help='How many steps for sampling (override config)')
parser.add_argument('--sample_offset', default = 0, type = int, help='Skip some iterations in the sampling (noisiest iters most unstable)')
parser.add_argument('--sample_algo', default = 'ddpm', help = 'What sampling algorithm (ddpm, cd, cd_stdz, ddim)')
parser.add_argument('--job_idx', default = -1, type = int, help = 'Split generation among different jobs')
parser.add_argument('--debug', action='store_true', default=False, help='Debugging options')
parser.add_argument('--save_folder_prepend', type=str, default=None, help='Optional text to append to training folder to separate outputs of training runs with the same config file')
parser.add_argument('--binning_file', type=str, default=None, help='Path to binning file') # added to account for new file structure
parser.add_argument('--sigma2', default=0.5, type=float, help='data sigma^2 value for c_skip and c_out weight calculations')
parser.add_argument('--num_sample', default=18, type=int, help='number of times to exeucute ODE solver step in sampling')
parser.add_argument('--rho', default=7., type=float, help='adjusts aggressiveness of n-to-time-step function')
parser.add_argument('--time_embed', default=None, type=str, help='pre-embedding time step rescaling method (identity, sin, scaled, sigma, log)')
parser.add_argument('--noise_sched', default='ddpm', type=str, help='type of noise schedule to follow in foward pass (ddpm, std)')

flags = parser.parse_args()

nevts = int(flags.nevts)
dataset_config = utils.LoadJson(flags.config)
emax = dataset_config['EMAX']
emin = dataset_config['EMIN']
cold_diffu = dataset_config.get('COLD_DIFFU', False)
cold_noise_scale = dataset_config.get("COLD_NOISE", 1.0)
training_obj = dataset_config.get('TRAINING_OBJ', 'noise_pred')
dataset_num = dataset_config.get('DATASET_NUM', 2)

sample_steps = dataset_config["NSTEPS"] if flags.sample_steps < 0 else flags.sample_steps

batch_size = flags.batch_size
shower_embed = dataset_config.get('SHOWER_EMBED', '')
orig_shape = ('orig' in shower_embed)
do_NN_embed = ('NN' in shower_embed)

if(not os.path.exists(flags.plot_folder)): 
    print("Creating plot directory " + flags.plot_folder)
    os.system("mkdir " + flags.plot_folder)

evt_start = 0
job_label = ""
if(flags.job_idx >= 0):
    if(flags.nevts <= 0):
        print("Must include number of events for split jobs")
        sys.exit(1)
    evt_start = flags.job_idx * flags.nevts
    job_label = "_job%i" % flags.job_idx




if flags.sample:
    # initialize save folder
    if flags.save_folder_prepend is None: checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    else: checkpoint_folder = '../models/{}/{}_{}/'.format(flags.save_folder_prepend,dataset_config['CHECKPOINT_NAME'],flags.model)
    if not os.path.exists(checkpoint_folder): os.makedirs(checkpoint_folder)

    energies = None
    data = None
    for i, dataset in enumerate(dataset_config['EVAL']):
        n_dataset = h5.File(os.path.join(flags.data_folder,dataset))['showers'].shape[0]
        if(evt_start >= n_dataset):
            evt_start -= n_dataset
            continue

        data_,e_ = utils.DataLoader(
            os.path.join(flags.data_folder,dataset),
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            nevts = flags.nevts,
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],
            dataset_num  = dataset_num,
            orig_shape = orig_shape,
            evt_start = evt_start
        )
        
        if(data is None): 
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))
        if(flags.nevts > 0 and data_.shape[0] == flags.nevts): break

    energies = np.reshape(energies,(-1,))
    if(not orig_shape): data = np.reshape(data,dataset_config['SHAPE_PAD'])
    else: data = np.reshape(data, (len(data), -1))

    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)

    print("DATA mean, std", torch.mean(torch_data_tensor), torch.std(torch_data_tensor))

    torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
    data_loader = torchdata.DataLoader(torch_dataset, batch_size = batch_size, shuffle = False)

    avg_showers = std_showers = E_bins = None
    if(cold_diffu or flags.model == 'Avg'):
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



    if(flags.model == "AE"):
        print("Loading AE from " + flags.model_loc)
        model = CaloAE(dataset_config['SHAPE_PAD'][1:], batch_size, config=dataset_config).to(device=device)

        saved_model = torch.load(flags.model_loc, map_location = device)
        if('model_state_dict' in saved_model.keys()): model.load_state_dict(saved_model['model_state_dict'])
        elif(len(saved_model.keys()) > 1): model.load_state_dict(saved_model)
        #model.load_state_dict(torch.load(flags.model_loc, map_location=device))

        generated = []
        for i,(E,d_batch) in enumerate(data_loader):
            E = E.to(device=device)
            d_batch = d_batch.to(device=device)
        
            gen = model(d_batch).detach().cpu().numpy()
            if(i == 0): generated = gen
            else: generated = np.concatenate((generated, gen))
            del E, d_batch
    elif(flags.model == "Diffu"):
        print("Loading Diffu model from " + flags.model_loc)

        shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]
        model = CaloDiffu(shape, config=dataset_config , training_obj = training_obj,NN_embed = NN_embed, nsteps = sample_steps,
                cold_diffu = cold_diffu, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins ).to(device = device)

        saved_model = torch.load(flags.model_loc, map_location = device)
        if('model_state_dict' in saved_model.keys()): model.load_state_dict(saved_model['model_state_dict'])
        elif(len(saved_model.keys()) > 1): model.load_state_dict(saved_model)

        generated = []
        start_time = time.time()
        for i,(E,d_batch) in enumerate(data_loader):
            if(E.shape[0] == 0): continue
            E = E.to(device=device)
            d_batch = d_batch.to(device=device)

            out = model.Sample(E, num_steps = sample_steps, cold_noise_scale = cold_noise_scale, sample_algo = flags.sample_algo,
                    debug = flags.debug, sample_offset = flags.sample_offset)


            if(flags.debug):
                gen, all_gen, x0s = out
                for j in [0,len(all_gen)//4, len(all_gen)//2, 3*len(all_gen)//4, 9*len(all_gen)//10, len(all_gen)-10, len(all_gen)-5,len(all_gen)-1]:
                    fout_ex = '{}/{}_{}_norm_voxels_gen_step{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, j, plt_exts[0])
                    make_histogram([all_gen[j].reshape(-1), data.reshape(-1)], ['Diffu', 'Geant4'], ['blue', 'black'], xaxis_label = 'Normalized Voxel Energy', 
                                    num_bins = 40, normalize = True, fname = fout_ex)

                    fout_ex = '{}/{}_{}_norm_voxels_x0_step{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, j, plt_exts[0])
                    make_histogram([x0s[j].reshape(-1), data.reshape(-1)], ['Diffu', 'Geant4'], ['blue', 'black'], xaxis_label = 'Normalized Voxel Energy', 
                                    num_bins = 40, normalize = True, fname = fout_ex)
            else: gen = out

        
            if(i == 0): generated = gen
            else: generated = np.concatenate((generated, gen))
            del E, d_batch
        end_time = time.time()
        print("Total sampling time %.3f seconds" % (end_time - start_time))
    elif(flags.model == "Consist"):
        print("Loading Consistency model from " + flags.model_loc)

        shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]
        model = CaloDiffu(shape, config=dataset_config, training_obj = training_obj, NN_embed = NN_embed, nsteps = dataset_config['NSTEPS'],
                                cold_diffu = cold_diffu, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins,
                                num_sample = flags.num_sample, sigma2 = flags.sigma2, ode_solver = None,
                                rho = flags.rho, time_embed = flags.time_embed, noise_sched = flags.noise_sched).to(device = device)

        saved_model = torch.load(flags.model_loc, map_location = device)
        if('model_state_dict' in saved_model.keys()): model.load_state_dict(saved_model['model_state_dict'])
        elif(len(saved_model.keys()) > 1): model.load_state_dict(saved_model)

        generated = []
        start_time = time.time()
        for i,(E,d_batch) in enumerate(data_loader):
            if(E.shape[0] == 0): continue
            E = E.to(device=device)
            d_batch = d_batch.to(device=device)

            out = model.Sample(E, num_steps = sample_steps, cold_noise_scale = cold_noise_scale, sample_algo = flags.sample_algo,
                    debug = flags.debug, sample_offset = flags.sample_offset)


            if(flags.debug):
                gen, all_gen, x0s = out
                for j in [0,len(all_gen)//4, len(all_gen)//2, 3*len(all_gen)//4, 9*len(all_gen)//10, len(all_gen)-10, len(all_gen)-5,len(all_gen)-1]:
                    fout_ex = '{}/{}_{}_norm_voxels_gen_step{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, j, plt_exts[0])
                    make_histogram([all_gen[j].reshape(-1), data.reshape(-1)], ['Diffu', 'Geant4'], ['blue', 'black'], xaxis_label = 'Normalized Voxel Energy', 
                                    num_bins = 40, normalize = True, fname = fout_ex)

                    fout_ex = '{}/{}_{}_norm_voxels_x0_step{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, j, plt_exts[0])
                    make_histogram([x0s[j].reshape(-1), data.reshape(-1)], ['Diffu', 'Geant4'], ['blue', 'black'], xaxis_label = 'Normalized Voxel Energy', 
                                    num_bins = 40, normalize = True, fname = fout_ex)
            else: gen = out

        
            if(i == 0): generated = gen
            else: generated = np.concatenate((generated, gen))
            del E, d_batch
        end_time = time.time()
        print("Total sampling time %.3f seconds" % (end_time - start_time))
    elif(flags.model == "Avg"):
        #define model just for useful fns
        model = CaloDiffu(dataset_config['SHAPE_PAD'][1:], nevts,config=dataset_config, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins ).to(device = device)

        generated = model.gen_cold_image(torch_E_tensor, cold_noise_scale).numpy()

    #print("GENERATED", np.mean(generated), np.std(generated), np.amax(generated), np.amin(generated))

    if(not orig_shape): generated = generated.reshape(dataset_config["SHAPE"])

    if(flags.debug):
        fout_ex = '{}/{}_{}_norm_voxels.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_exts[0])
        make_histogram([generated.reshape(-1), data.reshape(-1)], ['Diffu', 'Geant4'], ['blue', 'black'], xaxis_label = 'Normalized Voxel Energy', 
                        num_bins = 40, normalize = True, fname = fout_ex)

    generated,energies = utils.ReverseNorm(generated,energies,#[:nevts],
                                           shape=dataset_config['SHAPE'],
                                           logE=dataset_config['logE'],
                                           max_deposit=dataset_config['MAXDEP'],
                                           emax = dataset_config['EMAX'],
                                           emin = dataset_config['EMIN'],
                                           showerMap = dataset_config['SHOWERMAP'],
                                           dataset_num  = dataset_num,
                                           orig_shape = orig_shape,
                                           ecut = dataset_config['ECUT'],
                                           )

    energies = np.reshape(energies,(-1,1))
    if(dataset_num > 1):
        #mask for voxels that are always empty
        mask_file = os.path.join(flags.data_folder,dataset_config['EVAL'][0].replace('.hdf5','_mask.hdf5'))
        if(not os.path.exists(mask_file)):
            print("Creating mask based on data batch")
            mask = np.sum(data,0)==0

        else:
            with h5.File(mask_file,"r") as h5f:
                mask = h5f['mask'][:]
        generated = generated*(np.reshape(mask,(1,-1))==0)
    
    if(flags.generated == ""):
        fout = os.path.join(checkpoint_folder,'generated_{}_{}{}.h5'.format(dataset_config['CHECKPOINT_NAME'],flags.model, job_label))
    else:
        fout = flags.generated

    print("Creating " + fout)
    with h5.File(fout,"w") as h5f:
        dset = h5f.create_dataset("showers", data=1000*np.reshape(generated,(generated.shape[0],-1)), compression = 'gzip')
        dset = h5f.create_dataset("incident_energies", data=1000*energies, compression = 'gzip')


if(not flags.sample):


    geom_conv = None
    if(dataset_num <= 1):
        bins = XMLHandler(dataset_config['PART_TYPE'], flags.binning_file)
        geom_conv = GeomConverter(bins)

    def LoadSamples(fname):
        with h5.File(fname,"r") as h5f:
            generated = h5f['showers'][:flags.nevts]/1000.
            energies = h5f['incident_energies'][:flags.nevts]/1000.
        energies = np.reshape(energies,(-1,1))
        if(dataset_num <= 1):
            generated = geom_conv.convert(geom_conv.reshape(generated)).detach().numpy()
        generated = np.reshape(generated,dataset_config['SHAPE'])

        return generated,energies


    models = [flags.model]

    energies = []
    data_dict = {}
    for model in models:
        # initialize output folder
        if flags.save_folder_prepend is None: checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'], model)
        else: checkpoint_folder = '../models/{}/{}_{}/'.format(flags.save_folder_prepend,dataset_config['CHECKPOINT_NAME'], model)
        if not os.path.exists(checkpoint_folder): os.makedirs(checkpoint_folder)
        
        if(flags.generated == ""):
            f_sample = os.path.join(checkpoint_folder,'generated_{}_{}.h5'.format(dataset_config['CHECKPOINT_NAME'], model))
        else:
            f_sample = flags.generated
        if np.size(energies) == 0:
            data,energies = LoadSamples(f_sample)
            data_dict[utils.name_translate[model]]=data
        else:
            data_dict[utils.name_translate[model]]=LoadSamples(f_sample)[0]
    total_evts = energies.shape[0]


    data = []
    true_energies = []
    for dataset in dataset_config['EVAL']:
        with h5.File(os.path.join(flags.data_folder,dataset),"r") as h5f:
            start = evt_start
            end = start + total_evts
            show = h5f['showers'][start:end]/1000.
            if(dataset_num <=1 ):
                show = geom_conv.convert(geom_conv.reshape(show)).detach().numpy()
            data.append(show)
            true_energies.append(h5f['incident_energies'][start:end]/1000.)
            if(data[-1].shape[0] == total_evts): break


    data_dict['Geant4']=np.reshape(data,dataset_config['SHAPE'])
    print(data_dict['Geant4'].shape)
    print("Geant Avg", np.mean(data_dict['Geant4']))
    print("Generated Avg", np.mean(data_dict[utils.name_translate[model]]))
    true_energies = np.reshape(true_energies,(-1,1))
    model_energies = np.reshape(energies,(-1,1))
    #assert(np.allclose(data_energies, model_energies))



    #Plot high level distributions and compare with real values
    #assert np.allclose(true_energies,energies), 'ERROR: Energies between samples dont match'



    def HistERatio(data_dict,true_energies):
        

        ratios =  []
        feed_dict = {}
        for key in data_dict:
            dep = np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1)
            if('Geant' in key): feed_dict[key] = dep / true_energies.reshape(-1)
            else: feed_dict[key] = dep / model_energies.reshape(-1)

        binning = np.linspace(0.5, 1.5, 51)

        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Dep. energy / Gen. energy', logy=False,binning=binning, ratio = True, plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_ERatio_{}_{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        return feed_dict



    def ScatterESplit(data_dict,true_energies):
        

        fig,ax = SetFig("Gen. energy [GeV]","Dep. energy [GeV]")
        for key in data_dict:
            x = true_energies[0:500] if 'Geant' in key else model_energies[0:500]
            y = np.sum(data_dict[key].reshape(data_dict[key].shape[0],-1),-1)[0:500]

            ax.scatter(x, y, label=key)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(loc='best',fontsize=16,ncol=1)
        plt.tight_layout()
        if(len(flags.plot_label) > 0): ax.set_title(flags.plot_label, fontsize = 20, loc = 'right', style = 'italic')
        for plt_ext in plt_exts: fig.savefig('{}/FCC_Scatter_{}_{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))



    def AverageShowerWidth(data_dict):

        def GetMatrix(sizex,sizey, minval=-1,maxval=1, binning = None):
            nbins = sizex
            if(binning is None): binning = np.linspace(minval,maxval,nbins+1)
            coord = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
            matrix = np.repeat(np.expand_dims(coord,-1),sizey,-1)
            return matrix

        
        #TODO : Use radial bins
        #r_bins = [0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85]

        phi_matrix = GetMatrix(dataset_config['SHAPE'][2],dataset_config['SHAPE'][3], minval = -math.pi, maxval = math.pi)
        phi_matrix = np.reshape(phi_matrix,(1,1,phi_matrix.shape[0],phi_matrix.shape[1],1))
        
        
        r_matrix = np.transpose(GetMatrix(dataset_config['SHAPE'][3],dataset_config['SHAPE'][2]))
        r_matrix = np.reshape(r_matrix,(1,1,r_matrix.shape[0],r_matrix.shape[1],1))


        def GetCenter(matrix,energies,power=1):
            ec = energies*np.power(matrix,power)
            sum_energies = np.sum(np.reshape(energies,(energies.shape[0],energies.shape[1],-1)),-1)
            ec = np.reshape(ec,(ec.shape[0],ec.shape[1],-1)) #get value per layer
            ec = np.ma.divide(np.sum(ec,-1),sum_energies).filled(0)
            return ec

        def ang_center_spread(matrix, energies):
            #weighted average over periodic variabel (angle)
            #https://github.com/scipy/scipy/blob/v1.11.1/scipy/stats/_morestats.py#L4614
            #https://en.wikipedia.org/wiki/Directional_statistics#The_fundamental_difference_between_linear_and_circular_statistics
            cos_matrix = np.cos(matrix)
            sin_matrix = np.sin(matrix)
            cos_ec = GetCenter(cos_matrix, energies)
            sin_ec = GetCenter(sin_matrix, energies)
            ang_mean  = np.arctan2(sin_ec, cos_ec)
            R = sin_ec**2 + cos_ec**2
            eps = 1e-8
            R = np.clip(R, eps, 1.)

            ang_std = np.sqrt(-np.log(R))
            return ang_mean, ang_std


        def GetWidth(mean,mean2):
            width = np.ma.sqrt(mean2-mean**2).filled(0)
            return width
        
        feed_dict_phi = {}
        feed_dict_phi2 = {}
        feed_dict_r = {}
        feed_dict_r2 = {}
        
        for key in data_dict:
            feed_dict_phi[key], feed_dict_phi2[key] = ang_center_spread(phi_matrix, data_dict[key])
            feed_dict_r[key] = GetCenter(r_matrix,data_dict[key])
            feed_dict_r2[key] = GetWidth(feed_dict_r[key],GetCenter(r_matrix,data_dict[key],2))
            

        if(dataset_config['cartesian_plot']): 
            xlabel1 = 'x'
            f_str1 = "Eta"
            xlabel2 = 'y'
            f_str2 = "Phi"
        else: 
            xlabel1 = 'r'
            f_str1 = "R"
            xlabel2 = 'alpha'
            f_str2 = "Alpha"
        fig,ax0 = utils.PlotRoutine(feed_dict_r,xlabel='Layer number', ylabel= '%s-center of energy' % xlabel1, plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_{}EC_{}_{}.{}'.format(flags.plot_folder,f_str1,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        fig,ax0 = utils.PlotRoutine(feed_dict_phi,xlabel='Layer number', ylabel= '%s-center of energy' % xlabel2, plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_{}EC_{}_{}.{}'.format(flags.plot_folder,f_str2,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        fig,ax0 = utils.PlotRoutine(feed_dict_r2,xlabel='Layer number', ylabel= '%s-width' % xlabel1, plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_{}W_{}_{}.{}'.format(flags.plot_folder,f_str1,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        fig,ax0 = utils.PlotRoutine(feed_dict_phi2,xlabel='Layer number', ylabel= '%s-width (radians)' % xlabel2, plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_{}W_{}_{}.{}'.format(flags.plot_folder,f_str2,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))

        return feed_dict_r2

    def AverageELayer(data_dict):
        
        def _preprocess(data):
            preprocessed = np.reshape(data,(total_evts,dataset_config['SHAPE'][1],-1))
            preprocessed = np.sum(preprocessed,-1)
            #preprocessed = np.mean(preprocessed,0)
            return preprocessed
        
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Mean dep. energy [GeV]', plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_EnergyZ_{}_{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        return feed_dict

    def AverageER(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,3,1,2,4))
            preprocessed = np.reshape(preprocessed,(data.shape[0],dataset_config['SHAPE'][3],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed
            
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        if(dataset_config['cartesian_plot']): 
            xlabel = 'x-bin'
            f_str = "X"
        else: 
            xlabel = 'R-bin'
            f_str = "R"

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel=xlabel, ylabel= 'Mean Energy [GeV]', plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_Energy{}_{}_{}.{}'.format(flags.plot_folder,f_str, dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        return feed_dict
        
    def AverageEPhi(data_dict):

        def _preprocess(data):
            preprocessed = np.transpose(data,(0,2,1,3,4))
            preprocessed = np.reshape(preprocessed,(data.shape[0],dataset_config['SHAPE'][2],-1))
            preprocessed = np.sum(preprocessed,-1)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        if(dataset_config['cartesian_plot']): 
            xlabel = 'y-bin'
            f_str = "Y"
        else: 
            xlabel = 'alpha-bin'
            f_str = "Alpha"


        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel=xlabel, ylabel= 'Mean Energy [GeV]', plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_Energy{}_{}_{}.{}'.format(flags.plot_folder, f_str, dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        return feed_dict

    def HistEtot(data_dict):
        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            return np.sum(preprocessed,-1)

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

            
        binning = np.geomspace(np.quantile(feed_dict['Geant4'],0.01),np.quantile(feed_dict['Geant4'],1.0),20)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Deposited energy [GeV]', logy=True,binning=binning, plot_label = flags.plot_label)
        ax0.set_xscale("log")
        for plt_ext in plt_exts: fig.savefig('{}/FCC_TotalE_{}_{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        return feed_dict
        
    def HistNhits(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            return np.sum(preprocessed>0,-1)
        
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
            
        binning = np.linspace(np.quantile(feed_dict['Geant4'],0.0),np.quantile(feed_dict['Geant4'],1),20)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Number of hits', label_loc='upper right', binning = binning, ratio = True, plot_label = flags.plot_label )
        yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax0.yaxis.set_major_formatter(yScalarFormatter)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_Nhits_{}_{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        return feed_dict

    def HistVoxelE(data_dict):

        def _preprocess(data):
            return np.reshape(data, (-1))
        
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
            
        vmin = np.amin(feed_dict['Geant4'][feed_dict['Geant4'] > 0])
        binning = np.geomspace(vmin,np.quantile(feed_dict['Geant4'],1.0),50)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel='Voxel Energy [GeV]', logy= True, binning = binning, ratio = True, normalize = False, plot_label = flags.plot_label)
        ax0.set_xscale("log")
        for plt_ext in plt_exts: fig.savefig('{}/FCC_VoxelE_{}_{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        return feed_dict


    def HistMaxELayer(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],dataset_config['SHAPE'][1],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig,ax0 = utils.PlotRoutine(feed_dict,xlabel='Layer number', ylabel= 'Max voxel/Dep. energy', plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_MaxEnergyZ_{}_{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        return feed_dict

    def HistMaxE(data_dict):

        def _preprocess(data):
            preprocessed = np.reshape(data,(data.shape[0],-1))
            preprocessed = np.ma.divide(np.max(preprocessed,-1),np.sum(preprocessed,-1)).filled(0)
            return preprocessed


        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        binning = np.linspace(0,1,10)
        fig,ax0 = utils.HistRoutine(feed_dict,xlabel= 'Max. voxel/Dep. energy',binning=binning,logy=True, plot_label = flags.plot_label)
        for plt_ext in plt_exts: fig.savefig('{}/FCC_MaxEnergy_{}_{}.{}'.format(flags.plot_folder,dataset_config['CHECKPOINT_NAME'],flags.model, plt_ext))
        return feed_dict
        



    def plot_shower(shower, fout = "", title = "", vmax = 0, vmin = 0):
        #cmap = plt.get_cmap('PiYG')
        cmap = copy.copy(plt.get_cmap('viridis'))
        cmap.set_bad("white")

        shower[shower==0]=np.nan

        fig,ax = SetFig("x-bin","y-bin")
        if vmax==0:
            vmax = np.nanmax(shower[:,:,0])
            vmin = np.nanmin(shower[:,:,0])
            #print(vmin,vmax)
        im = ax.pcolormesh(range(shower.shape[0]), range(shower.shape[1]), shower[:,:,0], cmap=cmap,vmin=vmin,vmax=vmax)

        yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        #cbar.ax.set_major_formatter(yScalarFormatter)

        cbar=fig.colorbar(im, ax=ax,label='Dep. energy [GeV]',format=yScalarFormatter)
        
        
        bar = ax.set_title(title,fontsize=15)

        if(len(fout) > 0): fig.savefig(fout)
        return vmax, vmin



    def Plot_Shower_2D(data_dict):
        plt.rcParams['pcolor.shading'] ='nearest'
        layer_number = [10,44]



        for layer in layer_number:
            
            def _preprocess(data):
                preprocessed = data[:,layer,:]
                preprocessed = np.mean(preprocessed,0)
                preprocessed[preprocessed==0]=np.nan
                return preprocessed

            vmin=vmax=0
            nShowers = 5
            for ik,key in enumerate(['Geant4',utils.name_translate[flags.model]]):
                average = _preprocess(data_dict[key])

                fout_avg = '{}/FCC_{}2D_{}_{}_{}.{}'.format(flags.plot_folder,key,layer,dataset_config['CHECKPOINT_NAME'],flags.model, plt_exts[0])
                title = "{}, layer number {}".format(key,layer)
                plot_shower(average, fout = fout_avg, title = title)

                for i in range(nShowers):
                    shower = data_dict[key][i,layer]
                    fout_ex = '{}/FCC_{}2D_{}_{}_{}_shower{}.{}'.format(flags.plot_folder,key,layer,dataset_config['CHECKPOINT_NAME'],flags.model, i, plt_exts[0])
                    title = "{} Shower {}, layer number {}".format(key, i, layer)
                    vmax, vmin = plot_shower(shower, fout = fout_ex, title = title, vmax = vmax, vmin = vmin)


            

    do_cart_plots = (not dataset_config['CYLINDRICAL']) and dataset_config['SHAPE_PAD'][-1] == dataset_config['SHAPE_PAD'][-2]
    dataset_config['cartesian_plot'] = do_cart_plots
    print("Do cartesian plots " + str(do_cart_plots))
    high_level = []
    plot_routines = {
         'Energy per layer':AverageELayer,
         'Energy':HistEtot,
         '2D Energy scatter split':ScatterESplit,
         'Energy Ratio split':HistERatio,
         'Nhits':HistNhits,
         'VoxelE':HistVoxelE,
    }

    plot_routines['Shower width']=AverageShowerWidth        
    plot_routines['Max voxel']=HistMaxELayer
    plot_routines['Energy per radius']=AverageER
    plot_routines['Energy per phi']=AverageEPhi
    if(do_cart_plots):
        plot_routines['2D average shower']=Plot_Shower_2D

    print("Saving plots to "  + os.path.abspath(flags.plot_folder) )
    for plot in plot_routines:
        if '2D' in plot and flags.model == 'all':continue #skip scatter plots superimposed
        print(plot)
        if 'split' in plot:
            plot_routines[plot](data_dict,energies)
        else:
            high_level.append(plot_routines[plot](data_dict))
        
