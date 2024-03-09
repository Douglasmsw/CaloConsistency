from tkinter import N
from urllib.parse import _NetlocResultMixinBytes
import numpy as np
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from utils import *
from models import *

class CaloDiffu(nn.Module):
    """Diffusion based generative model"""
    def __init__(self, data_shape, config=None, R_Z_inputs = False, training_obj = 'noise_pred', nsteps = 400,
                    cold_diffu = False, E_bins = None, avg_showers = None, std_showers = None, NN_embed = None,
                    num_sample = None, ode_solver = 'back euler', sigma2 = 0.5, rho = 7., eps=0, time_embed = None,
                    noise_sched = 'ddpm'):
        super(CaloDiffu, self).__init__()
        self._data_shape = data_shape
        self.nvoxels = np.prod(self._data_shape)
        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1
        self.nsteps = nsteps
        self.cold_diffu = cold_diffu
        self.E_bins = E_bins
        self.avg_showers = avg_showers
        self.std_showers = std_showers
        self.training_obj = training_obj
        self.shower_embed = self.config.get('SHOWER_EMBED', '')
        self.fully_connected = ('FCN' in self.shower_embed)
        self.NN_embed = NN_embed
        self._N = num_sample
        self.ode_solver = ode_solver
        self.rho = rho
        self._eps = 0
        self.noise_sched = noise_sched

        if(torch.cuda.is_available()): self.device = torch.device('cuda')
        else: self.device = torch.device('cpu')      

        self._sigma2 = torch.tensor([sigma2]).to(self.device)

        supported = ['noise_pred', 'mean_pred', 'hybrid']
        is_obj = [s in self.training_obj for s in supported]
        if(not any(is_obj)):
            print("Training objective %s not supported!" % self.training_obj)
            exit(1)


        if config is None:
            raise ValueError("Config file not given")
        
        self.verbose = 1

        #Minimum and maximum maximum variance of noise
        self.beta_start = 0.0001
        self.beta_end = config.get("BETA_MAX", 0.02)

        #linear schedule
        schedd = config.get("NOISE_SCHED", "linear")
        self.discrete_time = True

        
        if("linear" in schedd): self.betas = torch.linspace(self.beta_start, self.beta_end, self.nsteps)
        elif("cosine" in schedd): 
            self.betas = cosine_beta_schedule(self.nsteps)
        elif("log" in schedd):
            self.discrete_time = False
            self.P_mean = -1.5
            self.P_std = 1.5
        else:
            print("Invalid NOISE_SCHEDD param %s" % schedd)
            exit(1)

        if(self.discrete_time):
            #precompute useful quantities for training
            self.alphas = 1. - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

            #shift all elements over by inserting unit value in first place
            alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

            self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

            self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.time_embed = config.get("TIME_EMBED", 'sin')
        if time_embed is not None: self.time_embed = time_embed
        
        self.E_embed = config.get("COND_EMBED", 'sin')
        cond_dim = config['COND_SIZE_UNET']
        layer_sizes = config['LAYER_SIZE_UNET']
        block_attn = config.get("BLOCK_ATTN", False)
        mid_attn = config.get("MID_ATTN", False)
        compress_Z = config.get("COMPRESS_Z", False)


        if(self.fully_connected):
            #fully connected network architecture
            self.model = FCN(cond_dim = cond_dim, dim_in = config['SHAPE_ORIG'][1], num_layers = config['NUM_LAYERS_LINEAR'],
                    cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin') )

            self.R_Z_inputs = False

            summary_shape = [[1,config['SHAPE_ORIG'][1]], [1], [1]]


        else:
            RZ_shape = config['SHAPE_PAD'][1:]

            self.R_Z_inputs = config.get('R_Z_INPUT', False)
            self.phi_inputs = config.get('PHI_INPUT', False)

            in_channels = 1

            self.R_image, self.Z_image = create_R_Z_image(self.device, scaled = True, shape = RZ_shape)
            self.phi_image = create_phi_image(self.device, shape = RZ_shape)

            if(self.R_Z_inputs): in_channels = 3

            if(self.phi_inputs): in_channels += 1

            calo_summary_shape = list(copy.copy(RZ_shape))
            calo_summary_shape.insert(0, 1)
            calo_summary_shape[1] = in_channels

            calo_summary_shape[0] = 1
            summary_shape = [calo_summary_shape, [1], [1]]


            self.model = CondUnet(cond_dim = cond_dim, out_dim = 1, channels = in_channels, layer_sizes = layer_sizes, block_attn = block_attn, mid_attn = mid_attn, 
                    cylindrical =  config.get('CYLINDRICAL', False), compress_Z = compress_Z, data_shape = calo_summary_shape,
                    cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin') )

        #print("\n\n Model: \n")
        #summary(self.model, summary_shape)

    #wrapper for backwards compatability
    def load_state_dict(self, d):
        if('noise_predictor' in list(d.keys())[0]):
            d_new = dict()
            for key in d.keys():
                key_new = key.replace('noise_predictor', 'model')
                d_new[key_new] = d[key]
        else: d_new = d

        return super().load_state_dict(d_new)

    def add_RZPhi(self, x):
        cats = [x]
        if(self.R_Z_inputs):

            batch_R_image = self.R_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)
            batch_Z_image = self.Z_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)

            cats+= [batch_R_image, batch_Z_image]
        if(self.phi_inputs):
            batch_phi_image = self.phi_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)

            cats += [batch_phi_image]

        if(len(cats) > 1):
            return torch.cat(cats, axis = 1)
        else: 
            return x
            
    
    def lookup_avg_std_shower(self, inputEs):
        idxs = torch.bucketize(inputEs, self.E_bins)  - 1 #NP indexes bins starting at 1 
        return self.avg_showers[idxs], self.std_showers[idxs]

    
    def noise_image(self, data = None, t = None, noise = None):

        if(noise is None): noise = torch.randn_like(data)

        if(t[0] <=0): return data

        if(self.discrete_time):
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
            out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise
            return out
        else:
            print("non discrete time not supported")
            exit(1)
    
    def cd_noise_image(self, data = None, t = None, noise = None, init_noise_sample = False, d_shape = None):

        if d_shape is None: d_shape = data.shape

        if(noise is None): 
            # generate random noise
            noise = torch.randn(d_shape).to(self.device)
            # scale variance by multiplying noise by t**2
            for _ in range( len(d_shape) - 1 ): t = t.unsqueeze(dim=-1)
            noise *= t**2
        
        # if performing initial sampling step, return pure noise scaled by max timestep
        if init_noise_sample is True: return noise
        
        assert noise.shape == data.shape

        out = data + noise
        return out

    def compute_loss(self, data, energy, noise = None, t = None,
                     loss_type = "l2", rnd_normal = None, energy_loss_scale = 1e-2,
                     ):
        if noise is None:
            noise = torch.randn_like(data)

        if(self.discrete_time): 
            if(t is None): t = torch.randint(0, self.nsteps, (data.size()[0],), device=data.device).long()
            x_noisy = self.noise_image(data, t, noise=noise)
            sigma = None
            sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)**2
        else:
            if(rnd_normal is None): rnd_normal = torch.randn((data.size()[0],), device=data.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            x_noisy = data + torch.reshape(sigma, (data.shape[0], 1,1,1,1)) * noise
            sigma2 = sigma**2


        t_emb = self.do_time_embed(t, self.time_embed, sigma)

        
        pred = self.pred(x_noisy, energy, t_emb)

        weight = 1.
        x0_pred = None
        if('hybrid' in self.training_obj ):

            c_skip = torch.reshape(1. / (sigma2 + 1.), (data.shape[0], 1,1,1,1))
            c_out = torch.reshape(1./ (1. + 1./sigma2).sqrt(), (data.shape[0], 1,1,1,1))
            weight = torch.reshape(1. + (1./ sigma2), (data.shape[0], 1,1,1,1))

            #target = (data - c_skip * x_noisy)/c_out


            x0_pred = pred = c_skip * x_noisy + c_out * pred
            target = data

        elif('noise_pred' in self.training_obj):
            target = noise
            weight = 1.
            if('energy' in self.training_obj): 
                sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
                sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
                x0_pred = (x_noisy - sqrt_one_minus_alphas_cumprod_t * pred)/sqrt_alphas_cumprod_t
        elif('mean_pred' in self.training_obj):
            target = data
            weight = 1./ sigma2
            x0_pred = pred


        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(target, pred)
        elif loss_type == 'l2':
            if('weight' in self.training_obj):
                loss = (weight * ((pred - data) ** 2)).sum() / (torch.mean(weight) * self.nvoxels)
            else:
                loss = torch.nn.functional.mse_loss(target, pred)

        elif loss_type == "huber":
            loss =torch.nn.functional.smooth_l1_loss(target, pred)
        else:
            raise NotImplementedError()

        if('energy' in self.training_obj):
            #sum total energy
            dims = [i for i in range(1,len(data.shape))]
            tot_energy_pred = torch.sum(x0_pred, dim = dims)
            tot_energy_data = torch.sum(data, dim = dims)
            loss_en = energy_loss_scale * torch.nn.functional.mse_loss(tot_energy_data, tot_energy_pred) / self.nvoxels
            loss += loss_en

        return loss

    
    def compute_distillation_loss(self, data, energy, noise = None, n = None, teacher_model = None,
                                    trained_model = None, loss_type = "l2", energy_loss_scale = 1e-2):
        noise = None
        sigma = None

        if self._N is None: raise TypeError("Model attempting to run distillation loss is not a student model")

        # uniformly sample n-values and convert to time steps
        if(n is None): n = torch.FloatTensor(data.shape[0]).uniform_(1, self._N-1)
        
        # perform ode solver step and compute student and teacher predictions
        student_pred, teacher_pred = self.perform_ode_step(data, n, noise, energy, trained_model, 
                                                            teacher_model, sigma, noise_sched = self.noise_sched)

        weight = 1.
        x0_pred = None

        # calculate loss, default to l2 loss
        if loss_type == 'l1':
            loss = weight * torch.nn.functional.l1_loss(student_pred, teacher_pred)
        elif loss_type == 'l2':
            loss = weight * torch.nn.functional.mse_loss(student_pred, teacher_pred)
        elif loss_type == "huber":
            loss = weight * torch.nn.functional.smooth_l1_loss(student_pred, teacher_pred)
        else:
            raise NotImplementedError()

        if('energy' in self.training_obj):
            #sum total energy
            dims = [i for i in range(1,len(data.shape))]
            tot_energy_pred = torch.sum(x0_pred, dim = dims)
            tot_energy_data = torch.sum(data, dim = dims)
            loss_en = energy_loss_scale * torch.nn.functional.mse_loss(tot_energy_data, tot_energy_pred) / self.nvoxels
            loss += loss_en

        return loss
    

    def do_time_embed(self, t = None, embed_type = "identity",  sigma = None,):
        if(embed_type == "identity") or (embed_type == 'sin'):
            return t
        if(self.discrete_time):
            if(embed_type == "scaled"):
                return t/self.nsteps
            if(sigma is None): 
                # identify tensor device so we can match index tensor t
                cumprod_device = self.sqrt_one_minus_alphas_cumprod.device 
                # move index tensor t to indexed tensor device before operation
                sigma = self.sqrt_one_minus_alphas_cumprod[t.to(cumprod_device)] 
            if(embed_type == "sigma"):
                return sigma.to(t.device)
            if(embed_type == "log"):
                return 0.5 * torch.log(sigma).to(t.device)
        else:
            if(embed_type == "log"):
                return 0.5 * torch.log(sigma).to(t.device)
            else:
                return sigma
    
    def calc_concat_coefs(self, t, eps = 0):
        if self._sigma2 is None: raise TypeError("Model attempting to run distillation loss is not a student model") 
        c_skip = self._sigma2 / ( (t - eps)**2 + self._sigma2 )
        c_out = ( torch.sqrt(self._sigma2) * (t - eps) ) / torch.sqrt( self._sigma2 + t**2 )
        return c_skip, c_out

    def pred(self, x, E, t_emb):
        if(self.NN_embed is not None): x = self.NN_embed.enc(x).to(x.device)
        out = self.model(self.add_RZPhi(x), E, t_emb)
        if(self.NN_embed is not None): out = self.NN_embed.dec(out).to(x.device)
        return out

    def pred_cd(self, x, E, t_emb, t):
    
        if(self.NN_embed is not None): x = self.NN_embed.enc(x).to(x.device)
        out = self.model(self.add_RZPhi(x), E, t_emb)
        if(self.NN_embed is not None): 
            x = self.NN_embed.dec(x).to(x.device)
            out = self.NN_embed.dec(out).to(x.device)

        c_skip, c_out = self.calc_concat_coefs(t, eps = 0)
        weighted_output = c_skip[:,None] * x + c_out[:,None] * out
        return weighted_output

    def denoise(self, x, E, t_emb):
        pred = self.pred(x, E, t_emb)
        if('mean_pred' in self.training_obj):
            return pred
        elif('hybrid' in self.training_obj):

            sigma2 = (t_emb**2).reshape(-1,1,1,1,1)
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

            return c_skip * x + c_out * pred


    @torch.no_grad()
    def p_sample(self, x, E, t, cold_noise_scale = 0., noise = None, sample_algo = 'ddpm', debug = False):
        #reverse the diffusion process (one step)

        t_emb = self.do_time_embed(t, self.time_embed)

        if(noise is None): 
            noise = torch.randn(x.shape, device = x.device)
            if(self.cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
                noise = self.gen_cold_image(E, cold_noise_scale, noise)

        if (sample_algo == 'ddpm') or (self.noise_sched == 'ddpm'):
            betas_t = extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)

            if(sample_algo == 'ddpm'): pred = self.pred(x, E, t_emb)
            else: pred = self.pred_cd(x, E, t_emb, t)

            if('noise_pred' in self.training_obj):
                noise_pred = pred
                x0_pred = None
            elif('mean_pred' in self.training_obj):
                x0_pred = pred
                noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t
            elif('hybrid' in self.training_obj):

                sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)**2
                c_skip = 1. / (sigma2 + 1.)
                c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

                x0_pred = c_skip * x + c_out * pred
                noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t
        else:
            # Sampling algo from https://arxiv.org/pdf/2303.01469.pdf
            # Use results from model to predict final output then re-up noise
            post_mean = self.pred_cd(x, E, t_emb, t)
            out = post_mean + torch.sqrt(t**2 - self._eps**2)[:,None] * noise
            if t[0] == 0: out = post_mean
            return out

        if(sample_algo == 'ddpm') or (self.noise_sched == 'ddpm'):
            # Sampling algo from https://arxiv.org/abs/2006.11239
            # Use results from our model (noise predictor) to predict the mean of posterior distribution of prev step
            post_mean = sqrt_recip_alphas_t * ( x - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            if t[0] == 0: return post_mean
            return post_mean + torch.sqrt(posterior_variance_t) * noise 
            #elif (sample_algo == 'ddpm'): return post_mean + torch.sqrt(posterior_variance_t) * noise 
            #else: return post_mean
        else:
            print("Algo %s not supported!" % sample_algo)
            exit(1)

        if(debug): 
            if(x0_pred is None):
                x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t
            return out, x0_pred
        return out

    def gen_cold_image(self, E, cold_noise_scale, noise = None):

        avg_shower, std_shower = self.lookup_avg_std_shower(E)

        if(noise is None):
            noise = torch.randn_like(avg_shower, dtype = torch.float32)

        cold_scales = cold_noise_scale

        return torch.add(avg_shower, cold_scales * (noise * std_shower))

    def perform_ode_step(self, data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched = 'ddpm'):
        if self.noise_sched == 'ddpm': discrete = True
        else: discrete = False

        if self.ode_solver == 'paper_euler': return self.paper_euler_solver(data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched, discrete)
        elif self.ode_solver == 'paper_heun': return self.paper_heun_solver(data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched, discrete)
        elif self.ode_solver == 'back_euler': return self.step_back_euler_solver(data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched, discrete)
        elif self.ode_solver == 'back_direct': return self.direct_step_back_solver(data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched, discrete)
        elif self.ode_solver == 'back_heun': return self.step_back_heun_solver(data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched, discrete)
        else: raise ValueError(f"\nInvalid ODE solver name input: '{self.ode_solver}'; must be one of 'paper_euler', 'paper_heun', 'back_euler', or 'back_heun'\n")

    def paper_euler_solver(self, data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched='ddpm', discrete=False):
            n_plus_1 = n + 1
            t_n_plus_1 = convert_n_to_t(n=n_plus_1, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)
            t_n = convert_n_to_t(n=n, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)

            # noise images and embed timesteps
            if noise_sched == 'ddpm': x_n_plus_1_noisy = self.noise_image(data, t_n_plus_1, noise)
            else: x_n_plus_1_noisy = self.cd_noise_image(data, t_n_plus_1, noise=noise)
            t_emb_n_plus_1 = self.do_time_embed(t_n_plus_1, self.time_embed, sigma)

            # student model prediction on original noised images
            student_pred = self.pred_cd(x_n_plus_1_noisy, energy, t_emb_n_plus_1, t_n_plus_1)

            # predict noise with one ODE step via pre-trained diffusion model
            ode_step_noise = trained_model.pred(x_n_plus_1_noisy, energy, t_emb_n_plus_1)
        
            # euler step
            #adj_traj = x_n_plus_1_noisy - (t_n - t_n_plus_1)[:,None] * t_n_plus_1[:,None] * ode_step_noise
            adj_traj = x_n_plus_1_noisy - (t_n - t_n_plus_1)[:,None] * t_n_plus_1[:,None] * ode_step_noise

            # teacher model prediction on adjascent image along trajectory
            t_emb_n = self.do_time_embed(t_n, self.time_embed, sigma)
            teacher_pred = teacher_model.pred_cd(adj_traj, energy, t_emb_n, t_n)

            return student_pred, teacher_pred

    def paper_heun_solver(self, data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched = 'ddpm', discrete=False):
        n_plus_1 = n + 1
        t_n_plus_1 = convert_n_to_t(n=n_plus_1, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)
        t_n = convert_n_to_t(n=n, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)
        
        # noise images and embed timesteps
        if noise_sched == 'ddpm': x_n_plus_1_noisy = self.noise_image(data, t_n_plus_1, noise)
        else: x_n_plus_1_noisy = self.cd_noise_image(data, t_n_plus_1, noise=noise)
        t_emb_n_plus_1 = self.do_time_embed(t_n_plus_1, self.time_embed, sigma)

        # student model prediction on original noised images
        student_pred = self.pred_cd(x_n_plus_1_noisy, energy, t_emb_n_plus_1, t_n_plus_1)

        # predict noise with one ODE step via pre-trained diffusion model
        ode_step_noise = trained_model.pred(x_n_plus_1_noisy, energy, t_emb_n_plus_1)
    
        # euler step
        adj_traj = x_n_plus_1_noisy - (t_n - t_n_plus_1)[:,None] * t_n_plus_1[:,None] * ode_step_noise

        # second prediction on t_{n-1}
        ode_step_2_noise = trained_model.pred(adj_traj, energy, t_n)

        # heun step
        adj_traj_heun = x_n_plus_1_noisy - 0.5 * (t_n - t_n_plus_1)[:,None] * t_n_plus_1[:,None] * (ode_step_noise + ode_step_2_noise)

        # teacher model prediction on adjascent image along trajectory
        t_emb_n = self.do_time_embed(t_n, self.time_embed, sigma)
        teacher_pred = teacher_model.pred_cd(adj_traj_heun, energy, t_emb_n, t_n)

        return student_pred, teacher_pred

    def step_back_euler_solver(self, data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched = 'ddpm', discrete = False):
        n_plus_1 = n + 1
        t_n_plus_1 = convert_n_to_t(n=n_plus_1, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)
        t_n = convert_n_to_t(n=n, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)

        # noise images and embed timesteps
        if noise_sched == 'ddpm': x_n_plus_1_noisy = self.noise_image(data, t_n_plus_1, noise)
        else: x_n_plus_1_noisy = self.cd_noise_image(data, t_n_plus_1, noise=noise)        
        t_emb_n_plus_1 = self.do_time_embed(t_n_plus_1, self.time_embed, sigma)

        # student model prediction on original noised images
        student_pred = self.pred_cd(x_n_plus_1_noisy, energy, t_emb_n_plus_1, t_n_plus_1)

        # predict noise with one ODE step via pre-trained diffusion model
        ode_step_noise = trained_model.pred(x_n_plus_1_noisy, energy, t_emb_n_plus_1)
    
        # euler step
        adj_traj = x_n_plus_1_noisy - (t_n_plus_1 - t_n)[:,None] * ode_step_noise

        # teacher model prediction on adjascent image along trajectory
        t_emb_n = self.do_time_embed(t_n, self.time_embed, sigma)
        teacher_pred = teacher_model.pred_cd(adj_traj, energy, t_emb_n, t_n)

        return student_pred, teacher_pred

    def direct_step_back_solver(self, data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched = 'ddpm', discrete = False):
        n_plus_1 = n + 1
        t_n_plus_1 = convert_n_to_t(n=n_plus_1, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)
        t_n = convert_n_to_t(n=n, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)

        # noise images and embed timesteps
        if noise_sched == 'ddpm': x_n_plus_1_noisy = self.noise_image(data, t_n_plus_1, noise)
        else: x_n_plus_1_noisy = self.cd_noise_image(data, t_n_plus_1, noise=noise)        
        t_emb_n_plus_1 = self.do_time_embed(t_n_plus_1, self.time_embed, sigma)

        # student model prediction on original noised images
        student_pred = self.pred_cd(x_n_plus_1_noisy, energy, t_emb_n_plus_1, t_n_plus_1)

        # predict noise with one ODE step via pre-trained diffusion model
        ode_step_noise = trained_model.pred(x_n_plus_1_noisy, energy, t_emb_n_plus_1)
    
        # euler step
        betas_t = extract(self.betas, t_n_plus_1, data.shape)
        scaled_betas_t = (t_n_plus_1 - t_n)[:,None] * betas_t
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_n_plus_1, data.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t_n_plus_1, data.shape)

        adj_traj = sqrt_recip_alphas_t * ( x_n_plus_1_noisy - scaled_betas_t * ode_step_noise  / sqrt_one_minus_alphas_cumprod_t)

        # teacher model prediction on adjascent image along trajectory
        t_emb_n = self.do_time_embed(t_n, self.time_embed, sigma)
        teacher_pred = teacher_model.pred_cd(adj_traj, energy, t_emb_n, t_n)

        return student_pred, teacher_pred

    def step_back_heun_solver(self, data, n, noise, energy, trained_model, teacher_model, sigma, noise_sched = 'ddpm', discrete = False):
        n_plus_1 = n + 1
        t_n_plus_1 = convert_n_to_t(n=n_plus_1, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)
        t_n = convert_n_to_t(n=n, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=discrete).to(data.device)
        
        # noise images and embed timesteps
        if noise_sched == 'ddpm': x_n_plus_1_noisy = self.noise_image(data, t_n_plus_1, noise)
        else: x_n_plus_1_noisy = self.cd_noise_image(data, t_n_plus_1, noise=noise)         
        t_emb_n_plus_1 = self.do_time_embed(t_n_plus_1, self.time_embed, sigma)

        # student model prediction on original noised images
        student_pred = self.pred_cd(x_n_plus_1_noisy, energy, t_emb_n_plus_1, t_n_plus_1)

        # predict noise with one ODE step via pre-trained diffusion model
        ode_step_1 = trained_model.pred(x_n_plus_1_noisy, energy, t_emb_n_plus_1)
    
        # euler step
        adj_traj_1 = x_n_plus_1_noisy - (t_n_plus_1 - t_n)[:,None] * ode_step_1

        # second order prediction on t_{n}
        t_emb_n = self.do_time_embed(t_n, self.time_embed, sigma)
        ode_step_2 = trained_model.pred(adj_traj_1, energy, t_emb_n)

        # heun step
        adj_traj_heun = x_n_plus_1_noisy - 0.5 * (t_n_plus_1 - t_n)[:,None] * (ode_step_1 + ode_step_2)

        # teacher model prediction on adjascent image along trajectory
        teacher_pred = teacher_model.pred_cd(adj_traj_heun, energy, t_emb_n, t_n)

        return student_pred, teacher_pred

    @torch.no_grad()
    def Sample(self, E, num_steps = 200, cold_noise_scale = 0., sample_algo = 'ddpm', debug = False, sample_offset = 0, sample_step = 1):
        """Generate samples from diffusion model.
        
        Args:
        E: Energies
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        
        Returns: 
        Samples.
        """

        print("SAMPLE ALGO : %s" % sample_algo)

        # Full sample (all steps)
        device = next(self.parameters()).device


        gen_size = E.shape[0]
        # start from pure noise (for each example in the batch)
        gen_shape = list(copy.copy(self._data_shape))
        gen_shape.insert(0, gen_size)

        #start from pure noise
        if self.noise_sched == 'ddpm': x_start = torch.randn(gen_shape, device=device)
        else:
            t = torch.full((gen_size,), self._N-1)
            x_start = self.cd_noise_image(data = None, t = t, noise = None, init_noise_sample = True, d_shape = gen_shape)

        if (self.cold_diffu): #cold diffu starts using avg images
            x_start = self.gen_cold_image(E, cold_noise_scale)


        start = time.time()


        x = x_start
        fixed_noise = None
        if('fixed' in sample_algo): 
            print("Fixing noise to constant for sampling!")
            fixed_noise = x_start
        xs = []
        x0s = []
        self.prev_noise = x_start

        if (sample_algo == 'ddpm'):        
            time_steps = list(range(0, num_steps - sample_offset, sample_step))
            time_steps.reverse()

            for time_step in time_steps:      
                times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
                out = self.p_sample(x, E, times, noise = fixed_noise, cold_noise_scale = cold_noise_scale, sample_algo = sample_algo, debug = debug)
                if(debug): 
                    x, x0_pred = out
                    xs.append(x.detach().cpu().numpy())
                    x0s.append(x0_pred.detach().cpu().numpy())
                else: x = out
        elif (sample_algo == 'cd') or (sample_algo == 'cd_stdz'):
            t = torch.full((gen_size,), self._N)
            n_range = torch.flip( torch.arange(1, self._N), dims=(-1,) )
            for n in n_range:
                n_vals = torch.full((gen_size,), n, device=device)
                if self.noise_sched == 'ddpm': t = convert_n_to_t(n=n_vals, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=True).to(self.device)
                else: t = convert_n_to_t(n=n_vals, N=self._N, T=self.nsteps, rho=self.rho, eps=0, d=1., discrete=False).to(self.device)
                x = self.p_sample(x, E, t, noise = fixed_noise, cold_noise_scale = cold_noise_scale, sample_algo = sample_algo, debug = debug)

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(gen_size,end - start), flush=True)
        if(debug):
            return x.detach().cpu().numpy(), xs, x0s
        else:   
            return x.detach().cpu().numpy()

    
        
