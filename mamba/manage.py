import os
import sys
import argparse

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
data_dir = root_folder + '/dataset'

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt
import copy

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, get_scheduler
from accelerate import Accelerator

from dynamics.orbit_dynamics import *
from optimization.rpod_scenario import *
from optimization.ocp import *


##############
# from mamba.S6model import DeepMambaModel
# from mamba.S6model2 import DeepMambaModel
# from mamba.S6model3 import DeepMambaModel
from mamba.S6model4 import Mamba_Traj
from mamba.mamba_ssm.models.config import MambaConfig
# from mamba.mamba_ssm.models.config import InferenceParams
from mamba.mamba_ssm.utils.generation import InferenceParams  #MambaTraj one
#############

# select device based on availability of GPU
verbose = False # set to True to get additional print statements
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Check device in manage.py(SSM):', device)

class RpodDataset(Dataset):
    # Create a Dataset object
    def __init__(self, data):
        self.data_stats = data['data_stats']
        self.data = data
        self.n_data, self.max_len, self.n_state = self.data['states'].shape
        self.n_action = self.data['actions'].shape[2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ix = torch.randint(self.n_data, (1,))
        states = torch.stack([self.data['states'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_state).float()
        actions = torch.stack([self.data['actions'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_action).float()
        rtgs = torch.stack([self.data['rtgs'][i, :]
                        for i in ix]).view(self.max_len, 1).float()
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long()
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long()

        horizons = self.data['data_param']['horizons'][ix].item()
        oe = np.transpose(self.data['data_param']['oe'][ix])
        time_discr = self.data['data_param']['time_discr'][ix].item()
        time_sec = self.data['data_param']['time_sec'][ix].reshape((1, self.max_len))
        ctgs = torch.stack([self.data['ctgs'][i, :]
                    for i in ix]).view(self.max_len, 1).float()
        return states, actions, rtgs, ctgs, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix
    
    def getix(self, ix):
        ix = [ix]
        states = torch.stack([self.data['states'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_state).float().unsqueeze(0)
        actions = torch.stack([self.data['actions'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_action).float().unsqueeze(0)
        rtgs = torch.stack([self.data['rtgs'][i, :]
                        for i in ix]).view(self.max_len, 1).float().unsqueeze(0)
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long().unsqueeze(0)
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long().unsqueeze(0)

        horizons = torch.tensor(self.data['data_param']['horizons'][ix].item())
        oe = torch.tensor(np.transpose(self.data['data_param']['oe'][ix])).unsqueeze(0)
        time_discr = torch.tensor(self.data['data_param']['time_discr'][ix].item())
        time_sec = torch.tensor(self.data['data_param']['time_sec'][ix].reshape((1, self.max_len))).unsqueeze(0)
        ctgs = torch.stack([self.data['ctgs'][i, :]
                    for i in ix]).view(self.max_len, 1).float()
        return states, actions, rtgs, ctgs, timesteps, attention_mask, oe, time_discr, time_sec, horizons, ix

    def get_data_size(self):
        return self.n_data

def get_train_val_test_data(state_representation, dataset_to_use, model_name):
    
    # Import and normalize torch dataset, then save data statistics
    torch_data, data_param = import_dataset_for_DT_eval_vXX(model_name)
    states_norm, states_mean, states_std = normalize(torch_data['torch_states'])
    actions_norm, actions_mean, actions_std = normalize(torch_data['torch_actions'])
    rtgs_norm, rtgs_mean, rtgs_std = torch_data['torch_rtgs'], None, None
    ctgs_norm, ctgs_mean, ctgs_std = torch_data['torch_ctgs'], None, None
    
    data_stats = {
        'states_mean' : states_mean,
        'states_std' : states_std,
        'actions_mean' : actions_mean,
        'actions_std' : actions_std,
        'rtgs_mean' : rtgs_mean,
        'rtgs_std' : rtgs_std,
        'ctgs_mean' : ctgs_mean,
        'ctgs_std' : ctgs_std
    }

    # Split dataset into training and validation
    n = int(0.9*states_norm.shape[0])
    train_data = {
        'states' : states_norm[:n, :],
        'actions' : actions_norm[:n, :],
        'rtgs' : rtgs_norm[:n, :],
        'ctgs' : ctgs_norm[:n, :],
        'data_param' : {
            'horizons' : data_param['horizons'][:n],
            'time_discr' : data_param['time_discr'][:n],
            'time_sec' : data_param['time_sec'][:n, :],
            'oe' : data_param['oe'][:n, :]
            },
        'data_stats' : data_stats
        }
    val_data = {
        'states' : states_norm[n:, :],
        'actions' : actions_norm[n:, :],
        'rtgs' : rtgs_norm[n:, :],
        'ctgs' : ctgs_norm[n:, :],
        'data_param' : {
            'horizons' : data_param['horizons'][n:],
            'time_discr' : data_param['time_discr'][n:],
            'time_sec' : data_param['time_sec'][n:, :],
            'oe' : data_param['oe'][n:, :]
            },
        'data_stats' : data_stats
        }
    
    # Create datasets
    train_dataset = RpodDataset(train_data)
    val_dataset = RpodDataset(train_data)
    test_dataset = RpodDataset(val_data)
    datasets = (train_dataset, val_dataset, test_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=4,
        num_workers=0,
    )
    eval_loader = DataLoader(
        val_dataset,
        sampler=torch.utils.data.RandomSampler(
            val_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=4,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=torch.utils.data.RandomSampler(
            test_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=1,
        num_workers=0,
    )
    dataloaders = (train_loader, eval_loader, test_loader)
    
    return datasets, dataloaders

def import_dataset_for_DT_eval_vXX(model_name):
    # Load the data
    print('Loading data from root/dataset', end='')
    # Loading the weights of the model to avoid the terminal warning
    states_cvx = torch.load(data_dir + '/torch_states_rtn_cvx.pth', weights_only=False)
    states_scp = torch.load(data_dir + '/torch_states_rtn_scp.pth', weights_only=False)
    actions_cvx = torch.load(data_dir + '/torch_actions_cvx.pth', weights_only=False)
    actions_scp = torch.load(data_dir + '/torch_actions_scp.pth', weights_only=False)
    rtgs_cvx = torch.load(data_dir + '/torch_rtgs_cvx.pth', weights_only=False)
    rtgs_scp = torch.load(data_dir + '/torch_rtgs_scp.pth', weights_only=False)
    ctgs_cvx = torch.load(data_dir + '/torch_ctgs_cvx.pth', weights_only=False)
    ctgs_scp = torch.load(data_dir + '/torch_ctgs_scp.pth', weights_only=False)

    data_param = np.load(data_dir + '/dataset-rpod-v05-param.npz')

    print('Completed, DATA IS NOT SHUFFLED YET.\n')
    data_param = {
        'horizons' : data_param['horizons'],
        'time_discr' :  data_param['dtime'],
        'time_sec' : data_param['time'],
        'oe' : data_param['oe']
    }

    # Only scp data
    torch_data = {
        'torch_states' : states_scp,
        'torch_actions' : actions_scp,
        'torch_rtgs' : rtgs_scp,
        'torch_ctgs' : ctgs_scp
    }

    # print(data_param['time_discr'].shape, data_param['time_sec'].shape, data_param['horizons'].shape, data_param['oe'].shape)
    return torch_data, data_param

def normalize(data, timestep_norm=False):
    # Normalize and return normalized data, mean and std
    if timestep_norm:
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0)
        data_norm = (data - data_mean)/(data_std + 1e-6)
    else:
        time_length, size_data = data.shape[1:]
        data_mean = torch.ones((time_length, size_data)) * data.view(-1,size_data).mean(dim=0)
        data_std = torch.ones((time_length, size_data)) * data.view(-1,size_data).std(dim=0)
        data_norm = (data - data_mean)/(data_std + 1e-6)

    return data_norm, data_mean, data_std

'''
Initializing the model while inference step
s6model 1/2/3
'''

def get_DT_model(model_name, train_loader, eval_loader):
    # DT model creation
    config = MambaConfig(
        state_dim=train_loader.dataset.n_state,     # State Dimension
        act_dim=train_loader.dataset.n_action,      # Action Dimension
        d_model=512,                                # Hidden dimension size   384/200/240
        n_layers=6,                                 # Number of stacked Mamba blocks
        max_ep_len=300,                             # Maximum Episode length
        d_state=128,                                # State size for Mamba
        d_conv=3,                                   # Convolution kernel size
        expand=2,                                   # Expansion factor
        dt_rank="auto",                             # Time discretization rank
        device=device,  
        dtype=torch.float32,                        # Precision
        )

    model = DeepMambaModel(config)    
    model_size = sum(t.numel() for t in model.parameters())
    print(f"S6 Model size: {model_size/1000**2:.1f}M parameters")
    model.to(device);

    # DT optimizer and accelerator
    optimizer = AdamW(model.parameters(), lr=3e-5)
    accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    

    accelerator.load_state(root_folder + '/mamba/saved_files/checkpoints_S6_l6_d128_dm512_scp/' + model_name)
    # accelerator.load_state(root_folder + '/mamba/saved_files/checkpoints_S64_l6_d128_dm512_test/' + model_name)
    # accelerator.load_state(root_folder + '/mamba/saved_files/checkpoints_S6_l6_d256_dm250_10p4steps/' + model_name)  #best one
    # accelerator.load_state(root_folder + '/mamba/saved_files/checkpoints_S63_l6_d16_dm512/' + model_name)

    return model.eval()

'''
Initializing the model while inference step
s6model4
'''

def get_DT_modelDM(model_name, train_loader, eval_loader):
    
    model = Mamba_Traj(
        state_dim=train_loader.dataset.n_state,     # State Dimension
        act_dim=train_loader.dataset.n_action,
        max_length=100, #seq_length
        max_ep_len=300,
        embedding_dim=384,
        d_model=512,
        n_layer=6,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        dropout=0,
        device=device,
        dtype=torch.float32
    )    
    model_size = sum(t.numel() for t in model.parameters())
    print(f"S6 Model size: {model_size/1000**2:.1f}M parameters")
    model.to(device);

    # DT optimizer and accelerator
    optimizer = AdamW(model.parameters(), lr=3e-5)
    accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    

    accelerator.load_state(root_folder + '/mamba/saved_files/checkpoints_S6De_l4_d128_dm384/' + model_name)
    # accelerator.load_state(root_folder + '/mamba/saved_files/checkpoints_S64_l6_d128_dm512_test/' + model_name)
    # accelerator.load_state(root_folder + '/mamba/saved_files/checkpoints_S6_l6_d256_dm250_10p4steps/' + model_name)  #best one
    # accelerator.load_state(root_folder + '/mamba/saved_files/checkpoints_S63_l6_d16_dm512/' + model_name)
    model.inference_params = InferenceParams(max_seqlen=300, max_batch_size=1)


    return model.eval()


def torch_check_koz_constraint(states_rtn, n_time):

    # Ellipse equation check for a single instant in the trajectory
    constr_koz = (states_rtn[:3]) @ (torch.from_numpy(EE_koz).float().to(device) @ states_rtn[:3])
    if (constr_koz < 1) and (n_time < dock_wyp_sample):
        constr_koz_violation = 1.
    else:
        constr_koz_violation = 0.

    return constr_koz_violation


############################################################################################################
############################# RNN based step Inference Archtecture -S6 ####################################################
'''
S6model works on this whihc uses just the state to predict the next action
v2 model: S6model2/3 where all are concatenated vertically to input
'''

def ssm_model_inference_dyn(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len 
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()

    # Initialize the autoregressive model
    batch_size = states_dyn.shape[0]

    inference_params = InferenceParams(batch_size=batch_size, n_layers= model.config.n_layers)    # model.
    conv_states, ssm_states = model.allocate_inference_cache(batch_size=batch_size, max_seqlen=n_time, dtype=torch.float32)
    
    # Pre-allocation for model states or other variables required for initialization
    # Filling tensors with dummy data to force memory allocation and initialization

    with torch.no_grad():
        dummy_input = torch.zeros_like(states_dyn[:, 0, :])
        dummy_action = torch.zeros_like(actions_dyn[:, 0, :])
        dummy_rtgs = torch.zeros_like(rtgs_dyn[:, 0, :])
        dummy_ctgs = torch.zeros_like(ctgs_dyn[:, 0, :])
        dummy_timesteps = torch.zeros_like(timesteps_i[:, 0])

        # Dummy pass to initialize caches and GPU memory
        _ = model.step(
            states=dummy_input,
            actions=dummy_action,
            returns_to_go=dummy_rtgs,
            constraints_to_go=dummy_ctgs,
            timesteps=dummy_timesteps,
            conv_states=conv_states,
            ssm_states=ssm_states,
            inference_params=inference_params
        )



    runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    
    ctgs_dyn[:,0,:] = ctgs_i[:,0,:]   #*ctg_perc
    if state_representation == 'roe':
        roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]
    

    # For loop trajectory generation using autoregressive step
    for t in np.arange(n_time):
        ##### Dynamics inference using autoregressive step        
        with torch.no_grad():
            
            '''
            For v1 models:
            where I am just using the state vector to get the action vector
            '''


            output_dyn = model.step(
                    states=states_dyn[:, t, :], 
                    actions=actions_dyn[:, t, :], 
                    returns_to_go=rtgs_dyn[:, t, :], 
                    constraints_to_go=ctgs_dyn[:, t, :], 
                    timesteps=timesteps_i[:, t],
                    conv_states = conv_states,
                    ssm_states = ssm_states,
                    inference_params = inference_params
                )
            
            '''
            Using forward methods
            '''
            # states_pred, action_pred = model(
            #         states=states_dyn[:, t, :], 
            #         actions=actions_dyn[:, t, :], 
            #         returns_to_go=rtgs_dyn[:, t, :], 
            #         constraints_to_go=ctgs_dyn[:, t, :], 
            #         timesteps=timesteps_i[:, t]
            #     )
            
            '''
            For v2 models:
            where I am using a stacked vector of all inputs and shifted action vector
            '''
            # if t == 0:
            #     output_dyn = model.step(
            #         states=states_dyn[:, t, :], 
            #         actions=actions_dyn[:, t, :], 
            #         returns_to_go=rtgs_dyn[:, t, :], 
            #         constraints_to_go=ctgs_dyn[:, t, :], 
            #         timesteps=timesteps_i[:, t],
            #         conv_states = conv_states,
            #         ssm_states = ssm_states,
            #         inference_params = inference_params
            #     )
            # else:
            #     output_dyn = model.step(
            #         states=states_dyn[:, t, :], 
            #         actions=actions_dyn[:, t-1, :],        # Shifted action is passed
            #         returns_to_go=rtgs_dyn[:, t, :], 
            #         constraints_to_go=ctgs_dyn[:, t, :], 
            #         timesteps=timesteps_i[:, t],
            #         conv_states = conv_states,
            #         ssm_states = ssm_states,
            #         inference_params = inference_params
            #     )
            
            (action_pred, conv_states, ssm_states) = output_dyn
        
        # Store predicted action
        action_dyn_t = action_pred[0, :]
        actions_dyn[:, t, :] = action_dyn_t
        dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Dynamics propagation of state variable 
        if t != n_time - 1:
            roe_dyn[:, t + 1] = stm[:, :, t] @ (roe_dyn[:, t] + cim[:, :, t] @ dv_dyn[:, t])
            rtn_dyn[:, t + 1] = psi[:, :, t + 1] @ roe_dyn[:, t + 1]
            if state_representation == 'roe':
                states_dyn_norm = (roe_dyn[:, t + 1] - data_stats['states_mean'][t + 1]) / (data_stats['states_std'][t + 1] + 1e-6)
            elif state_representation == 'rtn':
                states_dyn_norm = (rtn_dyn[:, t + 1] - data_stats['states_mean'][t + 1]) / (data_stats['states_std'][t + 1] + 1e-6)
            states_dyn[:, t + 1, :] = states_dyn_norm
            
            reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
            rtgs_dyn[:, t + 1, :] = rtgs_dyn[0, t] - reward_dyn_t
            viol_dyn = torch_check_koz_constraint(rtn_dyn[:, t + 1], t + 1)
            ctgs_dyn[:, t + 1, :] = ctgs_dyn[0, t] - viol_dyn
            actions_dyn[:, t + 1, :] = 0
        
        time_orb[:, t] = time_sec[:, t] / period_ref

    time_orb[:, n_time] = time_orb[:, n_time - 1] + dt / period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_dyn': rtn_dyn.cpu().numpy(),
        'roe_dyn': roe_dyn.cpu().numpy(),
        'dv_dyn': dv_dyn.cpu().numpy(),
        'time_orb': time_orb
    }

    return DT_trajectory, runtime_DT

'''
The below one was to verify my step setup vs getaction setup id correct; Yes it is
Never doubt yourself
'''

def ssm_model_inference_dyn_get_action(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len 
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()

    # Initialize the autoregressive model
    batch_size = states_dyn.shape[0]
    
    inference_params = InferenceParams(batch_size=batch_size, n_layers= model.config.n_layers)    # can also use model.infer...
    # Pre-allocation for model states or other variables required for initialization 
    # Filling tensors with dummy data to force memory allocation and initialization

    with torch.no_grad():
        dummy_input = torch.zeros_like(states_dyn[:, 0, :])
        dummy_action = torch.zeros_like(actions_dyn[:, 0, :])
        dummy_rtgs = torch.zeros_like(rtgs_dyn[:, 0, :])
        dummy_ctgs = torch.zeros_like(ctgs_dyn[:, 0, :])
        dummy_timesteps = torch.zeros_like(timesteps_i[:, 0])

        # Dummy pass to initialize caches and GPU memory
        _ = model.get_action(
            states=dummy_input,
            actions=dummy_action,
            returns_to_go=dummy_rtgs,
            constraints_to_go=dummy_ctgs,
            timesteps=dummy_timesteps,
            inference_params=inference_params
        )

    runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    
    ctgs_dyn[:,0,:] = ctgs_i[:,0,:]   #*ctg_perc
    if state_representation == 'roe':
        roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]
    

    # For loop trajectory generation using autoregressive step
    for t in np.arange(n_time):
        ##### Dynamics inference using autoregressive step 
        print(inference_params.seqlen_offset)       
        with torch.no_grad():
            output_dyn = model.get_action(
                    states=states_dyn[:, t, :], 
                    actions=actions_dyn[:, t, :], 
                    returns_to_go=rtgs_dyn[:, t, :], 
                    constraints_to_go=ctgs_dyn[:, t, :], 
                    timesteps=timesteps_i[:, t],
                    inference_params = inference_params
                )
            action_pred = output_dyn
        
        # Store predicted action
        action_dyn_t = action_pred[0, :]
        actions_dyn[:, t, :] = action_dyn_t
        dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Dynamics propagation of state variable 
        if t != n_time - 1:
            roe_dyn[:, t + 1] = stm[:, :, t] @ (roe_dyn[:, t] + cim[:, :, t] @ dv_dyn[:, t])
            rtn_dyn[:, t + 1] = psi[:, :, t + 1] @ roe_dyn[:, t + 1]
            if state_representation == 'roe':
                states_dyn_norm = (roe_dyn[:, t + 1] - data_stats['states_mean'][t + 1]) / (data_stats['states_std'][t + 1] + 1e-6)
            elif state_representation == 'rtn':
                states_dyn_norm = (rtn_dyn[:, t + 1] - data_stats['states_mean'][t + 1]) / (data_stats['states_std'][t + 1] + 1e-6)
            states_dyn[:, t + 1, :] = states_dyn_norm
            
            reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
            rtgs_dyn[:, t + 1, :] = rtgs_dyn[0, t] - reward_dyn_t
            viol_dyn = torch_check_koz_constraint(rtn_dyn[:, t + 1], t + 1)
            ctgs_dyn[:, t + 1, :] = ctgs_dyn[0, t] - viol_dyn
            actions_dyn[:, t + 1, :] = 0
        
        time_orb[:, t] = time_sec[:, t] / period_ref

    time_orb[:, n_time] = time_orb[:, n_time - 1] + dt / period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_dyn': rtn_dyn.cpu().numpy(),
        'roe_dyn': roe_dyn.cpu().numpy(),
        'dv_dyn': dv_dyn.cpu().numpy(),
        'time_orb': time_orb
    }

    return DT_trajectory, runtime_DT
'''
The following as per the Decision Mamba implementaiton RNN type
'''
def ssm_model_inference_dyn_get_actionRNN(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len 
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()

    # Pre-allocation for model states or other variables required for initialization 
    # Filling tensors with dummy data to force memory allocation and initialization

    with torch.no_grad():
        dummy_input = torch.zeros_like(states_dyn[:, 0, :])
        dummy_action = torch.zeros_like(actions_dyn[:, 0, :])
        dummy_rtgs = torch.zeros_like(rtgs_dyn[:, 0, :])
        dummy_ctgs = torch.zeros_like(ctgs_dyn[:, 0, :])
        dummy_timesteps = torch.zeros_like(timesteps_i[:, 0])

        # Dummy pass to initialize caches and GPU memory
        _ = model.get_action_RNN(
            states=dummy_input,
            actions=dummy_action,
            returns_to_go=dummy_rtgs,
            constraints_to_go=dummy_ctgs,
            timesteps=dummy_timesteps
        )

    runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    
    ctgs_dyn[:,0,:] = ctgs_i[:,0,:]   #*ctg_perc
    if state_representation == 'roe':
        roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]
    

    # For loop trajectory generation using autoregressive step
    for t in np.arange(n_time):
        ##### Dynamics inference using autoregressive step       
        with torch.no_grad():
            output_dyn = model.get_action_RNN(
                    states=states_dyn[:, t, :], 
                    actions=actions_dyn[:, t, :], 
                    returns_to_go=rtgs_dyn[:, t, :], 
                    constraints_to_go=ctgs_dyn[:, t, :], 
                    timesteps=timesteps_i[:, t]
                )
            state_preds, action_pred = output_dyn
        
        # Store predicted action
        action_dyn_t = action_pred[0, :]
        actions_dyn[:, t, :] = action_dyn_t
        dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Dynamics propagation of state variable 
        if t != n_time - 1:
            roe_dyn[:, t + 1] = stm[:, :, t] @ (roe_dyn[:, t] + cim[:, :, t] @ dv_dyn[:, t])
            rtn_dyn[:, t + 1] = psi[:, :, t + 1] @ roe_dyn[:, t + 1]
            if state_representation == 'roe':
                states_dyn_norm = (roe_dyn[:, t + 1] - data_stats['states_mean'][t + 1]) / (data_stats['states_std'][t + 1] + 1e-6)
            elif state_representation == 'rtn':
                states_dyn_norm = (rtn_dyn[:, t + 1] - data_stats['states_mean'][t + 1]) / (data_stats['states_std'][t + 1] + 1e-6)
            states_dyn[:, t + 1, :] = states_dyn_norm
            
            reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
            rtgs_dyn[:, t + 1, :] = rtgs_dyn[0, t] - reward_dyn_t
            viol_dyn = torch_check_koz_constraint(rtn_dyn[:, t + 1], t + 1)
            ctgs_dyn[:, t + 1, :] = ctgs_dyn[0, t] - viol_dyn
            actions_dyn[:, t + 1, :] = 0
        
        time_orb[:, t] = time_sec[:, t] / period_ref

    time_orb[:, n_time] = time_orb[:, n_time - 1] + dt / period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_dyn': rtn_dyn.cpu().numpy(),
        'roe_dyn': roe_dyn.cpu().numpy(),
        'dv_dyn': dv_dyn.cpu().numpy(),
        'time_orb': time_orb
    }

    return DT_trajectory, runtime_DT

'''
The following as per the Decision Mamba implementaiton Transformer type
'''
def ssm_model_inference_dyn_get_actionT(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len 
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()

    # Pre-allocation for model states or other variables required for initialization 
    # Filling tensors with dummy data to force memory allocation and initialization

    with torch.no_grad():
        dummy_input = torch.zeros_like(states_dyn[:, 0, :])
        dummy_action = torch.zeros_like(actions_dyn[:, 0, :])
        dummy_rtgs = torch.zeros_like(rtgs_dyn[:, 0, :])
        dummy_ctgs = torch.zeros_like(ctgs_dyn[:, 0, :])
        dummy_timesteps = torch.zeros_like(timesteps_i[:, 0])

        # Dummy pass to initialize caches and GPU memory
        _ = model.get_action_TWindow(
            states=dummy_input,
            actions=dummy_action,
            returns_to_go=dummy_rtgs,
            constraints_to_go=dummy_ctgs,
            timesteps=dummy_timesteps
        )

    runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    
    ctgs_dyn[:,0,:] = ctgs_i[:,0,:]   #*ctg_perc
    if state_representation == 'roe':
        roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]
    

    # For loop trajectory generation using autoregressive step
    for t in np.arange(n_time):
        ##### Dynamics inference using autoregressive step       
        with torch.no_grad():
            
            output_dyn = model.get_action_TWindow(
                    states=states_dyn[:, :t+1, :], 
                    actions=actions_dyn[:, :t+1, :], 
                    returns_to_go=rtgs_dyn[:, :t+1, :], 
                    constraints_to_go=ctgs_dyn[:, :t+1, :], 
                    timesteps=timesteps_i[:, :t+1]
                )
            
            state_preds, action_pred = output_dyn
        
        # Store predicted action
        action_dyn_t = action_pred
        actions_dyn[:, t, :] = action_dyn_t
        dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Dynamics propagation of state variable 
        if t != n_time-1:
            roe_dyn[:,t+1] = stm[:,:,t] @ (roe_dyn[:,t] + cim[:,:,t] @ dv_dyn[:,t])
            rtn_dyn[:,t+1] = psi[:,:,t+1] @ roe_dyn[:,t+1]
            if state_representation == 'roe':
                states_dyn_norm = (roe_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
            elif state_representation == 'rtn':
                states_dyn_norm = (rtn_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
            states_dyn[:,t+1,:] = states_dyn_norm
            
            reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
            rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - reward_dyn_t
            viol_dyn = torch_check_koz_constraint(rtn_dyn[:,t+1], t+1)
            ctgs_dyn[:,t+1,:] = ctgs_dyn[0,t] - viol_dyn
            actions_dyn[:,t+1,:] = 0
        
        time_orb[:, t] = time_sec[:, t]/period_ref
    time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_dyn' : rtn_dyn.cpu().numpy(),
        'roe_dyn' : roe_dyn.cpu().numpy(),
        'dv_dyn' : dv_dyn.cpu().numpy(),
        'time_orb' : time_orb
    }

    return DT_trajectory, runtime_DT

'''
this is one shot pass of total seq length and corresponding prediction of each time step in paraller inference step
'''

def ssm_model_inference_dyn_eval(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len 
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
    states_i = states_i.to(device)
    actions_i = actions_i.to(device)
    rtgs_i = rtgs_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()

    # Initialize the eval model
    batch_size = states_dyn.shape[0]
    with torch.no_grad():
        dummy_input = torch.zeros_like(states_dyn)
        dummy_action = torch.zeros_like(actions_dyn)
        dummy_rtgs = torch.zeros_like(rtgs_dyn)
        dummy_ctgs = torch.zeros_like(ctgs_dyn)
        dummy_timesteps = torch.zeros_like(timesteps_i)

        # Dummy pass to initialize caches and GPU memory
        _, _ = model(
            states=dummy_input,
            actions=dummy_action,
            returns_to_go=dummy_rtgs,
            constraints_to_go=dummy_ctgs,
            timesteps=dummy_timesteps
        )
    runtime0_DT = time.time()

    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    
    ctgs_dyn[:,0,:] = ctgs_i[:,0,:]   #*ctg_perc
    if state_representation == 'roe':
        roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]
    

    # For one shot trajectory generation
           
    with torch.no_grad():
        state_preds, action_preds = model(
            states=states_i,
            actions=actions_i,
            returns_to_go=rtgs_i,
            constraints_to_go=ctgs_i,
            timesteps=timesteps_i
        )          
            
    for t in range(n_time):
        action_dyn_t = action_preds[: , t, :]
        actions_dyn[:, t, :] = action_dyn_t
        dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]
        
        if state_representation == 'roe':
            roe_dyn[:, t] = (state_preds[:,t,:] * data_stats['states_std'][t]) + data_stats['states_mean'][t]
            rtn_dyn[:, t] = psi[:,:,t] @ roe_dyn[:,t]
        elif state_representation == 'rtn':
            rtn_dyn[:, t] = (state_preds[:,t,:] * data_stats['states_std'][t]) + data_stats['states_mean'][t]
            roe_dyn[:, t] = psi_inv[:,:,t] @ rtn_dyn[:,t]
        
        time_orb[:, t] = time_sec[:, t] / period_ref
        
    
    time_orb[:, n_time] = time_orb[:, n_time - 1] + dt / period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_dyn': rtn_dyn.cpu().numpy(),
        'roe_dyn': roe_dyn.cpu().numpy(),
        'dv_dyn': dv_dyn.cpu().numpy(),
        'time_orb': time_orb
    }

    return DT_trajectory, runtime_DT



def ssm_model_inference_dyn_state(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len 
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()

    # Initialize the autoregressive model
    batch_size = states_dyn.shape[0]
    inference_params = InferenceParams(batch_size=batch_size, n_layers = model.config.n_layers)   # From config file test_loader.dataset.batch_size
    conv_states, ssm_states = model.allocate_inference_cache(batch_size=batch_size, max_seqlen=n_time, dtype=torch.float32)
    
    # Pre-allocation for model states or other variables required for initialization
    # Filling tensors with dummy data to force memory allocation and initialization
    with torch.no_grad():
        dummy_input = torch.zeros_like(states_dyn[:, 0, :])
        dummy_action = torch.zeros_like(actions_dyn[:, 0, :])
        dummy_rtgs = torch.zeros_like(rtgs_dyn[:, 0, :])
        dummy_ctgs = torch.zeros_like(ctgs_dyn[:, 0, :])
        dummy_timesteps = torch.zeros_like(timesteps_i[:, 0])

        # Dummy pass to initialize caches and GPU memory
        _ = model.step(
            states = dummy_input,
            actions=dummy_action,
            returns_to_go=dummy_rtgs,
            constraints_to_go=dummy_ctgs,
            timesteps=dummy_timesteps,
            conv_states=conv_states,
            ssm_states=ssm_states,
            inference_params=inference_params
        )

    runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn = rtgs_i*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    
    ctgs_dyn = ctgs_i  
    actions_i = actions_i.to(device)
    actions_dyn = actions_i           # Setting up the values for all the actions as the valies from the sample
    
    if state_representation == 'roe':
        roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]
    

    # For loop trajectory generation using autoregressive step
    for t in np.arange(100):
        ########################################
        loop_start_time = time.time()
        ########################################
        
        ##### Dynamics inference using autoregressive step        
        with torch.no_grad():
            
            '''
            For v3 models:
            where I am just using the action, reward, constraint vector to get the state vector
            '''
            output_dyn = model.step(
                    states=states_dyn[:, t, :],
                    actions=actions_dyn[:, t, :], 
                    returns_to_go=rtgs_dyn[:, t, :], 
                    constraints_to_go=ctgs_dyn[:, t, :], 
                    timesteps=timesteps_i[:, t],
                    conv_states = conv_states,
                    ssm_states = ssm_states,
                    inference_params = inference_params
                )
                
            
            (state_pred, conv_states, ssm_states) = output_dyn

            # # Print profiling results
            # print(f"Profiling results for time step {t}:")
            # print(prof.key_averages().table(sort_by="cpu_time_total" if not torch.cuda.is_available() else "cuda_time_total"))
        
        ########################################
        runtime1_DT = time.time()
        runtime_DT = runtime1_DT - loop_start_time 
        # print(f'The time taken for {t}th time step inference is: {runtime_DT}')
        ########################################
        
        conv_states, ssm_states = conv_states, ssm_states
        dv_dyn[:, t] = (actions_dyn[:, t, :] * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Dynamics propagation of state variable 
        if t != n_time - 1:
            states_dyn[:, t + 1, :] = (state_pred[0, :] * (data_stats['states_std'][t + 1] + 1e-6)) + data_stats['states_mean'][t + 1] 
            if state_representation == 'roe':
                roe_dyn[:, t+1] = (states_dyn[:,t+1,:] * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]
                rtn_dyn[:, t+1] = psi[:,:,t+1] @ roe_dyn[:,t+1]
            elif state_representation == 'rtn':
                rtn_dyn[:, t+1] = (states_dyn[:,t+1,:] * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]
                roe_dyn[:, t+1] = psi_inv[:,:,t+1] @ rtn_dyn[:,t+1]
                
            reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
            rtgs_dyn[:, t + 1, :] = rtgs_dyn[0, t] - reward_dyn_t
            viol_dyn = torch_check_koz_constraint(rtn_dyn[:, t + 1], t + 1)
            ctgs_dyn[:, t + 1, :] = ctgs_dyn[0, t] - viol_dyn
            
        
        time_orb[:, t] = time_sec[:, t] / period_ref
        ########################################
        runtime2_DT = time.time()
        runtime_DT = runtime2_DT - runtime1_DT
        # print(f'The time taken for {t}th time step MODEL PROPAGATION is: {runtime_DT}')
        ########################################

    time_orb[:, n_time] = time_orb[:, n_time - 1] + dt / period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_dyn': rtn_dyn.cpu().numpy(),
        'roe_dyn': roe_dyn.cpu().numpy(),
        'dv_dyn': dv_dyn.cpu().numpy(),
        'time_orb': time_orb
    }

    return DT_trajectory, runtime_DT


def ssm_model_inference_ol(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
    ctgs_i = ctgs_i.view(1, n_time, 1)
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    horizons = horizons.item()
    oe = np.array(oe[0])
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
    time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
    stm = torch.from_numpy(stm).float().to(device)
    cim = torch.from_numpy(cim).float().to(device)
    psi = torch.from_numpy(psi).float().to(device)
    psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

    # Retrieve decoded states and actions for different inference cases
    roe_ol = torch.empty(size=(n_state, n_time), device=device).float()
    rtn_ol = torch.empty(size=(n_state, n_time), device=device).float()
    dv_ol = torch.empty(size=(n_action, n_time), device=device).float()
    states_ol = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_ol = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_ol = torch.empty(size=(1, n_time, 1), device=device).float()
    ctgs_ol = torch.empty(size=(1, n_time, 1), device=device).float()
    
    runtime0_DT = time.time()
    # Open-loop initialization
    states_ol[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_ol[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_ol[:,0,:] = rtg
    ctgs_ol[:,0,:] = ctgs_i[:,0,:]*ctg_perc

    if state_representation == 'roe':
        roe_ol[:, 0] = (states_ol[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_ol[:, 0] = psi[:,:,0] @ roe_ol[:,0]
    elif state_representation == 'rtn':
        rtn_ol[:, 0] = (states_ol[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_ol[:, 0] = psi_inv[:,:,0] @ rtn_ol[:,0]

    # Initialize the autoregressive model
    model.setup_autoregressive(batch_size=states_ol.shape[0], device=device)
    
    # For loop trajectory generation using autoregressive step
    for t in np.arange(n_time):
        ########################################
        loop_start_time = time.time()
        ########################################
        
        ##### Dynamics inference using autoregressive step        
        with torch.no_grad():
            # Start profiling here for autoregressive step
            # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
                # Perform one autoregressive step using the model
            action_pred = model.autoregressive_step(
                current_state=states_ol[:, t, :], 
                # current_action=actions_ol[:, t, :], 
                # current_rtgs=rtgs_ol[:, t, :], 
                # current_ctgs=ctgs_ol[:, t, :], 
                current_timestep=timesteps_i[:, t]
            )

            action_preds_ol = action_pred

        action_ol_t = action_preds_ol[0,:]
        actions_ol[:,t,:] = action_ol_t
        dv_ol[:, t] = (action_ol_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Compute states pred for open-loop model
        with torch.no_grad():
            states_pred = model.autoregressive_step(
                # current_state=states_ol[:, t, :], 
                current_action=actions_ol[:, t,:],
                # returns_to_go=rtgs_ol[:,:t+1,:],
                # constraints_to_go=ctgs_ol[:,:t+1,:],
                current_timestep=timesteps_i[:, t],
                
            )
            state_preds_ol = states_pred

        state_ol_t = state_preds_ol[0,:]

        # Open-loop propagation of state variable
        if t != n_time-1:
            states_ol[:,t+1,:] = state_ol_t
            if state_representation == 'roe':
                roe_ol[:,t+1] = (state_ol_t * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]
                rtn_ol[:,t+1] = psi[:,:,t+1] @ roe_ol[:,t+1]
            elif state_representation == 'rtn':
                rtn_ol[:,t+1] = (state_ol_t * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]
                roe_ol[:,t+1] = psi_inv[:,:,t+1] @ rtn_ol[:,t+1]

            reward_ol_t = - torch.linalg.norm(dv_ol[:, t])
            rtgs_ol[:,t+1,:] = rtgs_ol[0,t] - reward_ol_t
            viol_ol = torch_check_koz_constraint(rtn_ol[:,t+1], t+1)
            ctgs_ol[:,t+1,:] = ctgs_ol[0,t] - viol_ol
            actions_ol[:,t+1,:] = 0
        
        time_orb[:, t] = time_sec[:, t]/period_ref
    time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'rtn_ol' : rtn_ol.cpu().numpy(),
        'roe_ol' : roe_ol.cpu().numpy(),
        'dv_ol' : dv_ol.cpu().numpy(),
        'time_orb' : time_orb
    }

    return DT_trajectory, runtime_DT