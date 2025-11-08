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


############SSM
# from SSM.s4d import TrajectoryS4dModel 
from SSM.s4 import TrajectoryS4Model 
#############

# select device based on availability of GPU
verbose = False # set to True to get additional print statements
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Check device in manage.py(SSM):', device)
# mod = TrajectoryS4dModel(2,3)
# print(mod)

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
    

def get_DT_model(model_name, train_loader, eval_loader):
    # DT model creation

    state_dim=train_loader.dataset.n_state
    act_dim=train_loader.dataset.n_action
  
    # model = TrajectoryS4dModel(state_dim=state_dim, act_dim=act_dim) #    S4d model - minimal realization
    model = TrajectoryS4Model(state_dim=state_dim, act_dim=act_dim)    # S4/S4D model - full realization
    model_size = sum(t.numel() for t in model.parameters())
    print(f"S4 Model size: {model_size/1000**2:.1f}M parameters")
    model.to(device);

    # DT optimizer and accelerator
    optimizer = AdamW(model.parameters(), lr=3e-5)
    accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    
    accelerator.load_state(root_folder + '/SSM/saved_files/checkpoints_S4D_l12_d512/' + model_name)
    # accelerator.load_state(root_folder + '/SSM/saved_files/checkpoints_S4new/' + model_name)
    # accelerator.load_state(root_folder + '/SSM/saved_files/checkpoints_S4/' + model_name)

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
############################# Transformer Inference Archtecture ####################################################


# def ssm_model_inference_dyn(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
#     # Get dimensions and statistics from the dataset
#     n_state = test_loader.dataset.n_state
#     n_time = test_loader.dataset.max_len
#     n_action = test_loader.dataset.n_action
#     data_stats = copy.deepcopy(test_loader.dataset.data_stats)
#     data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
#     data_stats['states_std'] = data_stats['states_std'].float().to(device)
#     data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
#     data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

#     # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    
#     states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
#     ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
#     states_i = states_i.to(device)
#     rtgs_i = rtgs_i.to(device)
#     timesteps_i = timesteps_i.long().to(device)
#     attention_mask_i = attention_mask_i.long().to(device)
#     horizons = horizons.item()
#     oe = np.array(oe[0])
#     dt = dt.item()
#     time_sec = np.array(time_sec[0])
#     period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
#     time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
#     stm = torch.from_numpy(stm).float().to(device)
#     cim = torch.from_numpy(cim).float().to(device)
#     psi = torch.from_numpy(psi).float().to(device)
#     psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

#     # Retrieve decoded states and actions for different inference cases
#     roe_dyn = torch.empty(size=(n_state, n_time), device=device).float()
#     rtn_dyn = torch.empty(size=(n_state, n_time), device=device).float()
#     dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
#     states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
#     actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
#     rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
#     ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()

#     runtime0_DT = time.time()
#     # Dynamics-in-the-loop initialization
#     states_dyn[:,0,:] = states_i[:,0,:]
#     if rtg is None:
#         rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
#     else:
#         rtgs_dyn[:,0,:] = rtg
#     ctgs_dyn[:,0,:] = ctgs_i[:,0,:]*ctg_perc

#     if state_representation == 'roe':
#         roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
#         rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
#     elif state_representation == 'rtn':
#         rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
#         roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]
    
#     # For loop trajectory generation
#     for t in np.arange(n_time):
#         ########################################
#         loop_start_time = time.time()
#         ########################################
#         # print(f'the action dyn: {actions_dyn[:,t,:]}')
#         ##### Dynamics inference        
#         # Compute action pred for dynamics model
#         with torch.no_grad():
#             # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
#             output_dyn = model(
#                     states=states_dyn[:,:t+1,:],
#                     actions=actions_dyn[:,:t+1,:],
#                     returns_to_go=rtgs_dyn[:,:t+1,:],
#                     constraints_to_go=ctgs_dyn[:,:t+1,:],
#                     timesteps=timesteps_i[:,:t+1]
#                 )
#             # Print profiling results
#             # print(prof.key_averages().table(sort_by="cpu_time_total" if not torch.cuda.is_available() else "cuda_time_total"))    
#             (_, action_preds_dyn) = output_dyn
#         ########################################
#         runtime1_DT = time.time()
#         runtime_DT = runtime1_DT - loop_start_time 
#         print(f'The time taken for {t}th time step inference is: {runtime_DT}')
#         ########################################
#         action_dyn_t = action_preds_dyn[0,t]
#         actions_dyn[:,t,:] = action_dyn_t
#         # print(f'the action dyn after infernce: {actions_dyn[:,t,:]}')
#         dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

#         # Dynamics propagation of state variable 
#         if t != n_time-1:
#             roe_dyn[:,t+1] = stm[:,:,t] @ (roe_dyn[:,t] + cim[:,:,t] @ dv_dyn[:,t])
#             rtn_dyn[:,t+1] = psi[:,:,t+1] @ roe_dyn[:,t+1]
#             if state_representation == 'roe':
#                 states_dyn_norm = (roe_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
#             elif state_representation == 'rtn':
#                 states_dyn_norm = (rtn_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
#             states_dyn[:,t+1,:] = states_dyn_norm
            
#             reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t])
#             rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - reward_dyn_t
#             viol_dyn = torch_check_koz_constraint(rtn_dyn[:,t+1], t+1)
#             ctgs_dyn[:,t+1,:] = ctgs_dyn[0,t] - viol_dyn
#             actions_dyn[:,t+1,:] = 0
        
#         time_orb[:, t] = time_sec[:, t]/period_ref
#         ########################################
#         runtime2_DT = time.time()
#         runtime_DT = runtime2_DT - runtime1_DT
#         # print(f'The time taken for {t}th time step MODEL PROPOGATION is: {runtime_DT}')
#         ########################################

#     time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

#     # Pack trajectory's data in a dictionary and compute runtime
#     runtime1_DT = time.time()
#     runtime_DT = runtime1_DT - runtime0_DT
#     DT_trajectory = {
#         'rtn_dyn' : rtn_dyn.cpu().numpy(),
#         'roe_dyn' : roe_dyn.cpu().numpy(),
#         'dv_dyn' : dv_dyn.cpu().numpy(),
#         'time_orb' : time_orb
#     }

#     return DT_trajectory, runtime_DT


# def ssm_model_inference_ol(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
#     # Get dimensions and statistics from the dataset
#     n_state = test_loader.dataset.n_state
#     n_time = test_loader.dataset.max_len
#     n_action = test_loader.dataset.n_action
#     data_stats = copy.deepcopy(test_loader.dataset.data_stats)
#     data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
#     data_stats['states_std'] = data_stats['states_std'].float().to(device)
#     data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
#     data_stats['actions_std'] = data_stats['actions_std'].float().to(device)

#     # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
#     states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, dt, time_sec, horizons, ix = data_sample
#     ctgs_i = ctgs_i.view(1, n_time, 1)
#     states_i = states_i.to(device)
#     rtgs_i = rtgs_i.to(device)
#     timesteps_i = timesteps_i.long().to(device)
#     attention_mask_i = attention_mask_i.long().to(device)
#     horizons = horizons.item()
#     oe = np.array(oe[0])
#     dt = dt.item()
#     time_sec = np.array(time_sec[0])
#     period_ref = 2*np.pi/np.sqrt(mu_E/oe[0, 0]**3)
#     time_orb = np.zeros(shape=(1, n_time+1), dtype=float)
#     stm = torch.from_numpy(stm).float().to(device)
#     cim = torch.from_numpy(cim).float().to(device)
#     psi = torch.from_numpy(psi).float().to(device)
#     psi_inv = torch.linalg.solve(psi.permute(2,0,1), torch.eye(6, device=device)).permute(1,2,0).to(device)

#     # Retrieve decoded states and actions for different inference cases
#     roe_ol = torch.empty(size=(n_state, n_time), device=device).float()
#     rtn_ol = torch.empty(size=(n_state, n_time), device=device).float()
#     dv_ol = torch.empty(size=(n_action, n_time), device=device).float()
#     states_ol = torch.empty(size=(1, n_time, n_state), device=device).float()
#     actions_ol = torch.zeros(size=(1, n_time, n_action), device=device).float()
#     rtgs_ol = torch.empty(size=(1, n_time, 1), device=device).float()
#     ctgs_ol = torch.empty(size=(1, n_time, 1), device=device).float()
    
#     runtime0_DT = time.time()
#     # Open-loop initialization
#     states_ol[:,0,:] = states_i[:,0,:]
#     if rtg is None:
#         rtgs_ol[:,0,:] = rtgs_i[:,0,:]*rtg_perc
#     else:
#         rtgs_ol[:,0,:] = rtg
#     ctgs_ol[:,0,:] = ctgs_i[:,0,:]*ctg_perc

#     if state_representation == 'roe':
#         roe_ol[:, 0] = (states_ol[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
#         rtn_ol[:, 0] = psi[:,:,0] @ roe_ol[:,0]
#     elif state_representation == 'rtn':
#         rtn_ol[:, 0] = (states_ol[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
#         roe_ol[:, 0] = psi_inv[:,:,0] @ rtn_ol[:,0]

#     # For loop trajectory generation
#     for t in np.arange(n_time):

#         ##### Open-loop inference
#         # Compute action pred for open-loop model
#         with torch.no_grad():
#             output_ol = model(
#                     states=states_ol[:,:t+1,:],
#                     actions=actions_ol[:,:t+1,:],
#                     returns_to_go=rtgs_ol[:,:t+1,:],
#                     constraints_to_go=ctgs_ol[:,:t+1,:],
#                     timesteps=timesteps_i[:,:t+1]
#                 )
#             (_, action_preds_ol) = output_ol

#         action_ol_t = action_preds_ol[0,t]
#         actions_ol[:,t,:] = action_ol_t
#         dv_ol[:, t] = (action_ol_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

#         # Compute states pred for open-loop model
#         with torch.no_grad():
#             output_ol = model(
#                 states=states_ol[:,:t+1,:],
#                 actions=actions_ol[:,:t+1,:],
#                 returns_to_go=rtgs_ol[:,:t+1,:],
#                 constraints_to_go=ctgs_ol[:,:t+1,:],
#                 timesteps=timesteps_i[:,:t+1],
                
#             )
#             (state_preds_ol, _) = output_ol

#         state_ol_t = state_preds_ol[0,t]

#         # Open-loop propagation of state variable
#         if t != n_time-1:
#             states_ol[:,t+1,:] = state_ol_t
#             if state_representation == 'roe':
#                 roe_ol[:,t+1] = (state_ol_t * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]
#                 rtn_ol[:,t+1] = psi[:,:,t+1] @ roe_ol[:,t+1]
#             elif state_representation == 'rtn':
#                 rtn_ol[:,t+1] = (state_ol_t * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]
#                 roe_ol[:,t+1] = psi_inv[:,:,t+1] @ rtn_ol[:,t+1]

#             reward_ol_t = - torch.linalg.norm(dv_ol[:, t])
#             rtgs_ol[:,t+1,:] = rtgs_ol[0,t] - reward_ol_t
#             viol_ol = torch_check_koz_constraint(rtn_ol[:,t+1], t+1)
#             ctgs_ol[:,t+1,:] = ctgs_ol[0,t] - viol_ol
#             actions_ol[:,t+1,:] = 0
        
#         time_orb[:, t] = time_sec[:, t]/period_ref
#     time_orb[:, n_time] = time_orb[:, n_time-1] + dt/period_ref

#     # Pack trajectory's data in a dictionary and compute runtime
#     runtime1_DT = time.time()
#     runtime_DT = runtime1_DT - runtime0_DT
#     DT_trajectory = {
#         'rtn_ol' : rtn_ol.cpu().numpy(),
#         'roe_ol' : roe_ol.cpu().numpy(),
#         'dv_ol' : dv_ol.cpu().numpy(),
#         'time_orb' : time_orb
#     }

#     return DT_trajectory, runtime_DT



############################################################################################################
############################# RNN based step Inference Archtecture -S4/S4D ####################################################

def ssm_model_inference_dyn(model, test_loader, data_sample, stm, cim, psi, state_representation, rtg_perc=1., ctg_perc=1., rtg=None):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len 
    print('The length of passed traj:', n_time)
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
    model.setup_autoregressive(batch_size=states_dyn.shape[0], device=device)
    # Dummy pass to initialize caches and GPU memory
    with torch.no_grad():
        dummy_input = torch.zeros_like(states_dyn[:, 0, :])
        dummy_timesteps = torch.zeros_like(timesteps_i[:, 0])

        
        _ =  model.autoregressive_step(
                    current_state=dummy_input,  
                    current_timestep=dummy_timesteps
                )

    runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    ctgs_dyn[:,0,:] = ctgs_i[:,0,:]*ctg_perc

    if state_representation == 'roe':
        roe_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        rtn_dyn[:, 0] = psi[:,:,0] @ roe_dyn[:,0]
    elif state_representation == 'rtn':
        rtn_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
        roe_dyn[:, 0] = psi_inv[:,:,0] @ rtn_dyn[:,0]
    
   
    
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
                current_state=states_dyn[:, t, :], 
                # current_action=actions_dyn[:, t, :], 
                # current_rtgs=rtgs_dyn[:, t, :], 
                # current_ctgs=ctgs_dyn[:, t, :], 
                current_timestep=timesteps_i[:, t]
            )
            # #Print profiling results
            # print(f"Profiling results for time step {t}:")
            # print(prof.key_averages().table(sort_by="cpu_time_total" if not torch.cuda.is_available() else "cuda_time_total"))
        
        ########################################
        runtime1_DT = time.time()
        runtime_DT = runtime1_DT - loop_start_time 
        # print(f'The time taken for {t}th time step inference is: {runtime_DT}')
        ########################################
        
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