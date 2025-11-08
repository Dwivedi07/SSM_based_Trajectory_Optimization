import os
import sys
import argparse

art_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(root_folder)
sys.path.append(art_path)

'''
#TODO edit the description
root folder = /../transformermpc-ral24/freeflyer/ 
Place your data inside: dataset folder
Place your model weights: root_folder + '/decision_transformer/saved_files/checkpoints/' + model_name
'''

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import copy

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset, SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, get_scheduler
from accelerate import Accelerator

from decision_transformer.art import AutonomousFreeflyerTransformer
from dynamics.freeflyer import FreeflyerModel, check_koz_constraint
from optimization.ff_scenario import obs, safety_margin, robot_radius, table
import time

from mamba.S6modelff import Mamba_Traj
# from mamba.S6modelfffull import Mamba_Traj
from mamba.mamba_ssm.models.config import MambaConfig
from mamba.mamba_ssm.utils.generation import InferenceParams 

'''
subset data: only K timesteps extracted out of the total 100 timsteps
normal data: total 100 timsteps
'''
# normal
data_dir = root_folder + '/dataset'
data_dir_torch = root_folder + '/dataset/torch/v05'

#subset
# data_dir = root_folder + '/dataset_sub'
# data_dir_torch = root_folder + '/dataset_sub'


# select device based on availability of GPU
verbose = False # set to True to get additional print statements
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(device)

'''
    TODO: 
        - normalize:
            1) currently in the normalization without timestep dependency the dimension of "states_mean", "states_std",.... is (100xsize_data) just for portability with precedent version of "torch_inference...." --> TO BE REMOVED IN NEXT VERSIONS
'''


class RpodDataset(Dataset):
    # Create a Dataset object
    def __init__(self, data, mdp_constr, target=False):
        self.data_stats = data['data_stats']
        self.data = data
        self.n_data, self.max_len, self.n_state = self.data['states'].shape
        self.n_action = self.data['actions'].shape[2]
        self.mdp_constr = mdp_constr
        self.target = target

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
        goal = torch.stack([self.data['goal'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_state).float()
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long()
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long()

        time_discr = self.data['data_param']['time_discr'][ix].item()
        time_sec = self.data['data_param']['time_sec'][ix].reshape((1, self.max_len))

        if self.target == False:
            if not self.mdp_constr:
                return states, actions, rtgs, goal, timesteps, attention_mask, time_discr, time_sec, ix
            else:
                ctgs = torch.stack([self.data['ctgs'][i, :]
                            for i in ix]).view(self.max_len, 1).float()
                return states, actions, rtgs, ctgs, goal, timesteps, attention_mask, time_discr, time_sec, ix
        else:
            target_states = torch.stack([self.data['target_states'][i, :, :]
                            for i in ix]).view(self.max_len-1, self.n_state).float()
            target_actions = torch.stack([self.data['target_actions'][i, :, :]
                            for i in ix]).view(self.max_len, self.n_action).float()
            
            if not self.mdp_constr:
                return states, actions, rtgs, goal, target_states, target_actions, timesteps, attention_mask, time_discr, time_sec, ix
            else:
                ctgs = torch.stack([self.data['ctgs'][i, :]
                            for i in ix]).view(self.max_len, 1).float()
                return states, actions, rtgs, ctgs, goal, target_states, target_actions, timesteps, attention_mask, time_discr, time_sec, ix
    
    def getix(self, ix):
        ix = [ix]
        states = torch.stack([self.data['states'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_state).float().unsqueeze(0)
        actions = torch.stack([self.data['actions'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_action).float().unsqueeze(0)
        rtgs = torch.stack([self.data['rtgs'][i, :]
                        for i in ix]).view(self.max_len, 1).float().unsqueeze(0)
        goal = torch.stack([self.data['goal'][i, :, :]
                        for i in ix]).view(self.max_len, self.n_state).float().unsqueeze(0)
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long().unsqueeze(0)
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long().unsqueeze(0)

        time_discr = torch.tensor(self.data['data_param']['time_discr'][ix].item())
        time_sec = torch.tensor(self.data['data_param']['time_sec'][ix].reshape((1, self.max_len))).unsqueeze(0)

        if self.target == False:
            if not self.mdp_constr:
                return states, actions, rtgs, goal, timesteps, attention_mask, time_discr, time_sec, ix
            else:
                ctgs = torch.stack([self.data['ctgs'][i, :]
                            for i in ix]).view(self.max_len, 1).float()
                return states, actions, rtgs, ctgs, goal, timesteps, attention_mask, time_discr, time_sec, ix
        else:
            target_states = torch.stack([self.data['target_states'][i, :, :]
                            for i in ix]).view(self.max_len-1, self.n_state).float().unsqueeze(0)
            target_actions = torch.stack([self.data['target_actions'][i, :, :]
                            for i in ix]).view(self.max_len, self.n_action).float().unsqueeze(0)

            if not self.mdp_constr:
                return states, actions, rtgs, goal, target_states, target_actions, timesteps, attention_mask, time_discr, time_sec, ix
            else:
                ctgs = torch.stack([self.data['ctgs'][i, :]
                            for i in ix]).view(self.max_len, 1).float()
                return states, actions, rtgs, ctgs, goal, target_states, target_actions, timesteps, attention_mask, time_discr, time_sec, ix

    def get_data_size(self):
        return self.n_data


def get_train_val_test_data(mdp_constr, timestep_norm):

    # Import and normalize torch dataset, then save data statistics
    torch_data, data_param = import_dataset_for_DT_eval_vXX(mdp_constr)
    states_norm, states_mean, states_std = normalize(torch_data['torch_states'], timestep_norm)
    actions_norm, actions_mean, actions_std = normalize(torch_data['torch_actions'], timestep_norm)
    goal_norm, goal_mean, goal_std = normalize(torch_data['torch_goal'], timestep_norm)
    target_states_norm = states_norm[:,1:,:].clone().detach()
    target_actions_norm = actions_norm.clone().detach()
    if mdp_constr:
        rtgs_norm, rtgs_mean, rtgs_std = torch_data['torch_rtgs'], None, None
        ctgs_norm, ctgs_mean, ctgs_std = torch_data['torch_ctgs'], None, None
    else:
        rtgs_norm, rtgs_mean, rtgs_std = normalize(torch_data['torch_rtgs'], timestep_norm)
    
    data_stats = {
        'states_mean' : states_mean,
        'states_std' : states_std,
        'actions_mean' : actions_mean,
        'actions_std' : actions_std,
        'rtgs_mean' : rtgs_mean,
        'rtgs_std' : rtgs_std,
        'ctgs_mean' : ctgs_mean if mdp_constr else None,
        'ctgs_std' : ctgs_std if mdp_constr else None,
        'goal_mean' : goal_mean,
        'goal_std' : goal_std
    }

    # Split dataset into training and validation
    n = int(0.9*states_norm.shape[0])
    train_data = {
        'states' : states_norm[:n, :],
        'actions' : actions_norm[:n, :],
        'rtgs' : rtgs_norm[:n, :],
        'ctgs' : ctgs_norm[:n, :] if mdp_constr else None,
        'target_states' : target_states_norm[:n, :],
        'target_actions' : target_actions_norm[:n, :],
        'goal' : goal_norm[:n, :],
        'data_param' : {
            'time_discr' : data_param['time_discr'][:n],
            'time_sec' : data_param['time_sec'][:n, :]
            },
        'data_stats' : data_stats
        }
    val_data = {
        'states' : states_norm[n:, :],
        'actions' : actions_norm[n:, :],
        'rtgs' : rtgs_norm[n:, :],
        'ctgs' : ctgs_norm[n:, :] if mdp_constr else None,
        'target_states' : target_states_norm[n:, :],
        'target_actions' : target_actions_norm[n:, :],
        'goal' : goal_norm[n:, :],
        'data_param' : {
            'time_discr' : data_param['time_discr'][n:],
            'time_sec' : data_param['time_sec'][n:, :]
            },
        'data_stats' : data_stats
        }
    
    # Create datasets
    train_dataset = RpodDataset(train_data, mdp_constr)
    val_dataset = RpodDataset(val_data, mdp_constr)
    test_dataset = RpodDataset(val_data, mdp_constr)
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
            test_dataset, replacement=False, num_samples=5000),
        shuffle=False,
        pin_memory=True,
        batch_size=1,
        num_workers=0,
    )
    dataloaders = (train_loader, eval_loader, test_loader)
    
    return datasets, dataloaders

def import_dataset_for_DT_eval_vXX(mdp_constr):
    # Load the data
    print('Loading data from root/dataset/torch/...', end='')

    # data_dir = root_folder + '/dataset'
    # data_dir_torch = root_folder + '/dataset/torch/v05'
    states_cvx = torch.load(data_dir_torch + '/torch_states_cvx.pth')
    states_scp = torch.load(data_dir_torch + '/torch_states_scp.pth')
    actions_cvx = torch.load(data_dir_torch + '/torch_actions_cvx.pth')
    actions_scp = torch.load(data_dir_torch + '/torch_actions_scp.pth')
    rtgs_cvx = torch.load(data_dir_torch + '/torch_rtgs_cvx.pth')
    rtgs_scp = torch.load(data_dir_torch + '/torch_rtgs_scp.pth')
    ctgs_cvx = torch.load(data_dir_torch + '/torch_ctgs_cvx.pth')
    ctgs_scp = torch.load(data_dir_torch + '/torch_ctgs_scp.pth')
    data_param = np.load(data_dir + '/dataset-ff-v05-param.npz')

    print('Completed, DATA IS NOT SHUFFLED YET.\n')

    # Output dictionary
    if mdp_constr:
        #only scp data
        # perm = np.load(data_dir_torch + '/permutation.npy')
        torch_states = states_scp
        torch_actions = actions_scp
        torch_rtgs = rtgs_scp
        torch_ctgs = ctgs_scp
        torch_goal = torch.tensor(np.repeat(data_param['target_state'][:,None,:], torch_states.shape[1], axis=1))
        data_param = {
            'time_discr' : data_param['dtime'],
            'time_sec' : data_param['time']
        }
        # # both scp and cvx data
        # perm = np.load(data_dir_torch + '/permutation.npy')
        # torch_states = torch.concatenate((states_scp, states_cvx), axis=0)[perm]
        # torch_actions = torch.concatenate((actions_scp, actions_cvx), axis=0)[perm]
        # torch_rtgs = torch.concatenate((rtgs_scp, rtgs_cvx), axis=0)[perm]
        # torch_ctgs = torch.concatenate((ctgs_scp, ctgs_cvx), axis=0)[perm]
        # goal_timeseq = torch.tensor(np.repeat(data_param['target_state'][:,None,:], torch_states.shape[1], axis=1))
        # torch_goal = torch.concatenate((goal_timeseq, goal_timeseq), axis=0)[perm]
        # data_param = {
        #     'time_discr' : np.concatenate((data_param['dtime'], data_param['dtime']), axis=0)[perm],
        #     'time_sec' : np.concatenate((data_param['time'], data_param['time']), axis=0)[perm]
        # }
    else:
        torch_states = states_scp
        torch_actions = actions_scp
        torch_rtgs = rtgs_scp
        torch_ctgs = ctgs_scp
        torch_goal = torch.tensor(np.repeat(data_param['target_state'][:,None,:], torch_states.shape[1], axis=1))
        data_param = {
            'time_discr' : data_param['dtime'],
            'time_sec' : data_param['time']
        }

    torch_data = {
        'torch_states' : torch_states,
        'torch_actions' : torch_actions,
        'torch_rtgs' : torch_rtgs,
        'torch_ctgs' : torch_ctgs,
        'torch_goal' : torch_goal
    }


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

def get_DM_model(model_name, train_loader, eval_loader):
    # load model from the top
    # GSA MODEL
    model = Mamba_Traj(
        state_dim=train_loader.dataset.n_state,     # State Dimension
        act_dim=train_loader.dataset.n_action,
        max_length=100,               # Used when doing tranformer like implementation for inferenceseq_length
        max_ep_len=300,              # total time_steps in one traj sample
        embedding_dim=384,           # The first projection to shared embedding space
        d_model=428,                 # the embedding_dim is used to get voab_size then projected to d_model space
        n_layer=6,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,   # In trained model (checkpointsscp) it was true
        dropout=0,  
        device=device,
        dtype=torch.float32
    )  

    # #RCGSA MODEL
    # model = Mamba_Traj(
    #     state_dim=train_loader.dataset.n_state,     # State Dimension
    #     act_dim=train_loader.dataset.n_action,
    #     max_length=100,               # Used when doing tranformer like implementation for inferenceseq_length
    #     max_ep_len=300,              # total time_steps in one traj sample
    #     embedding_dim=384,           # The first projection to shared embedding space
    #     d_model=450,                 # the embedding_dim is used to get voab_size then projected to d_model space
    #     n_layer=6,
    #     ssm_cfg={},
    #     rms_norm=True,
    #     residual_in_fp32=True,
    #     fused_add_norm=True,   # In trained model (checkpointsscp) it was true
    #     dropout=0,  
    #     device=device,
    #     dtype=torch.float32
    # )   

    model_size = sum(t.numel() for t in model.parameters())
    print(f"S6 Model size: {model_size/1000**2:.1f}M parameters")
    model.to(device);

    # DT optimizer and accelerator
    optimizer = AdamW(model.parameters(), lr=3e-5)
    accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    

    accelerator.load_state(root_folder+'/dm/saved_files/checkpointsscp/'+ model_name)                      # GSA Model
    # accelerator.load_state(root_folder+'/dm/saved_files/checkpointsrcgsa/'+ model_name)                      # RCGSA Model
    model.inference_params = InferenceParams(max_seqlen=300, max_batch_size=1)
    return model.eval()

# someoneelse's model repo
def get_DM_model2(model_name, train_loader, eval_loader):
    
    model = Mamba_Traj(
        state_dim=train_loader.dataset.n_state,     # State Dimension
        act_dim=train_loader.dataset.n_action,
        d_model=384,                 # the embedding_dim is used to get voab_size then projected to d_model space
        n_layer=6,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,   # In trained model (checkpointsscp) it was true
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
    

    accelerator.load_state(root_folder+'/dm/saved_files/checkpointsscp3/'+ model_name)
    model.inference_params = InferenceParams(max_seqlen=300, max_batch_size=1)
    return model.eval()
'''
Transformer like inference
'''
def ssm_model_inference_dyn(model, test_loader, data_sample, rtg_perc=1., ctg_perc=1., rtg=None, ctg_clipped=True):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)
    data_stats['goal_mean'] = data_stats['goal_mean'].float().to(device)
    data_stats['goal_std'] = data_stats['goal_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    if test_loader.dataset.mdp_constr:
        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = data_sample
        ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
    else:
        states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = data_sample
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    goal_i = goal_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    obs_pos, obs_rad = torch.tensor(np.copy(obs['position'])).to(device), torch.tensor(np.copy(obs['radius'])).to(device)
    obs_rad = (obs_rad + robot_radius)*safety_margin
    ff_model = FreeflyerModel()
    Ak, B_imp = torch.tensor(ff_model.Ak).to(device).float(), torch.tensor(ff_model.B_imp).to(device).float()

    # Retrieve decoded states and actions for different inference cases
    xypsi_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    if test_loader.dataset.mdp_constr:
        ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    
    # Pre-allocation for model states or other variables required for initialization 
    # Filling tensors with dummy data to force memory allocation and initialization

    with torch.no_grad():
        dummy_input = torch.zeros_like(states_dyn[:, 0, :])
        dummy_action = torch.zeros_like(actions_dyn[:, 0, :])
        dummy_rtgs = torch.zeros_like(rtgs_dyn[:, 0, :])
        dummy_ctgs = torch.zeros_like(ctgs_dyn[:, 0, :])
        dummy_goal = torch.zeros_like(states_dyn[:, 0, :])
        dummy_timesteps = torch.zeros_like(timesteps_i[:, 0])

        _ = model.get_action_T(
            states=dummy_input,
            actions=dummy_action,
            returns_to_go=dummy_rtgs,
            constraints_to_go=dummy_ctgs,
            goal=dummy_goal,
            timesteps=dummy_timesteps
        )
            
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    if test_loader.dataset.mdp_constr:
        ctgs_dyn[:,0,:] = ctgs_i[:,0,:]*ctg_perc
    xypsi_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
    
    # For loop trajectory generation
    # Initialization with ground truth values for the first K timesteps
    
    runtime0_DT = time.time()
    # print('The first state passed to warmsatrt:',states_dyn[:,0,:])
    for t in np.arange(0, n_time):
        
        ##### Dynamics inference        
        # Compute action pred for dynamics model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_dyn = model.get_action_T(
                                states=states_dyn[:, :t+1, :], 
                                actions=actions_dyn[:, :t+1, :], 
                                returns_to_go=rtgs_dyn[:, :t+1, :], 
                                constraints_to_go=ctgs_dyn[:, :t+1, :], 
                                goal = goal_i[:,:t+1,:],
                                timesteps=timesteps_i[:, :t+1]
                            )
                state_preds_dyn, action_preds_dyn = output_dyn
                
            else:
                output_dyn = model(
                    states=states_dyn[:,:t+1,:],
                    actions=actions_dyn[:,:t+1,:],
                    goal=goal_i[:,:t+1,:],
                    returns_to_go=rtgs_dyn[:,:t+1,:],
                    timesteps=timesteps_i[:,:t+1],
                    attention_mask=attention_mask_i[:,:t+1],
                    return_dict=False,
                )
                (_, action_preds_dyn, _) = output_dyn

        action_dyn_t = action_preds_dyn
        actions_dyn[:,t,:] = action_dyn_t
        dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Dynamics propagation of state variable 
        if t != n_time-1:
            xypsi_dyn[:, t+1] = Ak @ (xypsi_dyn[:, t] + B_imp @ dv_dyn[:, t])
            states_dyn[:,t+1,:] = (xypsi_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
            
            if test_loader.dataset.mdp_constr:
                reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t], ord=1)
                rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - reward_dyn_t
                viol_dyn = torch_check_koz_constraint(xypsi_dyn[:,t+1], obs_pos, obs_rad)
                ctgs_dyn[:,t+1,:] = ctgs_dyn[0,t] - (viol_dyn if (not ctg_clipped) else 0)
            else:
                '''reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t], ord=1)
                rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - (reward_dyn_t/(data_stats['rtgs_std'][t]+1e-6))'''
                rtgs_dyn[:,t+1,:] = rtgs_i[0,t+1]
            actions_dyn[:,t+1,:] = 0

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'xypsi_dyn' : xypsi_dyn.cpu().numpy(),
        'dv_dyn' : dv_dyn.cpu().numpy(),
        'time' : time_sec
    }

    return DT_trajectory, runtime_DT

'''
RNN like inference
'''
def ssm_model_inference_dyn_RNN(model, test_loader, data_sample, rtg_perc=1., ctg_perc=1., rtg=None, ctg_clipped=True):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)
    data_stats['goal_mean'] = data_stats['goal_mean'].float().to(device)
    data_stats['goal_std'] = data_stats['goal_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    if test_loader.dataset.mdp_constr:
        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = data_sample
        ctgs_i = ctgs_i.view(1, n_time, 1).to(device) # probably not needed??
    else:
        states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = data_sample
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    goal_i = goal_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    obs_pos, obs_rad = torch.tensor(np.copy(obs['position'])).to(device), torch.tensor(np.copy(obs['radius'])).to(device)
    obs_rad = (obs_rad + robot_radius)*safety_margin
    ff_model = FreeflyerModel()
    Ak, B_imp = torch.tensor(ff_model.Ak).to(device).float(), torch.tensor(ff_model.B_imp).to(device).float()

    # Retrieve decoded states and actions for different inference cases
    xypsi_dyn = torch.empty(size=(n_state, n_time), device=device).float()
    dv_dyn = torch.empty(size=(n_action, n_time), device=device).float()
    states_dyn = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_dyn = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    if test_loader.dataset.mdp_constr:
        ctgs_dyn = torch.empty(size=(1, n_time, 1), device=device).float()
    
    # Pre-allocation for model states or other variables required for initialization 
    # Filling tensors with dummy data to force memory allocation and initialization

    with torch.no_grad():
        dummy_input = torch.zeros_like(states_dyn[:, 0, :])
        dummy_action = torch.zeros_like(actions_dyn[:, 0, :])
        dummy_rtgs = torch.zeros_like(rtgs_dyn[:, 0, :])
        dummy_ctgs = torch.zeros_like(ctgs_dyn[:, 0, :])
        dummy_goal = torch.zeros_like(states_dyn[:, 0, :])
        dummy_timesteps = torch.zeros_like(timesteps_i[:, 0])

        # Dummy pass to initialize caches and GPU memory    (states, actions, returns_to_go, constraints_to_go, goal, timesteps,**kwg)
        _ = model.get_action_RNN(
            states=dummy_input,
            actions=dummy_action,
            returns_to_go=dummy_rtgs,
            constraints_to_go=dummy_ctgs,
            goal = dummy_goal,
            timesteps=dummy_timesteps
        )


    # runtime0_DT = time.time()
    # Dynamics-in-the-loop initialization
    states_dyn[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_dyn[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_dyn[:,0,:] = rtg
    if test_loader.dataset.mdp_constr:
        ctgs_dyn[:,0,:] = ctgs_i[:,0,:]*ctg_perc
    xypsi_dyn[:, 0] = (states_dyn[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]
    
    # For loop trajectory generation
    # Initialization with ground truth values for the first K timesteps
    n_ground_truth = 1  # Number of timesteps to use ground truth: K = 8 works well
    runtime0_DT = time.time()
    
    for t in np.arange(n_ground_truth-1, n_time):
        
        ##### Dynamics inference        
        # Compute action pred for dynamics model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_dyn = model.get_action_RNN(
                    states=states_dyn[:, t, :], 
                    actions=actions_dyn[:, t, :], 
                    returns_to_go=rtgs_dyn[:, t, :], 
                    constraints_to_go=ctgs_dyn[:, t, :], 
                    goal = goal_i[:,t,:],
                    timesteps=timesteps_i[:, t]
                )
            
                state_preds_dyn, action_preds_dyn = output_dyn
                # output_dyn = model(
                #     states=states_dyn[:,:t+1,:],
                #     actions=actions_dyn[:,:t+1,:],
                #     goal=goal_i[:,:t+1,:],
                #     returns_to_go=rtgs_dyn[:,:t+1,:],
                #     constraints_to_go=ctgs_dyn[:,:t+1,:],
                #     timesteps=timesteps_i[:,:t+1],
                #     attention_mask=attention_mask_i[:,:t+1],
                #     return_dict=False,
                # )
                # (_, action_preds_dyn) = output_dyn
            else:
                output_dyn = model(
                    states=states_dyn[:,:t+1,:],
                    actions=actions_dyn[:,:t+1,:],
                    goal=goal_i[:,:t+1,:],
                    returns_to_go=rtgs_dyn[:,:t+1,:],
                    timesteps=timesteps_i[:,:t+1],
                    attention_mask=attention_mask_i[:,:t+1],
                    return_dict=False,
                )
                (_, action_preds_dyn, _) = output_dyn

        action_dyn_t = action_preds_dyn
        actions_dyn[:,t,:] = action_dyn_t
        dv_dyn[:, t] = (action_dyn_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Dynamics propagation of state variable 
        if t != n_time-1:
            xypsi_dyn[:, t+1] = Ak @ (xypsi_dyn[:, t] + B_imp @ dv_dyn[:, t])
            states_dyn[:,t+1,:] = (xypsi_dyn[:,t+1] - data_stats['states_mean'][t+1]) / (data_stats['states_std'][t+1] + 1e-6)
            
            if test_loader.dataset.mdp_constr:
                reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t], ord=1)
                rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - reward_dyn_t
                viol_dyn = torch_check_koz_constraint(xypsi_dyn[:,t+1], obs_pos, obs_rad)
                ctgs_dyn[:,t+1,:] = ctgs_dyn[0,t] - (viol_dyn if (not ctg_clipped) else 0)
            else:
                '''reward_dyn_t = - torch.linalg.norm(dv_dyn[:, t], ord=1)
                rtgs_dyn[:,t+1,:] = rtgs_dyn[0,t] - (reward_dyn_t/(data_stats['rtgs_std'][t]+1e-6))'''
                rtgs_dyn[:,t+1,:] = rtgs_i[0,t+1]
            actions_dyn[:,t+1,:] = 0

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'xypsi_dyn' : xypsi_dyn.cpu().numpy(),
        'dv_dyn' : dv_dyn.cpu().numpy(),
        'time' : time_sec
    }

    return DT_trajectory, runtime_DT

def torch_check_koz_constraint(states, obs_positions, obs_radii):

    constr_koz = torch.norm(states[None,:2] - obs_positions, 2, dim=1) - obs_radii
    constr_koz_violation = (1*(constr_koz <= 0)).sum().item()

    return constr_koz_violation

def ssm_model_inference_ol(model, test_loader, data_sample, rtg_perc=1., ctg_perc=1., rtg=None, ctg_clipped=True):
    # Get dimensions and statistics from the dataset
    n_state = test_loader.dataset.n_state
    n_time = test_loader.dataset.max_len
    n_action = test_loader.dataset.n_action
    data_stats = copy.deepcopy(test_loader.dataset.data_stats)
    data_stats['states_mean'] = data_stats['states_mean'].float().to(device)
    data_stats['states_std'] = data_stats['states_std'].float().to(device)
    data_stats['actions_mean'] = data_stats['actions_mean'].float().to(device)
    data_stats['actions_std'] = data_stats['actions_std'].float().to(device)
    data_stats['goal_mean'] = data_stats['goal_mean'].float().to(device)
    data_stats['goal_std'] = data_stats['goal_std'].float().to(device)

    # Unnormalize the data sample and compute orbital period (data sample is composed by tensors on the cpu)
    if test_loader.dataset.mdp_constr:
        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = data_sample
        ctgs_i = ctgs_i.view(1, n_time, 1)
    else:
        states_i, actions_i, rtgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = data_sample
    states_i = states_i.to(device)
    rtgs_i = rtgs_i.to(device)
    goal_i = goal_i.to(device)
    timesteps_i = timesteps_i.long().to(device)
    attention_mask_i = attention_mask_i.long().to(device)
    dt = dt.item()
    time_sec = np.array(time_sec[0])
    obs_pos, obs_rad = torch.tensor(np.copy(obs['position'])).to(device), torch.tensor(np.copy(obs['radius'])).to(device)
    obs_rad = (obs_rad + robot_radius)*safety_margin

    # Retrieve decoded states and actions for different inference cases
    xypsi_ol = torch.empty(size=(n_state, n_time), device=device).float()
    dv_ol = torch.empty(size=(n_action, n_time), device=device).float()
    states_ol = torch.empty(size=(1, n_time, n_state), device=device).float()
    actions_ol = torch.zeros(size=(1, n_time, n_action), device=device).float()
    rtgs_ol = torch.empty(size=(1, n_time, 1), device=device).float()
    if test_loader.dataset.mdp_constr:
        ctgs_ol = torch.empty(size=(1, n_time, 1), device=device).float()

    # Pre-allocation for model states or other variables required for initialization 
    # Filling tensors with dummy data to force memory allocation and initialization

    with torch.no_grad():
        dummy_input = torch.zeros_like(states_ol[:, 0, :])
        dummy_action = torch.zeros_like(actions_ol[:, 0, :])
        dummy_rtgs = torch.zeros_like(rtgs_ol[:, 0, :])
        dummy_ctgs = torch.zeros_like(ctgs_ol[:, 0, :])
        dummy_goal = torch.zeros_like(states_ol[:, 0, :])
        dummy_timesteps = torch.zeros_like(timesteps_i[:, 0])

        # Dummy pass to initialize caches and GPU memory    (states, actions, returns_to_go, constraints_to_go, goal, timesteps,**kwg)
        # Start profiling here for autoregressive step
        # with torch.autograd.profiler.profile(
            # use_cuda=torch.cuda.is_available()) as prof:
        # Perform one autoregressive step using the model
        _ = model.get_action_T(
            states=dummy_input,
            actions=dummy_action,
            returns_to_go=dummy_rtgs,
            constraints_to_go=dummy_ctgs,
            goal=dummy_goal,
            timesteps=dummy_timesteps
        )

    runtime0_DT = time.time()
    # Open-loop initialization
    states_ol[:,0,:] = states_i[:,0,:]
    if rtg is None:
        rtgs_ol[:,0,:] = rtgs_i[:,0,:]*rtg_perc
    else:
        rtgs_ol[:,0,:] = rtg
    if test_loader.dataset.mdp_constr:
        ctgs_ol[:,0,:] = ctgs_i[:,0,:]*ctg_perc

    xypsi_ol[:, 0] = (states_ol[:,0,:] * data_stats['states_std'][0]) + data_stats['states_mean'][0]

    # For loop trajectory generation
    for t in np.arange(n_time):

        ##### Open-loop inference
        # Compute action pred for open-loop model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_ol = model.get_action_T(
                                states=states_ol[:, :t+1, :], 
                                actions=actions_ol[:, :t+1, :], 
                                returns_to_go=rtgs_ol[:, :t+1, :], 
                                constraints_to_go=ctgs_ol[:, :t+1, :], 
                                goal = goal_i[:,:t+1,:],
                                timesteps=timesteps_i[:, :t+1]
                            )
                _, action_preds_ol = output_ol
            # else:
            #     output_ol = model(
            #         states=states_ol[:,:t+1,:],
            #         actions=actions_ol[:,:t+1,:],
            #         goal=goal_i[:,:t+1,:],
            #         returns_to_go=rtgs_ol[:,:t+1,:],
            #         timesteps=timesteps_i[:,:t+1],
            #         attention_mask=attention_mask_i[:,:t+1],
            #         return_dict=False,
            #     )
            #     (_, action_preds_ol, _) = output_ol

        action_ol_t = action_preds_ol
        actions_ol[:,t,:] = action_ol_t
        dv_ol[:, t] = (action_ol_t * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]

        # Compute states pred for open-loop model
        with torch.no_grad():
            if test_loader.dataset.mdp_constr:
                output_ol = model.get_action_T(
                                states=states_ol[:, :t+1, :], 
                                actions=actions_ol[:, :t+1, :], 
                                returns_to_go=rtgs_ol[:, :t+1, :], 
                                constraints_to_go=ctgs_ol[:, :t+1, :], 
                                goal = goal_i[:,:t+1,:],
                                timesteps=timesteps_i[:, :t+1]
                            )
                state_preds_ol, _ = output_ol
            # else:
            #     output_ol = model(
            #         states=states_ol[:,:t+1,:],
            #         actions=actions_ol[:,:t+1,:],
            #         goal=goal_i[:,:t+1,:],
            #         returns_to_go=rtgs_ol[:,:t+1,:],
            #         timesteps=timesteps_i[:,:t+1],
            #         attention_mask=attention_mask_i[:,:t+1],
            #         return_dict=False,
            #     )
            #     (state_preds_ol, _, _) = output_ol

        state_ol_t = state_preds_ol

        # Open-loop propagation of state variable
        if t != n_time-1:
            states_ol[:,t+1,:] = state_ol_t
            xypsi_ol[:,t+1] = (state_ol_t * data_stats['states_std'][t+1]) + data_stats['states_mean'][t+1]

            if test_loader.dataset.mdp_constr:
                reward_ol_t = - torch.linalg.norm(dv_ol[:, t], ord=1)
                rtgs_ol[:,t+1,:] = rtgs_ol[0,t] - reward_ol_t
                viol_ol = torch_check_koz_constraint(xypsi_ol[:,t+1], obs_pos, obs_rad)
                ctgs_ol[:,t+1,:] = ctgs_ol[0,t] - (viol_ol if (not ctg_clipped) else 0)
            else:
                rtgs_ol[:,t+1,:] = rtgs_i[0,t+1]
            actions_ol[:,t+1,:] = 0

    # Pack trajectory's data in a dictionary and compute runtime
    runtime1_DT = time.time()
    runtime_DT = runtime1_DT - runtime0_DT
    DT_trajectory = {
        'xypsi_ol' : xypsi_ol.cpu().numpy(),
        'dv_ol' : dv_ol.cpu().numpy(),
        'time' : time_sec
    }

    return DT_trajectory, runtime_DT

def plot_DT_trajectory(DT_trajectory, plot_orb_time = False, savefig = False, plot_dir = ''):
    # Trajectory data extraction
    xypsi_true = DT_trajectory['xypsi_true']
    xypsi_dyn = DT_trajectory['xypsi_dyn']
    xypsi_ol = DT_trajectory['xypsi_ol']
    dv_true = DT_trajectory['dv_true']
    dv_dyn = DT_trajectory['dv_dyn']
    dv_ol = DT_trajectory['dv_ol']
    time_sec = DT_trajectory['time']
    i = 0
    idx_pred = 0
    idx_plt = 0
    
    # position trajectory
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    p01 = ax1.plot(xypsi_true[0,:], xypsi_true[1,:], 'k-', linewidth=1.5, label='true', zorder=3)
    p02 = ax1.plot(xypsi_ol[0,:], xypsi_ol[1,:], 'b-', linewidth=1.5, label='pred o.l.', zorder=3)
    p03 = ax1.plot(xypsi_dyn[0,:], xypsi_dyn[1,:], 'g-', linewidth=1.5, label='pred dyn.', zorder=3)
    p1 = ax1.scatter(xypsi_true[0,0], xypsi_true[1,0], marker = 'o', linewidth=1.5, label='$t_0$', zorder=3)
    p2 = ax1.scatter(xypsi_true[0,-1], xypsi_true[1,-1], marker = '*', linewidth=1.5, label='$t_f$', zorder=3)
    #p3 = ax.scatter(xypsi_true[0,context2.shape[1]//9], xypsi_true[1,context2.shape[1]//9], xypsi_true[2,context2.shape[1]//9], marker = '*', linewidth=1.5, label='$t_{init}$')
    ax1.add_patch(Rectangle((0,0), table['xy_up'][0], table['xy_up'][1], fc=(0.5,0.5,0.5,0.2), ec='k', label='table', zorder=2.5))
    for n_obs in range(obs['radius'].shape[0]):
        label_obs = 'obs' if n_obs == 0 else None
        label_robot = 'robot radius' if n_obs == 0 else None
        ax1.add_patch(Circle(obs['position'][n_obs,:], obs['radius'][n_obs], fc='r', label=label_obs, zorder=2.5))
        ax1.add_patch(Circle(obs['position'][n_obs,:], obs['radius'][n_obs]+robot_radius, fc='r', alpha=0.2, label=label_robot, zorder=2.5))
    ax1.set_xlabel('X [m]', fontsize=10)
    ax1.set_ylabel('Y [m]', fontsize=10)
    ax1.grid(True)
    ax1.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'pos_{idx_plt}.png')
    plt.show()

    plt.figure(figsize=(20,5))
    for j in range(3):
        plt.subplot(1,3,j+1)
        plt.plot(time_sec[0], xypsi_true[j,:], 'k-', linewidth=1.5, label='true')
        plt.plot(time_sec[0], xypsi_ol[j,:], 'b-', linewidth=1.5, label='pred o.l.')
        #plt.vlines(time_sec[0][(context2.shape[1]//9)+1], np.min(xypsi_ol[j,:]), np.max(xypsi_ol[j,:]), label='t_{init}', linewidth=2, color='red')
        plt.plot(time_sec[0], xypsi_dyn[j,:], 'g-', linewidth=1.5, label='pred dyn')
        if j == 0:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \delta r_r$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 1:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$\delta r_t$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 2:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$\delta r_n$ [m]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'rtn_pos_{idx_plt}.png')
    plt.show()

    # velocity vs time
    plt.figure(figsize=(20,5))
    for j in range(3):
        plt.subplot(1,3,j+1)
        plt.plot(time_sec[0], xypsi_true[j+3,:], 'k-', linewidth=1.5, label='true')
        plt.plot(time_sec[0], xypsi_ol[j+3,:], 'b-', linewidth=1.5, label='pred o.l.')
        #plt.vlines(time_sec[0][(context2.shape[1]//9)+1], np.min(xypsi_ol[j+3,:]), np.max(xypsi_ol[j+3,:]), label='t_{init}', linewidth=2, color='red')
        plt.plot(time_sec[0], xypsi_dyn[j+3,:], 'g-', linewidth=1.5, label='pred dyn')
        if j == 0:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \delta v_r$ [m/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 1:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$\delta v_t$ [m/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 2:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$\delta v_n$ [m/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'rtn_vel_{idx_plt}.png')
    plt.show()
    ###### DELTA-V

    # components
    plt.figure(figsize=(20,5))
    for j in range(3):
        plt.subplot(1,3,j+1)
        plt.stem(time_sec[0], dv_true[j,:]*1000., 'k-', label='true')
        plt.stem(time_sec[0], dv_ol[j,:]*1000., 'b-', label='pred o.l.')
        plt.stem(time_sec[0], dv_dyn[j,:]*1000., 'g-', label='pred dyn.')
        #plt.vlines(time_sec[0][(context2.shape[1]//9)+1], np.min(dv_ol[j,:]*1000.), np.max(dv_ol[j,:]*1000.), label='t_{init}', linewidth=2, color='red')
        if j == 0:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \Delta \delta v_r$ [mm/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 1:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \Delta \delta v_t$ [mm/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
        elif j == 2:
            plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
            plt.ylabel('$ \Delta \delta v_n$ [mm/s]', fontsize=10)
            plt.grid(True)
            plt.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'delta_v_{idx_plt}.png')
    plt.show()

    # norm
    plt.figure()
    plt.stem(time_sec[0], la.norm(dv_true*1000., axis=0), 'k-', label='true')
    plt.stem(time_sec[0], la.norm(dv_ol*1000., axis=0), 'b-', label='pred o.l.')
    plt.stem(time_sec[0], la.norm(dv_dyn*1000., axis=0), 'g-', label='pred dyn')
    plt.xlabel('time [orbits]' if plot_orb_time else 'time [steps]', fontsize=10)
    plt.ylabel('$ || \Delta \delta v || $ [mm/s]', fontsize=10)
    plt.grid(True)
    plt.legend(loc='best', fontsize=10)
    if savefig and i==idx_pred:
        plt.savefig(plot_dir + f'delta_v_norm_{idx_plt}.png')
    plt.show()

    '''fig2, ax2 = plt.subplots(3,2,figsize=(20,15))
    ax2[0,0].plot(tt, states[:3,:].T)
    ax2[0,0].grid(True)
    ax2[0,0].set_ylabel('$ x, y, \psi$')
    ax2[0,0].set_xlabel('time [s]')
    ax2[0,1].plot(tt, states[3:,:].T)
    ax2[0,1].grid(True)
    ax2[0,1].set_ylabel('$ \dot{x}, \dot{y}, \dot{\psi}$')
    ax2[0,1].set_xlabel('time [s]')
    if traj_ref is not None:
        ax2[0,1].plot(tt_ref, states_ref[3:,:].T)

    ax2[1,0].stem(tt[:-1], actions[0,:], linefmt='C0', markerfmt='C0o')
    ax2[1,0].stem(tt[:-1], actions[1,:], linefmt='C1', markerfmt='C1o')
    ax2[1,0].stem(tt[:-1], actions[2,:], linefmt='C2', markerfmt='C2o')
    ax2[1,0].grid(True)
    ax2[1,0].set_ylabel('$\Delta V_{G}$')
    ax2[1,0].set_xlabel('time [s]')

    actions_B = (ff.R_BG(states[2,:-1]) @ actions[:,None,:].transpose(2,0,1))[:,:,0].T
    ax2[1,1].stem(tt[:-1], actions_B[0,:], linefmt='C0', markerfmt='C0o')
    ax2[1,1].stem(tt[:-1], actions_B[1,:], linefmt='C1', markerfmt='C1o')
    ax2[1,1].stem(tt[:-1], actions_B[2,:], linefmt='C2', markerfmt='C2o')
    ax2[1,1].grid(True)
    ax2[1,1].set_ylabel('$\Delta V_{B}$')
    ax2[1,1].set_xlabel('time [s]')

    actions_t2 = ff.param['Lambda_inv'] @ actions_B
    sep = np.max(actions_t)*1.1
    ax2[2,0].stem(tt[:-1], 0+actions_t[0,:], linefmt='C0', markerfmt='C0o', bottom=0)
    ax2[2,0].stem(tt[:-1], 2*sep+actions_t[1,:], linefmt='C1', markerfmt='C1o', bottom=2*sep)
    ax2[2,0].stem(tt[:-1], 4*sep+actions_t[2,:], linefmt='C2', markerfmt='C2o', bottom=4*sep)
    ax2[2,0].stem(tt[:-1], 6*sep+actions_t[3,:], linefmt='C3', markerfmt='C3o', bottom=6*sep)
    ax2[2,0].stem(tt[:-1], 0+actions_t2[0,:], linefmt='k', markerfmt='k*', bottom=0)
    ax2[2,0].stem(tt[:-1], 2*sep+actions_t2[1,:], linefmt='k', markerfmt='k*', bottom=2*sep)
    ax2[2,0].stem(tt[:-1], 4*sep+actions_t2[2,:], linefmt='k', markerfmt='k*', bottom=4*sep)
    ax2[2,0].stem(tt[:-1], 6*sep+actions_t2[3,:], linefmt='k', markerfmt='k*', bottom=6*sep)
    ax2[2,0].grid(True)
    ax2[2,0].set_ylabel('$\Delta V_{B}$')
    ax2[2,0].set_xlabel('time [s]')'''
