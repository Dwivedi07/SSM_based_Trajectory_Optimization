import os
import sys
import argparse
import random
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

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

# Define paths
data_dir_torch = root_folder + '/' + "dataset/torch/v05"  # Update with your dataset directory
data_dir_np = root_folder + '/' + "dataset"

# Load dataset from .pth files
states_cvx = torch.load(data_dir_torch + '/torch_states_cvx.pth')  # [400000, 100, state_dim]
states_scp = torch.load(data_dir_torch + '/torch_states_scp.pth')  # [400000, 100, state_dim]
actions_cvx = torch.load(data_dir_torch + '/torch_actions_cvx.pth')  # [400000, 100, action_dim]
actions_scp = torch.load(data_dir_torch + '/torch_actions_scp.pth')  # [400000, 100, action_dim]
rtgs_cvx = torch.load(data_dir_torch + '/torch_rtgs_cvx.pth')  # [400000, 100, reward_dim]
rtgs_scp = torch.load(data_dir_torch + '/torch_rtgs_scp.pth')  # [400000, 100, reward_dim]
ctgs_cvx = torch.load(data_dir_torch + '/torch_ctgs_cvx.pth')  # [400000, 100, constraint_dim]
ctgs_scp = torch.load(data_dir_torch + '/torch_ctgs_scp.pth')  # [400000, 100, constraint_dim]

# Load dataset parameters from .npz
data_param = np.load(data_dir_np + '/dataset-ff-v05-param.npz')
target_state = data_param['target_state']  # [400000, 6]
dtime = data_param['dtime']  # [400000, 1]
time = data_param['time']  # [400000, 100]


print("Loaded dataset with 12-timestep-long sequences!")
# Dataset properties
N, T, state_dim = states_cvx.shape  # Should be [400000, 100, state_dim]
_, _, action_dim = actions_cvx.shape
subset_length = 12  # Length of each extracted sequence
num_samples = 400000  # Define how many subsets to extract

# Storage for new dataset
sampled_states_cvx, sampled_states_scp = [], []
sampled_actions_cvx, sampled_actions_scp = [], []
sampled_rtgs_cvx, sampled_rtgs_scp = [], []
sampled_ctgs_cvx, sampled_ctgs_scp = [], []
sampled_target_states, sampled_dtime, sampled_time = [], [], []

# Generate random 12-timestep subsets
for _ in range(num_samples):
    traj_idx = random.randint(0, N - 1)  # Select a random trajectory
    t_start = random.randint(0, T - subset_length)  # Select a random start time

    # Extract 12-timestep subsets
    sampled_states_cvx.append(states_cvx[traj_idx, t_start:t_start + subset_length, :])
    sampled_states_scp.append(states_scp[traj_idx, t_start:t_start + subset_length, :])
    
    sampled_actions_cvx.append(actions_cvx[traj_idx, t_start:t_start + subset_length, :])
    sampled_actions_scp.append(actions_scp[traj_idx, t_start:t_start + subset_length, :])
    
    sampled_rtgs_cvx.append(rtgs_cvx[traj_idx, t_start:t_start + subset_length])
    sampled_rtgs_scp.append(rtgs_scp[traj_idx, t_start:t_start + subset_length])
    
    sampled_ctgs_cvx.append(ctgs_cvx[traj_idx, t_start:t_start + subset_length])
    sampled_ctgs_scp.append(ctgs_scp[traj_idx, t_start:t_start + subset_length])
    
    sampled_target_states.append(target_state[traj_idx])  # Target state remains the same
    sampled_dtime.append(dtime[traj_idx])  # Same dtime for subset
    sampled_time.append(time[traj_idx, t_start:t_start + subset_length])  # Extract subset of time

# Convert lists to tensors
sampled_states_cvx = torch.stack(sampled_states_cvx)
sampled_states_scp = torch.stack(sampled_states_scp)

sampled_actions_cvx = torch.stack(sampled_actions_cvx)
sampled_actions_scp = torch.stack(sampled_actions_scp)

sampled_rtgs_cvx = torch.stack(sampled_rtgs_cvx)
sampled_rtgs_scp = torch.stack(sampled_rtgs_scp)

sampled_ctgs_cvx = torch.stack(sampled_ctgs_cvx)
sampled_ctgs_scp = torch.stack(sampled_ctgs_scp)

sampled_target_states = np.array(sampled_target_states)
sampled_dtime = np.array(sampled_dtime)
sampled_time = np.array(sampled_time)

# Save new dataset
save_dir = root_folder + '/' + "dataset_sub"  # Update with your save path

torch.save(sampled_states_cvx, save_dir + '/torch_states_cvx.pth')
torch.save(sampled_states_scp, save_dir + '/torch_states_scp.pth')

torch.save(sampled_actions_cvx, save_dir + '/torch_actions_cvx.pth')
torch.save(sampled_actions_scp, save_dir + '/torch_actions_scp.pth')

torch.save(sampled_rtgs_cvx, save_dir + '/torch_rtgs_cvx.pth')
torch.save(sampled_rtgs_scp, save_dir + '/torch_rtgs_scp.pth')

torch.save(sampled_ctgs_cvx, save_dir + '/torch_ctgs_cvx.pth')
torch.save(sampled_ctgs_scp, save_dir + '/torch_ctgs_scp.pth')

np.savez_compressed(save_dir + '/dataset-ff-v05-param.npz',
                    target_state=sampled_target_states,
                    dtime=sampled_dtime,
                    time=sampled_time)

# Permutation
if states_cvx.shape[0] != states_scp.shape[0]:
    raise RuntimeError('Different dimensions of cvx and scp datasets.')
perm2 = np.random.permutation(states_cvx.shape[0]*2)
np.save(save_dir + '/permutation.npy', perm2)
perm = np.random.permutation(states_scp.shape[0])
np.save(save_dir + '/permutation_scp.npy', perm)

print("Saved new dataset with 12-timestep-long sequences!")
