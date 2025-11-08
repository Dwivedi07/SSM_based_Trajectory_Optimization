import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import time
import torch
import torch.utils.checkpoint
from torch import nn

import os
import sys
import argparse
import csv

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

#################################################
from mamba.S6model import DeepMambaModel
from mamba.mamba_ssm.models.config import MambaConfig
#################################################

from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, get_scheduler
from accelerate import Accelerator
from transformer.art import AutonomousRendezvousTransformer

from dynamics.orbit_dynamics import *
from optimization.rpod_scenario import *
from optimization.ocp import *


parser = argparse.ArgumentParser(description='transformer-rpod')
parser.add_argument('--data_dir', type=str, default='dataset',
                    help='defines directory from where to load files')
args = parser.parse_args()
args.data_dir = root_folder + '/' + args.data_dir

# select device based on availability of GPU
verbose = False # set to True to get additional print statements
use_lr_scheduler = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"Running on device: {device}\n")

# Simulation configuration

state_representation = 'rtn'

# load the data 
print('Loading data...', end='')

######### combines both scp and cvx part ######### 

# if state_representation == 'roe':
#     torch_states = torch.cat((
#         torch.load(args.data_dir + '/torch_states_roe_scp.pth', weights_only=False),
#         torch.load(args.data_dir + '/torch_states_roe_cvx.pth', weights_only=False)
#     ), 0)
# else:
#     torch_states = torch.cat((
#         torch.load(args.data_dir + '/torch_states_rtn_scp.pth', weights_only=False),
#         torch.load(args.data_dir + '/torch_states_rtn_cvx.pth', weights_only=False)
#     ), 0)

# torch_actions = torch.cat((
#     torch.load(args.data_dir + '/torch_actions_scp.pth', weights_only=False),
#     torch.load(args.data_dir + '/torch_actions_cvx.pth', weights_only=False)
# ), 0)

# torch_rtgs = torch.cat((
#     torch.load(args.data_dir + '/torch_rtgs_scp.pth', weights_only=False),
#     torch.load(args.data_dir + '/torch_rtgs_cvx.pth', weights_only=False)
# ), 0)

# n_data = torch_states.size(dim=0)
# n_time = torch_states.size(dim=1)  # this excludes the target state time
# n_state = torch_states.size(dim=2)
# n_action = torch_actions.size(dim=2)
# n_reward = 1

# n_constraint = 1
# torch_ctgs = torch.cat((
#     torch.load(args.data_dir + '/torch_ctgs_scp.pth', weights_only=False),
#     torch.load(args.data_dir + '/torch_ctgs_cvx.pth', weights_only=False)
# ), 0)

# n_mdp = n_state + n_action + n_reward + n_constraint

######### combines both scp and cvx part ######### 

############## only scp part ###############

if state_representation == 'roe':
    torch_states = torch.load(args.data_dir + '/torch_states_roe_scp.pth', weights_only=False)
else:
    torch_states = torch.load(args.data_dir + '/torch_states_rtn_scp.pth', weights_only=False)

torch_actions = torch.load(args.data_dir + '/torch_actions_scp.pth', weights_only=False)

torch_rtgs = torch.load(args.data_dir + '/torch_rtgs_scp.pth', weights_only=False)

n_data = torch_states.size(dim=0)
n_time = torch_states.size(dim=1)  # this excludes the target state time
n_state = torch_states.size(dim=2)
n_action = torch_actions.size(dim=2)
n_reward = 1

n_constraint = 1
torch_ctgs = torch.load(args.data_dir + '/torch_ctgs_scp.pth', weights_only=False)

n_mdp = n_state + n_action + n_reward + n_constraint

############## only scp part ###############
print('Completed\n')

# Normalize data
states_mean = torch_states.mean(dim=0)
states_std = (torch_states.std(dim=0) + 1e-6)

actions_mean = torch_actions.mean(dim=0)
actions_std = (torch_actions.std(dim=0) + 1e-6)

rtgs_mean = torch_rtgs.mean(dim=0)
rtgs_std = (torch_rtgs.std(dim=0) + 1e-6)

ctgs_mean = torch_ctgs.mean(dim=0)
ctgs_std = (torch_ctgs.std(dim=0) + 1e-6)

states_norm = ((torch_states - states_mean) / (states_std + 1e-6))
actions_norm = ((torch_actions - actions_mean) / (actions_std + 1e-6))

rtgs_norm = ((torch_rtgs - rtgs_mean) / (rtgs_std + 1e-6))
ctgs_norm = ((torch_ctgs - ctgs_mean) / (ctgs_std + 1e-6))

# Separate dataset in train and val data
n = int(0.9*n_data)
train_data = {'states':states_norm[:n, :], 'actions':actions_norm[:n, :], 'rtgs':rtgs_norm[:n, :], 'ctgs':ctgs_norm[:n, :]}
val_data = {'states':states_norm[n:, :], 'actions':actions_norm[n:, :], 'rtgs':rtgs_norm[n:, :], 'ctgs':ctgs_norm[n:, :]}

# RPDO data class
class RpodDataset(Dataset):
    # Create a Dataset object
    def __init__(self, split):
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.data = train_data if (self.split == 'train') or (self.split == 'val') else val_data
        self.n_data = len(self.data['states'])
        self.max_len = self.data['states'].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ix = torch.randint(self.n_data, (1,))
        states = torch.stack([self.data['states'][i, :, :]
                        for i in ix]).view(self.max_len, n_state).float()
        actions = torch.stack([self.data['actions'][i, :, :]
                        for i in ix]).view(self.max_len, n_action).float()
        rtgs = torch.stack([self.data['rtgs'][i, :]
                        for i in ix]).view(self.max_len, 1).float()
        ctgs = torch.stack([self.data['ctgs'][i, :]
                        for i in ix]).view(self.max_len, 1).float()
        timesteps = torch.tensor([[i for i in range(self.max_len)] for _ in ix]).view(self.max_len).long()
        attention_mask = torch.ones(1, self.max_len).view(self.max_len).long()
        return states, actions, rtgs, ctgs, timesteps, attention_mask, ix

    def get_data_size(self):
        return self.n_data

# Initialize dataset objects
train_dataset = RpodDataset('train')
test_dataset = RpodDataset('val')
states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, ix = train_dataset[0]

if verbose:
    print("states:", states_i.shape)
    print("actions:", actions_i.shape)
    print("rtgs:", rtgs_i.shape)
    print("ctgs:", rtgs_i.shape)
    print("timesteps:", timesteps_i.shape)
    print("attention_mask:", attention_mask_i.shape)

# create a DataLoader object for both train and test
train_loader = DataLoader(
    train_dataset,
    sampler=torch.utils.data.RandomSampler(
        train_dataset, replacement=True, num_samples=int(1e10)),
    shuffle=False,
    pin_memory=True,
    batch_size=32,
    num_workers=0,
)

eval_loader = DataLoader(
    test_dataset,
    sampler=torch.utils.data.RandomSampler(
        test_dataset, replacement=True, num_samples=int(1e10)),
    shuffle=False,
    pin_memory=True,
    batch_size=32,
    num_workers=0,
)

config = MambaConfig(
    state_dim=n_state,  # State Dimension
    act_dim=n_action,   # Action Dimension
    d_model=512,     # Hidden dimension size 384/250/128
    n_layers=6,     # Number of stacked Mamba blocks  6/3
    max_ep_len=300, # Maximum Episode length  # for all models it is 300 and for one 500 longs sequence
    d_state=128,     # State size for Mamba - Maximum is 256
    d_conv=3,       # Convolution kernel size  3/4
    expand=2,       # Expansion factor
    dt_rank="auto", # Time discretization rank
    device=device,  
    dtype=torch.float32,  # Precision
)


print('Intializing MAMBA Model\n')
model = DeepMambaModel(config) 
model_size = sum(t.numel() for t in model.parameters())
print(f"SSM size: {model_size/1000**2:.1f}M parameters")
print(model)
model.to(device);
optimizer = AdamW(model.parameters(), lr=7e-6)   # lr = 3e-5 
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)

# for now this is unused. Potentially we can implement learning rate schedules
num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
print(num_update_steps_per_epoch,'This is the length of train_dataLoader')
if use_lr_scheduler:
    num_training_steps = 12000
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=num_training_steps,
    )

# Eval function to plot results during training
eval_iters = 100
@torch.no_grad()  # dont compute the gradient during evaluation phase
def evaluate():
    model.eval()
    losses = []
    losses_state = []
    losses_action = []
    for step in range(eval_iters):
        data_iter = iter(eval_dataloader)
        states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, ix = next(data_iter)
        with torch.no_grad():
            state_preds, action_preds = model(
                states=states_i,
                actions=actions_i,
                returns_to_go=rtgs_i,
                constraints_to_go=ctgs_i,
                timesteps=timesteps_i
            )
        loss_i = torch.mean((action_preds - actions_i) ** 2)
        loss_i_state = torch.mean((state_preds[:,:-1,:] - states_i[:,1:,:]) ** 2)
        losses.append(accelerator.gather(loss_i + loss_i_state))
        losses_state.append(accelerator.gather(loss_i_state))
        losses_action.append(accelerator.gather(loss_i))
    loss = torch.mean(torch.tensor(losses))
    loss_state = torch.mean(torch.tensor(losses_state))
    loss_action = torch.mean(torch.tensor(losses_action))
    model.train()
    return loss.item(), loss_state.item(), loss_action.item()

print('\n======================')
print('Initializing training\n')
start_time_train = time.time()
# Training loop
eval_steps = 500
samples_per_step = accelerator.state.num_processes * train_loader.batch_size

model.train()
completed_steps = 0


# Initialize a list to store training losses
train_losses = []
eval_losses = []
eval_state_loss = []
eval_action_loss = []
# gradient_norms = []

for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_dataloader, start=0):
        if step>= 10000: 
            break
        else:
            with accelerator.accumulate(model):
                
                states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, ix = batch
                state_preds, action_preds = model(
                    states=states_i,
                    actions=actions_i,
                    returns_to_go=rtgs_i,
                    constraints_to_go=ctgs_i,
                    timesteps=timesteps_i
                )
        
                loss_i_state = torch.mean((state_preds[:,:-1,:] - states_i[:,1:,:]) ** 2)
                loss_i_action = torch.mean((action_preds - actions_i) ** 2)

                # loss_i_target = torch.mean((state_preds[:,-1,:] - target_state) ** 2)
                loss = loss_i_action + loss_i_state
                if step % 100 == 0:
                    accelerator.print(
                        {
                            #"lr": lr_scheduler.get_lr(),
                            "lr":  lr_scheduler.get_lr() if use_lr_scheduler else optimizer.param_groups[0]['lr'],
                            "samples": step * samples_per_step,
                            "steps": completed_steps,
                            "loss/train": loss.item(),
                        }
                    )

                # Store the training loss
                train_losses.append((completed_steps, loss.item()))

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if use_lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                if (step % (eval_steps)) == 0:
                    eval_loss, loss_state, loss_action = evaluate()
                    accelerator.print({"loss/eval": eval_loss, "loss/state": loss_state, "loss/action": loss_action})
                    eval_losses.append((completed_steps, eval_loss))  # Store evaluation loss
                    eval_state_loss.append((completed_steps, loss_state))  # Store evaluation loss
                    eval_action_loss.append((completed_steps, loss_action))  # Store evaluation loss
                    model.train()
                    accelerator.wait_for_everyone()
                if (step % eval_steps*10) == 0:
                    accelerator.save_state(root_folder + '/mamba/saved_files/checkpoints_S6_l6_d128_dm512_test/checkpoint_rtn_s6_train')
                        

end_time_train = time.time()
total_time = end_time_train - start_time_train

# Convert to hours, minutes, seconds
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"\n======================")
print(f"Training completed in: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)")

# Plot training loss vs. steps
# steps, norms = zip(*gradient_norms)
steps, losses = zip(*train_losses)
eval_steps, eval_loss_values = zip(*eval_losses)
eval_steps, eval_stae_loss_values = zip(*eval_state_loss)
eval_steps, eval_action_loss_values = zip(*eval_action_loss)

# Plot Gradient Norm
# plt.plot(steps, norms, label = 'Gradient Norm ')
# plt.xlabel("Training Steps")
# plt.ylabel("Gradient Norm")
# plt.title("Gradient Norm vs Training Steps")
# plt.legend()
# plt.grid(True)
# plt.savefig(root_folder + '/mamba/saved_files/checkpoints_S6_l6_d128_dm512_test/gradnorm.png')
# plt.show()

# Plot Training Loss

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
# plt.yscale('log')
# plt.xscale('log')
plt.title('Training Loss vs Steps')
plt.legend()
plt.grid(True)
plt.savefig(root_folder + '/mamba/saved_files/checkpoints_S6_l6_d128_dm512_test/trainloss.png')
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(eval_steps, eval_loss_values, label='Evaluation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
# plt.yscale('log')
# plt.xscale('log')
plt.title('Total Evaluation Loss vs Steps')
plt.legend()
plt.grid(True)
plt.savefig(root_folder + '/mamba/saved_files/checkpoints_S6_l6_d128_dm512_test/evalloss.png')
plt.show()




plt.figure(figsize=(10, 6))
plt.plot(eval_steps, eval_stae_loss_values, label='State Evaluation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
# plt.yscale('log')
# plt.xscale('log')
plt.title('State Evaluation Loss vs Steps')
plt.legend()
plt.grid(True)
plt.savefig(root_folder + '/mamba/saved_files/checkpoints_S6_l6_d128_dm512_test/evalstateloss.png')
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(eval_steps, eval_action_loss_values, label='Action Evaluation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
# plt.yscale('log')
# plt.xscale('log')
plt.title('Action Evaluation Loss vs Steps')
plt.legend()
plt.grid(True)
plt.savefig(root_folder + '/mamba/saved_files/checkpoints_S6_l6_d128_dm512_test/evalactionloss.png')
plt.show()


# Saving the data in a csv file

file_name = "checkpoints_S6_l6_d128_dm512_test_losses_data.csv"

# Data to be saved
data_to_save = [
    ("Step", "Train_Loss", steps, losses),
    ("Eval_Step", "Eval_Loss", eval_steps, eval_loss_values),
    ("Eval_Step", "State_Loss", eval_steps, eval_stae_loss_values),
    ("Eval_Step", "Action_Loss", eval_steps, eval_action_loss_values)
]

# Prepare data for saving
max_length = max(len(steps), len(eval_steps))

# Fill missing values with None for alignment
aligned_data = {
    "Step": list(steps) + [None] * (max_length - len(steps)),
    "Train_Loss": list(losses) + [None] * (max_length - len(losses)),
    "Eval_Step": list(eval_steps) + [None] * (max_length - len(eval_steps)),
    "Eval_Loss": list(eval_loss_values) + [None] * (max_length - len(eval_loss_values)),
    "State_Loss": list(eval_stae_loss_values) + [None] * (max_length - len(eval_stae_loss_values)),
    "Action_Loss": list(eval_action_loss_values) + [None] * (max_length - len(eval_action_loss_values))
}

# Save data to a single CSV file
with open(file_name, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Step", "Train_Loss", "Eval_Step", "Eval_Loss", "State_Loss", "Action_Loss"])
    # Write rows
    for i in range(max_length):
        writer.writerow([
            aligned_data["Step"][i],
            aligned_data["Train_Loss"][i],
            aligned_data["Eval_Step"][i],
            aligned_data["Eval_Loss"][i],
            aligned_data["State_Loss"][i],
            aligned_data["Action_Loss"][i]
        ])

print("Data saved to a CSV file successfully.")