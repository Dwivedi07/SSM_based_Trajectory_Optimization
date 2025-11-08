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
from mamba.S6model4 import Mamba_Traj
from mamba.mamba_ssm.models.config import MambaConfig
#################################################

from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, get_scheduler
from accelerate import Accelerator
from transformer.art import AutonomousRendezvousTransformer

from transformer.manage import get_train_val_test_data
# from mamba.manage import get_train_val_test_data


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

datasets, dataloaders = get_train_val_test_data(state_representation, 'both', 'transformer_model_name')
train_loader, eval_loader, test_loader = dataloaders
train_dataset, val_dataset, test_dataset = datasets 


print('Intializing MAMBA Model\n')
model = Mamba_Traj(
        state_dim=6,
        act_dim=3,
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
    num_training_steps = 12000000
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
        # states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, ix = next(data_iter)
        states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, time_discr, time_sec, horizons, ix = next(data_iter)
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
        if step>= 100000: 
            break
        else:
            with accelerator.accumulate(model):
                
                states_i, actions_i, rtgs_i, ctgs_i, timesteps_i, attention_mask_i, oe, time_discr, time_sec, horizons, ix = batch
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
                    # print(step, '000saved000')
                    accelerator.save_state(root_folder + '/mamba/saved_files/checkpoints_S6De_l4_d128_dm384/checkpoint_rtn_s6_train')   #checkpoints_S6De_l4_d128_dm384
                        

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
plt.savefig(root_folder + '/mamba/saved_files/checkpoints_S6De_l4_d128_dm384/trainloss.png')
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
plt.savefig(root_folder + '/mamba/saved_files/checkpoints_S6De_l4_d128_dm384/evalloss.png')
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
plt.savefig(root_folder + '/mamba/saved_files/checkpoints_S6De_l4_d128_dm384/evalstateloss.png')
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
plt.savefig(root_folder + '/mamba/saved_files/checkpoints_S6De_l4_d128_dm384/evalactionloss.png')
plt.show()


# Saving the data in a csv file

file_name = "checkpoints_S6De_l4_d128_dm384_losses_data.csv"

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