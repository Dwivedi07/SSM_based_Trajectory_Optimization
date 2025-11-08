import os
import sys
import time
import csv

art_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(root_folder)
sys.path.append(art_path)


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#S4 Modules
from SSM.models import TrajectoryFFS4Model
import s4.manage as DM_manager
from s4.manage import device

import torch
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler


# Initial parameters
model_name_saving = 'checkpoint_ff_S4_gsa'  # 
save_path = '/s4/saved_files/checkpointsscpS4/'
mdp_constr = True

'''
Load the data for training
'''
datasets, dataloaders = DM_manager.get_train_val_test_data(mdp_constr=mdp_constr, timestep_norm=False)
train_loader, eval_loader, test_loader = dataloaders
n_state = train_loader.dataset.n_state
n_data = train_loader.dataset.n_data
n_action = train_loader.dataset.n_action
n_time = train_loader.dataset.max_len

'''
Initilaizing the model
'''
# state_dim=n_state, act_dim=n_action
print('Intializing SSM-S4 Model\n')
model = TrajectoryFFS4Model(state_dim = n_state,
                            act_dim = n_action,
                            d_model=350,       # 128: S4
                            n_layers=6,
                            max_ep_len=100,
                            dropout=0.2,
                            prenorm=False,       
                            ) 

model_size = sum(t.numel() for t in model.parameters())
print(f"SSM size: {model_size/1000**2:.1f}M parameters")
# print(model)
model.to(device);
optimizer = AdamW(model.parameters(), lr=7e-6)   # lr = 3e-5 
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)


num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = 900000

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=num_training_steps,
)


# Eval function
eval_iters = 100
@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    losses_state = []
    losses_action = []
    for step in range(eval_iters):
        data_iter = iter(eval_dataloader)
        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, _, _, _ = next(data_iter)
        # Shift actions one step to the right
        actions_shifted = torch.zeros_like(actions_i)
        actions_shifted[:, 1:, :] = actions_i[:, :-1, :]

        with torch.no_grad():
            state_preds, action_preds = model(
                states=states_i,
                actions=actions_shifted,
                # returns_to_go=rtgs_i,
                # constraints_to_go=ctgs_i,
                goals=goal_i,
                timesteps=timesteps_i,
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

eval_steps = 500
samples_per_step = accelerator.state.num_processes * train_loader.batch_size


train_losses = []
eval_losses = []
eval_state_loss = []
eval_action_loss = []


model.train()
completed_steps = 0

for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_dataloader, start=0):
        if step>= 200000: 
            break
        else:
            with accelerator.accumulate(model):
                # torch.cuda.empty_cache()
                states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, _, _, _ = batch

                # Shift actions one step to the right and set action at t=0 to zero
                actions_shifted = torch.zeros_like(actions_i)  # Initialize with zeros
                actions_shifted[:, 1:, :] = actions_i[:, :-1, :]  # Shift actions to the right

                state_preds, action_preds = model(
                    states=states_i,
                    actions=actions_shifted,
                    # returns_to_go=rtgs_i,
                    # constraints_to_go=ctgs_i,
                    goals=goal_i,
                    timesteps=timesteps_i,
                )
                loss_i_action = torch.mean((action_preds - actions_i) ** 2)
                loss_i_state = torch.mean((state_preds[:,:-1,:] - states_i[:,1:,:]) ** 2)
                loss = loss_i_action + loss_i_state

                # Store the training loss
                train_losses.append((completed_steps, loss.item()))
                
                if step % 100 == 0:
                    accelerator.print(
                        {
                            "lr": lr_scheduler.get_lr(),
                            "samples": step * samples_per_step,
                            "steps": completed_steps,
                            "loss/train": loss.item(),
                        }
                    )
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
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
                if (step % (eval_steps*10)) == 0:
                    print('Saving model..')
                    accelerator.save_state(root_folder+save_path+model_name_saving)

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
plt.savefig(root_folder + save_path+'/trainloss.png')
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
plt.savefig(root_folder + save_path+ '/evalloss.png')
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
plt.savefig(root_folder + save_path+'/evalstateloss.png')
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
plt.savefig(root_folder + save_path+'/evalactionloss.png')
plt.show()


# Saving the data in a csv file

file_name = model_name_saving+ ".csv"

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