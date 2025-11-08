import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from transformers import DecisionTransformerConfig
from decision_transformer.art import AutonomousFreeflyerTransformer2
import torch
import decision_transformer.manage as ART_manager
from decision_transformer.manage import device
import time
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
import csv
# import wandb  # Import WandB


# Initial parameters
model_name_4_saving = 'checkpoint_ff_GSA'
save_path = '/decision_transformer/saved_files/checkpointsART/'
mdp_constr = True
datasets, dataloaders = ART_manager.get_train_val_test_data(mdp_constr=mdp_constr, timestep_norm=False)
train_loader, eval_loader, test_loader = dataloaders
n_state = train_loader.dataset.n_state
n_data = train_loader.dataset.n_data
n_action = train_loader.dataset.n_action
n_time = train_loader.dataset.max_len


# # Initialize WandB 
# wandb.init(project="Autonomous-Freeflyer-Transformer", name="training_run_1", config={
#     "learning_rate": 3e-5,
#     "epochs": 1,
#     "batch_size": train_loader.batch_size,
#     "hidden_size": 384,
#     "layers": 6,
#     "heads": 6
# })


# Transformer parameters
config = DecisionTransformerConfig(
    state_dim=n_state, 
    act_dim=n_action,
    hidden_size=384,
    max_ep_len=n_time,
    vocab_size=1,
    action_tanh=False,
    n_positions=1024,
    n_layer=6,
    n_head=6,
    n_inner=None,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    )
model = AutonomousFreeflyerTransformer2(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT size: {model_size/1000**2:.1f}M parameters")
model.to(device);


optimizer = AdamW(model.parameters(), lr=3e-5)
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)
num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = 10000000000

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=num_training_steps,
)

# To activate only when starting from a pretrained model
# accelerator.load_state(root_folder + '/decision_transformer/saved_files/checkpoints/' + model_name_4_dataset)

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
        with torch.no_grad():
            state_preds, action_preds = model(
                states=states_i,
                actions=actions_i,
                goal=goal_i,
                returns_to_go=rtgs_i,
                constraints_to_go=ctgs_i,
                timesteps=timesteps_i,
                attention_mask=attention_mask_i,
                return_dict=False,
            )
        loss_i = torch.mean((action_preds - actions_i) ** 2)
        loss_i_state = torch.mean((state_preds[:,:-1,:] - states_i[:,1:,:]) ** 2)
        losses.append(accelerator.gather(loss_i + loss_i_state))
        losses_state.append(accelerator.gather(loss_i_state))
        losses_action.append(accelerator.gather(loss_i))
    loss = torch.mean(torch.tensor(losses))
    loss_state = torch.mean(torch.tensor(losses_state))
    loss_action = torch.mean(torch.tensor(losses_action))

    # wandb.log({"loss/eval": loss.item(), "loss/state": loss_state.item(), "loss/action": loss_action.item()})


    model.train()
    return loss.item(), loss_state.item(), loss_action.item()

eval_loss, loss_state, loss_action = evaluate()
accelerator.print({"loss/eval": eval_loss, "loss/state": loss_state, "loss/action": loss_action})

# Training

eval_steps = 500
samples_per_step = accelerator.state.num_processes * train_loader.batch_size
#torch.manual_seed(4)

train_losses = []
eval_losses = []
eval_state_loss = []
eval_action_loss = []

model.train()
completed_steps = 0
# log = {
#     'loss':[],
#     'loss_state':[],
#     'loss_action':[]
# }
'''log = np.load(root_folder + '/decision_transformer/saved_files/checkpoints/' + model_name_4_saving + '/log.npz', allow_pickle=True)['log'].item()'''

print('\n======================')
print('Initializing training\n')
start_time_train = time.time()


for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_dataloader, start=0):
        if step >= 150000:
            break
        else:
            with accelerator.accumulate(model):
                states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, _, _, _ = batch
                state_preds, action_preds = model(
                    states=states_i,
                    actions=actions_i,
                    goal=goal_i,
                    returns_to_go=rtgs_i,
                    constraints_to_go=ctgs_i,
                    timesteps=timesteps_i,
                    attention_mask=attention_mask_i,
                    return_dict=False,
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
                    accelerator.save_state(root_folder+save_path+model_name_4_saving)

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

file_name = model_name_4_saving + ".csv"

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