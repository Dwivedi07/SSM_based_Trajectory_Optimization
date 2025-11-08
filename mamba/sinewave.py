import numpy as np
import os
import sys

import random
import matplotlib.pyplot as plt

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW

from mamba.mamba_ssm.modules.mamba_simple import Mamba
from mamba.mamba_ssm.models.config import MambaConfig
from mamba.mamba_ssm.models.config import InferenceParams

from accelerate import Accelerator

from torch.nn import MSELoss
from tqdm import tqdm

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"Running on device: {device}\n")

'''
MAMBA Block
'''
class MambaBlock(nn.Module):
    def __init__(self, config : MambaConfig, idx : int):
        super().__init__()

        self.mixer = Mamba(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    dt_rank=config.dt_rank,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                    dt_init=config.dt_init,
                    dt_scale=config.dt_scale,
                    dt_init_floor=config.dt_init_floor,
                    conv_bias=config.conv_bias,
                    bias=config.bias,
                    use_fast_path=config.use_fast_path,
                    layer_idx=idx,
                    device=config.device,
                    dtype=config.dtype
                )

        self.norm = (
            nn.LayerNorm(config.d_model, eps=config.norm_epsilon, device=config.device, dtype=config.dtype)
            if not config.rms_norm
            else RMSNorm(config.d_model, eps=config.norm_epsilon, device=config.device, dtype=config.dtype)
        )
    
    def forward(self, x, inference_params = None):
        '''
        Input:
            x : Stacked Inputs (B, L, D)

        output:
            y : (B, L, D)
        '''

        # output = self.mixer(x) + x
        output = self.mixer((self.norm(x))) + x

        return output

    def step(self, x, conv_state, ssm_state, inference_params = None):
        '''
        Input:
            x           : input token
            conv_state  : cache
            ssm_state   : cache
        
        '''
        # output, conv_state, ssm_state = self.mixer.step(x, conv_state, ssm_state)
        output, conv_state, ssm_state = self.mixer.step(self.norm(x), conv_state, ssm_state)
        output = output + x

        return output, conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=torch.float32):
        conv_state, ssm_state = self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

        return conv_state, ssm_state
    
    def _get_states_from_cache(self, inference_params, batch_size):
        conv_state, ssm_state = self.mixer._get_states_from_cache(inference_params, batch_size)

        return conv_state, ssm_state

'''
Deep Model
'''

class SineWavePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding_token = nn.Linear(config.state_dim, config.d_model)

        # Stack Mamba blocks
        self.layers = nn.ModuleList([MambaBlock(config, i) for i in range(config.n_layers)])

        self.norm = (
            nn.LayerNorm(config.d_model, eps=config.norm_epsilon, device=config.device, dtype=config.dtype)
            if not config.rms_norm
            else RMSNorm(config.d_model, eps=config.norm_epsilon, device=config.device, dtype=config.dtype)
        )

        # Prediction heads 
        self.output_layer = nn.Linear(config.d_model, config.state_dim)

    def forward(self, x, inference_params=None):
        '''
        X = (B,L,D)
        '''
        x = self.embedding_token(x)
        for i, layer in enumerate(self.layers):           
            x = layer(x, inference_params=inference_params)

        mamba_output = self.norm(x)
        # Generate predictions
        predictions = self.output_layer(mamba_output)

        '''
        Ouput shape also (B,L,D)
        '''
        return predictions
    
    def step(self, x, conv_states, ssm_states, inference_params=None):
        '''
        Here only one token is passed at a time
        '''
        x = self.embedding_token(x)
        for i, layer in enumerate(self.layers):           
            x, conv_states[i], ssm_states[i] = layer.step(x, conv_states[i], ssm_states[i], inference_params=inference_params)
        
        inference_params.seqlen_offset += 1
        x = self.norm(x)
        predictions = self.output_layer(x)

        return predictions, conv_states, ssm_states

    def step_T(self, x, inference_params=None):
        '''
        Here we can pass trasnformer like inference with mamba where we will
        feed last K time steps input and process thr K+1 input
        x = [1, L, D] 
        but we will feed only last K inputs x = [1, K, D]
        '''
        x = self.embedding_token(x)

        # TODO: PADDING FOR THE INPUT SEQUENCE

        for i, layer in enumerate(self.layers):           
            x = layer(x, inference_params=inference_params)
            
        mamba_output = self.norm(x)
        # Generate predictions
        predictions = self.output_layer(mamba_output)

        return predictions[:,-1,:]

    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=torch.float32):
        
        conv_states = [None] * len(self.layers)  
        ssm_states = [None] * len(self.layers)   

        # Iterate through each layer and call allocate_inference_cache for each layer
        for i, layer in enumerate(self.layers):
            conv_states[i], ssm_states[i] = layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

        return conv_states, ssm_states

    def _get_states_from_cache(self, inference_params, batch_size):
        conv_states = [None] * len(self.layers)  
        ssm_states = [None] * len(self.layers) 

        # Iterate through each layer and call _get_states_from_cache for each layer
        for i, layer in enumerate(self.layers):
            conv_states[i], ssm_states[i] = layer._get_states_from_cache(inference_params, batch_size)

        return conv_states, ssm_states



'''
Dataset
'''

class SineWaveDataset(Dataset):
    def __init__(self, seq_length, num_samples):
        self.seq_length = seq_length
        self.data = []
        self.targets = []
        for _ in range(num_samples):
            phase = np.random.rand() * 2 * np.pi
            x = np.linspace(phase, phase + 2 * np.pi, seq_length + 1)
            y = np.sin(x)
            self.data.append(y[:-1])
            self.targets.append(y[1:])
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32).unsqueeze(-1)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Parameters

data_dir = root_folder + '/mamba/sinedataset/' 
dataset = torch.load(data_dir + '/sine_wave_dataset.pth', weights_only=False)

# Split the dataset into training and evaluation sets
train_size = int(0.8 * len(dataset))  # 80% for training
eval_size = len(dataset) - train_size  # 20% for evaluation

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# Parameters
batch_size = 32

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    sampler=torch.utils.data.RandomSampler(
        train_dataset, replacement=True, num_samples=int(2e4)),
    shuffle=False,
    pin_memory=True,
    batch_size=32,
    num_workers=0,
)

eval_loader = DataLoader(
    eval_dataset,
    sampler=torch.utils.data.RandomSampler(
        eval_dataset, replacement=True, num_samples=int(2e4)),
    shuffle=False,
    pin_memory=True,
    batch_size=32,
    num_workers=0,
)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Print the sizes to confirm


'''
 Initlizaing the config file for the MAMBA model
'''
config = MambaConfig(
    state_dim=1,  # State Dimension
    act_dim = None,
    d_model=32,     # 16
    n_layers=3,     # 1
    max_ep_len=50, # 100
    d_state=4,     # 4
    d_conv=3,       # 3
    expand=2,       # 1
    dt_rank="auto", # Time discretization rank
    device=device,  
    dtype=torch.float32,  # Precision
)
save_path = root_folder + '/mamba/sinedataset/check/checkpoint_sine_model'

'''
Training
'''

# # print('Intializing MAMBA Model\n')
# model = SineWavePredictor(config) 
# model_size = sum(t.numel() for t in model.parameters())
# print(f"SSM size: {model_size:.1f} parameters")
# print(model)
# model.to(device);
# optimizer = AdamW(model.parameters(), lr=3e-4)   # lr = 3e-5 
# accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_loader, eval_loader
# )

# print(f"Total dataset size: {len(dataset)}")
# print(f"Training dataset size: {len(train_dataset)} and {len(train_dataloader)}")
# print(f"Evaluation dataset size: {len(eval_dataset)}")


# # Define the loss function
# criterion = MSELoss()

# # Initialize lists to store losses
# train_losses = []
# eval_losses = []

# num_epoch = 60
# # Training Loop
# for epoch in range(num_epoch):  # Number of epochs
#     model.train()  # Set the model to training mode
#     epoch_loss = 0.0

#     # Training phase
#     print(f"Epoch {epoch + 1}/{num_epoch}")
#     for batch in tqdm(train_dataloader, desc="Training"):
#         inputs, targets = batch  # Get the inputs and targets
#         inputs, targets = inputs.to(device), targets.to(device)  # Move to the correct device

#         optimizer.zero_grad()  # Reset gradients
#         predictions = model(inputs)  # Forward pass
#         loss = criterion(predictions, targets)  # Compute the loss
#         accelerator.backward(loss)  # Backpropagation
#         optimizer.step()  # Update model parameters

#         epoch_loss += loss.item()  # Accumulate batch loss

#     epoch_loss /= len(train_dataloader)
#     train_losses.append(epoch_loss)  # Save train loss for this epoch
#     print(f"Training Loss: {epoch_loss:.4f}")

#     # Evaluation phase
#     model.eval()  # Set the model to evaluation mode
#     eval_loss = 0.0
#     with torch.no_grad():
#         for batch in tqdm(eval_dataloader, desc="Evaluating"):
#             inputs, targets = batch  # Get the inputs and targets
#             inputs, targets = inputs.to(device), targets.to(device)  # Move to the correct device

#             predictions = model(inputs)  # Forward pass
#             loss = criterion(predictions, targets)  # Compute the loss
#             eval_loss += loss.item()  # Accumulate batch loss

#     eval_loss /= len(eval_dataloader)
#     eval_losses.append(eval_loss)  # Save eval loss for this epoch
#     print(f"Evaluation Loss: {eval_loss:.4f}")

#     # Optionally, save the model after every few epochs
#     if (epoch + 1) % 10 == 0:
#         accelerator.save_state(save_path)


# # Plot the training and evaluation losses
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_epoch+1), train_losses, label="Train Loss", color="blue")
# plt.plot(range(1, num_epoch+1), eval_losses, label="Eval Loss", color="orange")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Evaluation Loss")
# plt.legend()
# plt.grid()
# plt.savefig(os.path.join(root_folder, "mamba/sinedataset/loss_plot.png"))
# plt.show()

'''
Inference
Evaluating the model using forward pass for inference and plotting
'''

# print('Intializing MAMBA Model\n')
model = SineWavePredictor(config) 
model_size = sum(t.numel() for t in model.parameters())
print(f"SSM size: {model_size:.1f} parameters")
print(model)
model.to(device)
model.eval()  # Set the model to evaluation mode
optimizer = AdamW(model.parameters(), lr=3e-4)   # lr = 3e-5 
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)

accelerator.load_state(save_path)

model.eval()  
random_index = random.randint(0, len(eval_dataset) - 1)

# Retrieve the random sample
with torch.no_grad():
    input_sample, target_sample = eval_dataset[random_index]
    input_sample, target_sample = input_sample.to(device), target_sample.to(device)

    # Get the model's prediction for the random sample
    pred_sample = model(input_sample.unsqueeze(0))  # Add batch dimension
    pred_sample = pred_sample.squeeze().detach().cpu().numpy()

# Convert the input and target back to CPU for plotting
input_sample = input_sample.squeeze().cpu().numpy()
target_sample = target_sample.squeeze().cpu().numpy()
print(input_sample )
print('One forward pass',pred_sample)

# Plot the results,
plt.figure(figsize=(12, 6))
plt.plot(input_sample, label="Input (x_t)", color="blue", linestyle="--")
plt.plot(target_sample, label="True Target (y_t)", color="green")
plt.plot(pred_sample, label="Predicted (y_pred)", color="red",linestyle="--")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("Sine Wave Prediction - True vs Predicted")
plt.legend()
plt.grid()
plt.savefig(os.path.join(root_folder, "mamba/sinedataset/prediction_visualization.png"))
plt.show()

with torch.no_grad():
    # Randomly select an index from the evaluation dataset
    # Retrieve the random sample
    input_sample, target_sample = eval_dataset[random_index]
    input_sample, target_sample = input_sample.to(device), target_sample.to(device)

    # Use only the first point of the input as the initial condition
    initial_point = input_sample[0] 

    # Autoregressively generate the trajectory
    generated_trajectory = []
    current_input = initial_point  # Start with the initial point
    current_input = current_input.unsqueeze(0).unsqueeze(0)
    for _ in range(len(target_sample)):  # Generate for the entire target length
        # print(current_input.shape)
        prediction = model(current_input)  
        generated_trajectory.append(prediction.item())  
        current_input = prediction  

    # Convert the target to CPU for visualization
    target_sample = target_sample.squeeze().cpu().numpy()
    generated_trajectory = np.array(generated_trajectory)  # Convert to NumPy array

print('generated_trajectory ',generated_trajectory )
# Visualization
plt.figure(figsize=(10, 6))
plt.plot(range(len(target_sample)), target_sample, label="True Trajectory (Target)", color="green")
plt.plot(range(len(generated_trajectory)), generated_trajectory, label="Generated Trajectory (Prediction)", color="orange", linestyle="dashed")
plt.scatter(0, initial_point.item(), label="Initial Point", color="red", zorder=5)
plt.title("Autoregressive Trajectory Generation")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig(os.path.join(root_folder, "mamba/sinedataset/generation_visualization.png"))
plt.show()



batch_size = 1
inference_params = InferenceParams(batch_size=batch_size, n_layers = model.config.n_layers)   # From config file test_loader.dataset.batch_size
conv_states, ssm_states = model.allocate_inference_cache(batch_size=batch_size, max_seqlen=config.max_ep_len, dtype=torch.float32)

with torch.no_grad():
    input_sample, target_sample = eval_dataset[random_index]
    input_sample, target_sample = input_sample.to(device), target_sample.to(device)

    # Use only the first point of the input as the initial condition
    initial_point = input_sample[0] 

    # Autoregressively generate the trajectory
    generated_trajectory_r = []
    current_input = initial_point  # Start with the initial point
    current_input = current_input.unsqueeze(0).unsqueeze(0)
    for _ in range(len(target_sample)):  # Generate for the entire target length
        # print(current_input.shape)
        output_pred = model.step(
                    current_input, 
                    conv_states = conv_states,
                    ssm_states = ssm_states,
                    inference_params = inference_params
                )
        (pred_nexts, conv_states, ssm_states) = output_pred
 
        generated_trajectory_r.append(pred_nexts.cpu().numpy())  
        current_input = pred_nexts


    target_sample = target_sample.squeeze().cpu().numpy()
    generated_trajectory_r = np.array(generated_trajectory_r).flatten()  # Convert to NumPy array
    # Convert the target to CPU for visualization
    # generated_trajectory_r = np.array(generated_trajectory_r)  # Convert to NumPy array

print('generated_trajectory_r ',generated_trajectory_r )
# Visualization
plt.figure(figsize=(10, 6))
plt.plot(range(len(target_sample)), target_sample, label="True Trajectory (Target)", color="green")
plt.plot(range(len(generated_trajectory_r)), generated_trajectory_r, label="Generated Trajectory (Prediction)", color="orange", linestyle="dashed")
plt.scatter(0, initial_point.item(), label="Initial Point", color="red", zorder=5)
plt.title("Autoregressive R Trajectory Generation")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig(os.path.join(root_folder, "mamba/sinedataset/generation_visualization_r.png"))
plt.show()

'''
Transformer like inference with last K inputs
we will intiially pass first 50 inputs (as we trained on 50 while training) and then predict the next 51th 
and so on till 100th time step
'''

with torch.no_grad():
    input_sample, target_sample = eval_dataset[random_index]
    input_sample, target_sample = input_sample.to(device), target_sample.to(device)

    # Autoregressively generate the trajectory from t=51 to t = 100
    generated_trajectory_Tr = []
    current_input = input_sample  # Start with the initial point
    current_input = current_input.unsqueeze(0)[:, -2:, :]
    for _ in range(50+len(target_sample)):  
        output_pred = model.step_T(current_input)
        pred_nexts = output_pred

        generated_trajectory_Tr.append(pred_nexts.cpu().flatten())
       
        current_input = torch.cat((current_input, pred_nexts.unsqueeze(0)), dim = 1)[:, -2:, :]


    target_sample = target_sample.squeeze().cpu().numpy()
    generated_trajectory_r = np.array(generated_trajectory_Tr).flatten()  # Convert to NumPy array

generated_trajectory_Tr = torch.stack(generated_trajectory_Tr).numpy().flatten()
print('generated_trajectory_Tr ',generated_trajectory_Tr )
print('testing values',target_sample[-1], generated_trajectory_Tr[0])

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(range(len(target_sample)), target_sample, label="True Trajectory (Target)", color="green")
plt.plot(range(49, 49 + len(generated_trajectory_Tr)), generated_trajectory_Tr, label="Generated Trajectory tranformer like (Prediction)", color="orange", linestyle="dashed")
# plt.scatter(0, initial_point.item(), label="Initial Point", color="red", zorder=5)
plt.title("Autoregressive T Trajectory Generation")
plt.xlabel("Time Step")
plt.ylabel("Value")
# plt.legend()
plt.grid()
plt.savefig(os.path.join(root_folder, "mamba/sinedataset/generation_visualization_T.png"))
plt.show()
