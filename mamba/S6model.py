import os
import sys

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from mamba.mamba_ssm.modules.mamba_simple import Mamba
from mamba.mamba_ssm.models.config import MambaConfig
from mamba.mamba_ssm.utils.generation import InferenceParams


try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


'''
This is the structure :
- A DeepMambaModel is composed of several layers, which are MambaBlock.
- A MambaBlock is composed of a Mamba, a normalization, and a residual connection : MambaBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).

See Figure 3 of the paper - https://arxiv.org/pdf/2312.00752 (page 8) for a visual representation of a mamba.

'''

class DeepMambaModel(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        """
        A deep model with stacked Mamba blocks, initialized via a configuration.

        Args:
            config: Configuration object containing hyperparameters.
        """
        self.config = config
        self.n_state = config.state_dim
        self.n_action = config.act_dim
        self.d_model = config.d_model
        self.max_ep_len = config.max_ep_len

        # Embedding layers for each modality
        self.embed_state = nn.Linear(self.n_state, self.d_model)
        self.embed_action = nn.Linear(self.n_action, self.d_model)
        self.embed_return = nn.Linear(1, self.d_model)  # Returns-to-go
        self.embed_constraint = nn.Linear(1, self.d_model)  # Constraints-to-go
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.d_model)  # Time embeddings

        # Stack Mamba blocks
        self.layers = nn.ModuleList([MambaBlock(config, i) for i in range(config.n_layers)])

        self.norm = (
            nn.LayerNorm(config.d_model, eps=config.norm_epsilon, device=config.device, dtype=config.dtype)
            if not config.rms_norm
            else RMSNorm(config.d_model, eps=config.norm_epsilon, device=config.device, dtype=config.dtype)
        )
        # Prediction heads
        self.predict_action = torch.nn.Linear(self.d_model, self.n_action)
        self.predict_state = torch.nn.Linear(self.d_model, self.n_state) 

    def forward(self, states, actions, returns_to_go, constraints_to_go, timesteps, inference_params=None):
        """
        Forward pass through the DeepMambaModel.

        Args:
            - states: (B, L, state_dim)
            - actions: (B, L, act_dim)
            - returns_to_go: (B, L, 1)
            - constraints_to_go: (B, L, 1)
            - timesteps: (B, L)
        Returns:
            - states: (B, L, state_dim)
            - actions: (B, L, act_dim)
        """
        # Data Params
        batch_size, seq_length = timesteps.shape[0], timesteps.shape[1]   # this will not be true when I am passing L=1 : TODO
        # batch_size, seq_length = timesteps.shape[0], 1    # Use this: Modified if forward is used for inference
        d_model = self.config.d_model

        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns_to_go)
        constraint_embeddings = self.embed_constraint(constraints_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings so added to each 
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = return_embeddings + time_embeddings
        constraints_embeddings = constraint_embeddings + time_embeddings

        # following will make the sequence look like (R_1, C_1, s_1, a_1, R_2, C_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if seq_length != 1:
            stacked_inputs = (
                torch.stack((returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, 4 * seq_length, d_model)
            )
        else:
            stacked_inputs = (
                torch.stack((returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1)
            )

        # Mamba block takes: (B, L, D) format
        # Pass through stacked MambaBlocks
        x = stacked_inputs
        for i, layer in enumerate(self.layers):           
            x = layer(x, inference_params=inference_params)

        # Apply normalization
        x = self.norm(x)
        outputs = x.reshape(batch_size, seq_length, 4, d_model).permute(0, 2, 1, 3)   # Repermutation of the data

        # Predict next action and next state
        state_preds = self.predict_state(outputs[:, 3])   # predict next state given state and action
        action_preds = self.predict_action(outputs[:, 2]) # predict next action given state

        return state_preds, action_preds
    
    def step(self, states, actions, returns_to_go, constraints_to_go, timesteps, conv_states, ssm_states, inference_params=None):
        '''
        Step pass through the DeepMambaModel

        Args:
            - states: (B, L, state_dim)
            - actions: (B, L, act_dim)
            - returns_to_go: (B, L, 1)
            - constraints_to_go: (B, L, 1)
            - timesteps: (B, L)
            - conv_states: conv_state for each layer
            - ssm_states: ssm_state for each layer
        Returns:
            y : Prediction of next action to take: (B, L, act_dim)
            conve_states : updated cache
            ssm_states : updated cache
        '''
        # Data Params
        batch_size = timesteps.shape[0]
        seq_length = 1
        d_model = self.config.d_model

        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns_to_go)
        constraint_embeddings = self.embed_constraint(constraints_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings so added to each 
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = return_embeddings + time_embeddings
        constraints_embeddings = constraint_embeddings + time_embeddings

        # following will make the sequence look like (R_1, C_1, s_1, a_1, R_2, C_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = state_embeddings

        x= stacked_inputs
        for i, layer in enumerate(self.layers):           
            x, conv_states[i], ssm_states[i] = layer.step(x, conv_states[i], ssm_states[i], inference_params=inference_params)

        # Apply normalization
        x = self.norm(x)

        action_preds = self.predict_action(x)

        return action_preds, conv_states, ssm_states

    def get_action(self, states, actions, returns_to_go, constraints_to_go, timesteps, inference_params):
        '''
        Step pass through the DeepMambaModel

        Args:
            - states: (B, L, state_dim)
            - actions: (B, L, act_dim)
            - returns_to_go: (B, L, 1)
            - constraints_to_go: (B, L, 1)
            - timesteps: (B, L)
            - conv_states: conv_state for each layer
            - ssm_states: ssm_state for each layer
        Returns:
            y : Prediction of next action to take: (B, L, act_dim)
            conve_states : updated cache
            ssm_states : updated cache
        '''
        # Data Params
        batch_size = timesteps.shape[0]
        seq_length = 1
        d_model = self.config.d_model

        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns_to_go)
        constraint_embeddings = self.embed_constraint(constraints_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings so added to each 
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = return_embeddings + time_embeddings
        constraints_embeddings = constraint_embeddings + time_embeddings

        # following will make the sequence look like (R_1, C_1, s_1, a_1, R_2, C_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        returns_embeddings = returns_embeddings.unsqueeze(1)
        state_embeddings   = state_embeddings.unsqueeze(1)
        action_embeddings  = action_embeddings.unsqueeze(1)
        constraints_embeddings = constraints_embeddings.unsqueeze(1)

        stacked_inputs = (
                torch.stack((returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, 4 * seq_length, d_model)
            )

        ###
        # x= stacked_inputs
        # outputs = torch.zeros(batch_size, x.shape[1], d_model, dtype=x.dtype, device=x.device)
        # for i in range(x.shape[1]):
        #     ret_y = x[:, i:i+1]
        #     for j, layer in enumerate(self.layers):           
        #         ret_y = layer(ret_y, inference_params=inference_params)

        #     outputs[:, i] = ret_y[:, 0]
        #     inference_params.seqlen_offset += 1
        

        # outputs = outputs.reshape(batch_size, 1, 4, self.d_model).permute(0, 2, 1, 3)

        ###
        x= state_embeddings
        for layer in self.layers:           
            x = layer(x, inference_params=inference_params)

        outputs = x
        inference_params.seqlen_offset += 1
        ##

        outputs = self.norm(outputs)
        action_preds = self.predict_action(outputs)
        # action_preds = self.predict_action(outputs[:,-1])

        return action_preds
    
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
            x : Stacked Inputs (B, 4*L, D)

        output:
            y : (B, 4*L, D)
        '''
        # output = self.mixer(x, inference_params) + x
        output = self.mixer((self.norm(x)), inference_params) + x

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
