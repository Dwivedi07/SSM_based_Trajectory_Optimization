import os
import sys

import random
from dataclasses import field
from functools import partial
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from mamba.mamba_ssm.models.mixer_seq_simple2 import MixerModel
from mamba.mamba_ssm.utils.generation import InferenceParams


try:
    from mamba.mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba_Traj(nn.Module):
    def __init__(
        self,
        state_dim: int,      # state dimension in real dyanmics
        act_dim: int,        # action dimension in real dynamics
        embedding_dim: int,  # The first projection to shared embedding space
        d_model: int,        # the embedding_dim is used to get voab_size then projected to d_model space
        n_layer: int,        # Number of mamba blocks you want to keep
        ssm_cfg=None,
        rms_norm: bool=True,
        residual_in_fp32: bool=True,
        fused_add_norm: bool=True,
        initializer_cfg=None,
        device=None,
        dtype=None,
        dropout=0,
        max_length=12,     # Used when doing tranformer like implementation for inference
        max_ep_len=300,    # total time_steps in one traj sample
        action_tanh=True,
        reward_type='normal',
        time_embd=True,  #Initially it was false
        type_input='B4LD',
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.input_emb_size = embedding_dim
        self.d_model = d_model
        self.dropout = dropout
        self.reward_type = reward_type
        self.time_embd=time_embd
        self.type_input = type_input
        
        n_layer = n_layer
        ssm_cfg = ssm_cfg
        rms_norm = rms_norm
        residual_in_fp32 = residual_in_fp32
        fused_add_norm = fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}

        if self.time_embd:
            self.embed_timestep = nn.Embedding(max_ep_len, self.input_emb_size)
        
        self.ret_emb = nn.Linear(1, self.input_emb_size)  # Returns-to-go
        self.ctg_emb = nn.Linear(1, self.input_emb_size)  # Constraints-to-go
        self.goal_emb = nn.Linear(self.state_dim, self.input_emb_size)  # goal
        self.state_encoder = nn.Linear(self.state_dim, self.input_emb_size)
        self.action_embeddings = nn.Linear(self.act_dim, self.input_emb_size)

        self.backbone = MixerModel(
            d_model=self.d_model,
            # d_intermediate = 2*self.d_model,
            n_layer=n_layer,
            vocab_size=self.input_emb_size if type_input=='B4LD' else 4*self.input_emb_size,
            # dropout_val=self.dropout,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.d_model, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.d_model, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        
        self.inference_params = InferenceParams(max_seqlen=300, max_batch_size=1)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    # def forward(self, states, actions, returns_to_go, constraints_to_go, goal, timesteps, running=False, cache=None,
    #             attention_mask=None, position_ids=None, inference_params=None, num_last_tokens=0):
    def forward(self, states, actions, goal, timesteps, running=False, cache=None,
                attention_mask=None, position_ids=None, inference_params=None, num_last_tokens=0):
        '''
        Implementation of simple (g,s,a)_t for t = 0,1,...,N
            goals: goals - dimension 6
            states: current state - dimension 6
            actions: action at t - dimesnion 3
            returns_to_go - dimesion 1
            constraints_to_go - dimesnion 1
            timestep - its which step of traj this seq is
        
        '''

        del_r = 0   # delted del_r first timesteps-> Ensures only the last (sequence_len - del_r) timesteps are used.
        batch_size = states.shape[0]
        sequence_len = actions.shape[1]

        if running or sequence_len==1:
            del_r = 0
        state_embeddings = self.state_encoder(states[:,del_r:,...].reshape(-1, sequence_len-del_r, self.state_dim).type(torch.float32).contiguous())
        goal_embeddings = self.goal_emb(goal[:, :sequence_len-del_r, ...].reshape(-1, sequence_len-del_r, self.state_dim).type(torch.float32))
        action_embeddings = self.action_embeddings(actions[:,:sequence_len-del_r,...].reshape(-1, sequence_len-del_r, self.act_dim))
        # returns_embeddings = self.ret_emb(returns_to_go[:, :sequence_len-del_r, ...].reshape(-1, sequence_len-del_r, 1).type(torch.float32))
        # constraints_embeddings = self.ctg_emb(constraints_to_go[:, :sequence_len-del_r, ...].reshape(-1, sequence_len-del_r, 1).type(torch.float32))
        
        
        if self.time_embd:
            time_embeddings = self.embed_timestep(timesteps[:, :sequence_len-del_r])
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            # returns_embeddings = returns_embeddings + time_embeddings
            # constraints_embeddings = constraints_embeddings + time_embeddings
            goal_embeddings = goal_embeddings + time_embeddings

        if self.type_input == 'B4LD': # B4LD just means that i am stacking along time dimesnion: this string is just use to make cases, it canbe B3LD, b5LD etc
            # Model Type : (g, r, c, s, a)
            # u = torch.stack(
            #     (goal_embeddings, returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1
            # ).permute(0, 2, 1, 3).reshape(batch_size, 5*(sequence_len-del_r), self.input_emb_size)

            # Model Type : (g,s,a)

            u = torch.stack(
                (goal_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*(sequence_len-del_r), self.input_emb_size)
        elif self.type_input == 'BL4D':
            # u = torch.cat([returns_embeddings, constraints_embeddings, action_embeddings, state_embeddings], dim=-1)
            pass

        y = self.backbone(u)
        y = y.reshape(batch_size, sequence_len-del_r, 3, self.d_model).permute(0, 2, 1, 3)
        
        state_preds = self.predict_state(y[:,-1])    # predict next state given state and action
        action_preds = self.predict_action(y[:,-2])  # predict next action given state
    
        return state_preds, action_preds

    def get_action_RNN(self, states, actions, returns_to_go, constraints_to_go, goal, timesteps, position_ids=None, num_last_tokens=0, **kwargs):
        '''
        Implementation of simple (g,s,a)_t for t = 0,1,...,N
            goals: goals - dimension 6
            states: current state - dimension 6
            actions: action at t - dimesnion 3
            returns_to_go - dimesion 1
            constraints_to_go - dimesnion 1
            timestep - its which step of traj this seq is
        
        '''
        if len(states.shape)==3:
            bs = states.shape[0]
        else:
            bs = 1
        
        states = states.reshape(bs, -1, self.state_dim)[:,-1]
        actions = actions.reshape(bs, -1, self.act_dim)[:,-1]
        returns_to_go = returns_to_go.reshape(bs, -1, 1)[:,-1]
        constraints_to_go = constraints_to_go.reshape(bs, -1, 1)[:,-1]
        goal = goal.reshape(bs, -1, self.state_dim)[:,-1]
           

        state_embeddings = self.state_encoder(states)
        action_embeddings = self.action_embeddings(actions)
        returns_embeddings = self.ret_emb(returns_to_go)
        constraints_embeddings = self.ctg_emb(constraints_to_go)
        goal_embeddings = self.goal_emb(goal)

        if self.time_embd:
    
            time_embeddings = self.embed_timestep(timesteps)
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            returns_embeddings = returns_embeddings + time_embeddings
            constraints_embeddings = constraints_embeddings + time_embeddings
            goal_embeddings = goal_embeddings + time_embeddings
        
        returns_embeddings = returns_embeddings.unsqueeze(1)
        state_embeddings   = state_embeddings.unsqueeze(1)
        action_embeddings  = action_embeddings.unsqueeze(1)
        constraints_embeddings = constraints_embeddings.unsqueeze(1)
        goal_embeddings = goal_embeddings.unsqueeze(1)

        if self.type_input == 'B4LD':
            #Model Type 1: (g,r,c,s,a)

            u = torch.stack(
                (goal_embeddings, returns_embeddings, constraints_embeddings, action_embeddings, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(bs, 5, self.input_emb_size)

            #Model Type 1: (g,s,a)
            # u = torch.stack(
            #     (goal_embeddings, state_embeddings, action_embeddings), dim=1
            # ).permute(0, 2, 1, 3).reshape(bs, 3, self.input_emb_size)
        elif self.type_input == 'BL4D':
            # u = torch.cat([returns_embeddings, constraints_embeddings, action_embeddings, state_embeddings], dim=-1)
            pass

        outputs = torch.zeros(bs, u.shape[1], self.d_model, dtype=u.dtype, device=u.device)

        for i in range(u.shape[1]):
            ret_y = self.backbone(u[:, i:i+1], inference_params=self.inference_params)
            outputs[:, i] = ret_y[:, 0]
            self.inference_params.seqlen_offset += 1
            

        if self.type_input == 'B4LD':
            outputs = outputs.reshape(bs, 1, 5, self.d_model).permute(0, 2, 1, 3)

        state_preds = self.predict_state(outputs[:,-1]) 
        action_preds = self.predict_action(outputs[:,-2])
        
        return state_preds, action_preds

    def get_action_T(self, states, actions, returns_to_go, constraints_to_go, goal, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        if len(states.shape)==3:
            bs = states.shape[0]
        else:
            bs = 1
        K = 35    # the last K timesteps that i want to feed during the inference

        states = states.reshape(bs, -1, self.state_dim)
        actions = actions.reshape(bs, -1, self.act_dim)
        # returns_to_go = returns_to_go.reshape(bs, -1, 1)
        # constraints_to_go = constraints_to_go.reshape(bs, -1, 1)
        goal = goal.reshape(bs, -1, self.state_dim)

        if K is not None:
            if states.shape[1]<=K:
                Kt = states.shape[1]
                states = states[:,-Kt:]
                actions = actions[:,-Kt:]
                # returns_to_go = returns_to_go[:,-Kt:]
                # constraints_to_go = constraints_to_go[:,-Kt:]
                goal = goal[:,-Kt:]
                timesteps = timesteps.reshape(bs, -1)
                

                '''
                Pad states 
                '''

                # pad all tokens to sequence length
                states = torch.cat(
                    [torch.zeros((states.shape[0], Kt-states.shape[1], self.state_dim), device=states.device), states],
                    dim=1).to(dtype=torch.float32)
                actions = torch.cat(
                    [torch.zeros((actions.shape[0], Kt - actions.shape[1], self.act_dim),
                                device=actions.device), actions],
                    dim=1).to(dtype=torch.float32)
                # returns_to_go = torch.cat(
                #     [torch.zeros((returns_to_go.shape[0], Kt-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                #     dim=1).to(dtype=torch.float32)
                # constraints_to_go = torch.cat(
                #     [torch.zeros((constraints_to_go.shape[0], Kt-constraints_to_go.shape[1], 1), device=constraints_to_go.device), constraints_to_go],
                #     dim=1).to(dtype=torch.float32)
                timesteps = torch.cat(
                    [torch.zeros((timesteps.shape[0], Kt-timesteps.shape[0]), device=timesteps.device), timesteps],
                    dim=1
                ).to(dtype=torch.long)
                goal = torch.cat(
                    [torch.zeros((goal.shape[0], Kt-goal.shape[1], self.state_dim), device=states.device), goal],
                    dim=1).to(dtype=torch.float32)
                
            else:
                Kt = K
                states = states[:,-Kt:]
                actions = actions[:,-Kt:]
                # returns_to_go = returns_to_go[:,-Kt:]
                # constraints_to_go = constraints_to_go[:,-Kt:]
                goal = goal[:,-Kt:]
                timesteps = timesteps.reshape(bs, -1)
                

                '''
                Pad states 
                '''

                # pad all tokens to sequence length
                states = torch.cat(
                    [torch.zeros((states.shape[0], Kt-states.shape[1], self.state_dim), device=states.device), states],
                    dim=1).to(dtype=torch.float32)
                actions = torch.cat(
                    [torch.zeros((actions.shape[0], Kt - actions.shape[1], self.act_dim),
                                device=actions.device), actions],
                    dim=1).to(dtype=torch.float32)
                # returns_to_go = torch.cat(
                #     [torch.zeros((returns_to_go.shape[0], Kt-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                #     dim=1).to(dtype=torch.float32)
                # constraints_to_go = torch.cat(
                #     [torch.zeros((constraints_to_go.shape[0], Kt-constraints_to_go.shape[1], 1), device=constraints_to_go.device), constraints_to_go],
                #     dim=1).to(dtype=torch.float32)
                timesteps = torch.cat(
                    [torch.zeros((timesteps.shape[0], Kt-timesteps.shape[0]), device=timesteps.device), timesteps],
                    dim=1
                ).to(dtype=torch.long)
                goal = torch.cat(
                    [torch.zeros((goal.shape[0], Kt-goal.shape[1], self.state_dim), device=states.device), goal],
                    dim=1).to(dtype=torch.float32)
        # print(states.shape)
        # state_preds, action_preds= self.forward(
        #     states, actions, returns_to_go, constraints_to_go, goal, timesteps, running=True, **kwargs)
        state_preds, action_preds= self.forward(
            states, actions, goal, timesteps, running=True, **kwargs)
            
        return state_preds[:,states.shape[1]-1], action_preds[:,states.shape[1]-1]  #accessing last element N-1^th element


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)