import os
import sys

import random
from dataclasses import field
from functools import partial
import math
import random
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam


from mamba.mamba_ssm.modules.mamba_simple import Mamba
from mamba.mamba_ssm.modules.mamba2 import Mamba2
from mamba.mamba_ssm.modules.mha import MHA
from mamba.mamba_ssm.modules.block import Block
from mamba.mamba_ssm.utils.generation import GenerationMixin
from mamba.mamba_ssm.modules.mlp import GatedMLP
from mamba.mamba_ssm.utils.generation import InferenceParams

try:
    from mamba.mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


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

class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.residual_in_fp32 = residual_in_fp32

        #self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        d_intermediate = d_model
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        hidden_states = input_ids

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params = inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32
            )
        return hidden_states


class Mamba_Traj(nn.Module, GenerationMixin):
    def __init__(
        self,
        state_dim: int,      # state dimension in real dyanmics
        act_dim: int,        # action dimension in real dynamics
        d_model: int,        # the embedding_dim is used to get voab_size then projected to d_model space
        n_layer: int,        # Number of mamba blocks you want to keep
        ssm_cfg=None,
        rms_norm: bool=True,
        residual_in_fp32: bool=True,
        fused_add_norm: bool=True,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.d_model = d_model
        
        n_layer = n_layer
        ssm_cfg = ssm_cfg
        rms_norm = rms_norm
        residual_in_fp32 = residual_in_fp32
        fused_add_norm = fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.goal_emb = nn.Linear(self.state_dim, d_model)  # goal
        self.state_encoder = nn.Linear(self.state_dim, d_model)
        self.action_embeddings = nn.Linear(self.act_dim, d_model)

        self.embed_ln = nn.LayerNorm(d_model)

        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )

        self.predict_state = torch.nn.Linear(d_model, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(d_model, act_dim)] + ([nn.Tanh()]))
        )
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.inference_params = InferenceParams(max_seqlen=300, max_batch_size=1)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def forward(self, states, actions, goal, timesteps, returns_to_go = None, constraints_to_go = None, running=False, cache=None,
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

        batch_size = states.shape[0]
        sequence_len = actions.shape[1]

        state_embeddings = self.state_encoder(states)
        goal_embeddings = self.goal_emb(goal)
        action_embeddings = self.action_embeddings(actions)

        u = torch.stack(
            (goal_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*(sequence_len), self.d_model)
        
        u = self.embed_ln(u)
        y = self.backbone(u, inference_params=inference_params)
        y = y.reshape(batch_size, sequence_len, 3, self.d_model).permute(0, 2, 1, 3)
        
        state_preds = self.predict_state(y[:,-1])    # predict next state given state and action
        action_preds = self.predict_action(y[:,-2])  # predict next action given state
    
        return state_preds, action_preds

    def get_action_RNN(self, states, actions, goal, timesteps, returns_to_go = None, constraints_to_go = None, position_ids=None, num_last_tokens=0, **kwargs):
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
        goal = goal.reshape(bs, -1, self.state_dim)[:,-1]
        # returns_to_go = returns_to_go.reshape(bs, -1, 1)[:,-1]
        # constraints_to_go = constraints_to_go.reshape(bs, -1, 1)[:,-1]
           
        state_embeddings = self.state_encoder(states)
        action_embeddings = self.action_embeddings(actions)
        goal_embeddings = self.goal_emb(goal) 
        # returns_embeddings = self.ret_emb(returns_to_go)
        # constraints_embeddings = self.ctg_emb(constraints_to_go)
               
        state_embeddings   = state_embeddings.unsqueeze(1)
        action_embeddings  = action_embeddings.unsqueeze(1)
        goal_embeddings = goal_embeddings.unsqueeze(1)
        # returns_embeddings = returns_embeddings.unsqueeze(1)
        # constraints_embeddings = constraints_embeddings.unsqueeze(1)    

        # Model Type 1: (g,s,a)
        u = torch.stack(
            (goal_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(bs, 3, self.d_model)

        u = self.embed_ln(u)

        outputs_list = []
        # outputs = torch.zeros(bs, u.shape[1], self.d_model, dtype=u.dtype, device=u.device)

        for i in range(u.shape[1]):
            ret_y = self.backbone(u[:, i:i+1], inference_params=self.inference_params)
            # outputs[:, i] = ret_y[:, 0]
            outputs_list.append(ret_y[:, 0]) 
            self.inference_params.seqlen_offset += 1
            
        outputs = torch.stack(outputs_list, dim=1)
        outputs = outputs.reshape(bs, 1, 3, self.d_model).permute(0, 2, 1, 3)
        state_preds = self.predict_state(outputs[:,-1]) 
        action_preds = self.predict_action(outputs[:,-2])
        
        return state_preds, action_preds

    def get_action_T(self, states, actions, goal, timesteps, returns_to_go, constraints_to_go,  **kwargs):
        if len(states.shape)==3:
            bs = states.shape[0]
        else:
            bs = 1
        K = 10   # the last K timesteps that i want to feed during the inference

        states = states.reshape(bs, -1, self.state_dim)
        actions = actions.reshape(bs, -1, self.act_dim)
        goal = goal.reshape(bs, -1, self.state_dim)
        # returns_to_go = returns_to_go.reshape(bs, -1, 1)
        # constraints_to_go = constraints_to_go.reshape(bs, -1, 1)

        if states.shape[1]<=K:
            Kt = states.shape[1]
            states = states[:,-Kt:]
            actions = actions[:,-Kt:]
            goal = goal[:,-Kt:]
            # returns_to_go = returns_to_go[:,-Kt:]
            # constraints_to_go = constraints_to_go[:,-Kt:]
            # timesteps = timesteps.reshape(bs, -1)
        
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
            goal = torch.cat(
                [torch.zeros((goal.shape[0], Kt-goal.shape[1], self.state_dim), device=states.device), goal],
                dim=1).to(dtype=torch.float32)
            # returns_to_go = torch.cat(
            #     [torch.zeros((returns_to_go.shape[0], Kt-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
            #     dim=1).to(dtype=torch.float32)
            # constraints_to_go = torch.cat(
            #     [torch.zeros((constraints_to_go.shape[0], Kt-constraints_to_go.shape[1], 1), device=constraints_to_go.device), constraints_to_go],
            #     dim=1).to(dtype=torch.float32)
            # timesteps = torch.cat(
            #     [torch.zeros((timesteps.shape[0], Kt-timesteps.shape[0]), device=timesteps.device), timesteps],
            #     dim=1
            # ).to(dtype=torch.long)
            
        else:
            Kt = K
            states = states[:,-Kt:]
            actions = actions[:,-Kt:]
            goal = goal[:,-Kt:]
            # returns_to_go = returns_to_go[:,-Kt:]
            # constraints_to_go = constraints_to_go[:,-Kt:]
            # timesteps = timesteps.reshape(bs, -1)
            

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
            goal = torch.cat(
                [torch.zeros((goal.shape[0], Kt-goal.shape[1], self.state_dim), device=states.device), goal],
                dim=1).to(dtype=torch.float32)
            # returns_to_go = torch.cat(
            #     [torch.zeros((returns_to_go.shape[0], Kt-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
            #     dim=1).to(dtype=torch.float32)
            # constraints_to_go = torch.cat(
            #     [torch.zeros((constraints_to_go.shape[0], Kt-constraints_to_go.shape[1], 1), device=constraints_to_go.device), constraints_to_go],
            #     dim=1).to(dtype=torch.float32)
            # timesteps = torch.cat(
            #     [torch.zeros((timesteps.shape[0], Kt-timesteps.shape[0]), device=timesteps.device), timesteps],
            #     dim=1
            # ).to(dtype=torch.long)
            
        state_preds, action_preds= self.forward(
            states, actions, goal, returns_to_go, constraints_to_go, timesteps, running=True, **kwargs)
        
            
        return state_preds[:,states.shape[1]-1], action_preds[:,states.shape[1]-1]  #accessing last element N-1^th element

