import os
import sys

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class MambaConfig:
    """
    Configuration class for initializing Mamba and DeepMambaModel parameters.
    """
    def __init__(
        self,
        state_dim: int,  
        act_dim: int,
        d_model: int,
        n_layers: int,
        max_ep_len: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        residual_in_fp32: bool = False,
        fused_add_norm: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_ep_len = max_ep_len
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.conv_bias = conv_bias
        self.bias = bias
        self.use_fast_path = use_fast_path
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.device = device
        self.dtype = dtype



# Create inference parameters
class InferenceParams:
    def __init__(self, batch_size, n_layers, dtype=torch.float32):
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.dtype = dtype
        self.key_value_memory_dict = {}       # Dictionary to store states for each layer
        self.seqlen_offset = 0

    