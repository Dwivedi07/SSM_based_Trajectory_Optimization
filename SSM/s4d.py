"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from SSM.utilities import DropoutNd

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None
    

##################################################################################################################
###########################################TrajectoryOptimization Model ##########################################

class TrajectoryS4dModel(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        d_model=384,
        n_layers=12,
        max_ep_len=1000,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm
        # Embedding layers for each modality
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(act_dim, d_model)
        self.embed_return = nn.Linear(1, d_model)  # Returns-to-go
        self.embed_constraint = nn.Linear(1, d_model)  # Constraints-to-go
        self.embed_timestep = nn.Embedding(max_ep_len, d_model)  # Time embeddings

        # S4D layers
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))

        # Prediction heads
        self.predict_action = torch.nn.Linear(d_model, act_dim)
        self.predict_state = torch.nn.Linear(d_model, state_dim)  # Optional: for next state prediction

    def forward(self, states, actions, returns_to_go, constraints_to_go, timesteps):
        """
        Inputs:
        - states: (B, L, state_dim)
        - actions: (B, L, act_dim)
        - returns_to_go: (B, L, 1)
        - constraints_to_go: (B, L, 1)
        - timesteps: (B, L)
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        d_model = 384
        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns_to_go)
        constraint_embeddings = self.embed_constraint(constraints_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = return_embeddings + time_embeddings
        constraints_embeddings = constraint_embeddings + time_embeddings

        # print('Before combining new the embeddings: shape state:', state_embeddings.shape)
        # this makes the sequence look like (R_1, C_1 s_1, a_1, R_2, C_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 4 * seq_length, d_model)
        )
        # print('Before combining new the embeddings: shape stack:', stacked_inputs.shape)
        

        stacked_inputs = stacked_inputs.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        # Pass through stacked S4D layers
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = stacked_inputs
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)  # Prenorm

            z, _ = layer(z)  # S4D block
            z = dropout(z)
            stacked_inputs = z + stacked_inputs  # Residual connection

            if not self.prenorm:
                stacked_inputs = norm(stacked_inputs.transpose(-1, -2)).transpose(-1, -2)  # Postnorm

        s4d_outputs = stacked_inputs


        s4d_outputs = stacked_inputs.transpose(-1, -2)  # (B, d_model, L) -> (B, L, d_model)
        # print('shape of s4d stacked outputs:', s4d_outputs.shape)
        s4d_outputs = s4d_outputs.reshape(batch_size, seq_length, 4, d_model).permute(0, 2, 1, 3)
        # print('After brekaing one all modalities shape of s4d stacked outputs:', s4d_outputs.shape)
        # Predict next action and next state
        action_preds = self.predict_action(s4d_outputs[:, 2]) # predict next action given state
        state_preds = self.predict_state(s4d_outputs[:, 3])  # predict next state given state and action

        return state_preds, action_preds
