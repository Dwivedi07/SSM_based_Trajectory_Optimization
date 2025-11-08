import os
import sys
import argparse

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from collections import defaultdict
from typing import Optional, Mapping, Tuple, Union
import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# from s4.py
from SSM.s4 import S4Block 
from SSM.s4 import DropoutNd  
 

'''
To change the model:
    - self.N - d_state
    - self.Nlayers
    - model_saved_name
    - model_directory
    - inferenece type - ol/dyn
'''

###############################################################################################
########################################### FF S4D Model ##########################################

class TrajectoryFFS4DModel(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        d_model=384,
        n_layers=12,  # Number of S4 layers
        max_ep_len=1000,
        dropout=0.0,
        prenorm=False,
        predict_mode="both"  # "state", "action", or "both"
    ):
        super().__init__()

        self.prenorm = prenorm
        self.d_model = d_model
        self.predict_mode = predict_mode  # Defines inference behavior
        self.projection_dim = 2*d_model
        # Embedding layers for each input modality
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(act_dim, d_model)
        self.embed_goal = nn.Linear(state_dim, d_model)
        self.embed_timestep = nn.Embedding(max_ep_len, d_model)  # Time embeddings

        # **Projection layer** to combine goal, state, and action into a single feature
        self.feature_projection = nn.Linear(3 * d_model, self.projection_dim)

        # S4 layers
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            #S4D Block
            self.s4_layers.append(S4Block(self.projection_dim, dropout=dropout, transposed=True,
                                            mode='diag',
                                            init='diag-lin',
                                            bidirectional=False, 
                                            disc='zoh', 
                                            real_transform='exp'))
            self.norms.append(nn.LayerNorm(self.projection_dim))
            self.dropouts.append(DropoutNd(dropout))

        # Prediction heads
        self.predict_action = nn.Linear(self.projection_dim, act_dim)
        self.predict_state = nn.Linear(self.projection_dim, state_dim)

        # State storage for autoregressive inference
        self.autoregressive_states = None

    def forward(self, states, actions, goals, returns_to_go, constraints_to_go, timesteps):
        """
        Inputs:
        - states: (B, L, state_dim)
        - actions: (B, L, act_dim)
        - goals: (B, L, 1)
        - timesteps: (B, L)
        """

        batch_size, seq_length = states.shape[0], states.shape[1]

        # Step 1: Embed individual modalities
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        goal_embeddings = self.embed_goal(goals)
        time_embeddings = self.embed_timestep(timesteps)

        # Step 2: Combine embeddings
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        goal_embeddings += time_embeddings

        # Step 3: Concatenate and project into single feature space
        combined_features = torch.cat((goal_embeddings, state_embeddings, action_embeddings), dim=-1)
        projected_features = self.feature_projection(combined_features)  # Shape: (B, L, projection_dim)

        # Step 4: Pass through S4 layers
        stacked_inputs = projected_features.transpose(-1, -2)  # (B, L, projection_dim) -> (B, projection_dim, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = stacked_inputs
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)  # S4 block
            z = dropout(z)
            stacked_inputs = z + stacked_inputs  # Residual connection

            if not self.prenorm:
                stacked_inputs = norm(stacked_inputs.transpose(-1, -2)).transpose(-1, -2)

        # Step 5: Output predictions
        s4_outputs = stacked_inputs.transpose(-1, -2)  # (B, projection_dim, L) -> (B, L, projection_dim)

        # Training: predict both state and action
        state_preds = self.predict_state(s4_outputs)
        action_preds = self.predict_action(s4_outputs)

        return state_preds, action_preds

    def setup_autoregressive(self, batch_size, device):
        """Sets up the default states for autoregressive inference."""
        self.autoregressive_states = []
        for layer in self.s4_layers:
            layer.setup_step()
            self.autoregressive_states.append(
                layer.default_state(batch_size, device=device)
            )

    def autoregressive_step(self, current_goal, current_state=None, current_action=None, current_timestep=None):
        """
        Performs one autoregressive step using the `step()` method of S4 layers.
        
        Inputs:
        - current_goal: (B, 1)
        - current_state: (B, state_dim) [Used if predicting action]
        - current_action: (B, act_dim) [Used if predicting state]
        - current_timestep: (B,)

        Returns:
        - Either next state or next action prediction based on `predict_mode`
        """

        batch_size = current_goal.shape[0]
        
        # Step 1: Embed inputs
        goal_embedding = self.embed_goal(current_goal)
        time_embedding = self.embed_timestep(current_timestep)

        if self.predict_mode == "action":
            assert current_state is not None, "State input required for action prediction"
            state_embedding = self.embed_state(current_state) + time_embedding
            action_embedding = self.embed_action(current_action) + time_embedding
            goal_embedding = goal_embedding + time_embedding
            combined_features = torch.cat((goal_embedding, state_embedding, action_embedding), dim=-1)
        elif self.predict_mode == "state":
            assert current_action is not None, "Action input required for state prediction"
            state_embedding = self.embed_state(current_state) + time_embedding
            action_embedding = self.embed_action(current_action) + time_embedding
            goal_embedding = goal_embedding + time_embedding
            combined_features = torch.cat((goal_embedding, state_embedding, action_embedding), dim=-1)
        else:
            raise ValueError("Invalid `predict_mode`, choose 'state' or 'action'.")

        # Step 2: Project combined features
        projected_features = self.feature_projection(combined_features)

        # Step 3: Process through S4 layers
        for i, layer in enumerate(self.s4_layers):
            projected_features, new_state = layer.step(projected_features, self.autoregressive_states[i])
            self.autoregressive_states[i] = new_state

        # Step 4: Predict next state or action
        if self.predict_mode == "action":
            return self.predict_action(projected_features)
        else:
            return self.predict_state(projected_features)


###############################################################################################
########################################### FF S4 Model ##########################################
class TrajectoryFFS4Model(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        d_model=384,
        n_layers=12,  # Number of S4 layers
        max_ep_len=1000,
        dropout=0.0,
        prenorm=False,
        predict_mode="both"  # "state", "action", or "both"
    ):
        super().__init__()

        self.prenorm = prenorm
        self.d_model = d_model
        self.predict_mode = predict_mode  # Defines inference behavior
        self.projection_dim = 2*d_model
        # Embedding layers for each input modality
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(act_dim, d_model)
        self.embed_goal = nn.Linear(state_dim, d_model)
        self.embed_timestep = nn.Embedding(max_ep_len, d_model)  # Time embeddings

        # **Projection layer** to combine goal, state, and action into a single feature
        self.feature_projection = nn.Linear(3 * d_model, self.projection_dim)

        # S4 layers
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            #S4D Block
            self.s4_layers.append(S4Block(self.projection_dim, dropout=dropout, transposed=True,
                                            mode='diag',
                                            init='diag-lin',
                                            bidirectional=False, 
                                            disc='zoh', 
                                            real_transform='exp'))
            #S4 Block
            # self.s4_layers.append(S4Block(self.projection_dim, dropout=dropout, transposed=True))
            self.norms.append(nn.LayerNorm(self.projection_dim))
            self.dropouts.append(DropoutNd(dropout))

        # Prediction heads
        self.predict_action = nn.Linear(self.projection_dim, act_dim)
        self.predict_state = nn.Linear(self.projection_dim, state_dim)

        # State storage for autoregressive inference
        self.autoregressive_states = None

    def forward(self, states, actions, goals, timesteps):
        """
        Inputs:
        - states: (B, L, state_dim)
        - actions: (B, L, act_dim)
        - goals: (B, L, 1)
        - timesteps: (B, L)
        """

        batch_size, seq_length = states.shape[0], states.shape[1]

        # Step 1: Embed individual modalities
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        goal_embeddings = self.embed_goal(goals)
        time_embeddings = self.embed_timestep(timesteps)

        # Step 2: Combine embeddings
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        goal_embeddings += time_embeddings

        # Step 3: Concatenate and project into single feature space
        combined_features = torch.cat((goal_embeddings, state_embeddings, action_embeddings), dim=-1)
        projected_features = self.feature_projection(combined_features)  # Shape: (B, L, projection_dim)

        # Step 4: Pass through S4 layers
        stacked_inputs = projected_features.transpose(-1, -2)  # (B, L, projection_dim) -> (B, projection_dim, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = stacked_inputs
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)  # S4 block
            z = dropout(z)
            stacked_inputs = z + stacked_inputs  # Residual connection

            if not self.prenorm:
                stacked_inputs = norm(stacked_inputs.transpose(-1, -2)).transpose(-1, -2)

        # Step 5: Output predictions
        s4_outputs = stacked_inputs.transpose(-1, -2)  # (B, projection_dim, L) -> (B, L, projection_dim)

        # Training: predict both state and action
        state_preds = self.predict_state(s4_outputs)
        action_preds = self.predict_action(s4_outputs)

        return state_preds, action_preds

    def setup_autoregressive(self, batch_size, device):
        """Sets up the default states for autoregressive inference."""
        self.autoregressive_states = []
        for layer in self.s4_layers:
            layer.setup_step()
            self.autoregressive_states.append(
                layer.default_state(batch_size, device=device)
            )

    def autoregressive_step(self, current_goal, current_state=None, current_action=None, current_timestep=None):
        """
        Performs one autoregressive step using the `step()` method of S4 layers.
        
        Inputs:
        - current_goal: (B, 1)
        - current_state: (B, state_dim) [Used if predicting action]
        - current_action: (B, act_dim) [Used if predicting state]
        - current_timestep: (B,)

        Returns:
        - Either next state or next action prediction based on `predict_mode`
        """

        batch_size = current_goal.shape[0]
        
        # Step 1: Embed inputs
        goal_embedding = self.embed_goal(current_goal)
        time_embedding = self.embed_timestep(current_timestep)

        if self.predict_mode == "action":
            assert current_state is not None, "State input required for action prediction"
            state_embedding = self.embed_state(current_state) + time_embedding
            action_embedding = self.embed_action(current_action) + time_embedding
            combined_features = torch.cat((goal_embedding, state_embedding, action_embedding), dim=-1)
        elif self.predict_mode == "state":
            assert current_action is not None, "Action input required for state prediction"
            state_embedding = self.embed_state(current_state) + time_embedding
            action_embedding = self.embed_action(current_action) + time_embedding
            combined_features = torch.cat((goal_embedding, state_embedding, action_embedding), dim=-1)
        else:
            raise ValueError("Invalid `predict_mode`, choose 'state' or 'action'.")

        # Step 2: Project combined features
        projected_features = self.feature_projection(combined_features)

        # Step 3: Process through S4 layers
        for i, layer in enumerate(self.s4_layers):
            projected_features, new_state = layer.step(projected_features, self.autoregressive_states[i])
            self.autoregressive_states[i] = new_state

        # Step 4: Predict next state or action
        if self.predict_mode == "action":
            return self.predict_action(projected_features)
        else:
            return self.predict_state(projected_features)




###############################################################################################
########################################### FF S4 Full Model ##########################################
class TrajectoryFFS4FULLModel(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        d_model=384,
        n_layers=12,  # Number of S4 layers
        max_ep_len=1000,
        dropout=0.0,
        prenorm=False,
        predict_mode="both"  # "state", "action", or "both"
    ):
        super().__init__()

        self.prenorm = prenorm
        self.d_model = d_model
        self.predict_mode = predict_mode  # Defines inference behavior
        self.projection_dim = 2*d_model
        # Embedding layers for each input modality
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(act_dim, d_model)
        self.embed_goal = nn.Linear(state_dim, d_model)
        self.embed_return = nn.Linear(1, d_model)
        self.embed_constraint = nn.Linear(1, d_model)
        self.embed_timestep = nn.Embedding(max_ep_len, d_model)  # Time embeddings

        # **Projection layer** to combine return, constrs, goal, state, and action into a single feature
        self.feature_projection = nn.Linear(5 * d_model, self.projection_dim)

        # S4 layers
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            #S4D Block
            self.s4_layers.append(S4Block(self.projection_dim, dropout=dropout, transposed=True,
                                            mode='diag',
                                            init='diag-lin',
                                            bidirectional=False, 
                                            disc='zoh', 
                                            real_transform='exp'))
            #S4 Block
            # self.s4_layers.append(S4Block(self.projection_dim, dropout=dropout, transposed=True))
            self.norms.append(nn.LayerNorm(self.projection_dim))
            self.dropouts.append(DropoutNd(dropout))

        # Prediction heads
        self.predict_action = nn.Linear(self.projection_dim, act_dim)
        self.predict_state = nn.Linear(self.projection_dim, state_dim)

        # State storage for autoregressive inference
        self.autoregressive_states = None

    def forward(self, states, actions, goals, returns_to_go, constraints_to_go, timesteps):
        """
        Inputs:
        - states: (B, L, state_dim)
        - actions: (B, L, act_dim)
        - goals: (B, L, state_dim)
        - returns_to_go: (B, L, 1)
        - constranints_to_go: (B, L, 1)
        - timesteps: (B, L)
        """

        batch_size, seq_length = states.shape[0], states.shape[1]

        # Step 1: Embed individual modalities
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        goal_embeddings = self.embed_goal(goals)
        returns_embeddings = self.embed_return(returns_to_go)
        constraints_embeddings = self.embed_constraint(constraints_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Step 2: Combine embeddings
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        goal_embeddings += time_embeddings
        returns_embeddings += time_embeddings
        constraints_embeddings += time_embeddings

        # Step 3: Concatenate and project into single feature space
        combined_features = torch.cat((returns_embeddings, constraints_embeddings, goal_embeddings, state_embeddings, action_embeddings), dim=-1)
        projected_features = self.feature_projection(combined_features)  # Shape: (B, L, projection_dim)

        # Step 4: Pass through S4 layers
        stacked_inputs = projected_features.transpose(-1, -2)  # (B, L, projection_dim) -> (B, projection_dim, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = stacked_inputs
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)  # S4 block
            z = dropout(z)
            stacked_inputs = z + stacked_inputs  # Residual connection

            if not self.prenorm:
                stacked_inputs = norm(stacked_inputs.transpose(-1, -2)).transpose(-1, -2)

        # Step 5: Output predictions
        s4_outputs = stacked_inputs.transpose(-1, -2)  # (B, projection_dim, L) -> (B, L, projection_dim)

        # Training: predict both state and action
        state_preds = self.predict_state(s4_outputs)
        action_preds = self.predict_action(s4_outputs)

        return state_preds, action_preds

    def setup_autoregressive(self, batch_size, device):
        """Sets up the default states for autoregressive inference."""
        self.autoregressive_states = []
        for layer in self.s4_layers:
            layer.setup_step()
            self.autoregressive_states.append(
                layer.default_state(batch_size, device=device)
            )

    def autoregressive_step(self, current_goal, current_state=None, current_action=None, current_timestep=None):
        """
        Performs one autoregressive step using the `step()` method of S4 layers.
        
        Inputs:
        - current_goal: (B, 1)
        - current_state: (B, state_dim) [Used if predicting action]
        - current_action: (B, act_dim) [Used if predicting state]
        - current_timestep: (B,)

        Returns:
        - Either next state or next action prediction based on `predict_mode`
        """

        batch_size = current_goal.shape[0]
        
        # Step 1: Embed inputs
        goal_embedding = self.embed_goal(current_goal)
        time_embedding = self.embed_timestep(current_timestep)

        if self.predict_mode == "action":
            assert current_state is not None, "State input required for action prediction"
            state_embedding = self.embed_state(current_state) + time_embedding
            action_embedding = self.embed_action(current_action) + time_embedding
            combined_features = torch.cat((goal_embedding, state_embedding, action_embedding), dim=-1)
        elif self.predict_mode == "state":
            assert current_action is not None, "Action input required for state prediction"
            state_embedding = self.embed_state(current_state) + time_embedding
            action_embedding = self.embed_action(current_action) + time_embedding
            combined_features = torch.cat((goal_embedding, state_embedding, action_embedding), dim=-1)
        else:
            raise ValueError("Invalid `predict_mode`, choose 'state' or 'action'.")

        # Step 2: Project combined features
        projected_features = self.feature_projection(combined_features)

        # Step 3: Process through S4 layers
        for i, layer in enumerate(self.s4_layers):
            projected_features, new_state = layer.step(projected_features, self.autoregressive_states[i])
            self.autoregressive_states[i] = new_state

        # Step 4: Predict next state or action
        if self.predict_mode == "action":
            return self.predict_action(projected_features)
        else:
            return self.predict_state(projected_features)


###############################################################################################
########################################### RPO Model ##########################################


class TrajectoryS4Model(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        d_model=384,
        n_layers=12, # 12, 16, 20, 24
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
        '''
        For using the minimal realization of S4D we will use follwing arguments set as:
        S4(mode='diag', init='diag-lin', bidirectional=False, disc='zoh', real_transform='exp')
        '''
        for _ in range(n_layers):
            #for S4D
            self.s4_layers.append(S4Block(d_model, dropout=dropout, transposed=True,
                                            mode='diag',
                                            init='diag-lin',
                                            bidirectional=False, 
                                            disc='zoh', 
                                            real_transform='exp'))
            #for S4
            # self.s4_layers.append(S4Block(d_model, dropout=dropout, transposed=True))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))

        # Prediction heads
        self.predict_action = torch.nn.Linear(d_model, act_dim)
        self.predict_state = torch.nn.Linear(d_model, state_dim)  # Predict next state

        # State storage for autoregressive inference
        self.autoregressive_states = None
    
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
        time_before_layer = time.time()
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = stacked_inputs
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)  # Prenorm
            
            z, _ = layer(z)  # S4D block
            z = dropout(z)
            stacked_inputs = z + stacked_inputs  # Residual connection

            if not self.prenorm:
                stacked_inputs = norm(stacked_inputs.transpose(-1, -2)).transpose(-1, -2)  # Postnorm

        time_afterall_layer = time.time()
        tot_time_between_l = time_afterall_layer - time_before_layer
        # print(print(f'In this inference: Layer Pass time is: {tot_time_between_l}'))
        s4_outputs = stacked_inputs


        s4_outputs = stacked_inputs.transpose(-1, -2)  # (B, d_model, L) -> (B, L, d_model)
        # print('shape of s4d stacked outputs:', s4_outputs.shape)
        s4_outputs = s4_outputs.reshape(batch_size, seq_length, 4, d_model).permute(0, 2, 1, 3)
        # print('After brekaing one all modalities shape of s4d stacked outputs:', s4_outputs.shape)
        # Predict next action and next state
        
        state_preds = self.predict_state(s4_outputs[:, 3])   # predict next state given state and action
        action_preds = self.predict_action(s4_outputs[:, 2]) # predict next action given state

        return state_preds, action_preds

    def setup_autoregressive(self, batch_size, device):
        """Sets up the default states for autoregressive inference."""
        self.autoregressive_states = []
        for layer in self.s4_layers:
            layer.setup_step()
            self.autoregressive_states.append(
                layer.default_state(batch_size, device=device)
            )

    def autoregressive_step(self, current_state = None, current_action = None, current_timestep = None):
        """
        Performs one autoregressive step using the `step()` method of S4 layers.
        
        Inputs:
        - current_state: (B, state_dim)
        - current_action: (B, act_dim)
        - current_rtgs: (B, 1)
        - current_ctgs: (B, 1)
        - current_timestep: (B,)
        
        Returns:
        - state_pred: (B, state_dim)
        - action_pred: (B, act_dim)
        """

        batch_size = current_state.shape[0] if current_state is not None else current_action.shape[0]
        d_model = 384
        # Embed inputs
        if current_state is not None:
            state_embedding = self.embed_state(current_state)
        if current_action is not None: 
            action_embedding = self.embed_action(current_action)

        # state_embedding = self.embed_state(current_state)
        # action_embedding = self.embed_action(current_action)
        # rtg_embedding = self.embed_return(current_rtgs)
        # ctg_embedding = self.embed_constraint(current_ctgs)
        time_embedding = self.embed_timestep(current_timestep)

        # time embeddings are treated similar to positional embeddings
        if current_state is not None:
            state_embeddings = state_embedding + time_embedding
        if current_action is not None: 
            action_embeddings = action_embedding + time_embedding
        
        
        # returns_embeddings = rtg_embedding + time_embedding
        # constraints_embeddings = ctg_embedding + time_embedding

        # which works nice in an autoregressive sense since states predict actions
        # stacked_inputs = (
        #     torch.stack((returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1)
        # )
        stacked_inputs = state_embeddings if current_state is not None else action_embeddings

        # stacked_inputs = stacked_inputs.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        # Process through each S4 layer using step()
        for i, layer in enumerate(self.s4_layers):
            stacked_inputs, new_state = layer.step(stacked_inputs, self.autoregressive_states[i])
            self.autoregressive_states[i] = new_state  # Update the state

        # Predict the action and state from the last modality
        # state_pred = self.predict_state(stacked_inputs[:, 3, :])  # State is at index 2

        pred_val = self.predict_action(stacked_inputs) if current_state is not None else self.predict_state(stacked_inputs) 

        return pred_val
    
