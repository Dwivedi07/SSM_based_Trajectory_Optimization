import os
import sys

art_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
root_folder = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_folder)
sys.path.append(art_path)

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# SSM modules S6
import dm.manage as DM_manager
from dm.manage import device

# ART modules
import decision_transformer.manage as DT_manager

# FF dynamics modules
from dynamics.freeflyer import FreeflyerModel, ocp_no_obstacle_avoidance, ocp_obstacle_avoidance, check_koz_constraint, compute_constraint_to_go
from optimization.ff_scenario import n_obs, n_time_rpod, obs, table, robot_radius, safety_margin, dt, T
import decision_transformer.manage as DT_manager
from dynamics.FreeflyerEnv import FreeflyerEnv
from decision_transformer.art_closed_loop import AutonomousFreeflyerTransformerMPC, ConvexMPC, MyopicConvexMPC
from optimization.ff_scenario import N_STATE, N_ACTION, iter_max_SCP

import time
import itertools
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

'''
Forecasting analyisis of test dataset samples

'''


if __name__ =='__main__':

    transformer_ws = 'dyn' # 'dyn'/'ol'
    type_model_ = 'rc'   # 'gsa' / 'rc'
    
    if type_model_ == 'gsa':
        # GSA MODEL
        ssm_model_name = "checkpoint_ff_S6_gsa"  #S6 MODEL
        transformer_model_name = 'checkpoint_ff_GSA'  #ART
    else:
        # RCGSA MODEL
        ssm_model_name = "checkpoint_ff_S6_rcgsa"  #S6 MODEL
        transformer_model_name = 'checkpoint_ff_ctgrtg_art'  #ART

    import_config = DT_manager.transformer_import_config(transformer_model_name)
    mdp_constr=import_config['mdp_constr']
    timestep_norm=import_config['timestep_norm']

    '''
    Loading the dataset
    '''
    # Get the datasets and loaders from the torch data
    if type_model_ == 'gsa':
        datasets, dataloaders = DM_manager.get_train_val_test_data(mdp_constr=import_config['mdp_constr'], timestep_norm=import_config['timestep_norm'])
        train_loader, eval_loader, test_loader = dataloaders
        train_dataset, val_dataset, test_dataset = datasets
    else:
        datasets, dataloaders = DT_manager.get_train_val_test_data(mdp_constr=import_config['mdp_constr'], timestep_norm=import_config['timestep_norm'])
        train_loader, eval_loader, test_loader = dataloaders
        train_dataset, val_dataset, test_dataset = datasets
    

    # Loading the models
    model = DT_manager.get_DT_model(transformer_model_name, train_loader, eval_loader)
    model_S6 = DM_manager.get_DM_model(ssm_model_name, train_loader, eval_loader)

    N_data_test = test_loader.dataset.n_data
    data_stats = test_loader.dataset.data_stats


    ''' Initialize storage arrays '''
    test_dataset_idx = np.zeros(N_data_test, dtype=int)

    # Feasibility flags
    feasibility_list_CVX = np.full(N_data_test, False, dtype=bool)
    feasibility_list_DT = np.full(N_data_test, False, dtype=bool)
    feasibility_list_S6 = np.full(N_data_test, False, dtype=bool)

    # Costs
    ctgs0_cvx = np.zeros(N_data_test, dtype=float)
    J_list_CVX = np.zeros(N_data_test, dtype=float)
    J_list_DT = np.zeros(N_data_test, dtype=float)
    J_list_S6 = np.zeros(N_data_test, dtype=float)

    # J vectors
    J_vect_scp_list_CVX = np.zeros(shape=(N_data_test,), dtype=object)
    J_vect_scp_list_DT = np.zeros(shape=(N_data_test,), dtype=object)  # For storing arrays
    J_vect_scp_list_S6 = np.zeros(shape=(N_data_test,), dtype=object)  # For storing arrays

    # warmstart runtime metrics
    runtime_list_CVX = np.zeros(N_data_test, dtype=float)
    runtime_list_DT = np.zeros(N_data_test, dtype=float)
    runtime_list_S6 = np.zeros(N_data_test, dtype=float)

    # SCP runtime metrics
    runtime_scp_list_CVX = np.zeros(N_data_test, dtype=float)
    runtime_scp_list_DT = np.zeros(N_data_test, dtype=float)
    runtime_scp_list_S6 = np.zeros(N_data_test, dtype=float)

    # SCP Iterations
    iter_scp_list_CVX = np.zeros(N_data_test, dtype=int)
    iter_scp_list_DT = np.zeros(N_data_test, dtype=int)
    iter_scp_list_S6 = np.zeros(N_data_test, dtype=int)

    # Errors
    trajectory_rmse_list_DT = np.zeros(N_data_test, dtype=float)
    trajectory_rmse_list_S6 = np.zeros(N_data_test, dtype=float)
    control_error_list_DT = np.zeros(N_data_test, dtype=float)
    control_error_list_S6 = np.zeros(N_data_test, dtype=float)
    final_state_error_list_DT = np.zeros(N_data_test, dtype=float)
    final_state_error_list_S6 = np.zeros(N_data_test, dtype=float)

    k = 0
    # print(f"N_data_test from dataset: {test_loader.dataset.n_data}")
    # print(f"Total batches in test_loader: {len(test_loader)}")
    # print(f"Batch size: {test_loader.batch_size}")
    
    for idx, test_sample in enumerate(test_loader):
        out = {}
        test_dataset_idx[idx] = idx
        # Extract true trajectory
        if mdp_constr: #always true
            states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, dt, time_sec, ix = test_sample
            unnorm_states_i = np.empty_like(states_i)
            unnorm_actions_i = np.empty_like(actions_i)
            unnorm_goal_i = np.empty_like(goal_i)
            for t in range(states_i.shape[1]):
                unnorm_states_i[:, t, :] = (states_i[:, t, :] * data_stats['states_std'][t]) + data_stats['states_mean'][t]
                unnorm_actions_i[:, t, :] = (actions_i[:, t, :] * (data_stats['actions_std'][t]+1e-6)) + data_stats['actions_mean'][t]
                unnorm_goal_i[:, t, :] =  (goal_i[:, t, :] * (data_stats['goal_std'][t])) + data_stats['goal_mean'][t]
        
            states_i_np = unnorm_states_i.squeeze(0).T 
            actions_i_np = unnorm_actions_i.squeeze(0).T 
            goal_i_np = unnorm_goal_i.squeeze(0).T

        #Freeflyer model
        state_init = ((test_sample[0][0,0,:] * data_stats['states_std'][0]) + (data_stats['states_mean'][0])).cpu().numpy()
        state_final = ((test_sample[4][0,0,:] * data_stats['goal_std'][0]) + (data_stats['goal_mean'][0])).cpu().numpy()
        
        ffm = FreeflyerModel(verbose=True)
        dt = dt.item()


        ################################################# COnvex relaxation problem
        # Obstacles info
        obs_positions = obs['position']
        obs_radii = (obs['radius'] + robot_radius)*safety_margin
        # Solve Convex Problem
        runtime_CVX0 = time.time()
        traj_cvx, _, n_iter_cvx, feas_cvx = ocp_no_obstacle_avoidance(ffm, state_init, state_final)
        out['runtime_CVX'] = time.time() - runtime_CVX0
        states_ws_cvx, actions_ws_cvx = traj_cvx['states'], traj_cvx['actions_G']
        out['J_cvx'] = np.sum(la.norm(actions_ws_cvx, ord=1, axis=0))
        constr_cvx, constr_viol_cvx= check_koz_constraint(states_ws_cvx.T, obs_positions, obs_radii)
        ctgs_cvx = compute_constraint_to_go(states_ws_cvx.T, obs['position'], (obs['radius'] + robot_radius)*safety_margin)
        out['ctgs0_cvx'] = ctgs_cvx[0,0]

        # Solve SCP
        runtime0_scp_cvx = time.time()
        traj_scp_cvx, J_vect_scp_cvx, iter_scp_cvx, feas_scp_cvx = ocp_obstacle_avoidance(ffm, states_ws_cvx+np.array([0,0.,0,0,0,0]).reshape(-1,1), actions_ws_cvx, state_init, state_final)
        runtime1_scp_cvx = time.time()
        out['runtime_scp_cvx'] = runtime1_scp_cvx - runtime0_scp_cvx
        out['iter_scp_cvx'] = iter_scp_cvx
        out['J_vect_scp_cvx'] = J_vect_scp_cvx
        if np.char.equal(feas_scp_cvx,'optimal'):
            states_scp_cvx, actions_scp_cvx = traj_scp_cvx['states'], traj_scp_cvx['actions_G']
            constr_scp_cvx, constr_viol_scp_cvx = check_koz_constraint(states_scp_cvx.T, obs_positions, obs_radii)
            out['feasible_CVX'] = True
        else:
            out['feasible_CVX'] = False
        '''
        Storing CVX results in list
        '''
        ctgs0_cvx[idx] = out['ctgs0_cvx']
        runtime_list_CVX[idx] = out['runtime_CVX']
        runtime_scp_list_CVX[idx] = out['runtime_scp_cvx']
        iter_scp_list_CVX[idx] = out['iter_scp_cvx'] 
        J_vect_scp_list_CVX[idx] = out['J_vect_scp_cvx']
        J_list_CVX[idx] = out['J_cvx']
        feasibility_list_CVX[idx] = out['feasible_CVX'] 

        ################################################# 
        # inference for both models
        inference_func_S6 = getattr(DM_manager, 'ssm_model_inference_'+transformer_ws)
        inference_func_ART = getattr(DT_manager, 'torch_model_inference_'+transformer_ws)

        rtg = None 
        rtg = - np.sum(la.norm(actions_ws_cvx, ord=1, axis=0)) if mdp_constr else None # Not needed for case with (G,S,A)


        # Warm Start Trajectories
        DT_trajectory, runtime_DT = inference_func_ART(model, test_loader, test_sample, rtg_perc=1., ctg_perc=0., rtg=rtg, ctg_clipped=True)
        out['runtime_DT'] = runtime_DT
        out['J_DT'] = np.sum(la.norm(DT_trajectory['dv_' + transformer_ws], ord=1, axis=0))
        ################################################ ART
        # Warm start states & actions for ART
        states_ws_DT = np.append(DT_trajectory['xypsi_' + transformer_ws], (DT_trajectory['xypsi_' + transformer_ws][:,-1] + ffm.B_imp @ DT_trajectory['dv_' + transformer_ws][:, -1]).reshape((6,1)), 1)# set warm start
        actions_ws_DT = DT_trajectory['dv_' + transformer_ws]
        
        # Solve SCP for ART
        runtime0_scp_DT = time.time()
        traj_scp_DT, J_vect_scp_DT, iter_scp_DT, feas_scp_DT = ocp_obstacle_avoidance(ffm, states_ws_DT, actions_ws_DT, state_init, state_final)
        runtime_scp_DT = time.time() - runtime0_scp_DT        
        
        if np.char.equal(feas_scp_DT,'optimal'):
            # Save scp_DT in the output dictionary
            # Computing RMSE
            states_scp_DT, actions_scp_DT = traj_scp_DT['states'], traj_scp_DT['actions_G']
            states_scp_DT = states_scp_DT[:, :-1] # removing the last time step 101-> 100
            out['feasible_DT'] = True
            out['J_vect_scp_DT'] = J_vect_scp_DT
            out['iter_scp_DT'] = iter_scp_DT
            out['runtime_scp_DT'] = runtime_scp_DT

            # Compute error

            out['trajectory_rmse'] = np.sqrt(np.mean((states_scp_DT - states_i_np) ** 2))
            out['control_error'] = np.sqrt(np.mean((actions_scp_DT - actions_i_np) ** 2))
            out['final_state_error'] = np.sqrt(np.mean((states_scp_DT[:, -1] - goal_i_np[:, -1]) ** 2))
            

        else:
            out['feasible_DT'] = False

        ''' Store results in preallocated arrays '''
        trajectory_rmse_list_DT[idx] = out.get('trajectory_rmse', np.nan)
        control_error_list_DT[idx] = out.get('control_error', np.nan)
        final_state_error_list_DT[idx] = out.get('final_state_error', np.nan)
        feasibility_list_DT[idx] = out.get('feasible_DT')
        runtime_list_DT[idx] = out.get('runtime_DT', np.nan)
        runtime_scp_list_DT[idx] = out.get('runtime_scp_DT', np.nan)
        iter_scp_list_DT[idx] = out.get('iter_scp_DT', 20000)
        J_vect_scp_list_DT[idx] = out.get('J_vect_scp_DT', np.array([]))
        J_list_DT[idx] = out.get('J_DT',np.nan)
        


        ################################################ S6
        S6_trajectory, runtime_S6 = inference_func_S6(model_S6, test_loader, test_sample, rtg_perc=1., ctg_perc=0., rtg=rtg, ctg_clipped=True)
        out['runtime_S6'] = runtime_S6
        out['J_S6'] = np.sum(la.norm(S6_trajectory['dv_' + transformer_ws], ord=1, axis=0))
        # Warm start states & actions for S6
        states_ws_S6 = np.append(S6_trajectory['xypsi_' + transformer_ws], (S6_trajectory['xypsi_' + transformer_ws][:,-1] + ffm.B_imp @ S6_trajectory['dv_' + transformer_ws][:, -1]).reshape((6,1)), 1)# set warm start
        actions_ws_S6 = S6_trajectory['dv_' + transformer_ws]

        # Solve SCP for ART
        runtime0_scp_S6 = time.time()
        traj_scp_S6, J_vect_scp_S6, iter_scp_S6, feas_scp_S6 = ocp_obstacle_avoidance(ffm, states_ws_S6, actions_ws_S6, state_init, state_final)
        runtime_scp_S6 = time.time() - runtime0_scp_DT
        
        if np.char.equal(feas_scp_S6,'optimal'):
            # Save scp_S6 in the output dictionary
            # Computing RMSE
            states_scp_S6, actions_scp_S6 = traj_scp_S6['states'], traj_scp_S6['actions_G']
            states_scp_S6 = states_scp_S6[:, :-1]
            out['feasible_S6'] = True
            out['J_vect_scp_S6'] = J_vect_scp_S6
            out['iter_scp_S6'] = iter_scp_S6
            out['runtime_scp_S6'] = runtime_scp_S6

            # Compute errors
            out['trajectory_rmse'] = np.sqrt(np.mean((states_scp_S6 - states_i_np) ** 2))
            out['control_error'] = np.sqrt(np.mean((actions_scp_S6 - actions_i_np) ** 2))
            out['final_state_error'] = np.sqrt(np.mean((states_scp_S6[:, -1] - goal_i_np[:, -1]) ** 2))
            

        else:
            out['feasible_S6'] = False
        
        ################################################

        ''' Store results in preallocated arrays '''
        trajectory_rmse_list_S6[idx] = out.get('trajectory_rmse', np.nan)
        control_error_list_S6[idx] = out.get('control_error', np.nan)
        final_state_error_list_S6[idx] = out.get('final_state_error', np.nan)
        feasibility_list_S6[idx] = out.get('feasible_S6')
        runtime_list_S6[idx] = out.get('runtime_S6', np.nan)
        runtime_scp_list_S6[idx] = out.get('runtime_scp_S6', np.nan)
        iter_scp_list_S6[idx] = out.get('iter_scp_S6', 20000)
        J_vect_scp_list_S6[idx] = out.get('J_vect_scp_S6', np.array([]))
        J_list_S6[idx] = out.get('J_S6', np.nan)


        # Plotting

        # 3D position trajectory
        # print(len(states_ws_cvx[0,:]), len(states_ws_DT[0,:]), len(states_scp_DT[0,:]))
        ax = plt.figure(figsize=(12,8)).add_subplot()
        p1 = ax.plot(states_ws_cvx[0,:], states_ws_cvx[1,:], color='gray', linewidth=1.5, label='warm-start cvx', zorder=3)
        if states_scp_cvx is not None:
            p2 = ax.plot(states_scp_cvx[0,:], states_scp_cvx[1,:], color='royalblue', linewidth=1.5, label='scp-cvx', zorder=3)
        p3 = ax.plot(states_ws_DT[0,:], states_ws_DT[1,:], color='darkred', linewidth=1.5, label='warm-start ART-' + transformer_ws, zorder=3)
        p4 = ax.plot(states_ws_S6[0,:], states_ws_S6[1,:], color='darkgreen', linewidth=1.5, label='warm-start S6-' + transformer_ws, zorder=3)
        if states_scp_DT is not None:
            p5 = ax.plot(states_scp_DT[0,:], states_scp_DT[1,:], color='black', linewidth=1.5, label='scp-ART-' + transformer_ws, zorder=3)
        if states_scp_S6 is not None:
            p6 = ax.plot(states_scp_S6[0,:], states_scp_S6[1,:], color='violet', linewidth=1.5, label='scp-S6-' + transformer_ws, zorder=3)

        ax.add_patch(Rectangle((0,0), table['xy_up'][0], table['xy_up'][1], fc=(0.5,0.5,0.5,0.2), ec='k', label='table', zorder=2.5))
        for n_obs in range(obs['radius'].shape[0]):
            label_obs = 'obs' if n_obs == 0 else None
            label_robot = 'robot radius' if n_obs == 0 else None
            ax.add_patch(Circle(obs['position'][n_obs,:], obs['radius'][n_obs], fc='r', label=label_obs, zorder=2.5))
            ax.add_patch(Circle(obs['position'][n_obs,:], obs['radius'][n_obs]+robot_radius, fc='r', alpha=0.2, label=label_robot, zorder=2.5))
        ax.scatter(state_init[0], state_init[1], label='state init', zorder=3)
        ax.scatter(state_final[0], state_final[1], label='state final', zorder=3)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]', fontsize=10)
        ax.set_ylabel('Y [m]', fontsize=10)
        ax.grid(True)
        ax.legend(loc='best', fontsize=10)
        plt.savefig(f"{root_folder}/optimization/saved_files/prediction_analysis_S6FULL/plots/pos_3d_{idx}.png" )

        # Constraint satisfaction
        # plt.figure()
        # plt.plot(time_sec, constr_cvx.T, 'k', linewidth=1.5, label='warm-start cvx')
        # plt.plot(time_sec, constr_scp_cvx.T, 'b', linewidth=1.5, label='scp-cvx')
        # plt.plot(time_sec, constr_DT.T, c=[0.5,0.5,0.5], linewidth=1.5, label='warm-start SSM-' + transformer_ws)
        # if actions_scp_DT is not None:
        #     plt.plot(time_sec, constr_scp_DT.T, 'c', linewidth=1.5, label='scp-SSM-' + transformer_ws)
        # plt.plot(time_sec, np.zeros(n_time_rpod+1), 'r-', linewidth=1.5, label='koz')
        # plt.xlabel('time [orbits]', fontsize=10)
        # plt.ylabel('keep-out-zone constraint [-]', fontsize=10)
        # plt.grid(True)
        # plt.legend(loc='best', fontsize=10)
        # plt.savefig(root_folder + '/optimization/saved_files/prediction_analysis_S6FULL/plots/constr_{idx}.png"')

        if idx % 100 == 0:
            ''' Save results '''
            np.savez_compressed(root_folder + '/optimization/saved_files/prediction_analysis_S6FULL/pred_analysis_2_' + transformer_model_name + '_vs_' + ssm_model_name + '_test'+str(idx),
                    # Feasibility flags
                    feasibility_list_CVX = feasibility_list_CVX,
                    feasibility_list_DT = feasibility_list_DT,
                    feasibility_list_S6 = feasibility_list_S6,

                    # Costs
                    ctgs0_cvx = ctgs0_cvx,
                    J_list_CVX = J_list_CVX,
                    J_list_DT = J_list_DT,
                    J_list_S6 = J_list_S6,

                    # J vectors
                    J_vect_scp_list_CVX = J_vect_scp_list_CVX,
                    J_vect_scp_list_DT = J_vect_scp_list_DT,
                    J_vect_scp_list_S6 = J_vect_scp_list_S6,

                    # Warmstart runtime metrics
                    runtime_list_CVX = runtime_list_CVX,
                    runtime_list_DT = runtime_list_DT,
                    runtime_list_S6 = runtime_list_S6,

                    # SCP runtime metrics
                    runtime_scp_list_CVX = runtime_scp_list_CVX,
                    runtime_scp_list_DT = runtime_scp_list_DT,
                    runtime_scp_list_S6 = runtime_scp_list_S6,

                    # SCP Iterations
                    iter_scp_list_CVX = iter_scp_list_CVX,
                    iter_scp_list_DT = iter_scp_list_DT,
                    iter_scp_list_S6 = iter_scp_list_S6,

                    # Errors
                    trajectory_rmse_list_DT = trajectory_rmse_list_DT,
                    trajectory_rmse_list_S6 = trajectory_rmse_list_S6,
                    control_error_list_DT = control_error_list_DT,
                    control_error_list_S6 = control_error_list_S6,
                    final_state_error_list_DT = final_state_error_list_DT,
                    final_state_error_list_S6 = final_state_error_list_S6
                    )
        

    ''' Save results '''
    np.savez_compressed(root_folder + '/optimization/saved_files/prediction_analysis_S6FULL/pred_analysis_2_' + transformer_model_name + '_vs_' + ssm_model_name + '_test',
                    
                    # Feasibility flags
                    feasibility_list_CVX = feasibility_list_CVX,
                    feasibility_list_DT = feasibility_list_DT,
                    feasibility_list_S6 = feasibility_list_S6,

                    # Costs
                    ctgs0_cvx = ctgs0_cvx,
                    J_list_CVX = J_list_CVX,
                    J_list_DT = J_list_DT,
                    J_list_S6 = J_list_S6,

                    # J vectors
                    J_vect_scp_list_CVX = J_vect_scp_list_CVX,
                    J_vect_scp_list_DT = J_vect_scp_list_DT,
                    J_vect_scp_list_S6 = J_vect_scp_list_S6,

                    # Warmstart runtime metrics
                    runtime_list_CVX = runtime_list_CVX,
                    runtime_list_DT = runtime_list_DT,
                    runtime_list_S6 = runtime_list_S6,

                    # SCP runtime metrics
                    runtime_scp_list_CVX = runtime_scp_list_CVX,
                    runtime_scp_list_DT = runtime_scp_list_DT,
                    runtime_scp_list_S6 = runtime_scp_list_S6,

                    # SCP Iterations
                    iter_scp_list_CVX = iter_scp_list_CVX,
                    iter_scp_list_DT = iter_scp_list_DT,
                    iter_scp_list_S6 = iter_scp_list_S6,

                    # Errors
                    trajectory_rmse_list_DT = trajectory_rmse_list_DT,
                    trajectory_rmse_list_S6 = trajectory_rmse_list_S6,
                    control_error_list_DT = control_error_list_DT,
                    control_error_list_S6 = control_error_list_S6,
                    final_state_error_list_DT = final_state_error_list_DT,
                    final_state_error_list_S6 = final_state_error_list_S6
                    )
    # first set of forecastin run invloves following code
    # np.savez_compressed(root_folder + '/optimization/saved_files/prediction_analysis_S6/pred_analysis_' + transformer_model_name + '_vs_' + ssm_model_name + '_test',
    #                 runtime_list_CVX = runtime_list_CVX,
    #                 trajectory_rmse_list_DT = trajectory_rmse_list_DT,
    #                 control_error_list_DT = control_error_list_DT,
    #                 final_state_error_list_DT = final_state_error_list_DT,
    #                 feasibility_list_DT = feasibility_list_DT,
    #                 runtime_list_DT = runtime_list_DT,
    #                 runtime_scp_list_DT = runtime_scp_list_DT,
    #                 iter_scp_list_DT = iter_scp_list_DT,
    #                 J_vect_scp_list_DT = J_vect_scp_list_DT,

    #                 trajectory_rmse_list_S6 = trajectory_rmse_list_S6,
    #                 control_error_list_S6 = control_error_list_S6,
    #                 final_state_error_list_S6 = final_state_error_list_S6,
    #                 feasibility_list_S6 = feasibility_list_S6,
    #                 runtime_list_S6 = runtime_list_S6,
    #                 runtime_scp_list_S6 = runtime_scp_list_S6,
    #                 iter_scp_list_S6 = iter_scp_list_S6,
    #                 J_vect_scp_list_S6 = J_vect_scp_list_S6
    #                 )

    print("Results saved successfully.")

   
    



    
