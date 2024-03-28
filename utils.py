import os
import pandas as pd
import pickle
import pdb
import numpy as np
from pandapower.pypower.makeYbus import makeYbus
from gurobipy import GRB
import gurobipy as gp


def coeff_PV(T, G, NOCT, eta_conv_PV, p_coef):
    T_GT = T + (NOCT - 20) * G / 800
    Delta_T = T_GT - 25
    Delta_P = Delta_T * p_coef
    eta_T = 1 + (Delta_P/100)
    eta_G = G/1000
    return eta_T*eta_G*eta_conv_PV, eta_T, eta_G

def fund_recove_coeff(r, n):
    return r*(1+r)**n/((1+r)**n -1)

def get_r_x_ij_from_panda_power_network(net, con_indices):
    """
    Extracts the resistance (r) and reactance (x) for connections between buses in a power grid network.

    Parameters:
    - net: A PandaPower network object containing the complete model of the power grid, including buses, lines, transformers, and other components.
    - con_indices: An array of shape (number_of_connections, 2) specifying the pairs of bus indices for which the resistance and reactance are to be extracted. Each pair represents a connection between two buses, which can be either direct line connections or connections via transformers.

    Returns:
    - r_ij: A dictionary where keys are tuples representing connection indices (i, j) between two buses, and values are the resistances of those connections.
    - x_ij: A dictionary similar to r_ij but containing reactance values for the connections instead of resistances.

    The function uses the network's admittance matrix (Ybus) to calculate the impedance matrix (Zbus) by taking its inverse. The real part of Zbus gives the resistance (Rbus), and the imaginary part provides the reactance (Xbus) of the network. 
    The desired resistances and reactances for specific connections are then extracted based on the provided indices and stored in the r_ij and x_ij dictionaries.
    """
    ppci = net["_ppc"]
    baseMVA, bus, branch = ppci["baseMVA"], ppci["bus"], ppci["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Zbus = 1/Ybus.todense()
    Rbus = np.real(Zbus)
    Xbus = np.imag(Zbus)
    r_ij, x_ij = {}, {}
    for i,j in con_indices:
        r_ij[i,j] = -Rbus[i,j]
        x_ij[i,j] = -Xbus[i,j]
    return r_ij, x_ij

def define_power_flow_variables(model, idx, init):
    P_line, Q_line, I_square_line, V_square, b = {}, {}, {}, {}, {}
    for w in range(init['num_W']):
        for t in idx['N_T']:
            for i,j in idx['con']:
                P_line[i,j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"P_{i}_{j}_{t}_{w}",lb=-GRB.INFINITY)
                Q_line[i,j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"Q_{i}_{j}_{t}_{w}",lb=-GRB.INFINITY)
                I_square_line[i,j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"I_square_{i}_{j}_{t}_{w}")
            for j in idx['node']:
                V_square[j,t,w] = np.float64(1.) if j in idx['N_slack'] else model.addVar(vtype=GRB.CONTINUOUS, name=f"V_square_{j}_{t}_{w}")
            b[t, w] = model.addVar(vtype=GRB.BINARY, name="b_{t}_{w}")
    return P_line, Q_line, I_square_line, V_square, b

def define_load_variables(model, idx, data, init):
    P_L, Q_L, x_curl_L = {}, {}, {}
    for w in range(init['num_W']):
        for t in idx['N_T']:
            for j in idx['node']:
                if w == 0:
                    if j in data['load']:
                        P_L[j,t] = data['load'][j][t]
                    else:
                        P_L[j,t] = 0
                    Q_L[j,t] = 0        # Currently Q_L is everywhere 0
                x_curl_L[j,t,w] = model.addVar(vtype=GRB.BINARY, name=f"x_curl_L_{j}_{t}_{w}") if j in idx['N_curl_L'] else np.int(1)
    return P_L, Q_L, x_curl_L

def define_generator_variables(model, idx, init):
    P_G, Q_G, p_G, q_G, x_curl_G, w_curl_G, x_G = {}, {}, {}, {}, {}, {}, {}
    for w in range(init['num_W']):
        for t in idx['N_T']:
            for j in idx['node']:
                p_G[j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"p_G_{j}_{t}_{w}") if j in idx['N_SGDG'] else np.float64(0.)
                q_G[j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"q_G_{j}_{t}_{w}") if j in idx['N_SGDG'] else np.float64(0.)
            for k, j in enumerate(idx['N_G']):
                if t ==0 and w==0:
                    if init.get('fix_G_size') is not None:
                        P_G[j] = init['fix_G_size'][k]
                        x_G[j] = init['fix_G_loc'][k]
                    else:
                        P_G[j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"P_G_{j}",lb=init['P_limits'][0], ub=init['P_limits'][1])
                        x_G[j] = model.addVar(vtype=GRB.BINARY, name=f"x_G_{j}") #if j in N_G else np.int(0)
                Q_G[j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"Q_G_{j}_{t}_{w}",lb=-GRB.INFINITY)
                x_curl_G[j,t,w] = model.addVar(vtype=GRB.BINARY, name=f"x_curl_G_{j}_{t}_{w}")
                w_curl_G[j,t,w] = model.addVar(vtype=GRB.BINARY, name=f"w_curl_G_{j}_{t}_{w}")
    return P_G, Q_G, p_G, q_G, x_curl_G, w_curl_G, x_G

def define_H2_variables(model, idx, init):
    p_EL, p_FC, P_FC, P_EL, x_act_FC, H2_cons, H2_prod = {}, {}, {}, {}, {}, {}, {}
    for w in range(init['num_W']):
        for t in idx['N_T']:
            for j in idx['node']:
                p_FC[j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"p_FC_{j}_{t}_{w}", lb=0) if j in idx['N_H2'] else np.float64(0.)
                p_EL[j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"p_EL_{j}_{t}_{w}", lb=0) if j in idx['N_H2'] else np.float64(0.)
            for j in idx['N_H2']:
                x_act_FC[j,t,w] = model.addVar(vtype=GRB.BINARY, name=f"x_act_{j}_{t}_{w}")
                H2_prod[j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"H2_+_{j}_{t}_{w}")
                H2_cons[j,t,w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"H2_-_{j}_{t}_{w}")
    for k, j in enumerate(idx['N_H2']):
        if init.get('fix_FC') is not None:
            P_FC[j] = init['fix_FC'][k]
            P_EL[j] = init['fix_EL'][k]
        else:
            P_FC[j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"P_FC_{j}", lb=0, ub=init['P_limits_FC'][1])
            P_EL[j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"P_EL_{j}", lb=0, ub=init['P_limits_EL'][1])
    return p_EL, p_FC, P_FC, P_EL, x_act_FC, H2_prod, H2_cons

def define_power_flow_constraints(model, network, var, init):
    idx = network['idx']
    for w in range(init['num_W']):
        for t in idx['N_T']:
            for k in idx['node']:
                from_node = idx['con'][idx['con'][:, 0]==k]
                to_node = idx['con'][idx['con'][:, 1]==k]
                model.addConstr(gp.quicksum(var['P_line'][i,j,t,w] for i, j in from_node) + gp.quicksum(-var['P_line'][i,j,t,w] + network['r'][i,j]*var['I_square_line'][i,j,t,w] for i, j in to_node) + var['P_L'][k,t]*var['x_curl_L'][k,t,w] + var['p_EL'][k,t,w] == var['p_G'][k,t,w] + var['p_FC'][k,t,w], f"P_node_{k}_{t}_{w}")
                model.addConstr(gp.quicksum(var['Q_line'][i,j,t,w] for i, j in from_node) + gp.quicksum(-var['Q_line'][i,j,t,w] + network['x'][i,j]*var['I_square_line'][i,j,t,w] for i, j in to_node) + var['Q_L'][k,t]*var['x_curl_L'][k,t,w] == var['q_G'][k,t,w], f"Q_node_{k}_{t}_{w}")  # Used here the quadratic constraint  
            for i, j in idx['con']:
                model.addConstr(var['V_square'][i,t,w]-var['V_square'][j,t,w]-2*(network['r'][i,j]*var['P_line'][i,j,t,w]+network['x'][i,j]*var['Q_line'][i,j,t,w])+(network['r'][i,j]**2+network['x'][i,j]**2)*var['I_square_line'][i,j,t,w]==0, f"V_line_{i}_{j}_{t}_{w}")
                model.addConstr(var['I_square_line'][i,j,t,w]*var['V_square'][i,t,w] >= var['P_line'][i,j,t,w]*var['P_line'][i,j,t,w]+var['Q_line'][i,j,t,w]*var['Q_line'][i,j,t,w], f"I_line_{i}_{j}_{t}_{w}")
            model.addConstr(var['p_G'][idx['N_slack'][0],t,w] <= 1000 * var['b'][t,w], f"big_M_upper_{t}_{w}")
            model.addConstr(var['p_G'][idx['N_slack'][0],t,w] >= -1000 * (1 - var['b'][t,w]), f"big_M_lower_{t}_{w}")
    return model

def define_generator_constraints(model, network, data, var, init):
    idx = network['idx']
    for w in range(init['num_W']):
        for t in idx['N_T']:
            for k in idx['N_G']:
                model.addConstr(var['w_curl_G'][k,t,w]*var['P_G'][k]*data['c_pv'][t] == var['p_G'][k,t,w], f"p_curl_equality_{k}_{t}_{w}")
                model.addConstr(var['w_curl_G'][k,t,w]*var['Q_G'][k,t,w] == var['q_G'][k,t,w], f"q_curl_equality_{k}_{t}_{w}")
                model.addConstr(var['w_curl_G'][k,t,w]<=var['x_curl_G'][k,t,w], f"w_curl_binary1_{k}_{t}_{w}")
                model.addConstr(var['w_curl_G'][k,t,w]<=var['x_G'][k], f"w_curl_binary2_{k}_{t}_{w}")
                model.addConstr(var['x_curl_G'][k,t,w]+var['x_G'][k]-1<=var['w_curl_G'][k,t,w], f"w_curl_binary3_{k}_{t}_{w}")
    return model

def define_H2_constraints(model, network, data, var, init):
    idx = network['idx']
    for w in range(init['num_W']):
        for t in idx['N_T']:
            for j in idx['N_H2']:
                model.addConstr(var['H2_prod'][j,t,w] == init['alpha_EL']*var['p_EL'][j,t,w], f"H2_prod_{j}_{t}_{w}")
                model.addConstr(var['H2_cons'][j,t,w] == (init['alpha_FC']*var['P_FC'][j] + init['beta_FC']*var['p_FC'][j,t,w])*var['x_act_FC'][j,t,w], f"H2_cons_{j}_{t}_{w}")
                #model.addConstr(var['H2_prod'][j,t,w] == var['p_EL'][j,t,w], f"H2_prod_{j}_{t}_{w}")
                #model.addConstr(var['H2_cons'][j,t,w] == var['p_FC'][j,t,w]*var['x_act_FC'][j,t,w], f"H2_cons_{j}_{t}_{w}")
                model.addConstr(init['P_limits_EL'][0] <= var['p_EL'][j,t,w], f"p_EL_greater_0_{j}_{t}_{w}")
                model.addConstr(init['P_limits_FC'][0] <= var['p_FC'][j,t,w], f"p_FC_greater_0_{j}_{t}_{w}")
                model.addConstr(var['p_EL'][j,t,w] <= var['P_EL'][j], f"p_EL_smaller_max_{j}_{t}_{w}")
                model.addConstr(var['p_FC'][j,t,w] <= var['P_FC'][j]*var['x_act_FC'][j,t,w], f"p_FC_smaller_max_{j}_{t}_{w}")
    #if init.get('fix_FC') is not None:
       # print
    for j in idx['N_H2']:
        model.addConstr(0==gp.quicksum(data['weights'][t,w]*(var['H2_prod'][j,t,w] - var['H2_cons'][j,t,w]) for t in idx['N_T'] for w in range(init['num_W'])), f"H2_balance_{j}")
    return model

def define_limit_constraints(model, network, var, init):
    idx = network['idx']
    line_limits = network['line_limits']
    A = init['A']
    for w in range(init['num_W']):
        for t in idx['N_T']:
            for k in idx['node']:
                if k not in idx['N_slack']:
                    model.addConstr(init['V_square_limits'][0]<=var['V_square'][k,t,w], f"V_square_limits_{k}_{t}_{w}_1")
                    model.addConstr(var['V_square'][k,t,w]<=init['V_square_limits'][1], f"V_square_limits_{k}_{t}_{w}_2")
            for i,j in idx['line']:      
                model.addConstr(var['I_square_line'][i,j,t,w]<=line_limits[i,j], f"Line_limit_{i}_{j}_{t}_{w}")
            num = 1
            for j,k in enumerate(idx['N_DG']):
                model.addConstr(0<=var['p_G'][k,t,w], f"Diesel_lower_limit_{k}_{t}_{w}")
                if k in idx['N_DG_curtail']:
                    model.addConstr(var['p_G'][k,t,w]-init['DG_size'][j]*A[w,num]<=0.0, f"Diesel_limit_{k}_{t}_{w}")
                    num += 1
                else:
                    model.addConstr(var['p_G'][k,t,w]-init['DG_size'][j]<=0.0, f"Diesel_limit_{k}_{t}_{w}")
            #print(f"External_grid_limit_{idx['N_slack'][0]}_{t}_{w}")
            model.addConstr(init['p_limits_ext_grid'][0]*A[w,0]-var['p_G'][idx['N_slack'][0],t,w]<=0.0, f"External_grid_limit_lower_{idx['N_slack'][0]}_{t}_{w}")
            model.addConstr(var['p_G'][idx['N_slack'][0],t,w]-init['p_limits_ext_grid'][1]*A[w,0]<=0.0, f"External_grid_limit_upper_{idx['N_slack'][0]}_{t}_{w}")
    return model

def extract_values(network, var, init):
    idx = network['idx']
    buses_inverse = {value: key for key, value in network['buses'].items()}
    names_N_G = [buses_inverse[idx] for idx in idx['N_G']]
    names_N_H2 = [buses_inverse[idx] for idx in idx['N_H2']]
    from_bus = np.array([buses_inverse[idx]for idx in network['model'].line['from_bus'].values])
    to_bus = np.array([buses_inverse[idx]for idx in network['model'].line['to_bus'].values])
    line_names = [fr_bus + '-' + tr_bus for fr_bus, tr_bus in zip(from_bus, to_bus)]
    df_node_all, df_line_all = {}, {}
    for w in range(init['num_W']):
        for t in idx['N_T']:
            data_line = np.zeros((len(idx['line']), 3))
            data_node = np.zeros((len(idx['node']), 13))
            for j in idx['node']:
                if isinstance(var['V_square'][j,t,w], np.float64):
                    data_node[j, 0] = np.sqrt(var['V_square'][j,t,w])
                else:
                    data_node[j, 0] = np.sqrt(var['V_square'][j,t,w].x)
                if isinstance(var['p_G'][j,t,w], np.float64):
                    data_node[j, 2] = -var['p_G'][j,t,w]
                else:
                    data_node[j, 2] = -var['p_G'][j,t,w].x
                if isinstance(var['p_FC'][j,t,w], np.float64):
                    data_node[j, 3] = -var['p_FC'][j,t,w]
                else:
                    data_node[j, 3] = -var['p_FC'][j,t,w].x
                data_node[j, 4] = var['P_L'][j,t]
                if isinstance(var['p_EL'][j,t,w], np.float64):
                    data_node[j, 5] = var['p_EL'][j,t,w]
                else:
                    data_node[j, 5] = var['p_EL'][j,t,w].x
                data_node[:,1] = data_node[:,2] + data_node[:,3] + data_node[:,4] + data_node[:,5]
                if (j,t) in var['Q_G']:
                    data_node[j, 7] = var['Q_G'][j,t,w].x
                else:
                    data_node[j, 7] = 0
                if isinstance(var['q_G'][j,t,w], np.float64):
                    data_node[j, 8] = -var['q_G'][j,t,w]
                else:
                    data_node[j, 8] = -var['q_G'][j,t,w].x
                data_node[j, 9] = var['Q_L'][j,t]
                data_node[:, 6] = data_node[:,8] + data_node[:,9]
                if (j,t,w) in var['x_curl_G']:
                    data_node[j, 10] = var['x_curl_G'][j,t,w].x   # round()
                else:
                    data_node[j, 10] = np.nan
                if (j,t,w) in var['w_curl_G']:
                    data_node[j, 11] = var['w_curl_G'][j,t,w].x   # round()
                else:
                    data_node[j, 11] = np.nan
                if isinstance(var['x_curl_L'][j,t,w], np.int):       # defined for every node
                    data_node[j, 12] = np.nan
                else:
                    data_node[j, 12] = var['x_curl_L'][j,t,w].x  # round()
            for k, (i, j) in enumerate(idx['line']):
                if isinstance(var['P_line'][i,j,t,w], np.float64):
                    data_line[k, 0] = var['P_line'][i,j,t,w]
                else:
                    data_line[k, 0] = var['P_line'][i,j,t,w].x

                if isinstance(var['Q_line'][i,j,t,w], np.float64):
                    data_line[k, 1] = var['Q_line'][i,j,t,w]
                else:
                    data_line[k, 1] = var['Q_line'][i,j,t,w].x

                # if isinstance(var['I_square_line'][i,j,t,w], np.float64):
                #     data_line[k, 2] = np.sqrt(var['I_square_line'][i,j,t,w])
                # else:
                data_line[k, 2] = np.sqrt(var['I_square_line'][i,j,t,w].x)
                df_node = pd.DataFrame(columns=['|V|', 'P', 'p_G', 'p_FC', 'P_L', 'p_EL', 'Q', 'Q_G', 'q_G', 'Q_L', 'x_curl_G', 'w_curl_G', 'x_curl_L'], data=data_node, index=network['model'].bus['name'].values)
                df_line = pd.DataFrame(columns=['P_ij', 'Q_ij', '|I_ij|'], data=data_line, index=line_names)
                df_node_all[t, w] = df_node
                df_line_all[t, w] = df_line
    data_G = np.zeros((len(idx['N_G']), 2))
    data_H2 = np.zeros((len(idx['N_H2']), 2))
    for k, j in enumerate(idx['N_G']):
        if isinstance(var['P_G'][j], np.float64):
            data_G[k, 0] = var['P_G'][j]
            data_G[k, 1] = var['x_G'][j]
        else:
            data_G[k, 0] = var['P_G'][j].x
            data_G[k, 1] = var['x_G'][j].x
    for k, j in enumerate(idx['N_H2']):
        if isinstance(var['P_EL'][j], np.float64):
            data_H2[k, 0] = var['P_EL'][j]
        else:
            data_H2[k, 0] = var['P_EL'][j].x
        if isinstance(var['P_FC'][j], np.float64):
            data_H2[k, 1] = var['P_FC'][j]
        else:
            data_H2[k, 1] = var['P_FC'][j].x
        
    df_G = pd.DataFrame(columns=['P_G', 'x_G'], data=data_G, index=names_N_G)
    df_H2 = pd.DataFrame(columns=['P_EL', 'P_FC'], data=data_H2, index=names_N_H2)
    return df_node_all, df_line_all, df_G, df_H2


def calc_c_EL(n_elz, HHV, Delta_T=1):
    return n_elz*Delta_T/HHV

def dict_to_array(var_dict):
    # Extract unique values of j, t, w from the dictionary keys
    js = sorted(set(k[0] for k in var_dict.keys()))
    ts = sorted(set(k[1] for k in var_dict.keys()))
    ws = sorted(set(k[2] for k in var_dict.keys()))

    # Determine the dimensions of the array
    j_dim = len(js)
    t_dim = len(ts)
    w_dim = len(ws)

    # Initialize an empty numpy array of the appropriate size
    arr = np.empty((j_dim, t_dim, w_dim))

    # Populate the numpy array with values from the dictionary
    for j, t, w in var_dict.keys():
        if isinstance(var_dict[j, t, w], np.float64):
            arr[js.index(j), ts.index(t), ws.index(w)] = var_dict[j, t, w]
        else:
            arr[js.index(j), ts.index(t), ws.index(w)] = var_dict[j, t, w].X
    return arr

# def stack_seasons(data_sunny, data_rainy):
#     data = {}
#     keys = list(data_sunny.keys())
#     for key in keys:
#         if isinstance(data_sunny[key], dict):
#             loads = {}
#             for idx_L in list(data_sunny[key].keys()):
#                 loads[idx_L] = np.hstack((data_sunny[key][idx_L], data_rainy[key][idx_L]))
#             data[key] = loads
#         if key == 'weights':
#             data[key] = np.vstack((data_sunny[key], data_rainy[key]))/2.0
#         else:
#             data[key] = np.hstack((data_sunny[key], data_rainy[key]))
#     data['probs'] /= 2
#     return data