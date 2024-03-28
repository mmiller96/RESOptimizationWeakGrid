import numpy as np
from pandapower.networks import case33bw
import pandapower as pp
import pandas as pd
from pandapower import runpp
import gurobipy as gp
from gurobipy import GRB
import pdb
from pandapower.pypower.makeYbus import makeYbus
import os
import pickle

def get_r_x_ij_from_model(model, line_indices):
    ppci = model["_ppc"]
    baseMVA, bus, branch = ppci["baseMVA"], ppci["bus"], ppci["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Zbus = 1/Ybus.todense()
    Rbus = np.real(Zbus)
    Xbus = np.imag(Zbus)
    r_ij, x_ij = {}, {}
    for i,j in line_indices:
        r_ij[i,j] = -Rbus[i,j]
        x_ij[i,j] = -Xbus[i,j]
    return r_ij, x_ij

def get_variables(model, line_indices, trafo_indices):
    r_ij, x_ij = get_r_x_ij_from_model(model, line_indices)
    #p0, q0 = -model.load.p_mw.values, -model.load.q_mvar.values     # except p0 and q0
    V = np.full(len(model.bus), np.nan)
    P_G, Q_L, P_L, Q_G = np.zeros(len(model.bus)), np.zeros(len(model.bus)), np.zeros(len(model.bus)), np.zeros(len(model.bus))
    P_L[model.load['bus']] = model.load['p_mw']
    Q_L[model.load['bus']] = model.load['q_mvar']
    P_G[model.gen['bus'].values] = -model.gen['p_mw']
    P_G[model.ext_grid['bus']] = np.nan
    V[model.ext_grid['bus']] = model.ext_grid['vm_pu']*model.bus['vn_kv'][model.ext_grid['bus']]
    V[model.gen['bus']] = model.gen['vm_pu'].values*model.bus['vn_kv'][model.gen['bus']].values
    V_square = V**2
    bus = {'slack':model.ext_grid['bus'].values, 'gen':model.gen['bus'].values, 'load':model.load['bus'].values}
    Q_G[np.hstack((bus['gen'],bus['slack']))] = np.nan
    return r_ij, x_ij, P_L, Q_L, P_G, Q_G, V_square, bus


def load_model(name="testmodel"):
    if name == "testmodel":
        r_ohm_per_km = 0.642
        x_ohm_per_km = 0.083
        p_mw = 0.05
        q_mvar = 0.01
        model = pp.create_empty_network()
        b0 = pp.create_bus(model, vn_kv=20., name="0")
        b1 = pp.create_bus(model, vn_kv=20., name="1")
        b2 = pp.create_bus(model, vn_kv=20., name="2")
        b3 = pp.create_bus(model, vn_kv=20., name="3")
        b4 = pp.create_bus(model, vn_kv=20., name="4")
        b5 = pp.create_bus(model, vn_kv=0.4, name="5")
        pp.create_ext_grid(model, bus=b0, vm_pu=1.00, name="Grid Connection")
        pp.create_load(model, bus=b1, p_mw=p_mw, q_mvar=q_mvar, name="L1")
        pp.create_gen(model, bus=b2, p_mw=0.2, vm_pu=1.00, name="G2")
        pp.create_load(model, bus=b3, p_mw=p_mw, q_mvar=q_mvar, name="L3")
        pp.create_gen(model, bus=b4, p_mw=0.2, vm_pu=1.00, name="G4")
        pp.create_load(model, bus=b4, p_mw=p_mw, q_mvar=q_mvar, name="L4")
        pp.create_load(model, bus=b5, p_mw=p_mw, q_mvar=q_mvar, name="L5")
        # line01 = pp.create_line(model, from_bus=b0, to_bus=b1, length_km=1., name="Line(0,1)", std_type="NAYY 4x50 SE")    # r=0.642   x=0.083  c=210.0
        line01 = pp.create_line_from_parameters(model, from_bus=b0, to_bus=b1, length_km=1.0, r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km, c_nf_per_km=0.0, max_i_ka=1.4)
        line12 = pp.create_line_from_parameters(model, from_bus=b1, to_bus=b2, length_km=1.0, r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km, c_nf_per_km=0.0, max_i_ka=1.4)
        line23 = pp.create_line_from_parameters(model, from_bus=b2, to_bus=b3, length_km=1.0, r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km, c_nf_per_km=0.0, max_i_ka=1.4)
        line14 = pp.create_line_from_parameters(model, from_bus=b1, to_bus=b4, length_km=1.0, r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km, c_nf_per_km=0.0, max_i_ka=1.4)
        pp.create_transformer_from_parameters(model, hv_bus=b4, lv_bus=b5, 
                                      vn_hv_kv=20, vn_lv_kv=0.4, sn_mva=0.4,
                                      vk_percent=6, vkr_percent=0.5,
                                      pfe_kw=.0, i0_percent=0, shift_degree=0)
    elif name == "case33bw":
        model = case33bw()
    elif name == "puertoCarreno":
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, "net.pickle")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
    else:
        raise ValueError("Invalid value, model " + name + " does not exist.")
    model.line = model.line[model.line.in_service == True]
    line_indices = model.line[['from_bus', 'to_bus']].values
    V_base = model.bus.vn_kv.values
    I_base = 1/(np.sqrt(3) * np.max(V_base[line_indices], axis=1))
    node_indices = model.bus.name.values        # remove bus 0 
    node_indices = np.arange(len(model.bus))
    #node_indices = node_indices.astype(int)
    #node_indices = np.sort(node_indices)
    trafo_indices = model.trafo[['hv_bus', 'lv_bus']].values
    line_indices = np.vstack((line_indices, trafo_indices))
    runpp(model)
    r, x, P_L, Q_L, P_G, Q_G, V_square, bus = get_variables(model, line_indices, trafo_indices)
    trafo_limits = calc_trafo_limits(model, trafo_indices, r, x)
    line_limits = model.line['max_i_ka'].values/I_base
    return model, line_indices, node_indices, r, x, line_limits, trafo_limits, trafo_indices

def calc_trafo_limits(model, trafo_indices, r, x):
    z_trafo = model.trafo['sn_mva'].values
    trafo_limits = np.zeros((len(z_trafo), 2))
    for k, (i,j) in enumerate(trafo_indices):
        za = np.complex(r[i,j],x[i,j])
        angle = np.angle(za)
        p_trafo = z_trafo[k]*np.cos(angle)
        q_trafo = z_trafo[k]*np.sin(angle)
        trafo_limits[k,0] = p_trafo
        trafo_limits[k,1] = q_trafo
    return trafo_limits

def define_decision_variables(model, line_indices, node_indices, loads, N_L, N_G, N_slack, N_T, P_limits):
    P_line, Q_line, I_square_line, V_square, P_L, Q_L, P_G, Q_G, p_G, q_G, x_curl_L, x_curl_G, w_curl_G, x_G = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    N_SG = np.concatenate((N_slack, N_G))
    for t in N_T:
        for i,j in line_indices:
            P_line[i,j,t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"P_{i}_{j}_{t}",lb=-GRB.INFINITY)
            Q_line[i,j,t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"Q_{i}_{j}_{t}",lb=-GRB.INFINITY)
            I_square_line[i,j,t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"I_square_{i}_{j}_{t}")
        for j in node_indices:
            if j in loads:
                P_L[j,t] = loads[j][t]
            else:
                P_L[j,t] = 0
            Q_L[j,t] = 0
            p_G[j,t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"p_G_{j}_{t}") if j in N_SG else np.float64(0.)
            q_G[j,t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"q_G_{j}_{t}") if j in N_SG else np.float64(0.)
            x_curl_L[j,t] = model.addVar(vtype=GRB.BINARY, name=f"x_curl_L_{j}_{t}") if j in N_L else np.int(1)
            V_square[j,t] = np.float64(1.) if j in N_slack else model.addVar(vtype=GRB.CONTINUOUS, name=f"V_square_{j}_{t}")
        for j in N_G:
            if t ==0:
                P_G[j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"P_G_{j}",lb=P_limits[0], ub=P_limits[1])
                x_G[j] = model.addVar(vtype=GRB.BINARY, name=f"x_G_{j}") if j in N_G else np.int(0)
            Q_G[j,t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"Q_G_{j}",lb=-GRB.INFINITY)
            x_curl_G[j,t] = model.addVar(vtype=GRB.BINARY, name=f"x_curl_G_{j}_{t}")
            w_curl_G[j,t] = model.addVar(vtype=GRB.BINARY, name=f"w_curl_G_{j}_{t}")
    return model, P_line, Q_line, I_square_line, P_L, Q_L, P_G, Q_G, V_square, x_curl_L, x_curl_G, w_curl_G, x_G, p_G, q_G

def define_constraints(model, line_indices, node_indices, P_line, Q_line, I_square_line, P_L, Q_L, P_G, Q_G, V_square, r_line, x_line, x_curl_L, x_curl_G, w_curl_G, x_G, p_G, q_G, N_G, N_slack, V_square_limits, c_PV, N_T, line_limits, trafo_limits, trafo_indices):
    pdb.set_trace()
    for t in N_T:
        for k in node_indices:
            print(k)
            #pdb.set_trace()
            from_node = line_indices[line_indices[:, 0]==k]
            to_node = line_indices[line_indices[:, 1]==k]
            model.addConstr(gp.quicksum(P_line[i,j,t] for i, j in from_node) + gp.quicksum(-P_line[i,j,t] + r_line[i,j]*I_square_line[i,j,t] for i, j in to_node) + P_L[k,t]*x_curl_L[k,t] == p_G[k,t], f"P_node_{k}_{t}")
            model.addConstr(gp.quicksum(Q_line[i,j,t] for i, j in from_node) + gp.quicksum(-Q_line[i,j,t] + x_line[i,j]*I_square_line[i,j,t] for i, j in to_node) + Q_L[k,t]*x_curl_L[k,t] == q_G[k,t], f"Q_node_{k}_{t}")  # Used here the quadratic constraint  
            if k not in N_slack:
                model.addConstr(V_square_limits[0]<=V_square[k,t], f"V_square_limits_{k}_{t}_1")
                model.addConstr(V_square[k,t]<=V_square_limits[1], f"V_square_limits_{k}_{t}_2")
        for i, j in line_indices:
            model.addConstr(V_square[i,t]-V_square[j,t]-2*(r_line[i,j]*P_line[i,j,t]+x_line[i,j]*Q_line[i,j,t])+(r_line[i,j]**2+x_line[i,j]**2)*I_square_line[i,j,t]==0, f"V_line_{i}_{j}_{t}")
            model.addQConstr(I_square_line[i,j,t]*V_square[i,t] >= P_line[i,j,t]*P_line[i,j,t]+Q_line[i,j,t]*Q_line[i,j,t], f"I_line_{i}_{j}_{t}")
        for k in N_G:
            model.addConstr(w_curl_G[k,t]*P_G[k] == p_G[k,t]/c_PV[t], f"p_curl_equality_{k}_{t}")
            model.addConstr(w_curl_G[k,t]*Q_G[k,t] == q_G[k,t], f"q_curl_equality_{k}_{t}")
            model.addConstr(w_curl_G[k,t]<=x_curl_G[k,t], f"w_curl_binary1_{k}_{t}")
            model.addConstr(w_curl_G[k,t]<=x_G[k], f"w_curl_binary2_{k}_{t}")
            model.addConstr(w_curl_G[k,t]>=x_curl_G[k,t]+x_G[k]-1, f"w_curl_binary3_{k}_{t}")
        for k, (i,j) in enumerate(trafo_indices):
            model.addConstr(trafo_limits[k,0]>=r_line[i,j]*I_square_line[i,j,t], f"P_trafo_limit_{i}_{j}_{t}")
            model.addConstr(trafo_limits[k,1]>=x_line[i,j]*I_square_line[i,j,t], f"Q_trafo_limit_{i}_{j}_{t}")
        for k, (i,j) in enumerate(line_indices):        # REALLY JUST LINES
            model.addConstr(line_limits[k]>=I_square_line[i,j,t], f"Line_limit_{k}_{t}")
    return model

def get_values_from_optimizer(node_indices, line_indices, P_L, Q_L, P_G, Q_G, p_G, q_G, V_square, P_line, Q_line, I_square_line, x_curl_L, x_curl_G, w_curl_G, x_G, N_T):
    df_node_all, df_line_all = [], []
    for t in N_T:
        data_line = np.zeros((len(line_indices), 3))
        data_node = np.zeros((len(node_indices), 13))
        for j in node_indices:
            if isinstance(V_square[j,t], np.float64):
                data_node[j, 2] = np.sqrt(V_square[j,t])
            else:
                data_node[j, 2] = np.sqrt(V_square[j,t].x)
            data_node[j, 3] = P_L[j,t]
            data_node[j, 5] = Q_L[j,t]
            if j in P_G:
                data_node[j, 4] = P_G[j].x
            else:
                data_node[j, 4] = 0
            if (j,t) in Q_G:
                data_node[j, 6] = Q_G[j,t].x
            else:
                data_node[j, 6] = 0
            if isinstance(x_curl_L[j,t], np.int):       # defined for every node
                data_node[j, 7] = np.nan
            else:
                data_node[j, 7] = x_curl_L[j,t].x  # round()
            if (j,t) in x_curl_G:
                data_node[j, 8] = x_curl_G[j,t].x   # round()
            else:
                data_node[j, 8] = np.nan
            if j in x_G:
                data_node[j, 9] = x_G[j].x   # round()
            else:
                data_node[j, 9] = np.nan
            if (j,t) in w_curl_G:
                data_node[j, 10] = w_curl_G[j,t].x   # round()
            else:
                data_node[j, 10] = np.nan
            if isinstance(p_G[j,t], np.float64):
                data_node[j, 11] = -p_G[j,t]
            else:
                data_node[j, 11] = -p_G[j,t].x
            if isinstance(q_G[j,t], np.float64):
                data_node[j, 12] = -q_G[j,t]
            else:
                data_node[j, 12] = -q_G[j,t].x
        data_node[:,0] = data_node[:,3] + data_node[:,11]
        data_node[:,1] = data_node[:,5] + data_node[:,12]
        for k, (i, j) in enumerate(line_indices):
            if isinstance(P_line[i,j,t], np.float64):
                data_line[k, 0] = P_line[i,j,t]
            else:
                data_line[k, 0] = P_line[i,j,t].x

            if isinstance(Q_line[i,j,t], np.float64):
                data_line[k, 1] = Q_line[i,j,t]
            else:
                data_line[k, 1] = Q_line[i,j,t].x

            if isinstance(I_square_line[i,j,t], np.float64):
                data_line[k, 2] = np.sqrt(I_square_line[i,j,t])
            else:
                data_line[k, 2] = np.sqrt(I_square_line[i,j,t].x)
        df_node = pd.DataFrame(columns=['P', 'Q', '|V|', 'P_L', 'P_G', 'Q_L', 'Q_G', 'x_curl_L', 'x_curl_G', 'x_G', 'w_curl_G', 'p_G', 'q_G'], data=data_node)
        df_line = pd.DataFrame(columns=['P_ij', 'Q_ij', '|I_ij|'], data=data_line)
        df_node_all.append(df_node)
        df_line_all.append(df_line)
    return df_node_all, df_line_all

def transform_to_pu(model_net, line_indices, V_square, P_L, Q_L, P_G, Q_G):
        base = {}
        base['S'] = model_net.sn_mva
        base['V'] = model_net.bus.vn_kv.values
        #line_indices = list(r_line.keys())
        base['I'] = base['S'] / (np.sqrt(3) * np.max(base['V'][line_indices], axis=1))
        #Z_base = base['V']**2/base['S']
        # r_line = {key: value / Z_base[i] for i, (key, value) in enumerate(r_line.items())}
        # x_line = {key: value / Z_base[i] for i, (key, value) in enumerate(x_line.items())}
        P_L = P_L/base['S']
        Q_L = Q_L/base['S']
        P_G = P_G/base['S']
        Q_G = Q_G/base['S']
        V_square = V_square/base['V']**2
        return V_square,  P_L, Q_L, P_G, Q_G, base