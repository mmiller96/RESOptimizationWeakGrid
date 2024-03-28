import os
from gurobipy import GRB
from utils import define_power_flow_variables, define_load_variables, define_generator_variables, define_power_flow_constraints, define_generator_constraints, define_limit_constraints, extract_values, define_H2_variables, define_H2_constraints, dict_to_array
from loader import DataModelLoader
import gurobipy as gp
import os
import pickle
import numpy as np
from gurobipy import max_
import pdb
import pandas as pd

class PVHydroPlanningModel:
    def __init__(self, init):
        self.init = init
        self.directory = os.path.dirname(os.path.abspath(__file__))
        self.directory_data = os.path.join(self.directory, 'data')
        self.directory_network = os.path.join(self.directory, 'network')
        self.directory_results = os.path.join(self.directory, 'results')
        self.loader = DataModelLoader(self.directory_data, self.directory_network)

    def load_network_and_data(self):
        network = self.loader.load_network(self.init)
        data = self.loader.load_data(self.init, network)
        return network, data

    def create_or_load_model(self, name):
        model = gp.Model("PV planning model")
        model.Params.MIPGap = self.init['gap']
        return model
    
    def create_variables(self, model, network, data):
        idx = network['idx']
        P_line, Q_line, I_square_line, V_square, b =       define_power_flow_variables(model, idx, self.init)
        P_L, Q_L, x_curl_L =                            define_load_variables(model, idx, data, self.init)
        P_G, Q_G, p_G, q_G, x_curl_G, w_curl_G, x_G =   define_generator_variables(model, idx, self.init)
        p_EL, p_FC, P_FC, P_EL, x_act_FC, H2_prod, H2_cons = define_H2_variables(model, idx, self.init)
        var = {'P_line': P_line,'Q_line': Q_line,'I_square_line': I_square_line,'V_square': V_square,
                    'P_L': P_L,'Q_L': Q_L,'x_curl_L': x_curl_L,'P_G': P_G,'Q_G': Q_G,'p_G': p_G,'q_G': q_G,
                    'x_curl_G': x_curl_G,'w_curl_G': w_curl_G,'x_G': x_G,
                    'p_EL':p_EL, 'p_FC':p_FC, 'P_FC':P_FC, 'P_EL':P_EL, 'x_act_FC':x_act_FC, 'H2_prod':H2_prod, 'H2_cons':H2_cons, 'b':b}
        return var
    
    def create_constraints(self, model, network, data, var):
        model = define_power_flow_constraints(model, network, var, self.init)
        model = define_generator_constraints(model, network, data, var, self.init)
        model = define_H2_constraints(model, network, data, var, self.init)
        model = define_limit_constraints(model, network, var, self.init)
        return model

    def create_cost(self, model, data, network, var):
        idx = network['idx']
        C_inv_G = self.init['cost_inv_G']*gp.quicksum(var['x_G'][j]*var['P_G'][j] for j in idx['N_G'])/1000    # MW -> kW (*1000), to million -> *10^-6
        C_inv_FC = self.init['cost_inv_FC']*gp.quicksum(var['P_FC'][j] for j in idx['N_H2'])/1000    
        C_inv_EL = self.init['cost_inv_EL']*gp.quicksum(var['P_EL'][j] for j in idx['N_H2'])/1000
        C_inv = C_inv_G + C_inv_FC + C_inv_EL
        C_OM_G = self.init['cost_OM_G']*gp.quicksum(var['x_G'][j]*var['P_G'][j] for j in idx['N_G'])/1000    
        C_OM_FC = self.init['cost_OM_FC']*gp.quicksum(var['P_FC'][j] for j in idx['N_H2'])/1000              
        C_OM_EL = self.init['cost_OM_EL']*gp.quicksum(var['P_EL'][j] for j in idx['N_H2'])/1000
        C_OM = np.sum((1/(1+self.init['discount_rate']))for t in range(1, self.init['num_years']+1))*(C_OM_G + C_OM_FC + C_OM_EL)
        C_curl_L = self.init['cost_curl_L']*gp.quicksum(data['weights'][t,w]*var['P_L'][j,t] * (1-var['x_curl_L'][j,t,w]) for j in idx['node'] for t in idx['N_T'] for w in range(self.init['num_W']))/1000 
        C_ext_grid = gp.quicksum(data['weights'][t,w]*(self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,w] *var['b'][t,w] + 0.7*self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,w]*(1-var['b'][t,w])) for t in idx['N_T'] for w in range(self.init['num_W']))/1000      
        C_diesel = self.init['cost_diesel']*gp.quicksum(data['weights'][t,w]*var['p_G'][k,t,w] for k in idx['N_DG'] for t in idx['N_T'] for w in range(self.init['num_W']))/1000 
        C_sw = np.sum((self.init['num_horizon']/(1+self.init['discount_rate']))for t in range(1, self.init['num_years']+1))*(C_ext_grid + C_diesel + C_curl_L)
        C_curl_L0 = self.init['cost_curl_L']*gp.quicksum(data['probs'][t]*var['P_L'][j,t] * (1-var['x_curl_L'][j,t,0]) for j in idx['node'] for t in idx['N_T'])/1000 
        C_curl_L1 = self.init['cost_curl_L']*gp.quicksum(data['probs'][t]*var['P_L'][j,t] * (1-var['x_curl_L'][j,t,1]) for j in idx['node'] for t in idx['N_T'])/1000 
        C_curl_L2 = self.init['cost_curl_L']*gp.quicksum(data['probs'][t]*var['P_L'][j,t] * (1-var['x_curl_L'][j,t,2]) for j in idx['node'] for t in idx['N_T'])/1000 
        C_curl_L3 = self.init['cost_curl_L']*gp.quicksum(data['probs'][t]*var['P_L'][j,t] * (1-var['x_curl_L'][j,t,3]) for j in idx['node'] for t in idx['N_T'])/1000 
        C_curl_L4 = self.init['cost_curl_L']*gp.quicksum(data['probs'][t]*var['P_L'][j,t] * (1-var['x_curl_L'][j,t,4]) for j in idx['node'] for t in idx['N_T'])/1000 
        C_ext_grid0 = gp.quicksum((self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,0] *var['b'][t,0] + 0.7*self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,0]*(1-var['b'][t,0])) for t in idx['N_T'])/1000      
        C_ext_grid1 = gp.quicksum((self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,1] *var['b'][t,1] + 0.7*self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,1]*(1-var['b'][t,1])) for t in idx['N_T'])/1000     
        C_ext_grid2 = gp.quicksum((self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,2] *var['b'][t,2] + 0.7*self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,2]*(1-var['b'][t,2])) for t in idx['N_T'])/1000   
        C_ext_grid3 = gp.quicksum((self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,3] *var['b'][t,3] + 0.7*self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,3]*(1-var['b'][t,3])) for t in idx['N_T'])/1000    
        C_ext_grid4 = gp.quicksum((self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,4] *var['b'][t,4] + 0.7*self.init['cost_ext_grid']*var['p_G'][idx['N_slack'][0],t,4]*(1-var['b'][t,4])) for t in idx['N_T'])/1000    
        C_diesel0 = self.init['cost_diesel']*gp.quicksum(var['p_G'][k,t,0] for k in idx['N_DG'] for t in idx['N_T'])/1000         
        C_diesel1 = self.init['cost_diesel']*gp.quicksum(var['p_G'][k,t,1] for k in idx['N_DG'] for t in idx['N_T'])/1000         
        C_diesel2 = self.init['cost_diesel']*gp.quicksum(var['p_G'][k,t,2] for k in idx['N_DG'] for t in idx['N_T'])/1000         
        C_diesel3 = self.init['cost_diesel']*gp.quicksum(var['p_G'][k,t,3] for k in idx['N_DG'] for t in idx['N_T'])/1000         
        C_diesel4 = self.init['cost_diesel']*gp.quicksum(var['p_G'][k,t,4] for k in idx['N_DG'] for t in idx['N_T'])/1000         
        C = C_inv + C_OM + C_sw 
        cost = {'inv_G': C_inv_G, 'inv_FC': C_inv_FC, 'inv_EL': C_inv_EL, 'inv':C_inv, 
                'OM_G': C_OM_G, 'OM_FC': C_OM_FC, 'OM_EL': C_OM_EL, 'OM':C_OM, 
                'curl_L': C_curl_L, 'ext_grid': C_ext_grid,
                'diesel': C_diesel, 'sw':C_sw, 'C':C,
                'curl_L0': C_curl_L0, 'curl_L1': C_curl_L1, 'curl_L2': C_curl_L2, 'curl_L3': C_curl_L3, 'curl_L4': C_curl_L4,
                'ext_grid0': C_ext_grid0, 'ext_grid1': C_ext_grid1,'ext_grid2': C_ext_grid2, 'ext_grid3': C_ext_grid3, 'ext_grid4': C_ext_grid4,
                'diesel0': C_diesel0, 'diesel1': C_diesel1, 'diesel2': C_diesel2, 'diesel3': C_diesel3, 'diesel4': C_diesel4}  #, 'C_LL':C_LL}
        model.setObjective(cost['C'], GRB.MINIMIZE)
        return cost, model
    
    def save_attributes(self, network, var, cost, name):
        res_node, res_line, res_G, res_H2 = self.get_values(network, var)
        name = name + '.pkl'
        attr = {}
        for key, value in cost.items():
            attr[key] = value.getValue()
            print(key + ': ' + str(value.getValue()))
        attr['init'] = self.init
        idx = network['idx']
        attr['H2_prod'] = dict_to_array(var['H2_prod'])
        attr['H2_cons'] = dict_to_array(var['H2_cons'])
        attr['x_act_FC'] = dict_to_array(var['x_act_FC']) 
        attr['x_curl_G'] = dict_to_array(var['x_curl_G']) 
        attr['p_EL'] = dict_to_array(var['p_EL']) if len(idx['N_H2']) == 0 else dict_to_array(var['p_EL'])[idx['N_H2']] 
        attr['p_FC'] = dict_to_array(var['p_FC']) if len(idx['N_H2']) == 0 else dict_to_array(var['p_FC'])[idx['N_H2']] 
        attr['res_node'] = res_node
        attr['res_line'] = res_line
        attr['res_G'] = res_G
        attr['res_H2'] = res_H2
        path = os.path.join(self.directory_results, name) 
        with open(path, 'wb') as f:
            pickle.dump(attr, f)
        return attr
        
    def get_values(self, network, var):
        df_node, df_line, df_G, df_H2 = extract_values(network, var, self.init)
        return df_node, df_line, df_G, df_H2