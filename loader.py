import os
import pandas as pd
import numpy as np
import pickle
from utils import coeff_PV, get_r_x_ij_from_panda_power_network
import pdb

def load_initial(path_sizes=None):
    """
    Initializes configuration and parameters for a power grid simulation.
    
    Parameters:
    - path_sizes (str, optional): Path to a file containing predefined sizes and locations for certain components.
    
    Returns:
    - dict: A dictionary of initial settings and parameters for the optimization.
    
    The dictionary contains:
    - 'curl_L': List of load nodes that can be curtailed. These are potential points within the grid where demand can be adjusted to maintain stability.
    - 'curl_G': List of photovoltaic (PV) generator nodes that are capable of curtailment. Only at these nodes PVs can be placed.
    - 'pos_H2': Empty list reserved for specifying nodes where hydrogen components (Electrolyzer and Fuel Cell) can be installed.
    - 'N_ext_grid': Node representing the connection to the external power grid, also known as the slack bus, which balances the grid.
    - 'N_DG': Locations for diesel generators within the grid.
    - 'N_DG_curtail': Subset of Diesel generators that can have their output curtailed, providing flexibility in operation.
    - 'DG_size': Sizes of the Diesel generators in megawatts (MW).
    - 'P_limits', 'P_limits_FC', 'P_limits_EL', 'p_limits_ext_grid': Operational limits (in MW) for PV generators, Fuel Cells, Electrolyzers, and the external grid connection, respectively.
    - 'V_square_limits': Voltage limits (squared) at bus nodes, ensuring voltage stability within the grid.
    - 'NOCT': Nominal Operating Cell Temperature for PV cells, affecting performance.
    - 'n_con^PV': Efficiency of the PV inverter, converting DC to AC power.
    - 'p_coef': Power coefficient indicating how PV output changes with temperature.
    - 'n_elz': Efficiency of the Electrolyzer in converting electricity to hydrogen.
    - 'Q_H2': Higher heating value of hydrogen, indicative of its energy content.
    - 'alpha_FC', 'beta_FC', 'eta_FC': Parameters for Fuel Cell operation, including consumption coefficients and efficiency.
    - 'alpha_EL': Conversion coefficient of Electrolyzer from kWh -> kg 
    - 'cost_inv_*': Investment costs for PV, Fuel Cells, and Electrolyzers, denoted in $/kW.
    - 'cost_OM_*': Annual Operation and Maintenance costs for PV, Fuel Cells, and Electrolyzers.
    - 'cost_diesel': Fuel cost for diesel generators ($/kWh).
    - 'cost_curl_L': Cost associated with load curtailment ($/kWh).
    - 'cost_ext_grid': Cost for purchasing energy from the external grid ($/kWh).
    - 'num_years': Expected operational lifespan of components in years.
    - 'num_W': Number of contingency scenarios considered in the simulation.
    - 'prob_outage_ext_grid', 'prob_outage_diesel_generator': Probabilities of outage for the external grid and diesel generators, respectively.
    - 'num_samples': Number of representative scenarios for optimization, influencing accuracy and computational demand.
    - 'gap': Target optimality gap for the optimization process.
    - 'num_horizon': Number of time points in one year for hourly step size 
    - 'A': Matrix representing activation/deactivation of diesel generator and external grid functionalities. 
   
    If a path to predefined sizes and locations is provided, these are loaded and set for specific components.
    """

    init = {'curl_L': ['L6', 'L8', 'L9', 'L12', 'L13', 'L14','L17', 'L18', 'L19', 'L21',  
                       'L22', 'L25', 'L30', 'L34', 'L35', 'L36', 'L37', 'L38', 'L39', 'L40',
                       'L41', 'L42', 'N0.6', 'N0.7'],
            'curl_G': [],
            'pos_H2': [],
            #'curl_G': ['N0', 'N3', 'N7', 'N11', 'N18', 'N37', 'N25', 'N36', 'N40', 'N41'],
            #'pos_H2': ['N13', 'N18', 'N20', 'N37', 'N38', 'N39', 'N40', 'N41', 'N42'],                                          
            'N_ext_grid': ['N0.0'],
            'N_DG': ['N0.1', 'N0.2', 'N0.3', 'N0.4', 'N0.5'],
            'N_DG_curtail': ['N0.1', 'N0.2', 'N0.4'],
            'DG_size': [1.25, 1.0, 1.0, 0.6, 0.6],
            'P_limits': [0.2, 4.0], 'P_limits_FC': [0.0, 2.0], 'P_limits_EL': [0.0, 2.0], 'p_limits_ext_grid': [-4.3, 4.3],     # P limits
            'V_square_limits': [0.9**2, 1.1**2],
            'NOCT':45, 'n_con^PV':0.95, 'p_coef': -0.35,                                                                        # PV
            'n_elz':0.76, 'Q_H2':40.27,                                                                                         # Electrolyzer
            'alpha_FC': 0.004, 'beta_FC': 0.05, 'eta_FC':15.15,                                                                 # Fuel cell
            'cost_inv_G': 1002, 'cost_inv_FC': 3000, 'cost_inv_EL': 2000,                                                       # investment cost
            'cost_OM_G': 22.5, 'cost_OM_FC': 97.5, 'cost_OM_EL': 132.0,                                                         # OM cost
            'cost_diesel': 0.37,'cost_curl_L': 3.3, 'cost_ext_grid':0.19,                                                       # cost_ext_grid':0.051,
            'num_years': 20, 'num_W':5, 'prob_outage_ext_grid': 0.1129, 'prob_outage_diesel_generator': 0.0384,                 # Additional variables    
            'num_samples': 10, 'gap':0.01, 'discount_rate':0.03}                                                                  
    init['alpha_EL'] = init['n_elz']/init['Q_H2']  
    init['num_horizon'] = 24*365
    init['A'] = np.array([[1,1,1,1],
                          [0,1,1,1],
                          [0,0,1,1],
                          [0,1,0,1],
                          [0,1,1,0]])   
    if path_sizes is not None:    
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory , path_sizes)
        with open(file_path, 'rb') as f:
            attr = pickle.load(f)
        init['fix_G_size'] = attr['res_G']['P_G'].values
        init['fix_G_loc'] = attr['res_G']['x_G'].values
        # init['fix_FC'] = attr['res_H2']['P_FC'].values
        # init['fix_EL'] = attr['res_H2']['P_EL'].values
    return init




class DataModelLoader:
    def __init__(self, directory_data, directory_network):
        self.directory_data = directory_data
        self.directory_network = directory_network
    
    def load_network(self, init):
        file_path = os.path.join(self.directory_network, "net.pickle")
        file_buses_path = os.path.join(self.directory_network, "buses.pickle")
        with open(file_path, 'rb') as file:
            net = pickle.load(file)
        with open(file_buses_path, 'rb') as file:
            buses = pickle.load(file)
        net.line = net.line[net.line.in_service == True]
        idx = self.load_idx(net, buses, init)
        base = self.load_base_units(net, idx['line'])
        r, x = get_r_x_ij_from_panda_power_network(net, idx['con'])
        line_limits_values = (net.line['max_i_ka'].values/base['I'])**2
        line_limits = {(i, j): line_limits_values[k] for k, (i, j) in enumerate(idx['line'])}
        network = {'model':net, 'r': r, 'x': x, 'line_limits':line_limits, 'base': base, 'buses': buses, 'idx': idx}
        return network
    
    def load_idx(self, net, busses, init):
        node_idx = np.arange(len(net.bus))
        line_idx = net.line[['from_bus', 'to_bus']].values
        trafo_idx = net.trafo[['hv_bus', 'lv_bus']].values
        con_idx = np.vstack((line_idx, trafo_idx))
        N_slack = np.array([busses['N0.0']])
        N_DG = np.array([busses[bus_name]for bus_name in init['N_DG']])
        N_DG_curtail = np.array([busses[bus_name]for bus_name in init['N_DG_curtail']])
        N_G = np.array([busses[bus_name]for bus_name in init['curl_G']])
        N_H2 = np.array([busses[bus_name]for bus_name in init['pos_H2']])
        N_SGDG = np.concatenate((N_slack, N_G, N_DG))
        N_T = np.arange(init['num_samples'])
        DG_size = dict(zip(N_DG, init['DG_size']))
        N_curl_L = np.array([busses[bus_name]for bus_name in init['curl_L']])
        idx = {"node": node_idx, "line": line_idx, "trafo": trafo_idx, "con": con_idx, "N_slack": N_slack, 
               "N_T": N_T, "N_curl_L": N_curl_L, "N_G": N_G, "N_SGDG": N_SGDG, "N_H2": N_H2, 
               "N_DG":N_DG, "N_DG_curtail":N_DG_curtail, 'DG_size':DG_size}
        return idx
    
    def load_base_units(self, net, line_indices):
        V_base = net.bus.vn_kv.values
        I_base = 1/(np.sqrt(3) * np.max(V_base[line_indices], axis=1))
        base = {'V':V_base, 'I':I_base}
        return base
    
    def load_data(self, init, network):
        buses = network['buses']
        num_samples = init['num_samples']
        file_path = os.path.join(self.directory_data, "probs_"+ str(num_samples)+'.csv')
        df = pd.read_csv(file_path)
        T_values = df["T[°C]"].values
        G_values = df["G [W/m2]"].values
        prob_outage_ext_grid_and_generator = init['prob_outage_ext_grid']*init['prob_outage_diesel_generator']
        prob_normal_operation = 1 - init['prob_outage_ext_grid'] - 5*prob_outage_ext_grid_and_generator 
        prob_outages = np.array([prob_normal_operation, init['prob_outage_ext_grid'], prob_outage_ext_grid_and_generator, prob_outage_ext_grid_and_generator*2, prob_outage_ext_grid_and_generator*2])
        load_values = df.iloc[:, 2:-1].values
        prob_scenarios = np.array(df['prob'].values)[:, None]
        weights = prob_scenarios*prob_outages
        weights[:, init['num_W']:] = 0
        weights = weights/weights.sum()
        
        c_pv, eta_T, eta_G = coeff_PV(T_values, G_values, init['NOCT'], init['n_con^PV'], init['p_coef'])
        load_names = [s for s in buses.keys() if s.startswith('L')] + ['N0.6'] + ['N0.7']
        load_idx = np.array([buses[bus_name]for bus_name in load_names])
        load_dict = {key: val for val, key in zip(load_values.T, load_idx)}
        data = {'load': load_dict, 'T':T_values, 'G':G_values, 'weights': weights, 'c_pv':c_pv, 'probs': df['prob'].values}
        return data
    
