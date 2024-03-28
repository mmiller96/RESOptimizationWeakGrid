import pandas as pd
import pandapower as pp
import pickle
import pdb
import os
import numpy as np
from pandapower import runpp
from pandapower.pypower.makeYbus import makeYbus

def load_files(folder_path):
    file_path = os.path.join(folder_path, "Circuit Information.xlsx")
    file_load = os.path.join(folder_path, "cleaned_data_corrected.csv")
    trafo = pd.read_excel(file_path, sheet_name="Branches (transformers)")
    lines = pd.read_excel(file_path, sheet_name="Branches (lines)")
    df_loads = pd.read_csv(file_load).iloc[:,2:-1]
    loads = df_loads.iloc[0, :].values          # last two entries are Feeder 1# and 2#
    line_names = [['N0.0', 'N0'], ['N0','N0.1'], ['N0','N0.2'], ['N0','N0.3'], ['N0','N0.4'], ['N0','N0.5'], ['N0','N0.6'], ['N0','N0.7'], ['N0','N1'],  # N0.6 Feeder #2
                  ['N1','N2'], ['N2','DP1'], ['DP1','N3'], ['DP1','N4'], ['N4','N5'], ['N5','DP2'], ['DP2','N7'], ['N7','N8'],                          # N0.7 Feeder #3
                  ['N8','DP3'], ['DP3','N9'], ['DP3','DP4'], ['DP4','N11'], ['N11','N13'], ['N13','N14'], ['DP4','N10'], ['N11','N12'], ['N3','N15'], 
                  ['N15','N16'], ['N16','N17'], ['N17','N18'], ['N3','N19'], ['N19','N20'], ['N19','N21'], ['N21','N22'], ['N22','N23'], ['N23','N24'], 
                  ['N24','N25'], ['N25','N26'], ['N26','N27'], ['N25','N28'], ['N28','N29'], ['N29','N30'], ['N30','DP5'], ['DP5','N31'], ['DP5','N32'], 
                  ['N32','N33'], ['N33','N34'], ['N34','N35'], ['N35','N36'], ['N36','N37'], ['N37','N38'], ['N36','N39'], ['N39','N40'], ['N40','N41'], 
                  ['N41','N42'], ['DP2','N6']]
    trafo_names = [['N1', 'L1'], ['N2', 'L2'], ['N3', 'L3'], ['N4', 'L4'],['N6', 'L5'],['N10', 'L6'],['N12', 'L7'],['N5', 'L8'],['N7', 'L9'],
                   ['N8', 'L10'],['N9', 'L11'],['N11', 'L12'],['N13', 'L13'],['N14', 'L14'],['N15', 'L15'],['N16', 'L16'],['N17', 'L17'],['N18', 'L18'],
                   ['N25', 'L19'],['N26', 'L20'],['N27', 'L21'],['N19', 'L22'],['N21', 'L23'],['N22', 'L24'],['N23', 'L25'],['N24', 'L26'],['N20', 'L27'],
                   ['N28', 'L28'],['N29', 'L29'],['N30', 'L30'],['N32', 'L31'],['N33', 'L32'],['N34', 'L33'],['N35', 'L34'],['N39', 'L35'],['N40', 'L36'],
                   ['N41', 'L37'],['N42', 'L38'],['N31', 'L39'],['N36', 'L40'],['N37', 'L41'],['N38', 'L42']]
    return trafo, lines, loads, line_names, trafo_names

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

def create_buses(net, line_names, trafo_names):
    buses = {}
    line_names = np.hstack((line_names))
    trafo_names = np.hstack((trafo_names))
    buses = np.hstack((line_names, trafo_names))
    buses_unique = np.unique(buses)
    def custom_sort(item):
        if item.startswith('N'):
            return (0, int(item[1:].split('.')[0]))
        elif item.startswith('DP'):
            return (1, int(item[2:]))
        elif item.startswith('L'):
            return (2, int(item[1:]))
    buses_entries = sorted(buses_unique, key=custom_sort)
    swap_name = buses_entries[1]
    buses_entries[1] = buses_entries[0]
    buses_entries[0] = swap_name
    buses = dict(zip(buses_entries, np.arange(len(buses_entries))))
    for bus_name in buses.keys():
        if bus_name[0] == 'L':
            pp.create_bus(net, name=bus_name, vn_kv=0.208)
        else:
            pp.create_bus(net, name=bus_name, vn_kv=13.2)
    return net, buses

def bus_to_idx(line, buses):
    return [buses[line[0]], buses[line[1]]]

def create_lines(net, buses, line_names, lines):
    line_indices = [bus_to_idx(line, buses) for line in line_names]
    pp.create_line_from_parameters(net, from_bus=buses['N0.0'], to_bus=buses['N0'], length_km=1, 
            r_ohm_per_km=0.00001, x_ohm_per_km=0.000001, c_nf_per_km=0, max_i_ka=100, name="Line_{}{}".format(buses['N0.0'],buses['N0']))
    for k in range(1,8): # 1 EG + 5 DG + 2 Feeder (The external grid, the diesel generators and the feeders are modeled as separate nodes so that they are curtailable.)
        name = 'N0.' + str(k)
        pp.create_line_from_parameters(net, from_bus=buses['N0'], to_bus=buses[name], length_km=1, 
            r_ohm_per_km=0.00001, x_ohm_per_km=0.000001, c_nf_per_km=0, max_i_ka=100, name="Line_{}{}".format(buses['N0'],buses[name]))
    for k, (i, j) in enumerate(line_indices[8:]):   
        R_km = lines['RAC[Ohm/km]'].values[k] 
        X_km = lines['XAC[Ohm/km]'].values[k] 
        length = lines['Lenght [km]'].values[k] 
        ia = lines['Rated Current [kA]'].values[k]
        pp.create_line_from_parameters(net, from_bus=i, to_bus=j, length_km=length, 
            r_ohm_per_km=R_km, x_ohm_per_km=X_km, c_nf_per_km=0, max_i_ka=ia, name="Line_{}{}".format(i,j))
    return net

def create_trafo_and_loads(net, buses, trafo_names, trafo, loads):
    trafo_indices = [bus_to_idx(trafo_name, buses) for trafo_name in trafo_names] 
    for k, (i,j) in enumerate(trafo_indices):
        s_mva = trafo["S [kVA]"].values[k]/1000
        lv = trafo["LV [kV]"].values[k]
        hv = trafo["MV [kV]"].values[k]
        pp.create_transformer_from_parameters(net, name="Trafo_{}".format(k+1), hv_bus=i, lv_bus=j, 
                                          sn_mva=s_mva, vn_lv_kv=lv, vn_hv_kv=hv, vkr_percent=0.5, vk_percent=6, pfe_kw=0, i0_percent=0)
       
        pp.create_load(net, name="L{}".format(k+1), bus=j, p_mw=loads[k])
    pp.create_load(net, name="F#1", bus=buses['N0.6'], p_mw=loads[-1])
    pp.create_load(net, name="F#2", bus=buses['N0.7'], p_mw=loads[-2])
    return net

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_directory, 'network')
    net = pp.create_empty_network()
    trafo, lines, loads, line_names, trafo_names = load_files(folder_path)
    r_ij, x_ij, max_ia = {}, {}, {}
    net, buses = create_buses(net, line_names, trafo_names)
    pp.create_ext_grid(net, bus=buses['N0'], vm_pu=1.0, va_degree=0.0)
    net = create_lines(net, buses, line_names, lines)
    net = create_trafo_and_loads(net, buses, trafo_names, trafo, loads)
    line_indices = net.line[['from_bus', 'to_bus']].values
    trafo_indices = net.trafo[['hv_bus', 'lv_bus']].values
    line_indices = np.vstack((line_indices, trafo_indices))
    runpp(net)
    r_ij, x_ij = get_r_x_ij_from_model(net, line_indices)
    file_path = os.path.join(folder_path, "net.pickle")
    file_buses_dict_path = os.path.join(folder_path, "buses.pickle")
    with open(file_path, 'wb') as f:
        pickle.dump(net, f)
    with open(file_buses_dict_path, 'wb') as f:
        pickle.dump(buses, f)
    print(r_ij)         # should have no NAN entries