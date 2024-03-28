from model_class import PVHydroPlanningModel
from loader import load_initial
# import os
# import pickle
# import pdb

if __name__ == "__main__":   
   init = load_initial(path_sizes=None)
   model_planning = PVHydroPlanningModel(init)
   network, data = model_planning.load_network_and_data()
   name_model = "model_" + str(init['num_samples'])
   model = model_planning.create_or_load_model(name_model)
   var = model_planning.create_variables(model, network, data)
   model = model_planning.create_constraints(model, network, data, var)
   cost, model  = model_planning.create_cost(model, data, network, var)
   model.optimize()
   res_node, res_line, res_G, res_H2 = model_planning.get_values(network, var)
   attr = model_planning.save_attributes(network, var, cost, 'attr_'+str(init['num_samples']))
   # -->  Results can be retrieved wih the following code:   <--

   # directory = os.path.dirname(os.path.abspath(__file__))
   # directory_results = os.path.join(directory, 'results') 
   # path = os.path.join(directory_results, 'attr_10.pkl') 
   # with open(path, 'rb') as f:
   #    attr = pickle.load(f)
   # pdb.set_trace()