import sys
import networkx as nx
import numpy as np
sys.path.append('../')

from GPnetRegressor import GPnetRegressor
from GPnetClassifier import GPnetClassifier

#%%
lattice_m = 15
lattice_n = 15
G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],
                                     periodic= False)
#%%

const = np.log(2.3)
const_scale = np.log(2.1)
length_scale= np.log(1.5)
noise = np.log(0.01)


walk_length = np.log(4)

gpr = GPnetRegressor(Graph = G,
                     ntrain = 125,
                     ntest= 100,
                     theta = [const, walk_length, noise],
                     seed = 123,
                     kerneltype = "pstep_walk",
                     relabel_nodes = True)

#%%
gpr.plot_graph()
_ = gpr.predict()
gpr.plot_predict_2d()
#gpr.plot_predict_2d_old()


#%%
labels = (np.sin(0.5*gpr.pivot_distance(0))>0).replace({True: 1, False: -1})
train_nodes = gpr.training_nodes
test_nodes = gpr.test_nodes
train_labels = labels[train_nodes]



#%%
gpc = GPnetClassifier(Graph = G,
                      training_nodes = train_nodes, 
                      test_nodes = test_nodes,
                      training_values = train_labels,
                      theta = [np.log(2.1), np.log(2), np.log(0.1)],
                      seed = 321,
                      kerneltype = 'pstep_walk',
                      relabel_nodes = True)
#%%


#%%
gpc.predict()
gpc.plot_graph()
gpc.plot_latent()
gpc.plot_predict_graph()
gpc.plot_binary_prediction()