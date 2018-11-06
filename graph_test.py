import sys
import networkx as nx
import numpy as np
sys.path.append('../')

from GPnetRegressor import GPnetRegressor
from GPnetClassifier import GPnetClassifier

#%%
lattice_m = 10
lattice_n = 10
G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],
                                     periodic= False)
#%%

const = np.log(5)
const_scale = np.log(2.1)
length_scale= np.log(1.5)
noise = np.log(0.001)

walk_length = np.log(4)

gpr = GPnetRegressor(Graph = G,
                     ntrain = 60,
                     ntest= 40,
                     theta = [const, walk_length, noise],
                     seed = 123,
                     kerneltype = "pstep_walk",
                     relabel_nodes = True)

#%%
gpr.plot_graph()
_ = gpr.predict()
gpr.plot_predict_2d()