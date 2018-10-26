import sys
import networkx as nx
sys.path.append('../')

from GPnet import GPnetClassifier, GPnetRegressor
#%%
lattice_m = 5
lattice_n = 5
G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],
                                     periodic= False)
const = np.log(1)
const_scale = np.log(1)
length_scale= np.log(1.5)
noise = np.log(1)

nx.convert_node_labels_to_integers(G)

G.nodes()

gpr = GPnetRegressor(Graph = G,
                     ntrain = 18,
                     ntest= 7,
                     theta = [const, const_scale, length_scale, noise],
                     seed = 123,
                     relabel_nodes = True)

_ = gpr.predict()

gpr.plot_predict_2d()


#%%
N = 20   # numero punti training
n = 10 # numero punti test
deg = 5 #connectivity degree
seed=1412

G2 = nx.random_regular_graph(deg, N+n, seed=seed)

gpr_er = GPnetRegressor(Graph = G2, 
                   ntrain = N,
                   ntest= n,
                   theta = [const, const_scale, length_scale, noise],
                   seed = 123,
                   )


#%%

theta0 = np.linspace(-2, 2, 30)
theta1 = np.linspace(-3, 3, 31)
theta2 = np.linspace(-3, 3, 32)
theta3 = np.linspace(-1, 1, 33)

plots = {"const vs const_scale":    [[0,1], theta0, theta1, [const,const_scale]],
         "const vs length_scale":   [[0,2], theta0, theta2, [const,length_scale]],
         "const vs noise_scale":    [[0,3], theta0, theta3, [const,noise]],
         "const_scale vs length_scale": [[1,2], theta1, theta2, [const_scale,length_scale]],
         "const_scale vs noise_scale":  [[1,3], theta1, theta3, [const_scale,noise]],
         "length_scale vs noise_scale": [[2,3], theta2, theta3, [length_scale,noise]],
         }


gpr_er.plot_lml_landscape(plots, [const, const_scale, length_scale, noise])
