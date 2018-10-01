from __future__ import division
import numpy as np
import networkx as nx
from GPnet import GPnetRegressor, GPnetClassifier
import matplotlib.pylab as pl
from matplotlib import cm
#%%
lattice_m = 15
lattice_n = 15
N = 100     # numero punti training
n = 100    # numero punti test
ntest = 9
deg = 4 #connectivity degree
#%%
#G = nx.random_regular_graph(deg, N+n + 10)
G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],periodic= False)
G = nx.relabel_nodes(G, dict(zip(G,range(len(G.nodes)))))
#%%
a= GPnetRegressor(Graph = G, ntrain=N, ntest=n, theta=[1.36, 0.1, 0.01, 0.36])

#a= GPnetRegressor(totnodes=220, ntrain=N, ntest=n, theta=[1.36, 0.1, 0.01, 0.36], optimize=False)

a.predict()
#%%
#a.plot_latent()
a.plot_predict_2d()
#.plot_predict_graph()
#fstar, V = a.predict2()
#a.plot_graph()
#a.plot_predict_graph()
#a.plot_predict_2d()

#%%
            
# Plot LML landscape
theta0 = np.linspace(-10, 10, 100)
theta1 = np.linspace(-4, 4, 10)
theta2 = np.linspace(-4, 4, 40)
theta3 = np.linspace(-10, 10, 10)
theta4 = np.linspace(-10, 10, 10)

Theta1, Theta2 = np.meshgrid(theta1, theta2)
LML = [[a.logPosterior([1, Theta1[i, j], Theta2[i, j], 0.1, 0.1], a.training_nodes, a.t ) for i in range(Theta1.shape[0])] for j in range(Theta2.shape[1])]
LML = np.array(LML).T

#%%
#l.imshow(LML)
fig, ax = pl.subplots()
cax = ax.pcolor(theta1, theta2, LML, cmap=cm.viridis)
vmin, vmax = (-LML).min(), (-LML).max()
fig.colorbar(cax, ax=ax)

#%%
