from __future__ import division

import sys
sys.path.append('../')


import numpy as np
import networkx as nx
from GPnet import GPnetRegressor, GPnetClassifier
import matplotlib.pylab as pl
from matplotlib import cm
import pandas as pd
from numpy import unravel_index
#%%
lattice_m = 15
lattice_n = 15
N = 400    # numero punti training
n = 200    # numero punti test
ntest = 9
deg = 4 #connectivity degree

seed=1412
#%%
#G = nx.random_regular_graph(deg, N+n + 10)
G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],periodic= False)
G = nx.relabel_nodes(G, dict(zip(G,range(len(G.nodes)))))
#%%
#a= GPnetRegressor(Graph = G, ntrain=N*n, theta=[1.36, 0.1, 0.01, 0.36])

p0 = 0.1
p1 = 0.1
p2 = 0.2
p3 = 0.01

a= GPnetRegressor(totnodes=600, ntrain=N, ntest=n, theta=[p0, p1, p2, p3], optimize=False, seed = seed)
a.plot_graph()
a.calc_ktot()
#a.predict()
#%%
#a.plot_latent()
#a.plot_predict_2d()
#.plot_predict_graph()
#fstar, V = a.predict2()
#a.plot_graph()
#a.plot_predict_graph()
#a.plot_predict_2d()

#%%
L2 = np.linalg.cholesky(a.ktot + 1e-6 * np.eye(len(a.ktot)))
# f_prior = mu L*N(0,1)
f_prior = np.dot(L2, np.random.normal(size=(len(a.ktot), 5)))

new_training = pd.Series(index= a.training_nodes, data=f_prior[a.training_nodes][:,3])
#%%
a.pivot_flag = False
a.set_training_values(new_training)
a.predict()
a.plot_predict_2d()
#%%
#                 
# Plot LML landscape
theta0 = np.linspace(-10, 10, 10)
theta1 = np.linspace(-5, 2, 10)
theta2 = np.linspace(-5, 5, 10)
theta3 = np.linspace(-10, 10, 10)

#%%
Theta1, Theta2 = np.meshgrid(theta1, theta2)
LML0 = [[a.logPosterior([p0, Theta1[i, j], Theta2[i, j], p3], a.training_nodes, a.t ) for i in range(Theta1.shape[0])] for j in range(Theta2.shape[1])]
LML0 = -np.array(LML0).T

Theta0, Theta1 = np.meshgrid(theta0, theta1)
LML1 = [[a.logPosterior([Theta0[i,j], Theta1[i, j], p2, p3], a.training_nodes, a.t ) for i in range(Theta1.shape[0])] for j in range(Theta1.shape[1])]
LML1 = -np.array(LML1).T

Theta0, Theta3 = np.meshgrid(theta0, theta3)
LML2 = [[a.logPosterior([Theta0[i,j], p1, p2, Theta3[i,j]], a.training_nodes, a.t ) for i in range(Theta0.shape[0])] for j in range(Theta3.shape[1])]
LML2 = -np.array(LML2).T

Theta1, Theta3 = np.meshgrid(theta1, theta3)
LML3 = [[a.logPosterior([p0, Theta2[i,j], p2, Theta3[i,j]], a.training_nodes, a.t ) for i in range(Theta1.shape[0])] for j in range(Theta3.shape[1])]
LML3 = -np.array(LML3).T
#%%
#l.imshow(LML)
fig, ax = pl.subplots(2,2, dpi=150
                      )

fig.suptitle("LML landscapes")
cax0 = ax[0,0].pcolor(theta1, theta2, LML0, cmap=cm.viridis)
ax[0,0].set(xlabel="theta1", ylabel="theta2")
ax[0,0].plot([p1], [p2], marker='o', markersize=5, color="red")
vmin1, vmax1 = np.nanmin(LML0[LML0!= -np.inf]), np.nanmax(LML0[LML0!= np.inf])
fig.colorbar(cax0, ax=ax[0,0])

cax1 = ax[0,1].pcolor(theta0, theta1, LML1, cmap=cm.viridis)
ax[0,1].set(xlabel="theta0", ylabel="theta1")
ax[0,1].plot([p0], [p1], marker='o', markersize=5, color="red")
vmin1, vmax1 = np.nanmin(LML1[LML1!= -np.inf]), np.nanmax(LML1[LML1!= np.inf])
fig.colorbar(cax1, ax=ax[0,1])

cax2 = ax[1,0].pcolor(theta0, theta3, LML2, cmap=cm.viridis)
ax[1,0].set(xlabel="theta0", ylabel="theta3")
ax[1,0].plot([p0], [p3], marker='o', markersize=5, color="red")
vmin1, vmax1 = np.nanmin(LML2[LML2!= -np.inf]), np.nanmax(LML2[LML2!= np.inf])
fig.colorbar(cax2, ax=ax[1,0])

cax3 = ax[1,1].pcolor(theta1, theta3, LML3, cmap=cm.viridis)
ax[1,1].set(xlabel="theta1", ylabel="theta3")
ax[1,1].plot([p1], [p3], marker='o', markersize=5, color="red")
vmin1, vmax1 = np.nanmin(LML3[LML3!= -np.inf]), np.nanmax(LML3[LML3!= np.inf])
fig.colorbar(cax3, ax=ax[1,1])

#%%
