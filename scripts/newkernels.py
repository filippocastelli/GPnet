# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:22:42 2018

@author: filip
"""

#%%
import numpy as np
import scipy.linalg as sl
import scipy.sparse as ss

mat = np.array([[0, 1, 2], [1,0,3], [2,3,0]])

eigvals, eigvecs = np.linalg.eig(mat)

beta = 0.1

expD = np.diag(np.exp(beta*eigvals))
T = eigvecs.T
Tinv = np.linalg.inv(T)

expmat = np.dot(Tinv, np.dot(expD, T))

expmat2 = sl.expm(beta*mat)



#%%
deg = 4
N = 100
n = 50
lattice_m = 10
lattice_n = 10

import networkx as nx
#G = nx.generators.florentine_families_graph()
G = nx.generators.gnm_random_graph(10, 20)
#G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],periodic= False)
Gc = max(nx.connected_component_subgraphs(G), key = len)

A = nx.adjacency_matrix(Gc)
A1 = A.toarray()
D = np.diag(A1.sum(axis=0))

L = nx.laplacian_matrix(Gc)
Lcsc = ss.csc_matrix(L)
Lnorm = nx.normalized_laplacian_matrix(Gc)

L_spectrum = nx.laplacian_spectrum(Gc)
Lnorm_spectrum, Lnorm_eigs = np.linalg.eig(Lnorm.toarray())

l_scale = 0.6
K = sl.expm(l_scale * Lcsc)



#%%
import sys
sys.path.append("../")

from GPnet import GPnetRegressor
#%%
p0 = np.log(0.4)
p0box = [np.log(0.01), np.log(0.99)]
p1 = np.log(1)
p1box = [np.log(0.01), np.log(10)]

box = [p0box, p1box]
optimize={'method':'SLSQP', 'bounds':box}

#%% 
a = GPnetRegressor(Graph = Gc, ntrain =5, ntest=5, theta = [np.log(0.9), np.log(4)], relabel_nodes= True, optimize=optimize)
#a.kernel(a.training_nodes, a.training_nodes, [np.log(0.5),1], wantderiv=False)


a.predict()
a.plot_predict_2d()


#%%
theta0 = np.linspace(-2, -0.01, 30)
theta1 = np.linspace(-3, 3, 31)

theta =[theta0, theta1]


plots = {"const vs lambda":    [[0,1], theta0, theta1, [p0,p1]],
         "const vs lambdaq":    [[0,1], theta0, theta1, [p0,p1]],
         "const vs lambdaq1":    [[0,1], theta0, theta1, [p0,p1]],
         }

a.plot_lml_landscape(plots, [p0, p1])

