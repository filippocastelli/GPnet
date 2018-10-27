# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:22:42 2018

@author: filip
"""

#%%
import numpy as np
import scipy.linalg as sl
import scipy.sparse as ss
import pandas as pd

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
lattice_m = 20
lattice_n = 20

import networkx as nx
#G = nx.generators.florentine_families_graph()
#G = nx.generators.gnm_random_graph(10, 20)
G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],periodic= False)
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

from GPnetRegressor import GPnetRegressor
from GPnetClassifier import GPnetClassifier


#%%
p0 = np.log(0.3)
p0box = [np.log(0.2), np.log(0.99)]
p1 = np.log(5)
p1box = [np.log(0.01), np.log(10)]


theta0 = np.linspace(np.log(0.01), np.log(0.99), 4)
theta1 = np.linspace(np.log(0.01), np.log(10), 5)

theta =[theta0, theta1]




box = [p0box, p1box]
optimize={'method':'SLSQP', 'bounds':box}

#%% 
a = GPnetRegressor(Graph = Gc, ntrain =100, ntest=300, theta = [p0, p1], relabel_nodes= True, optimize=False)
#a.kernel(a.training_nodes, a.training_nodes, [np.log(0.5),1], wantderiv=False)

a.predict()
a.plot_predict_2d()
a.calc_ktot()

#%%
L2 = np.linalg.cholesky(a.ktot + 1e-6 * np.eye(len(a.ktot)))
# f_prior = mu L*N(0,1)
f_prior = a.ktot.mean() + np.dot(L2, np.random.normal(size=(len(a.ktot), 5)))

f_prior = np.where(f_prior<0, 0, f_prior)
new_training = pd.Series(index= a.training_nodes, data=f_prior[a.training_nodes][:,3])
#%%
a.pivot_flag = False
a.set_training_values(new_training)
a.predict()
#%%
theta0 = np.linspace(np.log(0.01), np.log(0.99), 20)
theta1 = np.linspace(np.log(0.1), np.log(10), 20)

theta =[theta0, theta1]

plots = {"const vs lambda":    [[0,1],  theta0, theta1, [p0,p1]],
         }

a.plot_lml_landscape(plots, [p0, p1])

