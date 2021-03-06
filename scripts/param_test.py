from __future__ import division
import numpy as np
import networkx as nx
import pandas as pd
from GPnet import GPnetRegressor
#%%
#randomregular
a = GPnetRegressor(totnodes = 400, ntrain=50, ntest=30, deg=3, theta=[0.1, 0.1, 0.1])
#%%
a.plot_graph()
a.predict()
a.plot_prior()
a.plot_post()
#%%
lengthscales = np.linspace(0.1, 100,   5)
constantscales = np.linspace(0.1, 10, 5)
noisescales = np.linspace(0.1, 100, 5)

#%%
dataframe = pd.DataFrame(columns = ["LengthScale", "ConstantScale", "Noise_Scale", "K_posdef","LogP"])

for l_scale in lengthscales:
    for c_scale in constantscales:
        for n_scale in noisescales:
            theta = np.array([c_scale, l_scale, n_scale])
            a.theta = theta
            a.predict()
            dataframe.loc[len(dataframe)] = [l_scale, c_scale, n_scale, a.is_pos_def(a.k), a.logp()]
            
#%% BARABASI-ALBERT
            
G = nx.barabasi_albert_graph(100, 3)
b = GPnetRegressor(G, ntrain=50, ntest=30, theta=[1,1,2])
b.predict()
#%%
dataframe = pd.DataFrame(columns = ["LengthScale", "ConstantScale", "Noise_Scale", "K_posdef","LogP"])

for l_scale in lengthscales:
    for c_scale in constantscales:
        for n_scale in noisescales:
            theta = np.array([c_scale, l_scale, n_scale])
            b.theta = theta
            b.predict()
            dataframe.loc[len(dataframe)] = [l_scale, c_scale, n_scale, b.is_pos_def(b.k), b.logp()]
            
            
