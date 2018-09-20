# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:13:17 2018

@author: filip
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as so
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import random
#%%
def net_logPosterior(theta,*args):
    Graph,distancematrix, data,t = args
    #k = kernel(data,data,theta,wantderiv=False)
    k = net_kernel(Graph,distancematrix, data,data,theta,wantderiv=False)
    L = np.linalg.cholesky(k)
    beta = np.linalg.solve(L.transpose(), np.linalg.solve(L,t))
    logp1 = -0.5*np.dot(t.transpose(),beta)
    logp2 = - np.sum(np.log(np.diag(L)))
    logp3 = - np.shape(data)[0] /2. * np.log(2*np.pi)
    logp = -0.5*np.dot(t.transpose(),beta) - np.sum(np.log(np.diag(L))) - np.shape(data)[0] /2. * np.log(2*np.pi)
    return -logp

def net_gradLogPosterior(theta,*args):
    #print(args)
    Graph,distancematrix, data,t = args
    theta = np.squeeze(theta)
    d = len(theta)
    #K = kernel(data,data,theta,wantderiv=True)
    K = net_kernel(Graph,distancematrix, data, data, theta, wantderiv=True)

    L = np.linalg.cholesky(np.squeeze(K[:,:,0]))
    invk = np.linalg.solve(L.transpose(),np.linalg.solve(L,np.eye(np.shape(data)[0])))
	
    dlogpdtheta = np.zeros(d)
    for d in range(1,len(theta)+1):
        dlogpdtheta[d-1] = 0.5*np.dot(t.transpose(), np.dot(invk, np.dot(np.squeeze(K[:,:,d]), np.dot(invk,t)))) - 0.5*np.trace(np.dot(invk,np.squeeze(K[:,:,d])))

    return -dlogpdtheta

def shortest_path_graph_distances(Graph):
    #shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(G))
    shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(Graph))
    dist = pd.DataFrame(shortest_paths_lengths).sort_index(axis=1)
    return dist

def squared_exponential(dist, params):
    return params[0]*np.exp(-.5 * (1/params[1]) * dist)

def net_kernel(Graph,graph_distance_matrix, nodes_a,nodes_b,theta,measnoise=1., wantderiv=True, print_theta=False):
    theta = np.squeeze(theta)
    theta = np.exp(theta)
    #graph_distance_matrix = shortest_path_graph_distances(Graph)
    nodelist = list(Graph.nodes)
    nodeset = set(nodes_a).union(set(nodes_b))
    nodes_to_drop = [x for x in nodelist if x not in nodeset]
    cols_to_drop = set(nodes_to_drop).union(set(nodes_b) - set(nodes_a))
    rows_to_drop = set(nodes_to_drop).union(set(nodes_a) - set(nodes_b))
    p = graph_distance_matrix.drop(cols_to_drop).drop(rows_to_drop, 1)
    distances = p.values**2
    
    d1 = len(nodes_a)
    d2 = len(nodes_b)
  
    k = theta[0] * np.exp(-0.5*distances)
    
    if wantderiv:
        K = np.zeros((d1,d2,len(theta)+1))
        # K[:,:,0] is the original covariance matrix
        K[:,:,0] = k + measnoise*theta[2]*np.eye(d1,d2)
        K[:,:,1] = k
        K[:,:,2] = -0.5*k*distances
        K[:,:,3] = theta[2]*np.eye(d1,d2)
        return K
    else:
        return k + measnoise*theta[2]*np.eye(d1,d2)
    
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
#%%
lattice_m = 15
lattice_n = 15
N = 100     # numero punti training
n = 90    # numero punti test
deg = 4 #connectivity degree
#%%
G = nx.random_regular_graph(deg, N+n + 10)
#G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],periodic= False)
G = nx.relabel_nodes(G, dict(zip(G,range(len(G.nodes)))))
random.seed(0)

#training_nodes = list(G.nodes)[0:N]
training_nodes = random.sample(list(G.nodes), N)
training_nodes.sort()

#test_nodes = list(G.nodes)[N:N+n]
test_nodes = random.sample((set(G.nodes) - set(training_nodes)), n)
test_nodes.sort()

othernodes = set(G.nodes) - set(training_nodes) - set(test_nodes)
othernodes = list(othernodes)
othernodes.sort()

dist = shortest_path_graph_distances(G)
#%%
#uso come funzione di prova la distanza dal nodo 0
pivot_distance = pd.Series(dict(nx.single_source_shortest_path_length(G,0))).sort_index()
t = pivot_distance[training_nodes]

#%%
lengthscale = 1
lengthscales = np.linspace(0.001, 100,   5)
constantscale = 1
constantscales = np.linspace(0, 1, 5)
noise_scale = -10
noisescales = np.linspace(0.0001, 100, 5)
theta = np.array([constantscale, lengthscale, noise_scale])


dataframe = pd.DataFrame(columns = ["LengthScale", "ConstantScale", "Noise_Scale", "K", "K**"])

for l_scale in lengthscales:
    for c_scale in constantscales:
        for n_scale in noisescales:
            theta = np.array([c_scale, l_scale, n_scale])
            k = net_kernel(G, dist, training_nodes, training_nodes, theta, wantderiv=False)
            kstarstar = net_kernel(G, dist, test_nodes, test_nodes,theta, wantderiv=False)
            
            is_pos_def(k)
            dataframe.loc[len(dataframe)] = [l_scale, c_scale, n_scale, is_pos_def(k), is_pos_def(kstarstar)]
            
            
#%%
k = net_kernel(G, dist, training_nodes, training_nodes, theta, wantderiv=False)
kstar = net_kernel(G, dist, test_nodes, training_nodes, theta, wantderiv=False)
kstarstar = net_kernel(G, dist, test_nodes, test_nodes,theta, wantderiv=False)
kstarstar_diag = np.diag(kstarstar)
#%%
L = np.linalg.cholesky(k)
invk = np.linalg.solve(L.transpose(),np.linalg.solve(L,np.eye(len(training_nodes))))

mean = np.squeeze(np.dot(kstar,np.dot(invk,t)))
var = kstarstar_diag - np.diag(np.dot(kstar,np.dot(invk,kstar.T)))
var = np.squeeze(np.reshape(var,(n,1)))
s = np.sqrt(var)

#pl.axis([-5, 5, -3, 3])
#%%
#L2 = np.linalg.cholesky(kstarstar + 1e-6*np.eye(n))
#Lk = np.linalg.solve(L, kstar.T)
#L2 = np.linalg.cholesky(kstarstar+ 1e-6*np.eye(n) - np.dot(Lk.T, Lk))

