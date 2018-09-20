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

def assign_nodes(Graph, N, n, seed=0):
    #training_nodes = list(G.nodes)[0:N]
    random.seed(seed)
    training_nodes = random.sample(list(Graph.nodes), N)
    training_nodes.sort()
    
    #test_nodes = list(G.nodes)[N:N+n]
    test_nodes = random.sample((set(Graph.nodes) - set(training_nodes)), n)
    test_nodes.sort()
    
    othernodes = set(Graph.nodes) - set(training_nodes) - set(test_nodes)
    othernodes = list(othernodes)
    othernodes.sort()
    
    return [training_nodes, test_nodes, othernodes]


class GPnetRegressor:
    def __init__(self, Graph=False, totnodes=False, ntrain=100, ntest=90, deg=4, train_values=False, seed=0, training_nodes=False, test_nodes=False, other_nodes=False):
        
        
        self.N = ntrain
        self.n = ntest
        self.deg = deg
        
        self.seed=0
        if totnodes==False:
            self.totnodes = self.N + self.n
        else:
            self.totnodes = totnodes
            
        
        if Graph==False:
            G = nx.random_regular_graph(deg, totnodes)
            self.Graph = nx.relabel_nodes(G, dict(zip(G,range(len(G.nodes)))))
            
        else:
            self.Graph = Graph
            self.training_nodes = training_nodes
            self.test_nodes = test_nodes
            self.other_nodes = other_nodes
        
        if training_nodes == False or test_nodes == False:
            print("assigning nodes randomly")
            print(self.N, " training nodes")
            print(self.n ,  " test nodes")
            print((self.totnodes - (self.N + self.n)), " idle nodes")
           
            self.assign_nodes()
        
        if train_values == False:
            pvtdist = self.pivot_distance(0)
            self.t = pvtdist[self.training_nodes]
        else:
            self.t = train_values
            
        self.calc_shortest_paths()
            
             
    def pivot_distance(self, pivot=0):
        pivot_distance = pd.Series(dict(nx.single_source_shortest_path_length(self.Graph,pivot))).sort_index()
        return pivot_distance
       
    def calc_shortest_paths(self):
        #shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(G))
        shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(self.Graph))
        self.dist = pd.DataFrame(shortest_paths_lengths).sort_index(axis=1)
        return
        
    def assign_nodes(self):
        
        if self.N + self.n > self.totnodes:
            raise ValueError("tot. nodes cannot be less than training nodes + test nodes")
        #training_nodes = list(G.nodes)[0:N]
        random.seed(self.seed)
        self.training_nodes = random.sample(list(self.Graph.nodes), self.N)
        self.training_nodes.sort()
        
        #test_nodes = list(G.nodes)[N:N+n]
        self.test_nodes = random.sample((set(self.Graph.nodes) - set(self.training_nodes)), self.n)
        self.test_nodes.sort()
        
        self.other_nodes = set(self.Graph.nodes) - set(self.training_nodes) - set(self.test_nodes)
        self.other_nodes = list(self.other_nodes)
        self.other_nodes.sort()
        return

    def plot(self):
        pl.figure(figsize = [10,9])
        #node positions
        self.plot_pos = nx.kamada_kawai_layout(self.Graph)
        #draw nodes
        nx.draw_networkx_nodes(self.Graph, self.plot_pos, with_labels=True, node_size=200, nodelist=self.training_nodes, node_color="r")
        nx.draw_networkx_nodes(self.Graph, self.plot_pos, with_labels=True, node_size=200, nodelist=self.test_nodes, node_color="g")
        nx.draw_networkx_nodes(self.Graph, self.plot_pos, with_labels=True, node_size=200, nodelist=self.other_nodes, node_color="b")
        #draw edges
        ec = nx.draw_networkx_edges(self.Graph, self.plot_pos, alpha=0.2)
        #legend
        labels = nx.draw_networkx_labels(self.Graph, pos=self.plot_pos, font_color='k')
        red_patch = mpatches.Patch(color='red', label='training nodes')
        blue_patch = mpatches.Patch(color='blue', label='test nodes')
        green_patch = mpatches.Patch(color='green', label='other nodes')
        pl.legend(handles=[red_patch, blue_patch, green_patch])
        

#%%
N = 100     # numero punti training
n = 90    # numero punti test
deg = 4 #connectivity degree
#%%

G = nx.random_regular_graph(4, N+n + 10)
#G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],periodic= False)
G = nx.relabel_nodes(G, dict(zip(G,range(len(G.nodes)))))


#%%
lengthscales = np.linspace(0.001, 100,   5)
constantscales = np.linspace(0, 1, 5)
noisescales = np.linspace(0.0001, 100, 5)


dataframe = pd.DataFrame(columns = ["LengthScale", "ConstantScale", "Noise_Scale", "K", "K**"])

for l_scale in lengthscales:
    for c_scale in constantscales:
        for n_scale in noisescales:
            theta = np.array([c_scale, l_scale, n_scale])
            k = net_kernel(G, dist, training_nodes, training_nodes, theta, wantderiv=False)
            kstarstar = net_kernel(G, dist, test_nodes, test_nodes,theta, wantderiv=False)
            
            is_pos_def(k)
            dataframe.loc[len(dataframe)] = [l_scale, c_scale, n_scale, is_pos_def(k), is_pos_def(kstarstar)]
            
            

