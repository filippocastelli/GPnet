import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import random
import numpy as np
import matplotlib.pylab as pl

#%%

def shortest_path_graph_distances(Graph):
    #shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(G))
    shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(Graph))
    dist = pd.DataFrame(shortest_paths_lengths).sort_index(axis=1)
    return dist

def squared_exponential(dist, params):
    return params[0]*np.exp(-.5 * (1/params[1]) * dist)

def net_kernel(nodes_a, nodes_b, Graph, params, graph_distance_matrix):
    nodelist = list(Graph.nodes)
    nodeset = set(nodes_a).union(set(nodes_b))
    nodes_to_drop = [x for x in nodelist if x not in nodeset]
    cols_to_drop = set(nodes_to_drop).union(set(nodes_b) - set(nodes_a))
    rows_to_drop = set(nodes_to_drop).union(set(nodes_a) - set(nodes_b))
    p = graph_distance_matrix.drop(cols_to_drop).drop(
            rows_to_drop, 1)
    distances = p.values
    return squared_exponential(distances, params)

#%%
lattice_m = 15
lattice_n = 15
N = 100     # numero punti training
n = 100    # numero punti test
deg = 4
s = 0.05    # noise variance
#%%
#G = nx.random_regular_graph(deg, N+n + 10)
G = nx.generators.lattice.grid_graph(dim = [lattice_m,lattice_n],periodic= False)
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

distances = shortest_path_graph_distances(G)
#%%

#nx.draw_networkx(G)
pl.figure(0, dpi=200, figsize=[12,7])
#node positions
pos = nx.kamada_kawai_layout(G)
#draw nodes
nx.draw_networkx_nodes(G, pos, with_labels=True, node_size=200, nodelist=training_nodes, node_color="r")
nx.draw_networkx_nodes(G, pos, with_labels=True, node_size=200, nodelist=test_nodes, node_color="g")
nx.draw_networkx_nodes(G, pos, with_labels=True, node_size=200, nodelist=othernodes, node_color="b")
#draw edges
ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
#legend
labels = nx.draw_networkx_labels(G, pos=pos, font_color='k')
red_patch = mpatches.Patch(color='red', label='training nodes')
blue_patch = mpatches.Patch(color='blue', label='test nodes')
green_patch = mpatches.Patch(color='green', label='other nodes')
pl.legend(handles=[red_patch, blue_patch, green_patch])
#%%
#uso come funzione di prova la distanza dal nodo 0
pivot_distance = pd.Series(dict(nx.single_source_shortest_path_length(G,0))).sort_index()
y = pivot_distance[training_nodes]


pl.figure(1,dpi=200, figsize=[12,7])
vmin = pivot_distance.min()
vmax = pivot_distance.max() 
cmap = pl.cm.inferno_r

sm = pl.cm.ScalarMappable(cmap=cmap, norm=pl.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = pl.colorbar(sm)

nx.draw_networkx_nodes(G,pos, node_color=pivot_distance,with_labels=True, node_size=200, cmap=pl.cm.inferno_r)
ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
labels = nx.draw_networkx_labels(G, pos=pos, font_color='w')
#%%
lengthscale = 10
constantscale = 1
kernel_parameters = [constantscale, lengthscale]

kernel_noise_variance = 0.001
K = net_kernel(training_nodes, training_nodes, G, kernel_parameters, distances)

# Ky = L L^T
L = np.linalg.cholesky(np.float64(K) + 0.005*np.eye(len(training_nodes)))

# mu = K*^T Ky^-1 y = K*T alpha
# alpha = Ky^-1 y = L^-T L^-1 y

# Lk = L^-1 K*
Lk = np.linalg.solve(L, net_kernel(training_nodes, test_nodes, G, kernel_parameters, distances))
# Lk^T = K*^T L^-T
# np.linalg.solve(L,y) = L^-1 y
# K*^T L^-T L^-1 y = K*^T Ky^-1 y
L_inv_y = np.linalg.solve(L, y)
alpha = np.dot(np.linalg.inv(L).T, L_inv_y)

mu = np.dot(Lk.T, L_inv_y )

loglikelihood = -0.5*np.dot(y.T, alpha) - (np.log(L.diagonal())).sum() - (N/2)*np.log(2*np.pi)

#K**
K_ = net_kernel(test_nodes, test_nodes, G,kernel_parameters, distances)   

#s2 = K** - K*^T Ky^-1 K*
# Lk^2 = Lk^T Lk = (L^-1 K*)^T (L^-1 K*) = K*^T L^-T L^-1 K* = K*^T K_y^-1 K*
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)
#%%
pl.figure(2,dpi=200, figsize=[12,7])

training_nodes_colors  = [pivot_distance[node] for node in training_nodes]

nx.draw_networkx_nodes(G,pos,nodelist=training_nodes, node_color=pivot_distance[(training_nodes)],with_labels=True, node_size=50, cmap=cmap)
nx.draw_networkx_nodes(G,pos,nodelist=othernodes, node_color="g", with_labels=True, node_size=200, cmap=cmap)
nx.draw_networkx_nodes(G,pos,nodelist=test_nodes, node_color=mu, with_labels=True, node_size=200, cmap=cmap)
ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
cbar = pl.colorbar(sm)
labels = nx.draw_networkx_labels(G, pos=pos, font_color='k')