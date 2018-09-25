from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as so
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import random

#%%
class GPnet:
    def __init__(self, Graph, totnodes, ntrain, ntest,deg, seed, training_nodes, training_values,
                     test_nodes, theta,optimize):
            self.N = ntrain
            self.n = ntest
            self.deg = deg
            self.seed=0
            
            self.optimize_flag = optimize
            
            self.theta = theta
            
            if totnodes==False:
                self.totnodes = self.N + self.n
            else:
                self.totnodes = totnodes
                
            if Graph==False:
                print("> Initializing Random Regular Graph")
                print(self.totnodes, "nodes")
                print("node degree", self.deg)
                G = nx.random_regular_graph(deg, totnodes)
                self.Graph = nx.relabel_nodes(G, dict(zip(G,range(len(G.nodes)))))
                
            else:
                self.Graph = Graph
                self.totnodes = len(Graph.nodes)
            self.training_nodes = training_nodes
            self.test_nodes = test_nodes
                #self.other_nodes = other_nodes
            
            if training_nodes == False or test_nodes == False:
                print("> Assigning Nodes Randomly ( seed =", self.seed, ")")
                print(self.N, " training nodes")
                print(self.n ,  " test nodes")
                print((self.totnodes - (self.N + self.n)), " idle nodes")
               
                self.random_assign_nodes()
            
                
            self.calc_shortest_paths()
            
            # END INIT #
            
    def pivot_distance(self, pivot=0):
        pivot_distance = pd.Series(dict(nx.single_source_shortest_path_length(self.Graph,pivot))).sort_index()
        return pivot_distance
       
    def calc_shortest_paths(self):
        #shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(G))
        shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(self.Graph))
        self.dist = pd.DataFrame(shortest_paths_lengths).sort_index(axis=1)
        return
        
    def random_assign_nodes(self):
        
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
        return self
    
    
    def is_pos_def(self,test_mat):
        return np.all(np.linalg.eigvals(test_mat) > 0)
        
    def kernel(self, nodes_a,nodes_b,theta,measnoise=1., wantderiv=True, print_theta=False):
        theta = np.squeeze(theta)
        theta = np.exp(theta)
        #graph_distance_matrix = shortest_path_graph_distances(Graph)
        nodelist = list(self.Graph.nodes)
        nodeset = set(nodes_a).union(set(nodes_b))
        nodes_to_drop = [x for x in nodelist if x not in nodeset]
        cols_to_drop = set(nodes_to_drop).union(set(nodes_b) - set(nodes_a))
        rows_to_drop = set(nodes_to_drop).union(set(nodes_a) - set(nodes_b))
        p = self.dist.drop(cols_to_drop).drop(rows_to_drop, 1)
        distances =(p.values/theta[1])**2
        
        d1 = len(nodes_a)
        d2 = len(nodes_b)
      
        #k = 2 +theta[0] * np.exp(-0.5*distances)
        k = (theta[0]**2) * np.exp(-0.5*distances)
        
        
        if wantderiv:
            K = np.zeros((d1,d2,len(theta)+1))
            # K[:,:,0] is the original covariance matrix
            K[:,:,0] = k + measnoise*theta[2]*np.eye(d1,d2)
            K[:,:,1] = 2*k
            K[:,:,2] = (k*distances)/theta[1]
            K[:,:,3] = theta[2]*np.eye(d1,d2)
            return K
        else:
            return k + measnoise*theta[2]*np.eye(d1,d2)
        
        

    def logp(self):
        return -self.logPosterior(self.theta, self.training_nodes, self.t)
    
    
    def plot_graph(self, filename=False):
        pl.figure(figsize = [10,9])
        #node positions
        self.plot_pos = nx.kamada_kawai_layout(self.Graph)
        #draw nodes
        nx.draw_networkx_nodes(self.Graph, self.plot_pos, with_labels=True, 
                               node_size=200, nodelist=self.training_nodes, 
                               node_color="r")
        nx.draw_networkx_nodes(self.Graph, self.plot_pos, with_labels=True, 
                               node_size=200, nodelist=self.test_nodes, 
                               node_color="g")
        nx.draw_networkx_nodes(self.Graph, self.plot_pos, with_labels=True, 
                               node_size=200, nodelist=self.other_nodes, 
                               node_color="b")
        #draw edges
        ec = nx.draw_networkx_edges(self.Graph, self.plot_pos, alpha=0.2)
        #legend
        labels = nx.draw_networkx_labels(self.Graph, pos=self.plot_pos, 
                                         font_color='k')
        red_patch = mpatches.Patch(color='red', label='training nodes')
        blue_patch = mpatches.Patch(color='blue', label='test nodes')
        green_patch = mpatches.Patch(color='green', label='other nodes')
        pl.legend(handles=[red_patch, blue_patch, green_patch])
        if type(filename) is str:
            pl.savefig(filename, bbox_inches='tight')
        return self
    
    def plot_prior(self, filename=False):
        L2 = np.linalg.cholesky(self.kstarstar + 1e-6*np.eye(self.n))
        #f_prior = mu L*N(0,1)
        f_prior = np.dot(L2, np.random.normal(size=(self.n,5)))
        pl.figure()
        pl.clf()
        pl.plot(self.test_nodes, f_prior)
        pl.title('5 estrazioni dalla dist. a priori')
        pl.xlabel('nodes')
        pl.ylabel('values')
        #pl.axis([-5, 5, -3, 3])
        if type(filename) is str:
            pl.savefig(filename, bbox_inches='tight')
        
        
    def plot_post(self, filename=False):
        Lk = np.linalg.solve(self.L, self.kstar.T)
        L2 = np.linalg.cholesky(self.kstarstar+ 1e-6*np.eye(self.n) - np.dot(Lk.T, Lk))
        
        #f_post = mu + L*N(0,1)
        f_post = self.mean.reshape(-1,1) + np.dot(L2, np.random.normal(size=(self.n,5)))
        pl.figure()
        pl.clf()
        pl.plot(self.test_nodes, f_post)
        pl.title('5 estrazioni dalla dist. a posteriori')
        pl.xlabel('nodes')
        pl.ylabel('values')
        #pl.axis([-5, 5, -3, 3])
        if type(filename) is str:
            pl.savefig(filename, bbox_inches='tight')
            
            
    def int_to_list(nodes):
        if type(nodes) == int:
            return [nodes]
        else:
            return nodes
            

class GPnetRegressor(GPnet):
    def __init__(self, Graph=False, totnodes=False, ntrain=100, ntest=90,
                     deg=4, seed=0, training_nodes=False, train_values=False, training_values=False,
                     test_nodes=False, theta = [0.1, 0.1, 0.1],
                     optimize=False):
        super(GPnetRegressor, self).__init__(Graph, totnodes, ntrain, ntest, deg, seed, training_nodes,train_values, test_nodes, theta, optimize)
        self.pivot_flag = False
        if train_values == False:
            self.pivot_flag = True
            self.pvtdist = self.pivot_distance(0)
            self.t = self.pvtdist[self.training_nodes]
        else:
            self.t = train_values
       
    def predict(self):
        self.optimize_params()
        
        self.k_not_posdef_flag = False
        self.kstar_not_posdef_flag = False
        #self.mean_t = np.mean(self.t)
        self.k =self.kernel(nodes_a = self.training_nodes, nodes_b = self.training_nodes, theta = self.theta, wantderiv=False)

        self.kstar = self.kernel(nodes_a = self.test_nodes, nodes_b=self.training_nodes, theta = self.theta, wantderiv=False, measnoise=False)
        self.kstarstar = self.kernel(nodes_a = self.test_nodes, nodes_b = self.test_nodes,theta = self.theta, wantderiv=False)

        self.kstarstar_diag = np.diag(self.kstarstar)
        
        if (not self.is_pos_def(self.k)):
            self.k_not_posdef_flag = True
            #raise ValueError("K is not positive definite")
            print("K not positive definite, aborting...")
            return self
        if (not self.is_pos_def(self.kstarstar)):
            self.kstar_not_posdef_flag = True
            #raise ValueError("K** is not positive definite")
            print("K** not positive definite, aborting...")
            return self

        self.L = np.linalg.cholesky(self.k)
        invk = np.linalg.solve(self.L.transpose(),np.linalg.solve(self.L,np.eye(len(self.training_nodes))))
        self.mean = np.squeeze(np.dot(self.kstar,np.dot(invk,self.t)))
        self.var = self.kstarstar_diag - np.diag(np.dot(self.kstar,np.dot(invk,self.kstar.T)))
        self.var = np.squeeze(np.reshape(self.var,(self.n,1)))
        self.s = np.sqrt(self.var)
        
        print("succesfully trained model")
    
    def oldlogPosterior(self, theta,*args):
        data,t = args
        k = self.kernel(data,data,theta,wantderiv=False)
        if self.is_pos_def(k) == False:
            return +999
        L = np.linalg.cholesky(k)
        beta = np.linalg.solve(L.transpose(), np.linalg.solve(L,t))
        logp = -0.5*np.dot(t.transpose(),beta) - np.sum(np.log(np.diag(L))) - np.shape(data)[0] /2. * np.log(2*np.pi)
        return -logp
    
    def logPosterior(self,theta,*args):
        data,t = args
        k = self.kernel(data,data,theta,wantderiv=False)
        try:
            L = np.linalg.cholesky(k)  # Line 2
        except np.linalg.LinAlgError:
            return np.inf
           # return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf
    
        #L = np.linalg.cholesky(k)
        alpha = np.linalg.solve(L, t )
        alpha.resize(len(alpha),1)
        t1 = t.values
        t1.resize(len(t1),1)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", t1, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= k.shape[0] / 2 * np.log(2 * np.pi)
        logp = log_likelihood_dims.sum(-1)  # sum over dimensions
        #beta = np.linalg.solve(L.transpose(), np.linalg.solve(L,t))
        #logp = -0.5*np.dot(t.transpose(),beta) - np.sum(np.log(np.diag(L))) - np.shape(data)[0] /2. * np.log(2*np.pi)
        return -logp
    
    
    def oldgradLogPosterior(self, theta,*args):
        data,t = args
        theta = np.squeeze(theta)
        d = len(theta)
        #K = kernel(data,data,theta,wantderiv=True)
        K = self.kernel(data, data, theta, wantderiv=True)
    
        L = np.linalg.cholesky(np.squeeze(K[:,:,0]))
        invk = np.linalg.solve(L.transpose(),np.linalg.solve(L,np.eye(np.shape(data)[0])))
    	
        dlogpdtheta = np.zeros(d)
        for d in range(1,len(theta)+1):
            dlogpdtheta[d-1] = 0.5*np.dot(t.transpose(),np.dot(invk, np.dot(np.squeeze(K[:,:,d]),np.dot(invk,t)))) - 0.5*np.trace(np.dot(invk,np.squeeze(K[:,:,d])))
    
        return -dlogpdtheta


    def gradLogPosterior(self, theta,*args):
        data,t = args
        theta = np.squeeze(theta)
        k = self.kernel(data,data,theta,wantderiv=True)
        try:
            L = np.linalg.cholesky(k[:,:,0])  # Line 2
        except np.linalg.LinAlgError:
            return -np.inf
           # return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf
    
        #L = np.linalg.cholesky(k)
        alpha = np.linalg.solve(L, t )
        tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
        tmp -= np.linalg.solve(L, np.eye(k.shape[0]))[:, :, np.newaxis]
        # Compute "0.5 * trace(tmp.dot(K_gradient))" without
        # constructing the full matrix tmp.dot(K_gradient) since only
        # its diagonal is required
        log_likelihood_gradient_dims = \
            0.5 * np.einsum("ijl,ijk->kl", tmp, k[:,:,1:])
        log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
        
        return -log_likelihood_gradient
    
    
    def optimize_params(self, gtol=1e-3, maxiter=200, disp=1):
        if self.optimize_flag == True:
            self.theta = so.fmin_cg(self.logPosterior, self.theta, 
                                    fprime=self.gradLogPosterior, 
                                    args=(self.training_nodes,self.t), 
                                    gtol=gtol,maxiter=200,disp=1)
        return self

    def gen_cmap(self):
        self.vmin = min(self.t.min(), self.mean.min())
        self.vmax = max(self.t.max(), self.mean.max())
        self.cmap = pl.cm.inferno_r

    def plot_result(self, filename=False):
        pl.figure()
        pl.clf()
        pl.plot(self.training_nodes, self.t, 'r+', ms=20)
        if self.pivot_flag == True:
            pl.plot(self.pvtdist)
        
        pl.gca().fill_between(self.test_nodes, self.mean-self.s, self.mean+self.s, color="#dddddd")
        pl.plot(self.test_nodes, self.mean, 'ro', ms=4)
        pl.plot(self.test_nodes, self.mean, 'r--', lw=2)
        pl.title('Valore medio e margini di confidenza')
        loglikelihood = self.logPosterior(self.theta,self.training_nodes,self.t)
        pl.title('Valore medio e margini a posteriori, (length scale: %.3f , constant scale: %.3f ,\
                                                        #noise variance: %.3f )\n Log-Likelihood: %.3f'
                                                        % (self.theta[1], self.theta[0], self.theta[2], loglikelihood))
        pl.xlabel('nodes')
        pl.ylabel('values')
        if type(filename) is str:
            pl.savefig(filename, bbox_inches='tight')
        #pl.axis([-5, 5, -3, 3])
        return self
        
    def plot_graph_with_values(self, filename=False):
        pl.figure(figsize=[10,9])
        #pl.figure(2,dpi=200, figsize=[12,7])
        
        training_nodes_colors  = [self.t[node] for node in self.training_nodes]
        self.gen_cmap()
        nx.draw_networkx_nodes(self.Graph,self.plot_pos,nodelist=self.training_nodes, node_color=self.t[(self.training_nodes)],with_labels=True, node_size=50, cmap=self.cmap)
        nx.draw_networkx_nodes(self.Graph,self.plot_pos,nodelist=self.other_nodes, node_color="gray", with_labels=True, node_size=200, cmap=self.cmap)
        nx.draw_networkx_nodes(self.Graph,self.plot_pos,nodelist=self.test_nodes, node_color=self.mean, with_labels=True, node_size=200, cmap=self.cmap)
        ec = nx.draw_networkx_edges(self.Graph, self.plot_pos, alpha=0.2)
        
        sm = pl.cm.ScalarMappable(cmap=self.cmap, norm=pl.Normalize(vmin=self.vmin, vmax=self.vmax))
        sm.set_array([])
        cbar = pl.colorbar(sm)
        
        labels = nx.draw_networkx_labels(self.Graph, pos=self.plot_pos, font_color='k')
        if type(filename) is str:
            pl.savefig(filename, bbox_inches='tight')
        return self
    
    
class GPnetClassifier(GPnet):
    
    
    def __init__(self, Graph=False, totnodes=False, ntrain=100, ntest=90,
                 deg=4, seed=0, training_nodes=False, train_values=False, training_values=False,
                 test_nodes=False, theta = [0.1, 0.1, 0.1],
                 optimize=False):
        super(GPnetClassifier, self).__init__(Graph, totnodes, ntrain, ntest, deg, seed, training_nodes,train_values, test_nodes, theta, optimize)
        self.pivot_flag = False
        if train_values == False:
            self.pivot_flag = True
            self.pvtdist = self.pivot_distance(0)
            self.t = self.pvtdist[self.training_nodes]
            self.binary_labels = (np.sin(0.5*self.pvtdist)>0).replace({True: 1, False: -1})
            self.training_labels = self.binary_labels[self.training_nodes]
        else:
                self.training_labels = train_values
                
    def logPosterior(self,theta,*args):
        Graph, distmatrix,data,targets = args
        (f,logq,a) = self.NRiteration(Graph, distmatrix, data,targets,theta)
        return -logq

    def NRiteration(self, data,targets,theta, tol=0.1, phif=1e100, scale=1.):
        #print("iteration")
        #pag 46 RASMUSSEN-WILLIAMS
        K = self.kernel( data, data, theta, wantderiv=False)
        #K = kernel(data,data,theta,wantderiv=False)
        n = np.shape(targets)[0]
        f = np.zeros((n,1))
#        tol = 0.1
#        phif = 1e100
#        scale = 1.
        count = 0
        targets = targets.values.reshape(n,1)
        while True:	
            
            count += 1
            s = np.where(f<0,f,0)
            W = np.diag(np.squeeze(np.exp(2*s - f) / ((np.exp(s) + np.exp(s-f))**2)))
            sqrtW = np.sqrt(W)
            # L = cholesky(B)
            L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW,np.dot(K,sqrtW)))
            p = np.exp(s)/(np.exp(s) + np.exp(s-f))
            b = np.dot(W,f) + 0.5*(targets+1) - p
            a = scale*(b - np.dot(sqrtW,np.linalg.solve(L.transpose(),np.linalg.solve(L,np.dot(sqrtW,np.dot(K,b))))))
            f = np.dot(K,a)
            oldphif = phif
            phif = np.log(p) -0.5*np.dot(f.transpose(),np.dot(np.linalg.inv(K),f)) - 0.5*np.sum(np.log(np.diag(L))) - np.shape(data)[0] /2. * np.log(2*np.pi)
            #print(phif)
            #print("loop",np.sum((oldphif-phif)**2))
            if (np.sum((oldphif-phif)**2) < tol):	
                break
            elif (count > 100):
                count = 0
                scale = scale/2.
    	
        s = -targets*f
        ps = np.where(s>0,s,0)
        #logq = -0.5*np.dot(a.transpose(),f) -np.sum(np.log(ps+np.log(np.exp(-ps) + np.exp(s-ps)))) - np.trace(np.log(L))
        logq = -0.5*np.dot(a.transpose(),f) -np.sum(np.log(ps+np.log(np.exp(-ps) + np.exp(s-ps)))) - sum(np.log(L.diagonal()))
        return (f,logq,a)


    def gradLogPosterior(self, theta,*args):
        data,targets = args
        theta = np.squeeze(theta)
        n = np.shape(targets)[0]
        K = self.kernel(data, data, theta, wantderiv=True)
        # K = kernel(data,data,theta,wantderiv=True)
        (f,logq,a) = self.NRiteration(data,targets,theta)
        s = np.where(f<0,f,0)
        W = np.diag(np.squeeze(np.exp(2*s - f) / ((np.exp(s) + np.exp(s-f))**2)))
        sqrtW = np.sqrt(W)
        L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW,np.dot(K[:,:,0],sqrtW)))
        
        R = np.dot(sqrtW,np.linalg.solve(L.transpose(),np.linalg.solve(L,sqrtW)))
        C = np.linalg.solve(L,np.dot(sqrtW,K[:,:,0]))
        p = np.exp(s)/(np.exp(s) + np.exp(s-f))
        hess = -np.exp(2*s - f) / (np.exp(s) + np.exp(s-f))**2
        s2 = -0.5*np.dot(np.diag(np.diag(K[:,:,0]) - np.diag(np.dot(C.transpose(),C))) , 2*hess*(0.5-p))
        
        targets = targets.values.reshape(n,1)
    
        gradZ = np.zeros(len(theta))
        for d in range(1,len(theta)+1):
            s1 = 0.5*(np.dot(a.transpose(),np.dot(K[:,:,d],a))) - 0.5*np.trace(np.dot(R,K[:,:,d]))	
            b = np.dot(K[:,:,d],(targets+1)*0.5-p)
            p = np.exp(s)/(np.exp(s) + np.exp(s-f))
            s3 = b - np.dot(K[:,:,0],np.dot(R,b))
            gradZ[d-1] = s1 + np.dot(s2.transpose(),s3)

        return -gradZ

    def predict(self, xstar,data,targets,theta):
        #vedi algoritmo 3.2 Rasmussen
        
        K = self.kernel(data,data,theta,wantderiv=False)
        n = np.shape(targets)[0]
        kstar = self.kernel(data,xstar,theta,wantderiv=False,measnoise=0)
        (f,logq,a) = self.NRiteration(data,targets,theta)
        targets = targets.values.reshape(n,1)
        s = np.where(f<0,f,0)
        #step 2
        W = np.diag(np.squeeze(np.exp(2*s - f) / ((np.exp(s) + np.exp(s-f))**2)))
        sqrtW = np.sqrt(W)
        L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW,np.dot(K,sqrtW)))
        p = np.exp(s)/(np.exp(s) + np.exp(s-f))
        fstar = np.dot(kstar.transpose(), (targets+1)*0.5 - p)
        v = np.linalg.solve(L,np.dot(sqrtW,kstar))	
        V = self.kernel(xstar,xstar,theta,wantderiv=False,measnoise=0)-np.dot(v.transpose(),v) 
        return (fstar,V)