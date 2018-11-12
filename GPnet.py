from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as so
import scipy.linalg as sl
import scipy.sparse as ss
import matplotlib.lines as mlines
import networkx as nx
import pandas as pd
import random
import time
from tqdm import tqdm

# %%
class iconshapes:
    # circles
    blue_circle = mlines.Line2D(
        [], [], color="blue", marker="o", linestyle="None", markersize=10
    )
    red_circle = mlines.Line2D(
        [], [], color="red", marker="o", linestyle="None", markersize=10
    )
    green_circle = mlines.Line2D(
        [], [], color="green", marker="o", linestyle="None", markersize=10
    )
    gray_circle = mlines.Line2D(
        [], [], color="gray", marker="o", linestyle="None", markersize=10
    )
    # triangles
    blue_triangle = mlines.Line2D(
        [], [], color="blue", marker="v", linestyle="None", markersize=10
    )
    red_triangle = mlines.Line2D(
        [], [], color="red", marker="v", linestyle="None", markersize=10
    )
    green_triangle = mlines.Line2D(
        [], [], color="green", marker="v", linestyle="None", markersize=10
    )
    # squares
    blue_square = mlines.Line2D(
        [], [], color="blue", marker="s", linestyle="None", markersize=10
    )
    red_square = mlines.Line2D(
        [], [], color="red", marker="s", linestyle="None", markersize=10
    )
    green_square = mlines.Line2D(
        [], [], color="green", marker="s", linestyle="None", markersize=10
    )


# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, self.lambdas)) + 1) / 2
# coefs = lstsq(A, b)[0]
# LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])
COEFS = np.array(
    [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
)[:, np.newaxis]


# %%
class GPnetBase:
    __metaclass_ = "GPnetBase"
    """ GPnetBase class cointains common attributes and methods for GPnetClassifier 
    and GPnetRegressor
    
    
    Attributes
    ----------
    
    Graph : network Graph
        NetworkX Graph on which regression/classification is made, if no graph
        is provided random regular graph is generated
    totnodes : int
        total number of nodes (for random graph generation)
    ntrain : int
        number of training nodes
    ntest : int
        number of test nodes
    deg : int
        connectivity degree (for random graph generation)
    seed : int
        seed for random number generation
    training_nodes: list
        list of nodes that are used for training
    test_nodes: list
        list of test nodes
    training_values: pandas Series (will be changed in future)
        training labels
    theta: list
        list of kernel parameters [a, b, c, d]
        a : constant term
        b : constant scale
        c : length scale
        d : noise term
        notice that kernel parameters are exponentiated, take np.log(theta) in
        advance
    optimize: bool
        if True activates the kernel parameter optimizer
    relabel_nodes: bool
        if True the nodes are relabelled to consecutive integers
    kerneltype: string
        "diffusion"
        "regularized_laplacian"
        "pstep_walk"
        
    Methods
    ----------
    calc_shortest_paths():
        calculates the shortest path matrix using Dijkstra's algorithm
    pivot_distance(pivot=0)
        returns pivot distance list respect to pivot
    random_assign_nodes():
        assigns nodes to training and test randomly, uses GPnet.seed
    kernel(nodes_a, nodes_b, theta, measnoise=1.0, wantderiv=True)
        calculates covariance matrix between nodes_a and nodes_b with
        theta parameters
    is_pos_def(test_mat):
        returns True if test_mat is positive definite
    logp()
        returns LogMarginalLikelihood
    plot_graph(filename=False):
        plots Graph with training/test/other labels
        if filename is defined saves plot as filename.png
    plot_prior():
        plots 5 extractions from prior process distribution
        if filename is defined saves plot as filename.png
    plot_post():
        plots 5 extractions from posterior process distribution
        if filename is defined saves plot as filename.png
    """

    def __init__(
        self,
        Graph,
        totnodes,
        ntrain,
        ntest,
        deg,
        seed,
        training_nodes,
        training_values,
        test_nodes,
        theta,
        optimize,
        relabel_nodes,
        kerneltype,
    ):
        self.N = ntrain
        self.n = ntest
        self.deg = deg
        self.seed = seed

        self.is_trained = False
        self.optimize = optimize

        self.theta = theta

        self.relabel_nodes = relabel_nodes
        self.kerneltype = kerneltype
        if totnodes == False:
            self.totnodes = self.N + self.n
        else:
            self.totnodes = totnodes

        if Graph == False:
            print("> Initializing Random Regular Graph")
            print(self.totnodes, "nodes")
            print("node degree", self.deg)
            G = nx.random_regular_graph(deg, totnodes)

        else:
            G = Graph
            self.totnodes = len(Graph.nodes)

        self.Graph = Graph

        self.orig_labels_dict = dict(zip(G, range(len(G.nodes))))
        self.orig_labels_invdict = dict(
            [[v, k] for k, v in self.orig_labels_dict.items()]
        )

        if relabel_nodes == True:
            print("> Relabeling nodes, orig. names stored in self.orig_labels_dict")
            self.Graph = nx.relabel_nodes(G, self.orig_labels_dict)

        self.training_nodes = training_nodes
        self.test_nodes = test_nodes
        # self.other_nodes = other_nodes

        if training_nodes == False or test_nodes == False:
            print("> Assigning Nodes Randomly ( seed =", self.seed, ")")
            print(self.N, " training nodes")
            print(self.n, " test nodes")
            print((self.totnodes - (self.N + self.n)), " idle nodes")

            self.random_assign_nodes()

        self.assign_other_nodes()
        self.calc_shortest_paths()

        # init plot stuff
        self.plot_pos = nx.kamada_kawai_layout(self.Graph)
        
        # shortest paths
        self.calc_pivot_distance()


        # END INIT #
        return

    def pivot_distance(self, pivot=0):
        pivot_distance = pd.Series(
            dict(nx.single_source_shortest_path_length(self.Graph, pivot))
        ).sort_index()
        pivot_distance.name = "pivot distance"
        return pivot_distance

    def calc_shortest_paths(self):
        # shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(G))
        shortest_paths_lengths = dict(nx.all_pairs_shortest_path_length(self.Graph))
        self.dist = pd.DataFrame(shortest_paths_lengths).sort_index(axis=1)
        return

    def random_assign_nodes(self):

        if self.N + self.n > self.totnodes:
            raise ValueError(
                "tot. nodes cannot be less than training nodes + test nodes"
            )
        # training_nodes = list(G.nodes)[0:N]
        random.seed(self.seed)
        self.training_nodes = random.sample(list(self.Graph.nodes), self.N)
        self.training_nodes.sort()

        # test_nodes = list(G.nodes)[N:N+n]
        self.test_nodes = random.sample(
            (set(self.Graph.nodes) - set(self.training_nodes)), self.n
        )
        self.test_nodes.sort()

        self.assign_other_nodes()
        return self

    def assign_other_nodes(self):

        self.other_nodes = (
            set(self.Graph.nodes) - set(self.training_nodes) - set(self.test_nodes)
        )
        self.other_nodes = list(self.other_nodes)
        self.other_nodes.sort()

        return self

    def optimize_params(self, gtol=1e-3, maxiter=200, disp=1):
        if self.optimize != False:
            print("> Optimizing parameters")
            print("method used: ", self.optimize["method"])
            print("bounds: ", self.optimize["bounds"])
            res = so.minimize(
                fun=self.logPosterior,
                x0=self.theta,
                args=(self.training_nodes, self.training_values),
                method=self.optimize["method"],
                bounds=self.optimize["bounds"],
                options={"disp": True},
            )
            self.theta = res["x"]
            print("new parameters: ", self.theta)
        return self

    def kernel(self, nodes_a, nodes_b, theta, measnoise=1.0, wantderiv=True):
        """
        Kernel Function
        ---------------
        
        k(nodes_a, nodes_b) = exp(a) + exp(b) * exp(-1/2 * (dist/exp(c))^2) + I*d
        
        with theta=[a,b,c,d]
        
        
        Parameters
        ----------
        
        nodes_a, nodes_b : list
            list of nodes between which the correlation matrix is calculated
        theta: 
            parameters, described aboce
        measnoise: 
            scale for measured noise ( just testing purposes )
        wantderiv:
            if True returns a k[len(nodes_a), len(nodes_b), len(theta) +1] ndarray
            k[:,:,0] is the covariance matrix
            K[:,:,j] are the the j-th partial derivatives respect to parameters
        """
        if not len(theta) == 1:
            theta = np.squeeze(theta)

        theta = np.exp(theta)
        # graph_distance_matrix = shortest_path_graph_distances(Graph)
        nodelist = list(self.Graph.nodes)
        nodeset = set(nodes_a).union(set(nodes_b))
        nodes_to_drop = [x for x in nodelist if x not in nodeset]
        cols_to_dropset = set(nodes_to_drop).union(set(nodes_b) - set(nodes_a))
        rows_to_dropset = set(nodes_to_drop).union(set(nodes_a) - set(nodes_b))

        cols_to_keepset = nodeset - cols_to_dropset
        rows_to_keepset = nodeset - rows_to_dropset

        if self.relabel_nodes == False:
            cols_to_drop = [self.orig_labels_dict[idx] for idx in cols_to_dropset]
            rows_to_drop = [self.orig_labels_dict[idx] for idx in rows_to_dropset]
            cols_to_keep = [self.orig_labels_dict[idx] for idx in cols_to_keepset]
            rows_to_keep = [self.orig_labels_dict[idx] for idx in rows_to_keepset]
        else:
            cols_to_drop = list(cols_to_dropset)
            rows_to_drop = list(rows_to_dropset)
            cols_to_keep = list(cols_to_keepset)
            rows_to_keep = list(rows_to_keepset)

        # need to keep track of node names somehow
        d1 = len(nodes_a)
        d2 = len(nodes_b)

        Lnorm = ss.csc_matrix(nx.normalized_laplacian_matrix(self.Graph))

        # maybe it's wrong
        #        Lnorm = Lnorm[:, cols_to_keep]
        #        Lnorm = Lnorm[rows_to_keep, :]
        # ok ofcourse it doesnt work
        kernel_list = ("diffusion", "regularized_laplacian", "pstep_walk")

        assert self.kerneltype in kernel_list, "kerneltype not implemented"
        if self.kerneltype == "diffusion":
            assert theta[0] < 1, "Lambda must be < 1" % theta[0]
            K = sl.expm(-theta[0] * Lnorm).toarray()
        elif self.kerneltype == "regularized_laplacian":
            K = sl.inv(np.eye(len(self.Graph.nodes())) + theta[0] * Lnorm).toarray()
        elif self.kerneltype == "pstep_walk":
            assert theta[0] >= 2, "a must be >=2" % theta[0]
            K = np.asarray(
                np.linalg.matrix_power(
                    theta[0] * np.eye(len(self.Graph.nodes())) - Lnorm, int(theta[1])
                )
            )

        # Lnorm2 = ss.csc_matrix(np.eye(len(self.Graph.nodes())) + theta[0]*nx.normalized_laplacian_matrix(self.Graph))
        # DIFFUSION PROCESS KERNEL

        # REGULARIZED LAPLACIAN KERNEL
        # K = sl.inv(np.eye(len(self.Graph.nodes())) + theta[0]*Lnorm)

        # P-STEP

        # K = sl.matrix_power(ss.csc_matrix(theta[0]*np.eye(len(self.Graph.nodes()))) - Lnorm, 3)
        # K = np.linalg.matrix_power(theta[0]*np.eye(len(nodelist)) - Lnorm.toarray(), 5)

        k = np.delete(K, cols_to_drop, axis=0)
        k = np.delete(k, rows_to_drop, axis=1)

        k = k + measnoise * theta[-1]

        return k

    @abstractmethod
    def logPosterior(self, theta, *args):
        raise NotImplementedError(
            "logPosterior() must be overridden by GPnetRegressor or GPnetClassifier"
        )

    def logp(self):
        return -self.logPosterior(self.theta, self.training_nodes, self.training_values)

    def plot_graph(self, filename=False):
        pl.figure(figsize=[15, 9])
        pl.title("Graph")
        # node positions
        # draw nodes
        nx.draw_networkx_nodes(
            self.Graph,
            self.plot_pos,
            with_labels=True,
            node_size=200,
            nodelist=self.training_nodes,
            node_color="r",
        )
        nx.draw_networkx_nodes(
            self.Graph,
            self.plot_pos,
            with_labels=True,
            node_size=200,
            nodelist=self.test_nodes,
            node_color="g",
        )
        nx.draw_networkx_nodes(
            self.Graph,
            self.plot_pos,
            with_labels=True,
            node_size=200,
            nodelist=self.other_nodes,
            node_color="b",
        )
        # draw edges
        ec = nx.draw_networkx_edges(self.Graph, self.plot_pos, alpha=0.2)
        # legend
        if self.relabel_nodes == True:
            labels = nx.draw_networkx_labels(
                self.Graph,
                labels=self.orig_labels_invdict,
                pos=self.plot_pos,
                font_color="k",
            )
        else:
            labels = nx.draw_networkx_labels(
                self.Graph, pos=self.plot_pos, font_color="k"
            )
        # legend
        training_patch = iconshapes.red_circle
        training_patch._label = "training nodes"
        test_patch = iconshapes.green_circle
        test_patch._label = "test nodes"
        other_patch = iconshapes.blue_circle
        other_patch._label = "other nodes"

        pl.legend(handles=[training_patch, test_patch, other_patch])

        if type(filename) is str:
            pl.savefig(filename, bbox_inches="tight")
        return self

    def plot_prior(self, filename=False):
        L2 = np.linalg.cholesky(self.kstarstar + 1e-6 * np.eye(self.n))
        # f_prior = mu L*N(0,1)
        f_prior = np.dot(L2, np.random.normal(size=(self.n, 5)))
        pl.figure()
        pl.clf()
        pl.plot(self.test_nodes, f_prior)
        pl.title("5 estrazioni dalla dist. a priori")
        pl.xlabel("nodes")
        pl.ylabel("values")
        # pl.axis([-5, 5, -3, 3])
        if type(filename) is str:
            pl.savefig(filename, bbox_inches="tight")

    def plot_post(self, filename=False):
        Lk = np.linalg.solve(self.L, self.kstar.T)
        L2 = np.linalg.cholesky(
            self.kstarstar + 1e-6 * np.eye(self.n) - np.dot(Lk.T, Lk)
        )

        # f_post = mu + L*N(0,1)
        f_post = self.fstar.reshape(-1, 1) + np.dot(
            L2, np.random.normal(size=(self.n, 5))
        )
        pl.figure()
        pl.clf()
        pl.plot(self.test_nodes, f_post)
        pl.title("5 estrazioni dalla dist. a posteriori")
        pl.xlabel("nodes")
        pl.ylabel("values")
        # pl.axis([-5, 5, -3, 3])
        if type(filename) is str:
            pl.savefig(filename, bbox_inches="tight")

    def plot_lml_landscape(self, plots, params, filename=False):
        pl.rcParams.update({"font.size": 5})
        plcols = 3
        #        if len(plots)%plcols != 0:
        #            plrows = len(plots)//plcols +1
        #        else:
        #            plrows = len(plots)//plcols
        plrows = len(plots) // plcols
        # print(plrows, " - ", plcols, "<")

        fig, ax = pl.subplots(plrows, plcols, dpi=300)
        fig.suptitle("LML landscapes", size=10)
        fig.subplots_adjust(wspace=0.3, hspace=0.4)
        for index, item in enumerate(plots):
            # print("Index: ", index)
            plot = plots[item]
            lml = self.lml_landscape(params, plot[0], plot[1], plot[2])
            idxmax = np.unravel_index(np.argmax(lml, axis=None), lml.shape)
            print(idxmax, lml[idxmax])
            idx1 = index // plcols
            idx2 = index % plcols
            if plrows == 0:
                idx = idx2
            else:
                idx = (idx1, idx2)

            # print(idx1, " - ", idx2)
            if plrows != 0:
                if len(plot) == 4:
                    cax = ax[idx].pcolor(plot[2], plot[1], lml)
                ax[idx].plot(
                    [plot[3][1]], [plot[3][0]], marker="o", markersize=5, color="red"
                )
                ax[idx].plot(
                    [plot[2][idxmax[0]]],
                    [plot[1][idxmax[1]]],
                    marker="o",
                    markersize=5,
                    color="blue",
                )
                ax[idx].set(
                    xlabel="theta" + str(plot[0][0]),
                    ylabel="theta" + str(plot[0][1]),
                    title=item,
                )
                # ax[idx1, idx2].set_title(item)
                fig.colorbar(cax, ax=ax[idx])
            else:
                if len(plot) == 4:
                    cax = pl.pcolor(plot[2], plot[1], lml)
                pl.plot(
                    [plot[3][1]], [plot[3][0]], marker="o", markersize=5, color="red"
                )
                pl.plot(
                    [plot[2][idxmax[0]]],
                    [plot[1][idxmax[1]]],
                    marker="x",
                    markersize=5,
                    color="blue",
                )
                pl.xlabel("theta" + str(plot[0][0]))
                pl.ylabel("theta" + str(plot[0][1]))
                pl.title(item)
                #                fig.set(
                #                        xlabel="theta" + str(plot[0][0]),
                #                        ylabel="theta" + str(plot[0][1]),
                #                        title=item,
                #                )
                pl.colorbar(cax)

    def lml_landscape(self, theta, axidx, ax1, ax2):
        print("> Calculating LML Landscape")
        lml = np.zeros([len(ax1), len(ax2)])
        for i in tqdm(range(len(ax1))):
            for j in range(len(ax2)):
                params = theta
                params[axidx[0]] = ax1[i]
                params[axidx[1]] = ax2[j]
                # print(axidx[0], axidx[1])

                lml[i, j] = -self.logPosterior(
                    params, self.training_nodes, self.training_values
                )
        return lml

    def set_training_values(self, training_values):
        self.training_values = training_values
        self.training_values.name = "training values"

    def calc_ktot(self):
        self.ktot = self.kernel(
            nodes_a=self.Graph.nodes,
            nodes_b=self.Graph.nodes,
            theta=self.theta,
            wantderiv=False,
        )
        
    def int_to_list(nodes):
        if type(nodes) == int:
            return [nodes]
        else:
            return nodes    
        
    def is_pos_def(self, test_mat):
        return np.all(np.linalg.eigvals(test_mat) > 0)
    
    
    def generate_df(self):
        fstar_series = pd.Series(index=self.test_nodes, data=self.fstar)
        s_series = pd.Series(index=self.test_nodes, data=self.s)
        
        try:
            self.predicted_probs
        except AttributeError:
            probs_series0 = pd.Series(index = self.test_nodes)
            probs_series1 = pd.Series(index = self.test_nodes)
            predicted_class_series = pd.Series(index = self.test_nodes)
        else:
            probs_series0 = pd.Series(index = self.test_nodes, data=self.predicted_probs.T[0])
            probs_series1 = pd.Series(index = self.test_nodes, data=self.predicted_probs.T[1])

        predicted_class_series = probs_series0.copy()
        predicted_class_series[predicted_class_series >0.5] = 1
        predicted_class_series[predicted_class_series <= 0.5] = -1            
        
        self.df = pd.DataFrame()
        self.df = self.df.assign(
            pvtdist=self.pvtdist,
            train_vals=self.training_values,
            fstar=fstar_series,
            variance_s=s_series,
            prob_0 = probs_series0,
            prob_1 = probs_series1,
            predicted_class = -predicted_class_series
        )
        
        return self
    
    def calc_pivot_distance(self):
        self.pvtdist = self.pivot_distance(list(self.Graph.nodes)[0])
        return self
        
