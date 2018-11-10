from GPnet import GPnetBase, iconshapes
import numpy as np
import networkx as nx
import matplotlib.pylab as pl
import pandas as pd

class GPnetRegressor(GPnetBase):
    """
    Class for Regressors
    
    
    Methods
    ---------
    
    predict():
        calculates predictions using training labels
    predict_RW():
        same thing, just implemented differently (to be removed)
    logPosterior(theta, data, labels):
        returns -log marginal likelihood
    gradlogposterior(theta, data, labels):
        returns - gradient(logposterior)
    optimize_params():
        optimizer
    plot_predict_2d(filename=False):
        plots post Gaussian Process in 2d fashion, with node number on x
        if filename is specified saves plot to 'filename.png'
    plot_predict_graph(filename=False):
        plots graph, node's color is proportional to process prediction
        if filename is specified saves plot to 'filename.png'
    set_training_values(training_values):
        set training values to training_values
        
    """

    def __init__(
        self,
        Graph=False,
        totnodes=False,
        ntrain=100,
        ntest=90,
        deg=4,
        seed=0,
        training_nodes=False,
        training_values=False,
        test_nodes=False,
        theta=[0.1, 0.1, 0.1],
        optimize=False,
        relabel_nodes=False,
        kerneltype = "diffusion",
    ):

        super(GPnetRegressor, self).__init__(
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
        )
        self.pivot_flag = False
        if training_values == False:
            self.pivot_flag = True
            self.pvtdist = self.pivot_distance(list(self.Graph.nodes)[0])
            self.training_values = self.pvtdist[self.training_nodes]
        else:
            self.training_values = training_values

        return
    


    def predict(self):
        # predicts the same exact results as GPnetRegressor.predict(), just reimplemented using Algorithm 2.1 in Rasmussen to make sure it was not the problem
        self.optimize_params()

        self.k_not_posdef_flag = False
        self.kstar_not_posdef_flag = False

        self.t_mean = np.mean(self.training_values)
        self.t_shifted = self.training_values - self.t_mean

        self.k = self.kernel(
            nodes_a=self.training_nodes,
            nodes_b=self.training_nodes,
            theta=self.theta,
            wantderiv=False,
        )

        self.kstar = self.kernel(
            nodes_a=self.test_nodes,
            nodes_b=self.training_nodes,
            theta=self.theta,
            wantderiv=False,
            measnoise=False,
        )
        self.kstarstar = self.kernel(
            nodes_a=self.test_nodes,
            nodes_b=self.test_nodes,
            theta=self.theta,
            wantderiv=False,
        )

        self.kstarstar_diag = np.diag(self.kstarstar)

        if not self.is_pos_def(self.k):
            self.k_not_posdef_flag = True
            # raise ValueError("K is not positive definite")
            print("K not positive definite, aborting...")
            return self
        if not self.is_pos_def(self.kstarstar):
            self.kstar_not_posdef_flag = True
            # raise ValueError("K** is not positive definite")
            print("K** not positive definite, aborting...")
            return self

        self.L = np.linalg.cholesky(self.k)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.t_shifted))
        self.fstar = np.dot(self.kstar, self.alpha) + self.t_mean
        self.v = np.linalg.solve(self.L, self.kstar.T)
        self.V = self.kstarstar_diag - np.dot(self.v.T, self.v)
        self.s = np.sqrt(np.diag(self.V))
        
        print("succesfully trained model")
        self.is_trained = True
        self.generate_df()
        
        return self

    def logPosterior(self, theta, *args):
        data, t = args

        K = self.kernel(data, data, theta, wantderiv=False)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -np.inf
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, t))
        logp = (
            -0.5 * np.dot(t.T, alpha)
            - np.sum(np.log(np.diag(L)))
            - K.shape[0] * 0.5 * np.log(2 * np.pi)
        )
        return -logp

    def gradLogPosterior(self, theta, *args):
        data, t = args
        theta = np.squeeze(theta)
        k = self.kernel(data, data, theta, wantderiv=True)
        try:
            L = np.linalg.cholesky(k[:, :, 0])  # Line 2
            K_inv = np.dot(np.linalg.inv(L).T, np.linalg.inv(L))
        except np.linalg.LinAlgError:
            return -np.inf
        # return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf

        # L = np.linalg.cholesky(k)
        alpha = np.linalg.solve(L, t)

        tmp = np.eye(k.shape[0]) * np.dot(alpha, alpha.T)
        # tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
        # tmp2 = np.linalg.solve(L, np.eye(k.shape[0]))[:, :, np.newaxis]
        tmp -= K_inv
        # Compute "0.5 * trace(tmp.dot(K_gradient))" without
        # constructing the full matrix tmp.dot(K_gradient) since only
        # its diagonal is required
        log_likelihood_gradient_dim = np.zeros([len(data), len(data), len(theta)])
        for i in range(0, len(theta)):
            log_likelihood_gradient_dim[:, :, i] = 0.5 * np.dot(tmp, k[:, :, i + 1])
            log_likelihood_gradient = np.trace(log_likelihood_gradient_dim, axis1=0)

        # log_likelihood_gradient_dims = 0.5 * np.einsum("ij,ijk->ijk", tmp, k[:, :, 1:])
        # log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
        print(log_likelihood_gradient)
        return -log_likelihood_gradient

    def generate_df(self):
        fstar_series = pd.Series(index = self.test_nodes, data=self.fstar)
        s_series = pd.Series(index = self.test_nodes, data = self.s)
        self.df = pd.DataFrame()
        self.df = self.df.assign(pvtdist = self.pvtdist, train_vals = self.training_values, fstar = fstar_series, variance_s = s_series)
        return self
    
    def gen_cmap(self):
        self.vmin = min(self.training_values.min(), self.fstar.min())
        self.vmax = max(self.training_values.max(), self.fstar.max())
        self.cmap = pl.cm.inferno_r
        
    def plot_predict_2d_old(self, filename=False):
        pl.figure(figsize=[15, 9])
        pl.clf()
        #pl.plot(self.training_nodes, self.training_values, "r+", ms=20)
#        pl.gca().fill_between(
#            self.test_nodes, self.fstar - self.s, self.fstar + self.s, color="#dddddd"
#        )
#        pl.plot(self.test_nodes, self.fstar, "ro", ms=4)
#        pl.plot(self.test_nodes, self.fstar, "r--", lw=2)
        errorbar_df = self.df.iloc[list(self.test_nodes)]
        pl.errorbar(errorbar_df.index,
                    errorbar_df["fstar"],
                    errorbar_df["variance_s"],
                    barsabove = True,
                    ecolor = "black",
                    linewidth = 1,
                    capsize = 5,
                    fmt = 'o')
        plot_df = self.df.iloc[list(self.training_nodes)]
        pl.plot(plot_df.index,
                plot_df["train_vals"].values,"r+",ms=20)
#        pl.errorbar(self.test_nodes, self.fstar,self.s,
#                    barsabove = True,
#                    ecolor = "black",
#                    linewidth = 1,
#                    capsize = 5,
#                    fmt = 'o')
        if self.pivot_flag == True:
            pvt_dist_df = self.df
            pl.plot(pvt_dist_df.index,
                    pvt_dist_df["pvtdist"].values)
        pl.title("Gaussian Process Mean and Variance")
        #loglikelihood = -self.logPosterior(self.theta, self.training_nodes, self.training_values)
        #        pl.title(
        #            "Valore medio e margini a posteriori\n(length scale: %.3f , constant scale: %.3f , noise variance: %.3f )\n Log-Likelihood: %.3f"
        #            % (self.theta[1], self.theta[0], self.theta[2], loglikelihood)
        #        )
#        pl.title(
#            "Valore medio e margini a posteriori\n(lambda: %.3f)" % (self.theta[0])
#        )
        pl.xlabel("nodes")
        pl.ylabel("values")
        if type(filename) is str:
            pl.savefig(filename, bbox_inches="tight")
        # pl.axis([-5, 5, -3, 3])
        return self

    def plot_predict_2d(self, filename=False):
        pl.figure(figsize=[15, 9])
        pl.clf()
        #pl.plot(self.training_nodes, self.training_values, "r+", ms=20)
#        pl.gca().fill_between(
#            self.test_nodes, self.fstar - self.s, self.fstar + self.s, color="#dddddd"
#        )
#        pl.plot(self.test_nodes, self.fstar, "ro", ms=4)
#        pl.plot(self.test_nodes, self.fstar, "r--", lw=2)
        errorbar_df = self.df.iloc[list(self.test_nodes)].sort_values(by =["pvtdist"])
        pl.errorbar(errorbar_df["pvtdist"].values,
                    errorbar_df["fstar"].values,
                    errorbar_df["variance_s"].values,
                    barsabove = True,
                    ecolor = "black",
                    linewidth = 1,
                    capsize = 5,
                    fmt = 'o')
        plot_df = self.df.iloc[list(self.training_nodes)].sort_values(by =["pvtdist"])
        pl.plot(plot_df["pvtdist"].values,
                plot_df["train_vals"].values,"r+",ms=20)
#        pl.errorbar(self.test_nodes, self.fstar,self.s,
#                    barsabove = True,
#                    ecolor = "black",
#                    linewidth = 1,
#                    capsize = 5,
#                    fmt = 'o')
        if self.pivot_flag == True:
            pvt_dist_df = self.df.sort_values(by = ["pvtdist"])
            pl.plot(pvt_dist_df["pvtdist"].values,
                    pvt_dist_df["pvtdist"].values)
        pl.title("Gaussian Process Mean and Variance")
        #loglikelihood = -self.logPosterior(self.theta, self.training_nodes, self.training_values)
        #        pl.title(
        #            "Valore medio e margini a posteriori\n(length scale: %.3f , constant scale: %.3f , noise variance: %.3f )\n Log-Likelihood: %.3f"
        #            % (self.theta[1], self.theta[0], self.theta[2], loglikelihood)
        #        )
#        pl.title(
#            "Valore medio e margini a posteriori\n(lambda: %.3f)" % (self.theta[0])
#        )
        pl.xlabel("nodes")
        pl.ylabel("values")
        if type(filename) is str:
            pl.savefig(filename, bbox_inches="tight")
        # pl.axis([-5, 5, -3, 3])
        return self
    
    def plot_predict_graph(self, filename=False):

        if self.is_trained == False:
            print("need to train a model first, use GPnetRegressor.predict()")
            return

        pl.figure(figsize=[15, 9])
        pl.title("Prediction plot")

        self.gen_cmap()
        nx.draw_networkx_nodes(
            self.Graph,
            self.plot_pos,
            nodelist=self.training_nodes,
            node_color=np.squeeze(self.training_values[(self.training_nodes)]),
            with_labels=True,
            node_size=200,
            cmap=self.cmap,
            node_shape="v",
        )

        nx.draw_networkx_nodes(
            self.Graph,
            self.plot_pos,
            nodelist=self.other_nodes,
            node_color="gray",
            with_labels=True,
            node_size=200,
            cmap=self.cmap,
        )

        nx.draw_networkx_nodes(
            self.Graph,
            self.plot_pos,
            nodelist=self.test_nodes,
            node_color=self.fstar,
            with_labels=True,
            node_size=200,
            cmap=self.cmap,
            node_shape="s",
        )

        ec = nx.draw_networkx_edges(self.Graph, self.plot_pos, alpha=0.2)

        sm = pl.cm.ScalarMappable(
            cmap=self.cmap, norm=pl.Normalize(vmin=self.vmin, vmax=self.vmax)
        )
        sm.set_array([])
        cbar = pl.colorbar(sm)

        labels = nx.draw_networkx_labels(self.Graph, pos=self.plot_pos, font_color="k")

        # legend
        training_patch = iconshapes.red_triangle
        training_patch._label = "training nodes"
        test_patch = iconshapes.green_square
        test_patch._label = "test nodes"
        other_patch = iconshapes.gray_circle
        other_patch._label = "other nodes"

        pl.legend(handles=[training_patch, test_patch, other_patch])

        if type(filename) is str:
            pl.savefig(filename, bbox_inches="tight")
        return self