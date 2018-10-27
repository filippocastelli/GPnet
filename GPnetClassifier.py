from GPnet import GPnetBase, LAMBDAS, COEFS, iconshapes
import numpy as np
import networkx as nx
import matplotlib.pylab as pl
from scipy.special import erf



class GPnetClassifier(GPnetBase):
    """
    Class for Classifiers
    
    
    Methods
    ---------
    predict():
        calculates predictions using training labels
    NRiteration(data, targets, theta, tol=0.1, phif=1e100, scale=1.):
        finds maximum f_star mode for Laplace Approximation
    logPosterior(theta, data, labels):
        returns -log marginal likelihood
    gradlogposterior(theta, data, labels):
        returns - gradient(logposterior)
    plot_latent(filename=False):
        plots latent Gaussian Process in 2d fashion, with node number on x
        if filename is specified saves plot to 'filename.png'
    plot_predict_graph(filename=False):
        plots graph, node's color is proportional to process prediction
        if filename is specified saves plot to 'filename.png'
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
    ):

        super(GPnetClassifier, self).__init__(
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
        )

        self.pivot_flag = False

        if training_values == False:
            print("no training labels where specified")
            print(
                "> Setting labels to (np.sin(0.6 * self.pvtdist) > 0).replace({True: 1, False: -1})"
            )
            self.pivot_flag = True
            self.pvtdist = self.pivot_distance(0)
            self.t = self.pvtdist[self.training_nodes]
            self.binary_labels = (np.sin(0.6 * self.pvtdist) > 0).replace(
                {True: 1, False: -1}
            )
            self.training_labels = self.binary_labels[self.training_nodes]

        else:
            self.training_labels = training_values

    def logPosterior(self, theta, *args):
        data, targets = args
        (f, logq, a) = self.NRiteration(data, targets, theta)
        return -logq

    def NRiteration(self, data, targets, theta, tol=0.1, phif=1e100, scale=1.0):
        # print("iteration")
        # pag 46 RASMUSSEN-WILLIAMS
        K = self.kernel(data, data, theta, wantderiv=False)
        # K = kernel(data,data,theta,wantderiv=False)
        n = np.shape(targets)[0]
        f = np.zeros((n, 1))
        #        tol = 0.1
        #        phif = 1e100
        #        scale = 1.
        count = 0
        targets = targets.values.reshape(n, 1)
        while True:

            count += 1
            # s = np.where(f < 0, f, 0)
            W = np.diag(np.squeeze(np.exp(-f) / (1 + np.exp(-f)) ** 2))

            sqrtW = np.sqrt(W)
            # L = cholesky(B)
            L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW, np.dot(K, sqrtW)))
            p = 1 / (1 + np.exp(-f))
            b = np.dot(W, f) + 0.5 * (targets + 1) - p
            a = scale * (
                b
                - np.dot(
                    sqrtW,
                    np.linalg.solve(
                        L.transpose(), np.linalg.solve(L, np.dot(sqrtW, np.dot(K, b)))
                    ),
                )
            )
            f = np.dot(K, a)
            oldphif = phif
            phif = (
                np.log(p)
                - 0.5 * np.dot(f.transpose(), np.dot(np.linalg.inv(K), f))
                - 0.5 * np.sum(np.log(np.diag(L)))
                - np.shape(data)[0] / 2.0 * np.log(2 * np.pi)
            )
            # print(phif)
            # print("loop",np.sum((oldphif-phif)**2))
            if np.sum((oldphif - phif) ** 2) < tol:
                break
            elif count > 100:
                count = 0
                scale = scale / 2.0

        s = -targets * f
        # ps = np.where(s > 0, s, 0)
        # logq = -0.5*np.dot(a.transpose(),f) -np.sum(np.log(ps+np.log(np.exp(-ps) + np.exp(s-ps)))) - np.trace(np.log(L))
        logq = (
            -0.5 * np.dot(a.transpose(), f)
            - np.sum(np.log(1 + np.log(1 + np.exp(-s))))
            - sum(np.log(L.diagonal()))
        )

        return (f, logq, a)

    def gradLogPosterior(self, theta, *args):
        data, targets = args
        theta = np.squeeze(theta)
        n = np.shape(targets)[0]
        K = self.kernel(data, data, theta, wantderiv=True)
        # K = kernel(data,data,theta,wantderiv=True)
        (f, logq, a) = self.NRiteration(data, targets, theta)
        s = np.where(f < 0, f, 0)
        W = np.diag(np.squeeze(np.exp(2 * s - f) / ((np.exp(s) + np.exp(s - f)) ** 2)))
        sqrtW = np.sqrt(W)
        L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW, np.dot(K[:, :, 0], sqrtW)))

        R = np.dot(sqrtW, np.linalg.solve(L.transpose(), np.linalg.solve(L, sqrtW)))
        C = np.linalg.solve(L, np.dot(sqrtW, K[:, :, 0]))
        p = np.exp(s) / (np.exp(s) + np.exp(s - f))
        hess = -np.exp(2 * s - f) / (np.exp(s) + np.exp(s - f)) ** 2
        s2 = -0.5 * np.dot(
            np.diag(np.diag(K[:, :, 0]) - np.diag(np.dot(C.transpose(), C))),
            2 * hess * (0.5 - p),
        )

        targets = targets.values.reshape(n, 1)

        gradZ = np.zeros(len(theta))
        for d in range(1, len(theta) + 1):
            s1 = 0.5 * (np.dot(a.transpose(), np.dot(K[:, :, d], a))) - 0.5 * np.trace(
                np.dot(R, K[:, :, d])
            )
            b = np.dot(K[:, :, d], (targets + 1) * 0.5 - p)
            p = np.exp(s) / (np.exp(s) + np.exp(s - f))
            s3 = b - np.dot(K[:, :, 0], np.dot(R, b))
            gradZ[d - 1] = s1 + np.dot(s2.transpose(), s3)

        return -gradZ

    def predict(self):
        # vedi algoritmo 3.2 Rasmussen
        if self.optimize != False:
            self.optimize_params()

        K = self.kernel(
            self.training_nodes, self.training_nodes, self.theta, wantderiv=False
        )
        n = np.shape(self.training_labels)[0]
        kstar = self.kernel(
            self.training_nodes,
            self.test_nodes,
            self.theta,
            wantderiv=False,
            measnoise=0,
        )
        (f, logq, a) = self.NRiteration(
            self.training_nodes, self.training_labels, self.theta
        )
        targets = self.training_labels.values.reshape(n, 1)
        s = np.where(f < 0, f, 0)
        # step 2
        W = np.diag(np.squeeze(np.exp(2 * s - f) / ((np.exp(s) + np.exp(s - f)) ** 2)))
        sqrtW = np.sqrt(W)
        L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW, np.dot(K, sqrtW)))
        p = np.exp(s) / (np.exp(s) + np.exp(s - f))
        self.fstar = np.squeeze(np.dot(kstar.transpose(), (targets + 1) * 0.5 - p))
        v = np.linalg.solve(L, np.dot(sqrtW, kstar))
        kstarstar = self.kernel(
            self.test_nodes, self.test_nodes, self.theta, wantderiv=False, measnoise=0
        ).diagonal()
        module_v = np.dot(v.transpose(), v)

        self.V = (kstarstar - module_v).diagonal()

        # V = self.kernel(self.test_nodes,self.test_nodes,self.theta,wantderiv=False,measnoise=0).diagonal()-np.dot(v.transpose(),v)

        alpha = np.tile((1 / (2 * self.V)), (5, 1))
        # gamma = LAMBDAS * fstar
        gamma = np.einsum("i,k->ik", LAMBDAS, self.fstar.T)
        lambdas_mat = np.tile(LAMBDAS, (len(self.test_nodes), 1)).T
        Vmat = np.tile(self.V, (5, 1))
        integrals = (
            np.sqrt(np.pi / alpha)
            * erf(gamma * np.sqrt(alpha / (alpha + lambdas_mat ** 2)))
            / (2 * np.sqrt(Vmat * 2 * np.pi))
        )
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()

        self.predicted_probs = np.vstack((1 - pi_star, pi_star)).T
        self.s = np.sqrt(self.V)

        print("succesfully trained model")
        self.is_trained = True

        return (self.fstar.T, self.V, self.predicted_probs)
        # return (fstar,V)

    def plot_latent(self, filename=False):
        pl.figure()
        pl.clf()
        pl.plot(self.training_nodes, self.training_labels, "r+", ms=20)

        pl.gca().fill_between(
            self.test_nodes, self.fstar - self.s, self.fstar + self.s, color="#dddddd"
        )
        pl.plot(self.test_nodes, self.fstar, "ro", ms=4)
        pl.plot(self.test_nodes, self.fstar, "r--", lw=2)

        loglikelihood = -self.logPosterior(self.theta, self.training_nodes, self.t)
        pl.title(
            "Latent Process Mean and Variance \n(length scale: %.3f , constant scale: %.3f , noise variance: %.3f )\n Log-Likelihood: %.3f"
            % (self.theta[1], self.theta[0], self.theta[2], loglikelihood)
        )
        pl.xlabel("nodes")
        pl.ylabel("values")
        if type(filename) is str:
            pl.savefig(filename, bbox_inches="tight")
        # pl.axis([-5, 5, -3, 3])
        return self

    def plot_predict_graph(self, filename=False):

        if self.is_trained == False:
            print("need to train a model first, use GPnetClassifier.predict()")
            return

        pl.figure(figsize=[15, 9])
        pl.title("Prediction plot")

        self.gen_cmap()
        nx.draw_networkx_nodes(
            self.Graph,
            self.plot_pos,
            nodelist=self.training_nodes,
            node_color=np.where(
                self.training_labels[(self.training_nodes)] > 0,
                self.training_labels[(self.training_nodes)],
                0,
            ),
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
            node_color=self.predicted_probs.T[0],
            with_labels=True,
            node_size=200,
            cmap=self.cmap,
            node_shape="s",
        )

        ec = nx.draw_networkx_edges(self.Graph, self.plot_pos, alpha=0.2)

        sm = pl.cm.ScalarMappable(cmap=self.cmap, norm=pl.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = pl.colorbar(sm)

        labels = nx.draw_networkx_labels(self.Graph, pos=self.plot_pos, font_color="k")

        # legend
        training_patch = iconshapes.red_triangle
        training_patch._label = "training nodes"
        test_patch = iconshapes.blue_square
        test_patch._label = "test nodes"
        other_patch = iconshapes.gray_circle
        other_patch._label = "other nodes"

        pl.legend(handles=[training_patch, test_patch, other_patch])

        if type(filename) is str:
            pl.savefig(filename, bbox_inches="tight")
        return self

    def gen_cmap(self):
        self.vmin = 0
        self.vmax = 1
        self.cmap = pl.cm.seismic
