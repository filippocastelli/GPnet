from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
#%%
f = lambda x: np.cos(.7*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()
#f = lambda x: np.tan(0.9*x).flatten()
#%%
def kernel(a, b, params):
    """ squared exponential """
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return params[0]*np.exp(-.5 * (1/params[1]) * sqdist)
#%%
N = 10         # numero punti training
n = 500       # numero punti test
s = 0.05    # noise variance
#%%
rng = np.random.RandomState(2)
X = rng.uniform(-5, 5, size=(N,1))
y = f(X) + s*np.random.randn(N)
#%%
lengthscale = 1
constantscale = 1

kernel_parameters = [constantscale, lengthscale]
kernel_noise_variance = 0.001
K = kernel(X, X, kernel_parameters)
# Ky = L L^T

#%%
L = np.linalg.cholesky(K + 0.005*np.eye(N))
#%%
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# mu = K*^T Ky^-1 y = K*T alpha
# alpha = Ky^-1 y = L^-T L^-1 y

# Lk = L^-1 K*
Lk = np.linalg.solve(L, kernel(X, Xtest, kernel_parameters))
# Lk^T = K*^T L^-T
# np.linalg.solve(L,y) = L^-1 y
# K*^T L^-T L^-1 y = K*^T Ky^-1 y
L_inv_y = np.linalg.solve(L, y)
alpha = np.dot(np.linalg.inv(L).T, L_inv_y)

mu = np.dot(Lk.T, L_inv_y )

loglikelihood = -0.5*np.dot(y.T, alpha) - (np.log(L.diagonal())).sum() - (N/2)*np.log(2*np.pi)

loglikelihood
#%%
#K**
K_ = kernel(Xtest, Xtest,kernel_parameters)   

#s2 = K** - K*^T Ky^-1 K*
# Lk^2 = Lk^T Lk = (L^-1 K*)^T (L^-1 K*) = K*^T L^-T L^-1 K* = K*^T K_y^-1 K*
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)
#%%
# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.title('Valore medio e margini di confidenza')

pl.title('Valore medio e margini a posteriori, (length scale: %.3f , constant scale: %.3f ,\
noise variance: %.3f )\n Log-Likelihood: %.3f'
         % (kernel_parameters[1], kernel_parameters[0], kernel_noise_variance, loglikelihood))

pl.savefig('predict.png', bbox_inches='tight')
pl.axis([-5, 5, -3, 3]);

#%%
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
#f_prior = mu L*N(0,1)
f_prior = np.dot(L, np.random.normal(size=(n,10)))
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('10 estrazioni dalla dist. a priori')
pl.axis([-5, 5, -3, 3])
pl.savefig('prior.png', bbox_inches='tight')
#%%
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
#f_post = mu + L*N(0,1)
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('10 estrazioni dalla dist. a posteriori')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')