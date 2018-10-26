import numpy as np
import matplotlib.pyplot as pl
from matplotlib import cm
import scipy.optimize as so

#%%

N = 10         # numero punti training
n = 500       # numeroy punti test
s = 0.05    # noise variance
#%%
rng = np.random.RandomState(2)
x = rng.uniform(-5, 5, size=(N,1))
xstar = np.linspace(-5, 5, n).reshape(-1,1)

f = lambda x: np.cos(.7*x).flatten()
t = f(x) + s*np.random.randn(N)
t = t.reshape(N,1)

#%%

def kernel(data1,data2,theta,wantderiv=True,measnoise=1.):
    theta = np.squeeze(theta)
    theta = np.exp(theta)
    # Squared exponential
    if np.ndim(data1) == 1:
        d1 = np.shape(data1)[0]
        n = 1
        data1 = data1*np.ones((d1,1))
        data2 = data2*np.ones((np.shape(data2)[0],1))
    else:
        (d1,n) = np.shape(data1)

    d2 = np.shape(data2)[0]
    sumxy = np.zeros((d1,d2))
    for d in range(n):
        D1 = np.transpose([data1[:,d]]) * np.ones((d1,d2))
        D2 = [data2[:,d]] * np.ones((d1,d2))
        sumxy += (D1-D2)**2*theta[d+1]

    k = theta[0] * np.exp(-0.5*sumxy)
    if wantderiv:
        K = np.zeros((d1,d2,len(theta)+1))
        # K[:,:,0] is the original covariance matrix
        K[:,:,0] = k + measnoise*theta[2]*np.eye(d1,d2)
        K[:,:,1] = k
        K[:,:,2] = -0.5*k*sumxy
        K[:,:,3] = theta[2]*np.eye(d1,d2)
        return K
    else:
        return k + measnoise*theta[2]*np.eye(d1,d2)
    
    
#%%
        
def logPosterior(theta,*args):
    data,t = args
    k = kernel(data,data,theta,wantderiv=False)
    L = np.linalg.cholesky(k)
    beta = np.linalg.solve(L.transpose(), np.linalg.solve(L,t))
    logp = -0.5*np.dot(t.transpose(),beta) - np.sum(np.log(np.diag(L))) - np.shape(data)[0] /2. * np.log(2*np.pi)
    return -logp

#%%
    

def gradLogPosterior(theta,*args):
    #print(args)
    data,t = args
    theta = np.squeeze(theta)
    d = len(theta)
    K = kernel(data,data,theta,wantderiv=True)
    L = np.linalg.cholesky(np.squeeze(K[:,:,0]))
    invk = np.linalg.solve(L.transpose(),np.linalg.solve(L,np.eye(np.shape(data)[0])))
    dlogpdtheta = np.zeros(d)
    for d in range(1,len(theta)+1):
        dlogpdtheta[d-1] = 0.5*np.dot(t.transpose(), np.dot(invk, np.dot(np.squeeze(K[:,:,d]), np.dot(invk,t)))) - 0.5*np.trace(np.dot(invk,np.squeeze(K[:,:,d])))

    return -dlogpdtheta


#%%
    
lengthscale = 1
constantscale = 1
noise_scale = 1
theta = np.array([constantscale,lengthscale,  noise_scale])

theta = so.fmin_cg(logPosterior, theta, fprime=gradLogPosterior, args=(x,t), gtol=1e-4,maxiter=200,disp=1)




#%%

theta0 = np.linspace(-3, 3, 100)
theta1 = np.linspace(-3, 3, 100)

Theta0, Theta1 = np.meshgrid(theta0, theta1)
LML = [[logPosterior([Theta0[i, j], Theta1[i, j], theta[2]], x, t ) for i in range(Theta0.shape[0])] for j in range(Theta1.shape[1])]
LML = np.squeeze(np.array(LML).T)

fig, ax = pl.subplots()
cax = ax.pcolor(theta0, theta1, -LML, cmap=cm.viridis)
vmin, vmax = (-LML).min(), (-LML).max()
fig.colorbar(cax, ax=ax)
pl.plot(theta[0], theta[1], 'ro')
pl.xlabel("const_scale")
pl.ylabel("length_scale")
pl.title("Landscape della LogMarginalLikelihood")

#%%
k = kernel(x,x,theta,wantderiv=False)
kstar = kernel(xstar,x, theta, wantderiv=False, measnoise=False)
kstarstar = kernel(xstar,xstar,theta,wantderiv=False)
kstarstar_diag = np.diag(kstarstar)

#inversione K
L = np.linalg.cholesky(k)
invk = np.linalg.solve(L.transpose(),np.linalg.solve(L,np.eye(np.shape(x)[0])))

#previsione
mean = np.squeeze(np.dot(kstar,np.dot(invk,t)))
var = kstarstar_diag - np.diag(np.dot(kstar,np.dot(invk,kstar.T)))
var = np.squeeze(np.reshape(var,(n,1)))
s = np.sqrt(var)

#%%

pl.plot(x, t, 'r+', ms=20)
pl.plot(xstar, f(xstar), 'b-')
pl.gca().fill_between(xstar.flat, mean-s, mean+s, color="#dddddd")
pl.plot(xstar, mean, 'r--', lw=2)
pl.title('Valore medio e margini di confidenza')
loglikelihood = logPosterior(theta, x,t)
pl.xlabel('x')
pl.ylabel('y')
title = pl.title('Valore medio e margini a posteriori\n(length scale: %.3f , constant scale: %.3f ,\
#noise variance: %.3f )\n Log-Likelihood: %.3f'
        % (theta[1], theta[0], theta[2], loglikelihood))
    
#%%
L2 = np.linalg.cholesky(kstarstar + 1e-6*np.eye(n))
#f_prior = mu L*N(0,1)
f_prior = np.dot(L2, np.random.normal(size=(n,10)))
pl.plot(xstar, f_prior)
pl.xlabel('x')
pl.ylabel('y')
title = pl.title('10 estrazioni dalla dist. a priori')

#%%

Lk = np.linalg.solve(L, kstar.T)
L2 = np.linalg.cholesky(kstarstar+ 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
#f_post = mu + L*N(0,1)
f_post = mean.reshape(-1,1) + np.dot(L2, np.random.normal(size=(n,10)))
pl.plot(xstar, f_post)
pl.title('10 estrazioni dalla dist. a posteriori')
pl.xlabel('x')
ylab = pl.ylabel('y')


#%%

def logPosterior(theta,*args):
    data,targets = args
    (f,logq,a) = NRiteration(data,targets,theta)
    return -logq

def NRiteration(data,targets,theta):
    #pag 46 RASMUSSEN-WILLIAMS
    K = kernel(data,data,theta,wantderiv=False)
    n = np.shape(targets)[0]
    f = np.zeros((n,1))
    tol = 0.1
    phif = 1e100
    scale = 1.
    count = 0
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


#%%
    
def gradLogPosterior(theta,*args):
    data,targets = args
    theta = np.squeeze(theta)
    n = np.shape(targets)[0]
    K = kernel(data,data,theta,wantderiv=True)
    (f,logq,a) = NRiteration(data,targets,theta)
    s = np.where(f<0,f,0)
    W = np.diag(np.squeeze(np.exp(2*s - f) / ((np.exp(s) + np.exp(s-f))**2)))
    sqrtW = np.sqrt(W)
    L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW,np.dot(K[:,:,0],sqrtW)))
    
    R = np.dot(sqrtW,np.linalg.solve(L.transpose(),np.linalg.solve(L,sqrtW)))
    C = np.linalg.solve(L,np.dot(sqrtW,K[:,:,0]))
    p = np.exp(s)/(np.exp(s) + np.exp(s-f))
    hess = -np.exp(2*s - f) / (np.exp(s) + np.exp(s-f))**2
    s2 = -0.5*np.dot(np.diag(np.diag(K[:,:,0]) - np.diag(np.dot(C.transpose(),C))) , 2*hess*(0.5-p))

    gradZ = np.zeros(len(theta))
    for d in range(1,len(theta)+1):
        s1 = 0.5*(np.dot(a.transpose(),np.dot(K[:,:,d],a))) - 0.5*np.trace(np.dot(R,K[:,:,d]))	
        b = np.dot(K[:,:,d],(targets+1)*0.5-p)
        p = np.exp(s)/(np.exp(s) + np.exp(s-f))
        s3 = b - np.dot(K[:,:,0],np.dot(R,b))
        gradZ[d-1] = s1 + np.dot(s2.transpose(),s3)

    return -gradZ

#%%
    
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])
COEFS = np.array(
    [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
)[:, np.newaxis]

#%%

def predict(xstar,data,targets,theta):
    K = kernel(data,data,theta,wantderiv=False)
    n = np.shape(targets)[0]
    kstar = kernel(data,xstar,theta,wantderiv=False,measnoise=0)
    (f,logq,a) = NRiteration(data,targets,theta)
    s = np.where(f<0,f,0)
    W = np.diag(np.squeeze(np.exp(2*s - f) / ((np.exp(s) + np.exp(s-f))**2)))
    sqrtW = np.sqrt(W)
    L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW,np.dot(K,sqrtW)))
    p = np.exp(s)/(np.exp(s) + np.exp(s-f))
    fstar = np.dot(kstar.transpose(), (targets+1)*0.5 - p)
    v = np.linalg.solve(L,np.dot(sqrtW,kstar))	
    kstarstar = kernel(
        xstar, xstar, theta, wantderiv=False, measnoise=0
    ).diagonal()
    module_v = np.dot(v.transpose(), v)

    V = (kstarstar - module_v).diagonal()


    alpha = np.tile((1 / (2 * V)), (5, 1))
    # gamma = LAMBDAS * fstar
    gamma = np.einsum("i,k->ik", LAMBDAS,np.squeeze(fstar))
    lambdas_mat = np.tile(LAMBDAS, (len(xstar), 1)).T
    Vmat = np.tile(V, (5, 1))
    int1 = np.sqrt(np.pi / alpha)
    int2 = np.vstack(gamma) * np.sqrt(alpha / (alpha + np.vstack(LAMBDAS) ** 2))
    int2_a = erf(int2)
    int3 = (2 * np.sqrt(Vmat * 2 * np.pi))
    integrals = (
        np.sqrt(np.pi / alpha)
        * erf(gamma * np.sqrt(alpha / (alpha + lambdas_mat ** 2)))
        / (2 * np.sqrt(Vmat * 2 * np.pi))
    )
    pi_star = (COEFS * integrals).sum(axis=0) + .5 * COEFS.sum()

    s = np.sqrt(V)
    
    mat = np.zeros((len(fstar), 4))
    mat[:,0] = fstar[:,0]
    mat[:,1] = V
    mat[:,2] = (1-pi_star)
    mat[:,3] = (pi_star)

    return(mat)
    
#%%
    
N = 20
n= 500

#%%

rng = np.random.RandomState(4)
x = np.vstack(rng.uniform(-5, 5, N))
xstar = np.linspace(-5, 5, n).reshape(-1,1)
training_labels = np.vstack(np.where(f(x)<0.5, -1, 1))

#%%

pl.plot(xstar, f(xstar), ms=20)
pl.plot(x, training_labels, 'bo')


#%%
logPosterior(theta, x, training_labels)

#%%
lengthscale = 3
constantscale = 0.4
noise_scale = 0.1

#theta =np.zeros((3,1))
theta[0] = constantscale
theta[1] = lengthscale
theta[2] = noise_scale

newtheta = so.fmin_cg(logPosterior, theta, fprime=gradLogPosterior, args=(x,training_labels), gtol=1e-4,maxiter=100,disp=1, full_output=0)
print(newtheta, logPosterior(newtheta,x,training_labels))


#$$
predict(xstar, x, training_labels, theta)