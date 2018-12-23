import theano.tensor as tt
import pymc3 as pm
import pandas as pd
import numpy as np

#%%

#myModel = pm.Model()
#
#with myModel:
#    logits = pm.Normal("Logits", mu = 0, sd = 1)
#    p = tt.exp(logits)/(1 + tt.exp(logits))
#    observed = pm.Binomial("Observed", p = p, n = 1, observed = [0,0,1,1,1,1,1,1])
#    trace = pm.sample(2000)
#    
#pm.traceplot(trace)

#%%
SEED = 326402

np.random.seed(SEED)


#%%

dati = pd.DataFrame(index = ['gene0','gene1', 'gene2', 'gene3','gene4',
                             'gene5','gene6', 'gene7', 'gene8', 'gene9'],
                    data = {'paziente1':    [1,0,1,1,0,0,1,1,0,1],
                            'paziente2':    [1,1,1,1,0,0,1,1,1,1],
                            'paziente3':    [1,0,1,1,0,0,1,1,0,1],
                            'paziente4':    [0,0,1,1,0,0,1,1,0,0],
                            'paziente5':    [1,0,1,1,0,0,1,1,0,1],
                            'paziente6':    [0,1,1,1,0,0,1,1,1,0],
                            'paziente7':    [1,0,1,1,0,0,1,1,0,1],
                            'paziente8':    [1,0,1,1,0,0,1,1,0,1],
                            'paziente9':    [1,0,1,1,0,0,1,1,0,1],
                            'paziente10':   [1,0,1,1,0,0,1,1,0,1],
                            'paziente11':   [0,1,1,1,0,0,1,1,1,0],
                            'paziente12':   [1,0,1,1,0,0,1,1,0,1],
                            'paziente13':   [0,1,1,1,0,0,1,1,1,0]
                            
                            }
                    )
#%%
    
model = pm.Model()

sigma0 = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])
    
SEED = 12345

with model:
    packed_L = pm.LKJCholeskyCov('Packed_L', n=len(dati), eta = 2., sd_dist = pm.HalfCauchy.dist(1))
    L = pm.expand_packed_triangular(len(dati), packed_L)
#    sigma = pm.Wishart("Sigma", nu = 1, V = sigma0, shape = (3,3))
    sigma = pm.Deterministic('Sigma', L.dot(L.T))
#    mu = pm.Normal('mu', 0., 10., shape=3, testval = dati.mean(axis=1))
    logits = pm.MvNormal("Logits", mu = 0.5*np.ones(len(dati)), cov = sigma, shape = (1,len(dati)))
    p = pm.Deterministic('p', tt.exp(logits)/(1 + tt.exp(logits)))
    observed = pm.Binomial("Observed", p = p, n = 1, observed = dati.values.T)
    trace  = pm.sample(random_seed = SEED, cores = 1)