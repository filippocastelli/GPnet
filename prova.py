from __future__ import division
import numpy as np
import networkx as nx
import pandas as pd
from GPnet import GPnetRegressor, GPnetClassifier
#%%
#randomregular
a= GPnetRegressor(totnodes = 100, ntrain=50, ntest=30, deg=3, theta=[0.1, 0.1, 0.1])
fstar1, V1 = a.predict()
fstar, V = a.predict2()
a.plot_graph()
a.plot_predict_graph()
a.plot_predict_2d()

            
