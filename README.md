# GPnet
Gaussian Process Regression and Classification on network-like topologies

Code still in pre-alpha concept testing, based on the code from Chapter 18 of Machine Learning: An Algorithmic Perspective (2nd Edition) by Stephen Marsland, implements the algorithms in Chapter 2 and 3 of Gaussian Processes for Machine Learning by C.E. Rasmussen.


GPnet at this date consists of four demos: `GPR.py`, `GPC.py`, `GPCnet.py`,`GPRnet.py` in a somewhat-functioning state.

* **GPR and GPC**: implementations of respectively Gaussian Process Regression and Gaussian Process Classification with tow-dimensional real functions: nothing new.
* **GPRnet and GPCnet**: implementations of GPR and GPC using a shortest distance metric on an undirected network.

The only kernel available at the moment is a _squared exponential_ kernel, more kernels and custom kernel composition will be added in the future.
Some basic parameter optimization is provided by `scipy.optimize`.
 
## Dependencies
GPnet demos require `networkx`, `pandas` , `numpy`, `matplotlib`, `scipy.optimize`, and `random` to work: nothing too exotic.

## What's the future of this repo? 
Some concept testing is required to estabilish if GPnet will make sense as a standalone Gaussian Process library like `GPy` or if it will continue as a network-friendly extension of pre-existent libraries. 
Work to do depends on what future will be decided for this project: if the standalone option is chosen then GPnet will be given a proper user-friendly face with proper class definitions, the possibility of composite kernel creation with fine-tuning and optimization, if GPnet is to be integrated in some other pre-existent ecosystem work will be focused on kernel definition and customization.

readme to be updated soon (as the rest of the code, rn it's a f-in mess)
