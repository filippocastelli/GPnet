# GPnet
Gaussian Process Regression and Classification on Graphs

The code is loosely inspired on the code from Chapter 18 of Machine Learning: An Algorithmic Perspective (2nd Edition) by Stephen Marsland and implements the algorithms in Chapters 2 and 3 of Gaussian Processes for Machine Learning by C.E. Rasmussen.

`GPnet` at the moment exposes two classes: `GPnetRegressor` and `GPnetClassifier`
* **GPnetRegressor** provides basic regression functionality for functions defined on graph nodes
* **GPnetClassfier** provides classification of test nodes, given a set of -1/+1 labels for the training nodes

The only kernel available at the moment is a _squared exponential_ kernel, more kernels and custom kernel composition will be added in the future

- update -
kernel composition is already possible via custom function definitions: "soft" kernel composition is yet to be implemented.

Some basic parameter optimization is provided by `scipy.optimize`.
 
## Dependencies
GPnet demos require `networkx`, `pandas` , `numpy`, `matplotlib`, `scipy.optimize`, and `random` to work: nothing too exotic.

## What's the future of this repo? 
Possible future work directions include:
* including support for multiple and custom-defined kernels
* fixing numerical issues with covariance matrix estimation and parameter optimization
