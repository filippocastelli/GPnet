import numpy as np


def SquaredExp(Graph,graph_distance_matrix, nodes_a,nodes_b,theta,measnoise=1., wantderiv=True, print_theta=False):
    theta = np.squeeze(theta)
    #theta = np.exp(theta)
    #graph_distance_matrix = shortest_path_graph_distances(Graph)
    nodelist = list(Graph.nodes)
    nodeset = set(nodes_a).union(set(nodes_b))
    nodes_to_drop = [x for x in nodelist if x not in nodeset]
    cols_to_drop = set(nodes_to_drop).union(set(nodes_b) - set(nodes_a))
    rows_to_drop = set(nodes_to_drop).union(set(nodes_a) - set(nodes_b))
    p = graph_distance_matrix.drop(cols_to_drop).drop(rows_to_drop, 1)
    distances = p.values
    
    d1 = len(nodes_a)
    d2 = len(nodes_b)
  
    k = theta[0] * np.exp(-0.5*distances)
    
    if wantderiv:
        K = np.zeros((d1,d2,len(theta)+1))
        # K[:,:,0] is the original covariance matrix
        K[:,:,0] = k + measnoise*theta[2]*np.eye(d1,d2)
        K[:,:,1] = k
        K[:,:,2] = -0.5*k*distances
        K[:,:,3] = theta[2]*np.eye(d1,d2)
        return K
    else:
        return k + measnoise*theta[2]*np.eye(d1,d2)
    
def gRBF(Graph,graph_distance_matrix, nodes_a,nodes_b, theta ,measnoise=1., wantderiv=True, print_theta=False):
    theta = np.squeeze(theta)
    theta = np.exp(theta)
    #graph_distance_matrix = shortest_path_graph_distances(Graph)
    nodelist = list(Graph.nodes)
    nodeset = set(nodes_a).union(set(nodes_b))
    nodes_to_drop = [x for x in nodelist if x not in nodeset]
    cols_to_drop = set(nodes_to_drop).union(set(nodes_b) - set(nodes_a))
    rows_to_drop = set(nodes_to_drop).union(set(nodes_a) - set(nodes_b))
    p = graph_distance_matrix.drop(cols_to_drop).drop(rows_to_drop, 1)
    distances = p.values
    
    d1 = len(nodes_a)
    d2 = len(nodes_b)
  
    k = np.exp(theta[0]) * np.exp(-0.5*(distances)*theta[1])
    
    if wantderiv:
        K = np.zeros((d1,d2,len(theta)+1))
        # K[:,:,0] is the original covariance matrix
        K[:,:,0] = k + measnoise*theta[2]*np.eye(d1,d2)
        K[:,:,1] = k
        K[:,:,2] = -0.5*k*distances
        K[:,:,3] = theta[2]*np.eye(d1,d2)
        return K
    else:
        return k + measnoise*theta[2]*np.eye(d1,d2)
    
    
def Periodic(Graph,graph_distance_matrix, nodes_a,nodes_b,theta,measnoise=1., wantderiv=True):
    theta = np.squeeze(theta)
    #theta = np.exp(theta)
    #graph_distance_matrix = shortest_path_graph_distances(Graph)
    nodelist = list(Graph.nodes)
    nodeset = set(nodes_a).union(set(nodes_b))
    nodes_to_drop = [x for x in nodelist if x not in nodeset]
    cols_to_drop = set(nodes_to_drop).union(set(nodes_b) - set(nodes_a))
    rows_to_drop = set(nodes_to_drop).union(set(nodes_a) - set(nodes_b))
    p = graph_distance_matrix.drop(cols_to_drop).drop(rows_to_drop, 1)
    distances = p.values
    
    d1 = len(nodes_a)
    d2 = len(nodes_b)
  
    k = theta[0] * np.exp(-2.0*theta[1]*np.sin(np.pi*distances/theta[2])**2)
    #k = theta[0] * np.exp(- 2.0*np.sin(np.pi*distances**2)**2/(theta[1]**2)) 
    
    if wantderiv:
        K = np.zeros((d1,d2,len(theta)+1))
        # K[:,:,0] is the original covariance matrix
        K[:,:,0] = k + measnoise*theta[2]*np.eye(d1,d2)
        K[:,:,1] = k*theta[1]
        K[:,:,2] = -k*np.cos((2*np.pi*distances)/theta[2])*(np.pi*distances/(theta[2]**2))
        K[:,:,3] = theta[3]*np.eye(d1,d2)
        return K
    else:
        return k + measnoise*theta[3]*np.eye(d1,d2)

def logp_fun(fun, theta,*args):
    Graph,distancematrix, data,t = args
    #k = kernel(data,data,theta,wantderiv=False)
    k = fun(Graph,distancematrix, data,data,theta,wantderiv=False)
    L = np.linalg.cholesky(k)
    beta = np.linalg.solve(L.transpose(), np.linalg.solve(L,t))
    logp = -0.5*np.dot(t.transpose(),beta) - np.sum(np.log(np.diag(L))) - np.shape(data)[0] /2. * np.log(2*np.pi)
    return -logp

def gradlogp_fun(fun, theta,*args):
    #print(args)
    Graph,distancematrix, data,t = args
    theta = np.squeeze(theta)
    d = len(theta)
    #K = kernel(data,data,theta,wantderiv=True)
    K = fun(Graph,distancematrix, data, data, theta, wantderiv=True)

    L = np.linalg.cholesky(np.squeeze(K[:,:,0]))
    invk = np.linalg.solve(L.transpose(),np.linalg.solve(L,np.eye(np.shape(data)[0])))
	
    dlogpdtheta = np.zeros(d)
    for d in range(1,len(theta)+1):
        dlogpdtheta[d-1] = 0.5*np.dot(t.transpose(), np.dot(invk, np.dot(np.squeeze(K[:,:,d]), np.dot(invk,t)))) - 0.5*np.trace(np.dot(invk,np.squeeze(K[:,:,d])))

    return -dlogpdtheta

#def kernel4(data1,data2,theta,wantderiv=True,measnoise=1.):
#	theta = np.squeeze(theta)
#	# Periodic 
#	if np.shape(data1)[0] == len(data1):
#		d1 = np.shape(data1)[0]
#		n = 1
#	else:
#		(d1,n) = np.shape(data1)
#	d2 = np.shape(data2)[0]
#	sumxy = np.zeros((d1,d2))
#	for d in range(n):
#		D1 = np.transpose([data1[:,d]]) * np.ones((d1,d2))
#		D2 = [data2[:,d]] * np.ones((d1,d2))
#		sumxy += (D1-D2)**2
#
#	k = theta[0]**2 * np.exp(- 2.0*np.sin(np.pi*sumxy)**2/(theta[1]**2)) 
#
#	if wantderiv:
#		K = np.zeros((d1,d2,len(theta)+1))
#		K[:,:,0] = k + measnoise*theta[2]**2*np.eye(d1,d2)
#		K[:,:,1] = 2.0 *k /theta[0]
#		K[:,:,2] = 4.0*k*np.sin(np.pi*sumxy)**2/(theta[2]**3)
#		K[:,:,3] = 2.0*theta[2]*np.eye(d1,d2)
#		return K
#	else:	
#		return k + measnoise*theta[2]**2*np.eye(d1,d2)
#
#def kernel3(data1,data2,theta,wantderiv=True,measnoise=1.):
#	theta = np.squeeze(theta)
#	# Periodic and a squared exponential
#	if np.shape(data1)[0] == len(data1):
#		d1 = np.shape(data1)[0]
#		n = 1
#	else:
#		(d1,n) = np.shape(data1)
#	d2 = np.shape(data2)[0]
#	sumxy = np.zeros((d1,d2))
#	for d in range(n):
#		D1 = np.transpose([data1[:,d]]) * np.ones((d1,d2))
#		D2 = [data2[:,d]] * np.ones((d1,d2))
#		sumxy += (D1-D2)**2
#
#	k = theta[0]**2 * np.exp(-sumxy/(2.0*theta[1]**2) - 2.0*np.sin(np.pi*sumxy)**2/(theta[2]**2)) 
#
#	#print k
#	#print measnoise*theta[2]**2*np.eye(d1,d2)
#	if wantderiv:
#		K = np.zeros((d1,d2,len(theta)+1))
#		K[:,:,0] = k + measnoise*theta[2]**2*np.eye(d1,d2)
#		K[:,:,1] = 2.0 *k /theta[0]
#		K[:,:,2] = k*sumxy/(theta[1]**3)
#		K[:,:,3] = -4.0*k*np.sin(np.pi*sumxy)**2/(theta[2]**3)
#		K[:,:,4] = 2.0*theta[3]*np.eye(d1,d2)
#		return K
#	else:	
#		return k + measnoise*theta[2]**2*np.eye(d1,d2)
#
#def kernel2(data1,data2,theta,wantderiv=True,measnoise=1.):
#	# Uses exp(theta) to ensure positive hyperparams
#	theta = np.squeeze(theta)
#	theta = np.exp(theta)
#	# Squared exponential
#	if np.ndim(data1) == 1:
#		d1 = np.shape(data1)[0]
#		n = 1
#		data1 = data1*np.ones((d1,1))
#		data2 = data2*np.ones((np.shape(data2)[0],1))
#	else:
#		(d1,n) = np.shape(data1)
#
#	d2 = np.shape(data2)[0]
#	sumxy = np.zeros((d1,d2))
#	for d in range(n):
#		D1 = np.transpose([data1[:,d]]) * np.ones((d1,d2))
#		D2 = [data2[:,d]] * np.ones((d1,d2))
#		sumxy += (D1-D2)**2*theta[d+1]
#
#	k = theta[0] * np.exp(-0.5*sumxy) 
#	#k = theta[0]**2 * np.exp(-sumxy/(2.0*theta[1]**2)) 
#
#	#print k
#	#print measnoise*theta[2]**2*np.eye(d1,d2)
#	if wantderiv:
#		K = np.zeros((d1,d2,len(theta)+1))
#		K[:,:,0] = k + measnoise*theta[2]*np.eye(d1,d2)
#		K[:,:,1] = k 
#		K[:,:,2] = -0.5*k*sumxy
#		K[:,:,3] = theta[2]*np.eye(d1,d2)
#		return K
#	else:	
#		return k + measnoise*theta[2]*np.eye(d1,d2)
#
#def kernel(data1,data2,theta,wantderiv=True,measnoise=1.):
#	theta = np.squeeze(theta)
#	# Squared exponential and periodic
#	if np.shape(data1)[0] == len(data1):
#		d1 = np.shape(data1)[0]
#		n = 1
#	else:
#		(d1,n) = np.shape(data1)
#	d2 = np.shape(data2)[0]
#	sumxy = np.zeros((d1,d2))
#	for d in range(n):
#		D1 = np.transpose([data1[:,d]]) * np.ones((d1,d2))
#		D2 = [data2[:,d]] * np.ones((d1,d2))
#		sumxy += (D1-D2)
#
#	k = theta[0]**2 * np.exp(-sumxy**2/(2.0*theta[1]**2)) + np.exp(-2.*np.sin(theta[2]*np.pi*(sumxy))**2/theta[3]**2)
#
#	if wantderiv:
#		K = np.zeros((d1,d2,len(theta)+1))
#		K[:,:,0] = k + measnoise*theta[4]**2*np.eye(d1,d2)
#		K[:,:,1] = 2.0 *k /theta[0]
#		K[:,:,2] = k*sumxy**2/(theta[1]**3)
#		K[:,:,3] = -4.0/(theta[3]**2)*np.pi*sumxy*np.sin(theta[2]*np.pi*sumxy)*np.cos(theta[2]*np.pi*sumxy)*np.exp(-2.*np.sin(theta[2]*np.pi*(sumxy))**2/theta[3]**2)
#		K[:,:,4] = 4.0*np.sin(theta[2]*np.pi*sumxy)**2/(theta[3]**3)*np.exp(-2.*np.sin(theta[2]*np.pi*(sumxy))**2)
#		K[:,:,5] = 2.0*theta[4]*np.eye(d1,d2)
#		return K
#	else:	
#return k + measnoise*theta[3]**2*np.eye(d1,d2)