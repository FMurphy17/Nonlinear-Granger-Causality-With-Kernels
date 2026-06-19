import numpy as np

"""
Stack a numpy vector by a lag 

Parameters
    x : np.ndarray - The input vector.
    l : int - The lag.
    contemporaneous: bool - include lag=0

Returns
  y - stacked vector 
  x - time-shifted response
"""
def stack_by_lag(x, l, contemporaneous=False):
    if l == 0:
        return x, x
    if contemporaneous:
        y = np.empty((x.shape[0] - l, l+1)) # Add lag at time t 
        for i in range(x.shape[0] - l):
            y[i] = x[i:i + l+1] # Add lag at time t
    else:
        y = np.empty((x.shape[0] - l, l))
        for i in range(x.shape[0] - l):
            y[i] = x[i:i + l] 

    return y,x[l:]


def init_causality(XX, m=1, contemporaneous=False):
    nn, nvar = XX.shape
    for i in range(nvar):
        # z-score the data along the columns
        XX[:,i] = (XX[:,i] - np.mean(XX[:,i])) / np.std(XX[:,i])       
    X,x = stack_by_lag(XX[:,0], m, contemporaneous)
    x=x[:,None]
    for i in range(1,XX.shape[1]):
        Xi,xi=stack_by_lag(XX[:,i], m, contemporaneous)
        X = np.hstack([X,Xi])
        x = np.hstack([x,xi[:,None]])
    return X, x

def create_connections(nvar):
    cnnx = []
    for i in range(nvar):
        for j in range(nvar):
            cnnx.append((i,j))
    return cnnx

def create_connections_contemporaneous(nvar, lags):
    cnnx = []
    for i in range(nvar):
        for j in range(nvar):
            for k in range(lags+1):
                cnnx.append((i,j,k)) # (cause, effect, lag)
    return cnnx