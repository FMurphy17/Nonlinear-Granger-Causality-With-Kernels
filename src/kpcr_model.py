import numpy as np
import scipy
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from sklearn.preprocessing import StandardScaler

"""
Creates lagged features for regression using NumPy.

Parameters:
    Y (ndarray): 2D array where columns are time series and rows are time steps.
    c (int): Index of the column to predict.
    n_lags (int): Number of lagged time steps to use as features.

Returns:
    tuple: (X, y_target) where X is the feature matrix and y_target is the target vector.
"""
def create_lagged_features(Y, c, n_lags=1):
    Y = np.asarray(Y)
    num_rows, num_cols = Y.shape

    X = np.hstack([Y[i:num_rows - n_lags + i, :] for i in range(n_lags)])
    y_target = Y[n_lags:, c]

    #remove mean
    X = X-np.mean(X,axis=0)[None,:]
    y_target=y_target[:,None]-np.mean(y_target)
    return X, y_target

"""
Filter out small eigenvalues. 
"""
def filter(L,V,th,maxeigpercent=0.5):
    cut=False
    indx=np.argsort(-L)
    L=L[indx]
    V=V[:,indx]
    ind0=np.where(L>np.max(L)*th)[0]
    ind = ind0[0:int(V.shape[1]*maxeigpercent)]
    if len(ind)<len(ind0):
        cut=True
    return L[ind],V[:,ind],cut

def centerK(K):
    # Center the kernel matrix
    N = K.shape[0]
    one_N = np.ones((N, N)) / N
    return K - one_N @ K - K @ one_N + one_N @ K @ one_N

"""
Nystrom method for large datasets.
"""
def _nystrom(y,
             X,
             kernel,
             num_inducing,
             eps_filter=1e-6,
             jitter=1e-9,
             kernel_params=None):
    
    if kernel_params is None:
        kernel_params = {}

    num = np.minimum(X.shape[0],num_inducing)
    ## Random inducing points
    random_indices = np.random.choice(X.shape[0], size=num, replace=False)
    Xind = X[random_indices]#random row
    # Set lengthscale to average distance between centers as per lsngc paper
    A = kernel(X,Xind,**kernel_params) 
    B = kernel(Xind,Xind,**kernel_params) 
    L = cholesky(B+np.eye(B.shape[0])*jitter, lower=True)  # B = L L^T
    B_inv_sqrt = solve_triangular(L, np.eye(B.shape[0]), lower=True).T  # B^{-1/2} = L^{-T}
    phi = A @ B_inv_sqrt  # Compute A * B^{-1/2} - 
    #center basis function
    phi =  phi-np.mean(phi,axis=0)[None,:]
    #eigenval-eigenvec decomp
    L1,V1= np.linalg.eigh(phi.T@phi)
    Lu,Vu,cut=filter(L1,V1,eps_filter)
    dof = Vu.shape[1]#degree of freedom
    ypred = Vu@np.diag(1/Lu)@Vu.T@ phi.T@y
    error = y-phi@ypred
    return error,dof

"""
Standard kernelised method. 
""" 
def _kernelised(y,
                X,
                kernel,
                eps_filter=1e-6,
                jitter=1e-6,
                kernel_params=None):
    
    if kernel_params is None:
        kernel_params = {}

    #center Kernel for unrestricted model           
    Ku = centerK( kernel(X,X, **kernel_params) )
    #eigenval-eigenvec
    L1,V1= np.linalg.eigh(Ku)
    #filter small eigenvalues
    Lu,Vu,cut=filter(L1,V1,eps_filter)
    if cut==True:
         print("Try to change Kernel lengthscale")
    #print(Lu)
    Kuf = Vu@np.diag(Lu)@Vu.T
    #projection matrix
    Pu = np.sum([Vu[:,[i]]@Vu[:,[i]].T for i in range(Vu.shape[1])],axis=0)
    error = (y-Pu@y)
    dof = Vu.shape[1]
    return error,dof,Pu,Kuf

"""
KPCR method 

Parameters:
    TSdata0: time-series data: [num_obs, num_ts]
    kernel_func: choice of kernel function
    n_lags: lag order
    test: Test for GC with f-test or correlation-based index
    alpha: Significance level for hypothesis test
    eps_filter: constant for filtering our small eigenvalues
    Nystrom: Use Nystrom approximation or standard kernelised method
    num_inducing: number of inducing points to use for Nystrom approx.
    jitter: jitter for Nystrom method
    ell_coeff: coefficient C to use for lengthscale heuristic - \ell = C(n_tm)

Returns:
    - Pvalues: p-values (f-test) or Pearson's correlation coefficients (corr)
    - Decisions: decision from test: 1: presence of causal edge, 0: absence of causal edge
"""
def KPCR(TSdata0,
         kernel_func,
         n_lags=1, 
         test="f-test",
         alpha=0.05,
         eps_filter=1e-6,
         Nystrom=True,
         num_inducing=30,
         jitter=1e-9,
         ell_coeff=1):
    #scale the data
    TSdata = TSdata0.copy()
    ns = TSdata.shape[1]# number timeseries
    scaler = StandardScaler()
    TSdata = scaler.fit_transform(TSdata)    
    #bonferroni correction 
    alphac = alpha/(ns*(ns-1))    
    #save results here
    CR = pd.DataFrame(np.zeros((ns,ns)), columns=np.arange(ns), index=np.arange(ns))
    Pvalues = pd.DataFrame(np.zeros((ns,ns)), columns=np.arange(ns), index=np.arange(ns))
    Decisions  = pd.DataFrame(np.zeros((ns,ns)), columns=np.arange(ns), index=np.arange(ns))
    #run the tests
    for t1 in range(ns):
        Xtmp,ytmp = create_lagged_features(TSdata, t1, n_lags=n_lags)
        lengthscale = ell_coeff*Xtmp.shape[1] 
        if Nystrom==False:
           erroru,dofu,_,Kuf = _kernelised(ytmp,Xtmp,kernel_func,eps_filter=eps_filter,kernel_params={"lengthscale": lengthscale})
        else:
           erroru,dofu = _nystrom(ytmp,Xtmp,kernel_func,num_inducing,eps_filter=eps_filter,jitter=jitter,kernel_params={"lengthscale": lengthscale})
            
        for t2 in range(ns): 
            if t1!=t2:
                #remove covariates relative to timeserties t2
                Xtmp1 = np.delete(Xtmp,np.arange(t2,TSdata.shape[1]*n_lags,ns),axis=1)
                if Nystrom==False:
                   errorr,dofr,Pr,_ = _kernelised(ytmp,Xtmp1,kernel_func,eps_filter=eps_filter,kernel_params={"lengthscale": lengthscale})
                else:
                   errorr,dofr = _nystrom(ytmp,Xtmp1,kernel_func,num_inducing,eps_filter=eps_filter,jitter=jitter,kernel_params={"lengthscale": lengthscale})

                if dofu<=dofr:
                    print("Restricted model has more (or equal) degree-of-freedom than unrestricted")
                    print("Try to change Kernel lengthscale or increase num_inducing if Nystrom")

                if test=="f-test":
                    delta = (np.sum(errorr**2)-np.sum(erroru**2))/np.sum(erroru**2)
                    df1=len(ytmp)-dofu
                    df2=len(ytmp)-dofr
                    deltadf = df2-df1
                    p_value = 1 - scipy.stats.f.cdf(np.real(delta*df1/(deltadf)),deltadf, df1)
                    dec = (p_value<alphac)+0
                    rr= None
                elif test=="corr":
                    if Nystrom==False:
                        Ktilde = (np.eye(Pr.shape[0])-Pr)@Kuf@(np.eye(Pr.shape[0])-Pr)
                        Ltilde,Vtilde= np.linalg.eigh((Ktilde+Ktilde.T)/2)    
                        Lf,Vf,_=filter(Ltilde,Vtilde,eps_filter)  
                        corr = [list(scipy.stats.pearsonr((ytmp-Pr@ytmp).flatten(), Vf[:,i])) for i in range(Vf.shape[1])]
                        corr=np.vstack(corr)
                        rr = corr[:,0]**2
                        pp = corr[:,1]
                        thb=alphac/Vf.shape[1]
                        indpr=pp<thb;
                        rr = np.sum(rr[indpr])
                        dec =(rr>0)+0
                        p_value = rr#this is not the p_value
                    else:
                        print("Not implemented correlation test with Nystrom")

                Pvalues.iloc[t1,t2]=p_value               
                Decisions.iloc[t1,t2]=dec
            
    return Pvalues.T, Decisions.T
            