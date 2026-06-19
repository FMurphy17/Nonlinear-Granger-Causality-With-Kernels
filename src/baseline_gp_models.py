import numpy as np
import GPy as GPy
import pandas as pd
from scipy.stats import chi2 
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from .utils import init_causality


class GPModel:
    def __init__(self, data, num_samples, lags=1, contemporaneous=False):
        self.data = data
        self.contemporaneous = contemporaneous
        self.X, self.Y = init_causality(data, lags, contemporaneous)
        self.num_samples = num_samples
        self.lags = lags
        self.nvar = data.shape[1] # number of time series

        self.covariate_names = [] #tuples of (ts, lag)
        if contemporaneous:
            for i in range(self.nvar):
                for j in range(self.lags, -1, -1):
                    self.covariate_names.append(str((i, j)))
        else:
             for i in range(self.nvar):
                for j in range(self.lags, 0, -1):
                    self.covariate_names.append(str((i, j)))
        self.covariate_names = np.array(self.covariate_names)
        self.Xdf = pd.DataFrame(self.X, columns=self.covariate_names)

    """
    Zaremba, A. B., & Peters, G. W. (2022.  doi.org/10.1007/s11009-022-09928-3
    Likelihood Ratio Test (LRT) method for determining Granger Causality using a Gaussian Process (GP).

    Returns:
    G: Causal matrix 
        - 1: Presence of causal edge
        - 0: Absence of causal edge
    """
    def LRT(self):
        G = np.zeros((self.nvar, self.nvar))
        for i in range(self.nvar):
            Y_i = self.Y[:, i:i+1] 

            # Unrestricted model
            # create GP Model, which uses all covariates
            m2 = GPy.models.GPRegression(self.X, Y_i, kernel=GPy.kern.RBF(len(self.X[1]), ARD=True))
            m2.kern.variance.constrain_fixed(1.0) 
            m2.optimize()
            loglike2 = - m2.objective_function() #full model
        
            # Restricted model -- drop out a different covariate for each iteration (causal series)
            def process_j(j):
                X_m1 = np.delete(self.X, np.arange(j*self.lags, (j+1)*self.lags), axis=1)

                m1 = GPy.models.GPRegression(X_m1, Y_i, kernel=GPy.kern.RBF(len(X_m1[1]), ARD=True))
                m1.kern.variance.constrain_fixed(1.0)
                m1.optimize()
                
                loglike1 = -m1.objective_function()
                deltal = -2*(loglike1 - loglike2)
                p_val = chi2.sf(deltal, len(m2.param_array) - len(m1.param_array))
                print(p_val)

                return 1 if p_val < 0.05 else 0

            # Run all j in parallel
            results = Parallel(n_jobs=-1)(delayed(process_j)(j) for j in range(self.nvar))

            # Store the causal outcomes
            for j, val in enumerate(results):
                G[j, i] = val
                    
        # fill diagonal with 0s to not consider self-causal relationships
        np.fill_diagonal(G, 0)

        return G

    """
    Amblard, P. O., Michel, O. J. J., & Richard, C., et al. (2012). doi.org/10.1109/ICASSP.2012.6288635
    Assesses Granger Causality via the difference between log evidence of restricted and unrestricted model.

    Returns:
    G: Causal matrix 
        - 1: Presence of causal edge
        - 0: Absence of causal edge
    """
    def log_evidence(self):
        G = np.zeros((self.nvar, self.nvar))
        for i in range(self.nvar):
            Y_i = self.Y[:, i:i+1] 

            # Unrestricted model
            # create GP Model, which uses all covariates
            m2 = GPy.models.GPRegression(self.X, Y_i, kernel=GPy.kern.RBF(len(self.X[1]), ARD=False))
            m2.kern.variance.constrain_fixed(1.0) 
            m2.optimize()
            
            # Restricted model -- drop out a different covariate for each iteration (causal series)
            for j in range(self.nvar):
                X_m1 = self.X.copy()
                X_m1 = np.delete(X_m1, [range(j*self.lags, (j+1)*self.lags)], axis = 1)
                                
                m1 = GPy.models.GPRegression(X_m1, Y_i, kernel=GPy.kern.RBF(len(X_m1[1]), ARD=False))
                m1.kern.variance.constrain_fixed(1.0) 
                m1.optimize()
                
                # compute the difference between log evidences
                loglike1 = -m1.objective_function() #reduced model
                loglike2 = -m2.objective_function() #full model
                deltal = (loglike2-loglike1)
                
                reject = False    
                if deltal > 0:
                    reject = True

                if reject == True:
                    G[j, i] = 1
                    
        # fill diagonal with 0s to not consider self-causal relationships
        np.fill_diagonal(G, 0)

        return G