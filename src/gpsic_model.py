import numpy as np
import GPy as GPy
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from scipy.stats import halfcauchy
from collections import defaultdict
import ast
from collections import Counter
from .utils import init_causality

"""
Leave-one-out pseudo log likelihood from Rasmussen 5.4.2

Returns:
    Array of shape (N,), where N = model.X.shape[0].
"""
def loo_per_point_logpdf(model):
    K = model.kern.K(model.X)
    K = K + np.eye(K.shape[0]) * float(model.Gaussian_noise.variance[0])
    y = model.Y.flatten()
    mu = np.zeros_like(y)
    
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - mu))
    K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K.shape[0])))
    diag_invK = np.diag(K_inv)
    
    sigma2_i = 1.0 / diag_invK
    mu_i = y - alpha / diag_invK

    logp_i = -0.5*np.log(2*np.pi*sigma2_i) - 0.5*(((y - mu_i)**2)/sigma2_i)
    return logp_i


def ddf_score(m2, parameters):
    noise_variance = parameters[-1]
    variance = parameters[0]    
    lengthscales = parameters[1:-1]
    
    m2.rbf.variance = variance
    m2.Gaussian_noise.variance = noise_variance
    m2.rbf.lengthscale = lengthscales
    dfs = m2.gradient
    return m2.log_likelihood(), dfs


def ddf_bic_penal(parameters, n, epsilon=0.1):
    lengthscales = parameters[1:-1] 
    inv_leng = 1/lengthscales ** 2
    num = -2 *lengthscales * epsilon
    denom = (1+ lengthscales**2 * epsilon)**2
    x = (-(0.5)* np.log(n)*(num/denom))
    return -np.log(n)*np.sum(inv_leng/(inv_leng+epsilon))/2, np.hstack([np.array([0]), x, np.array([0])])


def ddf_myobj(m2, x, epsilon, num_samples, tau):
    v, ddfv = ddf_score(m2, x)
    v2, ddfv2 = ddf_bic_penal(x, n=num_samples, epsilon=epsilon)
    return -(v + v2), -(ddfv + ddfv2)

"""
Wrapper for optimizing GP log-marginal likelihood with SIC penalization
"""
class Wrapper:
    def __init__(self, m2, num_samples, tau):
        self.cache = {}
        self.m2 = m2
        self.num_samples = num_samples
        self.tau = tau

    def __call__(self, x, epsilon, *args):
        fun, grad = ddf_myobj(self.m2, x, epsilon, self.num_samples, self.tau)
        self.cache['grad'] = grad
        return fun

    def jac(self, x, epsilon, *args):
        return self.cache.pop('grad')

class GPSICModel:
    def __init__(self, data, num_samples, lags=1, contemporaneous=False):
        self.data = data
        self.contemporaneous = contemporaneous
        self.X, self.Y = init_causality(data, lags, contemporaneous)
        self.num_samples = num_samples
        self.lags = lags
        self.nvar = data.shape[1] # Number of time series

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
    Resets class variables/restructures data after running optimize lag loo
    """
    def reset_state(self, opt_lag):
        self.X, self.Y = init_causality(self.data, opt_lag, self.contemporaneous)
        self.lags = opt_lag
        self.covariate_names = [] #tuples of (ts, lag)
        if self.contemporaneous:
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
    Smooth Information Criterion (SIC) method for determining Granger Causality using a Gaussian Process (GP).

    Returns:
        gc_matrix: Causality matrix with 0: absent link, 1: present link (Rows: cause, Cols: effect)
    """
    def SIC(self, model_type="GPSIC"):
        pos = []
        neg = []
        for i in range(len(self.Y[1])):
            Y_i = self.Y[:, i:i+1] # Response time series
            kernel=GPy.kern.RBF(self.Xdf.shape[1], ARD=True)
            gpmodel = GPy.models.GPRegression(self.X, Y_i, kernel=kernel)
            tau = halfcauchy.rvs(0, 1, 1) 
            bounds = [[0.01, 1e4]] * len(gpmodel.param_array)
            bounds[0] = [1.00, 1.01] # bounds on the variance
            init_param = gpmodel.param_array
            threshold = 50 

            """
            Method from Cui, C., Banelli, P., Djuric, P. M. (2022). doi.org/10.23919/EUSIPCO55093.2022.9909923
            - GP ARD model without information criterion
            """
            if model_type == "GP":
                gpmodel.optimize_restarts(5, verbose=False)

            elif model_type == "GPSIC":
                wrapper = Wrapper(gpmodel, self.num_samples-self.lags, tau)
                T = np.hstack((np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10]), np.logspace(0, -8, 40)))
                for eps in T: 
                    res = minimize(wrapper, init_param, jac=wrapper.jac, bounds=bounds, method='L-BFGS-B', options={'disp': False}, args=[eps])
                    init_param = res.x

            labels = [ast.literal_eval(s) for s in self.covariate_names] # covariate names as tuples
            for idx, i2 in enumerate(gpmodel.rbf.lengthscale):
                # Exclude self-causality to be consistent with baselines
                if i2 <= threshold: 
                    if labels[idx][0] != i:  
                        pos.append((labels[idx][0], i))
                elif  i2 > threshold:  
                    if labels[idx][0] != i:
                        neg.append((labels[idx][0],i))

        pos = set(pos)
        neg = set([tup for tup in neg if tup not in pos])
        gc_matrix = np.zeros((self.nvar, self.nvar))
        for x in pos:
            gc_matrix[x] = 1
        return gc_matrix


    """
    GPSIC-based method for identifying lagged and contemporaneous edges

    Returns:
        G_t: Causality matrix (Rows: cause, Cols: effect) with 
            0: no link
            1: directed link 
            2: conflict
            3: undirected link 
    """
    def SIC_contemporaneous(self):
        adj_l = np.zeros((self.nvar, self.nvar, self.lags + 1))
        adj_c = np.zeros((self.nvar, self.nvar, self.lags + 1))
        threshold = 50 

        contempcols = [s for s in self.covariate_names if s.strip("() ").split(",")[1].strip() == "0"]
        for i in range(len(self.Y[1])):
            # Exclude from X the column that matches Y, i.e. the observation which we are estimating 
            X_i = self.Xdf.drop(['('+ str(i) + ', 0)'], axis=1)
            Y_i = self.Y[:, i:i+1] # Response variable

            # G'' - includes contemporaneous links
            kernel_c = GPy.kern.RBF(X_i.shape[1], ARD=True)

            gpmodel_c = GPy.models.GPRegression(X_i.to_numpy(), Y_i, kernel=kernel_c)
            tau2 = halfcauchy.rvs(0, 1, 1) 
            bounds_c = [[0.01, 1e4]] * len(gpmodel_c.param_array)
            bounds_c[0] = [1.0, 1.01] # bounds on the variance
            init_param_c = gpmodel_c.param_array

            wrapper_c = Wrapper(gpmodel_c, self.num_samples, tau2)
            T = np.hstack((np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10]), np.logspace(0, -8, 40)))
            for eps in T: 
                res = minimize(wrapper_c, init_param_c, jac=wrapper_c.jac, bounds=bounds_c, method='L-BFGS-B', options={'disp': False}, args=[eps])
                init_param_c = res.x

            labels_contemp = X_i.columns
            for idx, i2 in enumerate(gpmodel_c.rbf.lengthscale):
                # Include self-causality to test for autocorrelation detection
                if i2 <= threshold:
                    adj_c[i, ast.literal_eval(labels_contemp[idx])[0], ast.literal_eval(labels_contemp[idx])[1]] = 1
            
            parents = np.argwhere(adj_c[:, :, :] == 1)
            lagged_parents = parents[(parents[:,2] > 0) & (parents[:,0] == i)]  

            # G' - only lagged links
            if len(lagged_parents > 0):
                X_lagged = self.Xdf.drop(contempcols, axis=1) 
                lps = [str(tuple(t[1:])) for t in lagged_parents]
                lps = list(set(lps))
                X_lagged = self.Xdf[lps]
                labels_lagged = X_lagged.columns

                kernel_lagged=GPy.kern.RBF(X_lagged.shape[1], ARD=True)
                gpmodel_lagged = GPy.models.GPRegression(X_lagged.to_numpy(), Y_i, kernel=kernel_lagged)
                tau = halfcauchy.rvs(0, 1, 1) 
                bounds = [[0.01, 1e4]] * len(gpmodel_lagged.param_array)
                bounds[0] = [1.0, 1.01] # bounds on the variance
                init_param = gpmodel_lagged.param_array

                wrapper_l = Wrapper(gpmodel_lagged, self.num_samples, tau)
                T = np.hstack((np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10]), np.logspace(0, -8, 40)))
                for eps in T: 
                    res = minimize(wrapper_l, init_param, jac=wrapper_l.jac, bounds=bounds, method='L-BFGS-B', options={'disp': False}, args=[eps])
                    init_param = res.x

                for idx, i2 in enumerate(gpmodel_lagged.rbf.lengthscale):
                    # Include self-causality to test for autocorrelation detection
                    if i2 <= threshold:
                        adj_l[i, ast.literal_eval(labels_lagged[idx])[0], ast.literal_eval(labels_lagged[idx])[1]] = 1

        G_final = np.zeros_like(adj_l)
        G_final[:, :, 0] = adj_c[:, :, 0]
        G_final[:, :, 1:] = adj_l[:, :, 1:]

        # Edge orientation
        # Search for lagged triples
        e0 = np.argwhere(G_final[:,:,0] == 1)
        pairs = [tuple(idx) for idx in e0]  
        pairs_set = set(pairs)  
        undirected = [p for p in pairs if (p[1], p[0]) in pairs_set] # Checks that both a->b and b->a are present

        lagged_triples = []
        for (a,b) in undirected:
            hits = np.argwhere(G_final[b, :, :] == 1)
            hits = hits[hits[:,1] > 0]  # only check lagged connections
            for (col, layer) in hits:
                lagged_triples.append(((a,b,0), (b,col,layer)))

        # Filter any lagged triples that are in the diamond exception -> cannot orient based on these 
        counts = Counter((i[0][0], i[1][1], i[1][2]) for i in lagged_triples)
        lagged_triples = [i for i in lagged_triples if counts[(i[0][0], i[1][1], i[1][2])] == 1]

        # Keep track of edge orientations for the contemporaneous edges 
        edge_orientation_count = {e: 0 for e in undirected} 

        """
        Collider Phase
        - Check if lagged edge from G_final triple is in G''.
        """ 
        e_count = 0
        while e_count < len(lagged_triples):
            edges = lagged_triples[e_count]
            # Check if Vi -> Vk and Vi -> Vj in lagged triples: if yes, cannot orient
            vi_vj = (edges[0][0], edges[1][1], edges[1][2])
            if G_final[vi_vj[0], vi_vj[1], vi_vj[2]] == 1.:
                # This is a shieleded triple. Leave undirected
                lagged_triples.remove(edges)
                continue
            # Check whether Vi -> Vj in G''
            if adj_c[vi_vj[0], vi_vj[1], vi_vj[2]] == 1.:
                # False Positive -> orient as collider
                collider = (edges[0][1], edges[0][0], edges[0][2])
                edge_orientation_count[(collider[1], collider[0])] += 1
                lagged_triples.remove(edges)
                # Remove contemporaneous edge from list once directed (or marked as conflicting)
                if (edges[0][0], edges[0][1]) in undirected:
                    undirected.remove((edges[0][0], edges[0][1]))
                if (edges[0][1], edges[0][0]) in undirected:
                    undirected.remove((edges[0][1], edges[0][0]))
            else:
                # else Vk not collider
                e_count += 1

        """
        Rule Orientation Phase (if possible orient remaining contemp. links according to R1) 
        R1) Everything remaining in lagged_triples is non-collider. Check for conflicts and orient. 
        R1.5) Identify any remaining contemporaneous triples Vi_t -> Vk_t - Vj_t where Vi not adjacent to Vj (unshileded triple)
            - This structure rules out collider, so orient Vk_t -> Vj_t
        """
        # R1
        for edges in lagged_triples:
            noncollider = edges[0]
            edge_orientation_count[(noncollider[1], noncollider[0])] += 1
            # Removed edges that were directed as non-collider (or marked as conflicting)
            if (edges[0][0], edges[0][1]) in undirected:
                    undirected.remove((edges[0][0], edges[0][1]))
            if (edges[0][1], edges[0][0]) in undirected:
                undirected.remove((edges[0][1], edges[0][0]))

        grouped = defaultdict(dict)
        for (x, y), v in edge_orientation_count.items():
            key = tuple(sorted((x, y)))
            grouped[key][(x, y)] = v

        majority_pairs = []
        conflicts = []
        und = []
        for key, vals in grouped.items():
            # ensure both directions exist, missing defaults to 0
            forward = vals.get(key, 0)
            reverse = vals.get((key[1], key[0]), 0)
            # if one direction > 0 and the other 0, assign edge to positive direction
            if (forward > 0 and reverse == 0) or (reverse > 0 and forward == 0):
                majority_pairs.append(
                    (key if forward > reverse else (key[1], key[0]))
                )
            # if both directions > 0 mark with conflict
            elif forward > 0 and reverse > 0:
                conflicts.append(key)  
            else: # both = 0, undirected      
                und.append(key)

        for mp in majority_pairs:
            G_final[mp[1], mp[0], 0] = 1
            G_final[mp[0], mp[1], 0] = 0
        for c in conflicts:
            G_final[c[1], c[0], 0] = 2
            G_final[c[0], c[1], 0] = 2
        for u in und:
            G_final[u[1], u[0], 0] = 3
            G_final[u[0], u[1], 0] = 3

        # R1.5 - can only be applied under the assumed scm
        while len(undirected) > 0:
            a = undirected[0][0]
            b = undirected[0][1]
            # 3rd/4th conditions will never occur under the assumed scm (no contemporaneous colliders & acyclic)
            contemp_causes_b = np.argwhere((G_final[b, :, 0] == 1) & (G_final[:, b, 0] == 0) & (G_final[a, :, 0] == 0) & (G_final[:, a, 0] == 0)).flatten()
            contemp_causes_a = np.argwhere((G_final[a, :, 0] == 1) & (G_final[:, a, 0] == 0) & (G_final[b, :, 0] == 0) & (G_final[:, b, 0] == 0)).flatten()
            if len(contemp_causes_b) > 0 and len(contemp_causes_a) > 0:
                # This is a conflicting edge 
                G_final[a,b,0] = 2
                G_final[b,a,0] = 2
            elif len(contemp_causes_a) == 0 and len(contemp_causes_b) > 0:
                G_final[b,a,0] = 1
                G_final[a,b,0] = 0
            elif len(contemp_causes_b) == 0 and len(contemp_causes_a) > 0:
                G_final[a,b,0] = 1
                G_final[b,a,0] = 0
            undirected.remove((a, b))
            undirected.remove((b, a))

        G_final[np.arange(G_final.shape[0]), np.arange(G_final.shape[1]), :] = 0 # Zero out self-causal links for evaluation
        G_t = np.transpose(G_final, (1, 0, 2))
        G_t = G_t.reshape(G_t.shape[0], -1) 
        return G_t   

    """
    Cross-validation method for lag order selection using log-pseudo-likelihood as described in Rasmussen Ch. 5.4.2

    Parameters:
        one_se=True applies the 'one-standard-error' rule to pick the smallest p within 1 SE of the best mean
    
    Returns: 
        best_lag: Lag order with maximum LOO likelihood
    """
    def optimize_lag_loo(self, n_lags, one_se=True, contemporaneous=False):
        results = []
        prev_params = None
        possible_lags = np.arange(1, n_lags+1)

        for m in possible_lags:
            if contemporaneous:
                X, Y =  init_causality(self.data, m, contemporaneous=True)
                covariate_names = [] #tuples of (ts, lag)
                for i in range(self.nvar):
                    for j in range(m, -1, -1):
                        covariate_names.append(str((i, j)))
                covariate_names = np.array(covariate_names)
                Xdf = pd.DataFrame(X, columns=covariate_names)
            else:
                X, Y =  init_causality(self.data, m)
            # Fit reg model for each time series in the system and then average over all
            logp_list = []
            mean_loo_list = []
            se_loo_list = []
            for i in range(len(Y[1])):
                if contemporaneous: # drop out the covariate that is being predicted
                    X = Xdf.drop(['('+ str(i) + ', 0)'], axis=1).to_numpy()
                Y_i = Y[:, i:i+1] # we want to predict for each time series
                kernel=GPy.kern.RBF(len(X[1]), ARD=True)
                gpmodel = GPy.models.GPRegression(X, Y_i, kernel=kernel)
                
                tau = halfcauchy.rvs(0, 1, 1) 
                bounds = [[0.01, 1e4]] * len(gpmodel.param_array)
                bounds[0] = [1.0, 1.01] # bounds on the variance
                init_param = gpmodel.param_array

                # Model optimization
                wrapper = Wrapper(gpmodel, self.num_samples, tau)
                T = np.hstack((np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10]), np.logspace(0, -8, 40)))
                for eps in T: 
                    res = minimize(wrapper, init_param, jac=wrapper.jac, bounds=bounds, method='L-BFGS-B', options={'disp': False}, args=[eps])
                    init_param = res.x
                logp_i = loo_per_point_logpdf(gpmodel)
                mean_loo = float(np.mean(logp_i))
                se_loo = float(np.std(logp_i) / np.sqrt(len(logp_i)))

                logp_list.append(logp_i)
                mean_loo_list.append(mean_loo)
                se_loo_list.append(se_loo)

            
            results.append({
                "lag": m,
                "mean_loo": np.mean(np.array(mean_loo_list)),
                "se_loo": np.mean(np.array(se_loo_list)),
                "n": int(len(logp_list[0])),
            })

        # Select p
        # Best by mean_loo
        best = max(results, key=lambda d: d["mean_loo"])
        selected = best
        if one_se:
            # Smallest p whose mean_loo >= best.mean_loo - best.se_loo
            threshold = best["mean_loo"] - best["se_loo"]
            eligible = sorted([r for r in results if r["mean_loo"] >= threshold], key=lambda d: d["lag"])
            if len(eligible) > 0:
                selected = eligible[0]
        best = selected

        self.reset_state(best["lag"]) # Reset the class variables with new lag
        return best["lag"]
