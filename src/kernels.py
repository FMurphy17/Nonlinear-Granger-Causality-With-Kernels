import GPy

def rbf_kernel(X1, X2, lengthscale=0.5):
    """Computes the RBF kernel matrix."""
    K=GPy.kern.RBF(X1.shape[1],lengthscale=lengthscale,ARD=False)
    return K.K(X1,X2)

def matern(X1, X2, lengthscale=0.5):
    """Computes the RBF kernel matrix."""
    K=GPy.kern.Matern32(X1.shape[1],lengthscale=lengthscale,ARD=False)
    return K.K(X1,X2)

def pol_kernel(X1, X2, p=2):
    K=GPy.kern.Poly(X1.shape[1],order=p)
    return K.K(X1,X2)

def linear_kernel(X1, X2):
    return X1.T@X2
