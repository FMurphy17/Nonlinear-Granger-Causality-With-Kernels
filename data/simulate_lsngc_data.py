import numpy as np
import csv
import os

"""
Genrate data according to the simulated models described in Wismüller, A., Dsouza, A. M., Vosoughi, M. A., et al. (2021). 
doi.org/10.1038/s41598-021-87316-6
"""

np.random.seed(10)

def generate_data(N, exp, writer):
    exp_nvar = {'nonlinear5': 5, 'linear5': 5, '3fanout': 3, '3fanin': 3}
    X = np.random.randn(exp_nvar[str(exp)], N)
    r = np.sqrt(2)
    sigma = 0.0001

    if exp == "nonlinear5":
        # Nonlinear 5 ts system
        for i in range(3, N):
            X[0, i] = X[0, i] + 0.95 * r * X[0, i - 1] - 0.9025 * X[0, i - 2] + np.random.randn() * sigma
            X[1, i] = X[1, i] + 0.5 * X[0, i - 2] ** 2 + np.random.randn() * sigma
            X[2, i] = X[2, i] - 0.4 * X[0, i - 3] + np.random.randn() * sigma
            X[3, i] = X[3, i] - 0.5 * X[0, i - 2] ** 2 + 0.5 * r * X[3, i - 1] + 0.25 * r * X[4, i - 1] + np.random.randn() * sigma
            X[4, i] = X[4, i] - 0.5 * r * X[3, i - 1] + 0.5 * r * X[4, i - 1] + np.random.randn() * sigma

    elif exp == "linear5":
        # Linear 5 ts system
        for i in range(3, N):
            X[0, i] = X[0, i] + 0.95 * r * X[0, i - 1] - 0.9025 * X[0, i - 2] + np.random.randn() * sigma
            X[1, i] = X[1, i] + 0.5 * X[0, i - 2] + np.random.randn() * sigma
            X[2, i] = X[2, i] - 0.4 * X[0, i - 3] + np.random.randn() * sigma
            X[3, i] = X[3, i] - 0.5 * X[0, i - 2] + 0.25 * r * X[3, i - 1] + 0.25 * r * X[4, i - 1] + np.random.randn() * sigma
            X[4, i] = X[4, i] - 0.25 * r * X[3, i - 1] + 0.25 * r * X[4, i - 1] + np.random.randn() * sigma

    elif exp == "3fanout":
        # fan-out
        X[0, 0] = np.random.uniform(0, 1, 1)  
        X[1, 0] = np.random.uniform(0, 1, 1)  
        X[2, 0] = np.random.uniform(0, 1, 1)  
        for i in range(1,N):
            X[0, i] = X[0, i-1]*(4 - 4*X[0,i-1] - 0*X[1, i-1] - 0*X[2, i-1])
            X[1, i] = X[1, i-1]*(3.1 - 0.21*X[0,i-1] - 3.1*X[1, i-1] - 0*X[2, i-1])
            X[2, i] = X[2, i-1]*(2.12 - -0.636*X[0,i-1] - 0*X[1, i-1] - 2.12*X[2, i-1])

    elif exp == "3fanin":
        # fan-in
        X[0, 0] = np.random.uniform(0, 1, 1)  
        X[1, 0] = np.random.uniform(0, 1, 1)  
        X[2, 0] = np.random.uniform(0, 1, 1)  
        W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
        for i in range(1,N):
            X[0, i] = X[0, i-1]*(4 - 4*X[0,i-1] - 0*X[1, i-1] - 0*X[2, i-1])
            X[1, i] = X[1, i-1]*(3.6 - 0*X[0,i-1] - 3.6*X[1, i-1] - 0*X[2, i-1])
            X[2, i] = X[2, i-1]*(2.12 - 0.636*X[0,i-1] - -0.636*X[1, i-1] - 2.12*X[2, i-1])
    
    data = X[:, 50:].T # dropout first 50 data points
    if np.isnan(data).any():
        return False
    else:
        writer.writerows(data)
        return True

if __name__ == "__main__":
    PATH = os.getcwd()
    for i in [250, 500]:
        num_samples = i
        dropout = num_samples + 50

        name = "nonlinear5"
        mc_samples = 100

        if not os.path.exists(PATH + "/" + name):
            os.makedirs(PATH + "/" + name)
        f = open(PATH + "/" + name + '/n' + str(num_samples) + "_mc" + str(mc_samples) + ".csv", 'w', newline='')
        writer = csv.writer(f)

        i = 0
        while i < mc_samples:
            res = generate_data(dropout, name, writer)
            if res == False:
                i = i
            else:
                i += 1

        f.close()
