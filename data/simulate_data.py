import numpy as np
import sys
import csv
import os
import networkx as nx
import pandas as pd
import scipy.integrate as intgr
import matplotlib.pyplot as plt


np.random.seed(10)

def mediator(N):
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    #W1, W2, W3 = np.random.standard_t(df=2, size=N), np.random.standard_t(df=2, size=N), np.random.standard_t(df=2, size=N)
    for n in range(N-1):
        q1[n+1] = np.sin(q2[n]) + 0.001*W1[n]
        q2[n+1] = np.cos(q3[n]) + 0.01*W2[n]
        q3[n+1] = 0.5*q3[n] + 0.1*W3[n]
    return q1, q2, q3

def confounder(N):
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    #W1, W2, W3 = np.random.standard_t(df=2, size=N), np.random.standard_t(df=2, size=N), np.random.standard_t(df=2, size=N)
    for n in range(N-1):
        q1[n+1] = np.sin(q1[n] + q3[n]) + 0.01*W1[n]
        q2[n+1] = np.cos(q2[n] - q3[n]) + 0.01*W2[n]
        q3[n+1] = 0.5*q3[n] + 0.1*W3[n]
    return q1, q2, q3

def synergistic_collider(N):
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    #W1, W2, W3 = np.random.standard_t(df=2, size=N), np.random.standard_t(df=2, size=N), np.random.standard_t(df=2, size=N)
    for n in range(N-1):
        q1[n+1] = np.sin(q2[n] * q3[n]) + 0.001*W1[n]
        q2[n+1] = 0.5*q2[n] + 0.1*W2[n]
        q3[n+1] = 0.5*q3[n] + 0.1*W3[n]
    return q1, q2, q3

def redundant_collider(N):
    q1, q2, q3 = np.zeros(N), np.zeros(N), np.zeros(N)
    W1, W2, W3 = np.random.normal(0, 1, N), np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    for n in range(N-1):
        q1[n+1] = 0.3*q1[n] + (np.sin(q2[n]*q3[n]) + 0.001*W1[n])
        q2[n+1] = 0.5*q2[n] + 0.1*W2[n]
        q3[n+1] = q2[n+1]
    return q1, q2, q3

def moran_effect(N, r1=3.4, r2=2.9, s1=0.4, s2=0.35, D1=4, D2=4, psi1=0.5, psi2=0.6, R0=1, N0=0.5):
    # Initialize arrays for the state variables with zeros
    R1 = np.zeros(N)
    R2 = np.zeros(N)
    N1 = np.zeros(N)
    N2 = np.zeros(N)
    v = np.random.normal(0, 1, N)
    
    # Initial values for R1, R2, N1, and N2
    R1[0] = R0
    R2[0] = R0
    N1[0] = N0
    N2[0] = N0

    # Simulate the system over N time steps
    for t in range(4, N-1):
        
        R1[t+1] = r1 * N1[t] * (1 - N1[t]) * np.exp(-psi1 * v[t])
        N1[t+1] = s1 * N1[t] + max(R1[t - D1], 0)
        
        R2[t+1] = r2 * N2[t] * (1 - N2[t]) * np.exp(-psi2 * v[t])
        N2[t+1] = s2 * N2[t] + max(R2[t - D2], 0)
    
    return R1, N1, R2, N2, v

def logistic(N, q10 = 0.2, q20=0.4, r1=3.8, r2=3.5, beta2to1=0.02, beta1to2=0.1):
    # Initialize arrays for the state variables with zeros 
    q1 = np.zeros(N)
    q2 = np.zeros(N)
    q1[0] = q10 
    q2[0] = q20 

    # Simulate the system over N time steps
    for t in range(0, N-1):
        q1[t+1] = q1[t] * (r1 - r1 * q1[t] - beta2to1 * q2[t]) + np.random.normal(0, 1e-8)
        q2[t+1] = q2[t] * (r2 - r2 * q2[t] - beta1to2 * q1[t]) + np.random.normal(0, 1e-8)
    return q1, q2

def stochastic_linear(N):
    # Initialize arrays for the state variables with zeros
    q1 = np.zeros(N)
    q2 = np.zeros(N)
    
    # Initialize arrays for the noise terms with random normal values
    eta1 = np.random.normal(0, 1, N)
    eta2 = np.random.normal(0, 1, N)
    
    # Simulate the system over N time steps
    for t in range(1, N-1):
        q1[t+1] = 0.95 * np.sqrt(2) * q1[t] - 0.9025 * q1[t-1] + eta1[t]
        q2[t+1] = 0.5 * q1[t-1] + eta2[t]
    return q1, q2
    

def stochastic_nonlinear(N):
    # Initialize arrays for the state variables with zeros
    q1 = np.zeros(N)
    q2 = np.zeros(N)
    
    # Initialize arrays for the noise terms with random normal values
    eta1 = np.random.normal(0, np.sqrt(0.4), N)
    eta2 = np.random.normal(0, np.sqrt(0.4), N)
    
    # Simulate the system over N time steps
    for t in range(1, N-1):
        q1[t+1] = 3.4 * q1[t] * (1 - q1[t]**2) * np.exp(-q1[t-1]**2) + eta1[t]
        q2[t+1] = 3.4 * q2[t] * (1 - q2[t]**2) * np.exp(-q2[t]**2) + q1[t-1] * q2[t] / 2 + eta2[t]
    return q1, q2

def synchronization(N, expid):
    q1 = np.full(N, 1e-5)
    q2 = np.full(N, 1e-5)
    q3 = np.full(N, 1e-5)

    eta1 = np.random.normal(0, 1e-5, N)
    eta2 = np.random.normal(0, 1e-5, N)
    eta3 = np.random.normal(0, 1e-5, N)
    # No Coupling
    if expid == "no_coupling":
        c1_2 = 0
        c12_3 = 0
    # One-way intermediate coupling
    elif expid == "1wayintermediate":
        c1_2 = 0.1
        c12_3 = 0
    # One-way strong coupling
    elif expid == "1waystrong":
        c1_2 = 1
        c12_3 = 0
    # Two-way strong coupling
    elif expid == "2waystrong":
        c1_2 = 0
        c12_3 = 1
    # Three-way strong coupling
    else:
        c1_2 = 1
        c12_3 = 1

    for t in range(1,N-1):
        q1[t+1] = 3.68 * q1[t] * (1 - q1[t]) + eta1[t]
        q2[t+1] = 3.67 * (q2[t] + c1_2 * q1[t])/(1 + c1_2) * (1 - (q2[t] + c1_2 * q1[t])/(1 + c1_2)) + eta2[t]
        q3[t+1] = 3.78 * (q3[t] + c12_3 * q1[t] + c12_3 * q2[t])/(1 + 2 * c12_3) * (1 - (q3[t] + c12_3 * q1[t] + c12_3 * q2[t])/(1 + 2 * c12_3)) + eta3[t]

    return q1, q2, q3


def eight_species(N, noise=0.005):
    
    # Initialize arrays for the state variables with zeros
    q1,q2,q3,q4,q5,q6,q7,q8 = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)
    
    # Initialize arrays for the noise terms with random normal values
    eta1,eta2,eta3,eta4,eta5,eta6,eta7,eta8 = np.random.normal(0, noise, N), np.random.normal(0, 0.005, N), np.random.normal(0, 0.005, N), np.random.normal(0, 0.005, N), np.random.normal(0, 0.005, N), np.random.normal(0, 0.005, N), np.random.normal(0, 0.005, N), np.random.normal(0, 0.005, N)

    # Set initial conditions
    q1[0], q2[0], q3[0], q4[0], q5[0], q6[0], q7[0], q8[0] = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    
    # Simulate the system over N time steps
    for t in range(0, N-1):
        q1[t+1] = q1[t] * (3.9 - 3.9 * q1[t]) + eta1[t]
        q2[t+1] = q2[t] * (3.5 - 3.5 * q2[t]) + eta2[t]
        q3[t+1] = q3[t] * (3.62 - 3.62 * q3[t] - 0.35 * q1[t] - 0.35 * q2[t]) + eta3[t]
        q4[t+1] = q4[t] * (3.75 - 3.75 * q4[t] - 0.35 * q2[t]) + eta4[t]
        q5[t+1] = q5[t] * (3.65 - 3.65 * q5[t] - 0.35 * q3[t]) + eta5[t]
        q6[t+1] = q6[t] * (3.72 - 3.72 * q6[t] - 0.35 * q3[t]) + eta6[t]
        q7[t+1] = q7[t] * (3.57 - 3.57 * q7[t] - 0.35 * q6[t]) + eta7[t]
        q8[t+1] = q8[t] * (3.68 - 3.68 * q8[t] - 0.35 * q6[t]) + eta8[t]

    return q1, q2, q3, q4, q5, q6, q7, q8


if __name__ == "__main__":
    PATH = os.getcwd()
    for n in [250, 500]:
        num_samples = n 
        num_samples_dropout = num_samples + 50 # 50 points dropout
        mc_samples = 100

        name = "synergistic_collider"
        if not os.path.exists(PATH + "/" + name ):
            os.makedirs(PATH + "/" + name)
        f = open(PATH + "/" + name +'/n' + str(num_samples) + "_mc" + str(mc_samples) + ".csv", 'w', newline='')
        writer = csv.writer(f)
        i = 0
        while i < mc_samples:    
            q1, q2, q3 = synergistic_collider(num_samples_dropout)
            if np.any(np.isnan(q1) | np.isinf(q1)) or np.any(np.isnan(q2) | np.isinf(q2)) or np.any(np.isnan(q3) | np.isinf(q3)):
                print("Error in generating data, rerunning MC run!")
            else:
                # Discard first 50 time points 
                q1 = q1[50:]
                q2 = q2[50:]
                q3 = q3[50:]
                X = np.vstack((q1, q2, q3))
                writer.writerows(X.T)
                i += 1
        f.close()