import numpy as np
import pandas as pd
import networkx as nx
import random
import os
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI

"""
Generate data using tigramite library. Can be used for simulating systems with both lagged and contemporaneous edges. 
"""
PATH = os.getcwd()
seed = 10
np.random.seed(seed)
random.seed(seed)

def sample_edges(num_vars, L, max_lag=5, p_contemp=0.3):
    edges = []
    G0 = nx.DiGraph()
    G0.add_nodes_from(range(num_vars))  # ensure all nodes exist

    while len(edges) < L:
        i, j = random.sample(range(num_vars), 2)

        if random.random() < p_contemp:
            tau = 0
            # Prevent contemporaneous confounders:
            # node j can have at most one incoming contemporaneous edge
            if G0.in_degree(j) == 0:
                G0.add_edge(i, j)
                if nx.is_directed_acyclic_graph(G0):
                    if (i, j, tau) not in edges:
                        edges.append((i, j, tau))
                else:
                    G0.remove_edge(i, j)  # reject cycle
            # else: skip if j already has an incoming lag-0 edge
        else:
            tau = random.randint(1, max_lag)
            if (i, j, tau) not in edges:
                edges.append((i, j, tau))

    return edges

auto_coeff = 0.8
coeff = 0.4
T = 500
nvar = 5
burnin = 100
Ntotal = T + burnin
L = int(np.floor(1.5 * nvar)) # number of cross-links 
maxLag = 1
def lin(x): return x
def nonlin(x): return (1 + 5*x*np.exp(-(x**2)/20))#*x

cnnx = []
for i in range(nvar):
    for j in range(nvar):
        if j != i:
            cnnx.append((i,j))

# Randomly pick L links
edges = sample_edges(nvar, L, max_lag=maxLag, p_contemp=0.3) # Set p_contemp=0 for simulating lag only systems
print("edges: ", edges)
print("Num edges: ", len(edges))

links = {}
for l in range(nvar):
    links[l] = [((l, -1), auto_coeff, lin)] + [((cause, -lag), coeff, nonlin) for (cause, eff, lag) in edges if eff == l]
print("Links: ", links)

mc_runs = 10
all_data = []
random_state = np.random.RandomState(seed)
m = 0
while m < mc_runs:
    try:
        # Specify noise distributions
        noises = [random_state.randn for j in links.keys()]
        # student t noise
        # noises = [
        #     (lambda size, rs=random_state: rs.standard_t(df=2, size=size))
        #     for _ in links.keys()
        # ]
        data, nonstationarity_indicator = toys.structural_causal_process(
            links=links, T=T, noises=noises, seed=seed)
        print("Nonstationary: ", nonstationarity_indicator)
        T, N = data.shape
        if m == 0:
            print(data)

        all_data.append(data)
        m += 1
    except:
        print("Failed to generate")
all_data = np.array(all_data).reshape(mc_runs * T, N)
print(all_data.shape)

# Initialize dataframe object, specify variable names
var_names = [r'$X^{%d}$' % j for j in range(N)]
dataframe = pp.DataFrame(data, var_names=var_names)

df_pandas = pd.DataFrame(
    all_data
)
print(df_pandas)

PATH_CONTEMPORANEOUS = os.path.join(PATH, 'contemporaneous')
name = str(nvar) + "nonlinear_lag" + str(maxLag)
if not os.path.exists(PATH_CONTEMPORANEOUS):
    os.makedirs(PATH_CONTEMPORANEOUS)
df_pandas.to_csv(PATH_CONTEMPORANEOUS + "/" + name + "_n" + str(T) + ".csv", index=False, header=None)

T_new = 250
rows_per_exp = T
indices_to_keep = np.concatenate([
    np.arange(i*rows_per_exp, i*rows_per_exp + T_new)
    for i in range(mc_runs)
])
df_new = df_pandas.iloc[indices_to_keep, :]
df_new.to_csv(PATH_CONTEMPORANEOUS +  "/" + name + "_n" + str(T_new) + ".csv", index=False, header=None)