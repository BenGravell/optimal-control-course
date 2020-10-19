import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt

# Problem data
n, m = 500, 50  # Number of states, inputs
T = 25  # Finite time horizon

npr.seed(1)

# Transition probabilities
P = npr.rand(n, n, m)
# Normalize to get a row-stochastic matrix for each P[:, :, i] where i indexes the action
for i in range(m):
    for j in range(n):
        P[j, :, i] /= np.sum(P[j, :, i])




# Stage costs
G = np.zeros([n, m, T])

# Terminal cost
GT = np.zeros(n)

# Initialize

# Recurse backwards in time
for t in range(T, -1, -1):
    for i in range(n):
        Ju = np.full(m, np.inf)
        for j in range():
            pu =
            pu[] =
            Ju[]
        u_opt[i, t] = np.argmin(Ju)
        J[i, t] = Ju[u_opt]