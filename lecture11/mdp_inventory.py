import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt

# Problem data
C = 6  # Warehouse capacity
n, m = C + 1, C + 1  # Number of states, inputs
T = 50  # Finite time horizon

npr.seed(1)

# Transition probabilities
p = np.array([0.1, 0.2, 0.7])  # Demand pdf, Prob[wt = 2, 1, 0]


# Stage costs
cs = 0.1  # Unit stock storage cost
cf = 1.0  # Fixed cost

G = np.zeros([n, m, T])

for t in range(T):
    for i in range(n):
        for j in range(m):
            if j > 0:
                G[i, j, t] = cs*i + cf
            else:
                G[i, j, t] = cs*i

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