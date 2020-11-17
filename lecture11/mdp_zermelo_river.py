import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt

# River grid size
river_wid, river_len = 20, 20
island_size = 2

n = river_wid*river_len  # Number of states
m = 4  # Number of inputs
T = 30  # Time horizon


# State space
X = np.zeros([river_wid, river_len])  # Free-flow river
island_slice_wid = slice(int(river_wid/2-island_size), int(river_wid/2+island_size))
island_slice_len = slice(int(river_len/2-island_size), int(river_len/2+island_size))
X[island_slice_wid, island_slice_len] = 1  # Island
X[:, 0] = 2  # Waterfall


# State-and-input-dependent state transition probabilities
# Input-dependent probabilities when state is in free-flow of river
p_free = np.array([[0.6, 0.1, 0.3, 0.0],
                   [0.1, 0.6, 0.3, 0.0],
                   [0.1, 0.1, 0.8, 0.0],
                   [0.1, 0.1, 0.2, 0.6]])

# Input-dependent probabilities when state is on top, bottom boundaries
p_tpbt = np.array([[0.6, 0.1, 0.3, 0.0],
                   [0.1, 0.6, 0.3, 0.0],
                   [0.1, 0.1, 0.8, 0.0],
                   [0.1, 0.1, 0.2, 0.6]])

# Input-dependent probabilities when state is on corner boundaries
# p_crnr =

# TODO - finish, incomplete!


# Stage costs
G = np.zeros([river_wid, river_len])
G[island_slice_wid, island_slice_len] = 1


# Dynamic programming
J = np.zeros([river_wid, river_len, T+1])
u_opt = np.zeros([river_wid, river_len, T])


# initialize
J[:, :, -1] = G

# Recurse backwards in time
for t in range(T, -1, -1):
    for i in range(river_wid):
        for j in range(river_len):
            Ju = np.full(m, np.inf)
            for k in range(m):
                # Map local up-down-left-right state to global river state
                pu = np.zeros([river_wid, river_len])
                # Loop over the 4 possible up-down-left-right after-states
                for kk, (ii, jj) in enumerate(((1, 0), (-1, 0), (0, -1), (0, 1))):
                    # Map local coordinates to global coordinates and clip to bounds
                    ix, jy = max(min(i+ii, river_wid), 0), max(min(j+jj, river_len), 0)
                    pu[ix, jy] = p_free[k, kk]

                Ju[k] = G[i, j] + np.sum(pu*J[i, j, t+1])
            u_opt[i, j, t] = np.argmin(Ju)
            J[i, j, t] = Ju[u_opt]


