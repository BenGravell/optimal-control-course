import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Problem data
height, width = 20, 20
m = 2  # Number of inputs
p = 4  # Number of disturbances

# States
states = np.zeros([height, width])
# Target states
target_states = [[4, 4], [16, 9], [9, 14]]
for target_state in target_states:
    states[target_state] = 1

# Stage costs
stage_cost = np.zeros([height, width, m])
# Waiting cost
stage_cost[:, :, 0] = 1
# Stopping cost
stopping_costs = [-120, -70, -150]
for target_state, stopping_cost in zip(target_states, stopping_costs):
    stage_cost[target_state[0], target_state[1], 1] = stopping_cost

# Dynamic programming
# Optimal cost functions
J = np.zeros([height, width])
J_done = 0
# Optimal policy
pi_opt = np.zeros([height, width])


def state_transition(i, j, l):
    # Disturbance state transitions
    ds = (0, 1), (0, -1), (-1, 0), (1, 0)
    d = ds[l]
    return i+d[0], j+d[1]


def expected_cost(i, j):
    expected_cost = np.zeros(m)
    cost = np.zeros([m, p])
    num_nbrs = 4 - (i == 0 or i == height) - (j == 0 or j == width)
    for k in range(m):
        for l in range(p):
            if k == 0:
                ii, jj = state_transition(i, j, l)
                if not(0 <= ii < height and 0 <= jj < width):
                    continue
                J_current = J[ii, jj]
            else:
                J_current = 0
            # Evaluate sum of current and future costs
            cost[k] = stage_cost[i, j, k] + J_current
            # Evaluate expected sum of current and future costs
            expected_cost[k] += (1/num_nbrs)*cost[k, l]
    return expected_cost


# Value iteration
converged = False
convergence_threshold = 1e-3
while not converged:
    J_old = np.copy(J)
    for i in range(height):
        for j in range(width):
            # Minimize over expected sum of current and future costs
            J[i, j] = np.min(expected_cost(i, j))
    if la.norm(J - J_old) < convergence_threshold:
        converged = True

# Find the optimal policy
for i in range(height):
    for j in range(width):
        # Minimize over expected sum of current and future costs
        pi_opt[i, j] = np.argmin(expected_cost(i, j))

# Plot optimal policy
plt.figure()
plt.imshow(pi_opt)

# Plot optimal cost
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(np.arange(height), np.arange(width))
surf = ax.plot_surface(X, Y, J, cmap=cm.coolwarm, linewidth=0, antialiased=False)
