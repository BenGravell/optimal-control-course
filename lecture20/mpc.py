"""
mpc.py
Runs 100 steps of MPC on an example system with 6 masses connected by springs.
"""

import cvxpy as cp
import numpy as np
import numpy.random as npr
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt
from time import time
from functools import reduce
import warnings
from warnings import warn
warnings.simplefilter('always', UserWarning)


def mdot(*args):
    """Multiple dot product."""
    return reduce(np.dot, args)


def mpc_iteration(x, G, H, Q, R, T, infeasible_behavior='deactivate_constraints'):
    n, m = Q.shape[0], R.shape[0]

    # Solve open-loop convex optimization problem
    # Decision variables
    X = cp.Variable((T, n))  # Predicted state trajectory
    U = cp.Variable((T, m))  # Planned control action sequence
    # Objective
    state_objective = cp.sum([cp.quad_form(X[i], Q) for i in range(T)])  # Penalty on state deviations
    input_objective = cp.sum([cp.quad_form(U[i], R) for i in range(T)])  # Penalty on input magnitudes
    objective = cp.atoms.affine.add_expr.AddExpression([state_objective, input_objective])
    # Constraints
    dynamics_constraint = cp.vec(X) == cp.matmul(G, x) + cp.matmul(H, cp.vec(U.T))
    state_constraint_upr = cp.max(X, axis=0) <= x_max
    state_constraint_lwr = cp.min(X, axis=0) >= x_min
    input_constraint_upr = cp.max(U, axis=0) <= u_max
    input_constraint_lwr = cp.min(U, axis=0) >= u_min
    constraints = [dynamics_constraint, state_constraint_upr, state_constraint_lwr,
                   input_constraint_upr, input_constraint_lwr]
    # Form the problem and solve
    problem = cp.Problem(cp.Minimize(objective), constraints)
    failed = False
    try:
        problem.solve()
    except Exception:
        failed = True
    if problem.status in ['infeasible', 'infeasible_inaccurate']:
        failed = True
    if failed:
        if infeasible_behavior == 'terminate':
            raise Exception('Problem is infeasible! Relax constraints.')
        elif infeasible_behavior == 'deactivate_constraints':
            warn('Problem is infeasible! Deactivating state and input constraints for this problem.')
            constraints = [dynamics_constraint]
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve()

    # Implement only first input in the sequence
    return U[0].value


def plot_trajectory(t_hist, x_hist, idx_show=None, label_string=None, ylim=None, x_upr=None, x_lwr=None):
    T, n = x_hist.shape
    if idx_show is None:
        idx_show = np.arange(n)
    if label_string is None:
        label_string = 'x'
    fig, ax = plt.subplots(nrows=len(idx_show))
    for i in idx_show:
        ax[i].step(t_hist, x_hist[:, i], lw=2, color='C0')
        if x_upr is not None:
            ax[i].step(t_hist, x_upr[i]*np.ones(nsteps), lw=2, color='tab:grey', linestyle='--', alpha=0.8)
        if x_lwr is not None:
            ax[i].step(t_hist, x_lwr[i]*np.ones(nsteps), lw=2, color='tab:grey', linestyle='--', alpha=0.8)
        ax[i].set_ylim(ylim)
        ax[i].set_ylabel(r'$%s_%d$' % (label_string, i+1), rotation=0, labelpad=20)
    ax[-1].set_xlabel('Time')
    fig.tight_layout()
    return fig, ax


# Problem data
seed = None
npr.seed(seed)
spring_constant = 1
damping_constant = 0
a = -2*spring_constant
b = -2*damping_constant
c = spring_constant
d = damping_constant

n = 12  # State dimension
n2 = int(n/2)
m = 3  # Input dimension

# Continuous time system matrices
Acts1 = np.hstack([np.zeros([n2, n2]), np.eye(n2)])
Acts2 = np.array([[a, c, 0, 0, 0, 0, b, d, 0, 0, 0, 0],
                  [c, a, c, 0, 0, 0, d, b, d, 0, 0, 0],
                  [0, c, a, c, 0, 0, 0, d, b, d, 0, 0],
                  [0, 0, c, a, c, 0, 0, 0, d, b, d, 0],
                  [0, 0, 0, c, a, c, 0, 0, 0, d, b, d],
                  [0, 0, 0, 0, c, a, 0, 0, 0, 0, d, b]])
Acts = np.vstack([Acts1, Acts2])

Bcts1 = np.zeros([n2, m])
Bcts2 = np.array([[1, 0, 0],
                  [-1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, -1, 0],
                  [0, 0, -1]])
Bcts = np.vstack([Bcts1, Bcts2])

# Convert to discrete-time system
ts = 0.5  # sampling time
A = sla.expm(ts*Acts)
B = np.dot(la.solve(Acts, (sla.expm(ts*Acts)-np.eye(n))), Bcts)

# Penalty matrices
Q = sla.block_diag(4*np.diag(np.ones(n2)), np.diag(np.ones(n2)))
R = np.eye(m)

# State and control limits
x_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
x_min = -x_max
u_max = np.array([8.0, 5.0, 3.0])
u_min = -0.5*u_max

# Disturbance trajectory
nsteps = 100
w = npr.rand(nsteps, n)-0.5
w[:, 0:n2] = 0  # Do not apply any disturbance to the first n/2 states

# Initial state
x0 = npr.randn(n)

# MPC parameters
T = 10  # time horizon

# Allocate history matrices
x_hist = np.zeros([nsteps, n])  # state
u_hist = np.zeros([nsteps, m])  # input
J_hist = np.zeros(nsteps)  # stage cost

# Initial state and input trajectories
X = np.zeros([T, n])
U = np.zeros([T, m])
x = x0

# Dynamics block matrices
G = np.zeros([n*T, n])
H = np.eye(n*T)
BB = np.kron(np.eye(T), B)
for i in range(T):
    G[i*n:n*(i+1)] = la.matrix_power(A, i+1)
for i in range(T):
    for j in range(T):
        if i > j:
            H[i*n:n*(i+1), j*n:n*(j+1)] = la.matrix_power(A, i-j)
H = np.dot(H, BB)

wall_clock_start = time()
for i in range(nsteps):
    print('Timestep: %3d' % i)
    # Use an MPC state-feedback policy to generate an input
    u = mpc_iteration(x, G, H, Q, R, T)

    # Record state, input, cost
    x_hist[i] = x
    u_hist[i] = u
    J_hist[i] = mdot(x.T, Q, x) + mdot(u.T, R, u)

    # State update
    x = np.dot(A, x) + np.dot(B, u) + w[i]
wall_clock_end = time()
wall_clock_elapsed = wall_clock_end-wall_clock_start
print('Solve time = %.3f seconds' % wall_clock_elapsed)

# Plotting
plt.close('all')
plt.style.use('fivethirtyeight')
t_hist = np.arange(nsteps)
plot_trajectory(t_hist, x_hist, idx_show=[0, 1, 2], label_string='x', x_upr=x_max, x_lwr=x_min)
plot_trajectory(t_hist, u_hist, idx_show=[0, 1, 2], label_string='u', x_upr=u_max, x_lwr=u_min)
