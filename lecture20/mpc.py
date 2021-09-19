"""
mpc.py
Runs linear-quadratic MPC on an example system with 6 masses connected by springs.
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


def quadform(x, Q):
    return np.dot(x, np.dot(Q, x))


def cont2discrete(a, b, dt):
    # This does the same thing as the function scipy.signal.cont2discrete with method="zoh"

    # Build an exponential matrix
    em_upper = np.hstack((a, b))

    # Need to stack zeros under the a and b matrices
    em_lower = np.hstack((np.zeros((b.shape[1], a.shape[1])),
                          np.zeros((b.shape[1], b.shape[1]))))

    em = np.vstack((em_upper, em_lower))
    ms = sla.expm(dt*em)

    # Dispose of the lower rows
    ms = ms[:a.shape[0], :]

    # Split
    ad = ms[:, 0:a.shape[1]]
    bd = ms[:, a.shape[1]:]
    return ad, bd


def make_block_dynamics(A, B):
    n, m = B.shape
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
    return G, H


def tridiag(n, a, b, c=None):
    if c is None:
        c = b
    return np.diag(a*np.ones(n)) + np.diag(b*np.ones(n-1), 1) + np.diag(c*np.ones(n-1), -1)


def make_system(num_springs=6, spring_constant=1.0, damping_constant=0.01, DT=0.2):
    a = -2*spring_constant
    b = -2*damping_constant
    c = spring_constant
    d = damping_constant

    # Continuous time system matrices
    Acts = np.block([[np.zeros([num_springs, num_springs]), np.eye(num_springs)],
                     [tridiag(num_springs, a, c), tridiag(num_springs, b, d)]])

    strengths = np.array([1.0, 0.8, 0.5, 0.4, 0.6, 1.0])
    Bcts = np.vstack([np.zeros([num_springs, num_springs]), np.diag(strengths)])

    # Convert to discrete-time system
    A, B = cont2discrete(Acts, Bcts, DT)
    return A, B


def make_penalty(num_springs=6, pos_penalty=4.0, vel_penalty=1.0, con_penalty=1.0):
    n, m = 2*num_springs, num_springs
    Q = sla.block_diag(np.diag(pos_penalty*np.ones(num_springs)), np.diag(vel_penalty*np.ones(num_springs)))
    R = con_penalty*np.eye(m)
    S = np.zeros([n, m])
    return Q, R, S


def make_disturbance_hist(nsteps, scale=2.0):
    w_hist = DT*scale*(npr.rand(nsteps, n)-0.5)
    w_hist[:, 0:int(n/2)] = 0  # Do not apply any disturbance to the first n/2 states (positions)
    return w_hist


class System:
    def __init__(self, A, B, C=None, D=None):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self._G = None
        self._H = None

    @property
    def G(self):
        if self._G is None:
            self._G, self._H = make_block_dynamics(self.A, self.B)
        return self._G

    @property
    def H(self):
        if self._H is None:
            self._G, self._H = make_block_dynamics(self.A, self.B)
        return self._H

    def step(self, x, u, w, t=None):
        return np.dot(self.A, x) + np.dot(self.B, u) + w

    def observe(self, x, u, v, t=None):
        if self.C is None or self.D is None:
            raise ValueError('System needs C and D for observations!')
        return np.dot(self.C, x) + np.dot(self.D, u) + v


class StageCost:
    def __init__(self, Q, R, S):
        self.Q = Q
        self.R = R
        self.S = S
        self.QRS = np.block([[Q, S],
                             [S.T, R]])

    def cost(self, x, u, t=None):
        xu = np.hstack([x, u])
        return quadform(xu, self.QRS)


class Bound:
    def __init__(self, x_max, x_min, u_max, u_min):
        self.x_max = x_max
        self.x_min = x_min
        self.u_max = u_max
        self.u_min = u_min


class MPC_Problem_CVX:
    def __init__(self, system, stage_cost, bound, T):
        self.system = system
        self.stage_cost = stage_cost
        self.bound = bound
        self.T = T
        n, m = self.stage_cost.Q.shape[0], self.stage_cost.R.shape[0]
        self.X = cp.Variable((T, n))  # Predicted state trajectory
        self.U = cp.Variable((T, m))  # Planned control action sequence

    def make_objective(self, Q, R):
        state_objective = cp.sum([cp.quad_form(self.X[i], Q) for i in range(self.T)])  # Penalty on state deviations
        input_objective = cp.sum([cp.quad_form(self.U[i], R) for i in range(self.T)])  # Penalty on input magnitudes
        objective = cp.sum([state_objective, input_objective])
        return objective

    def make_dynamics_constraint(self, x, G, H):
        dynamics_constraint = cp.vec(self.X) == cp.matmul(G, x) + cp.matmul(H, cp.vec(self.U.T))
        return [dynamics_constraint]

    def make_box_constraint(self, X, x_max, x_min):
        constraint_upr = cp.max(X, axis=0) <= x_max
        constraint_lwr = cp.min(X, axis=0) >= x_min
        return [constraint_upr, constraint_lwr]

    def make_state_constraint(self):
        return self.make_box_constraint(self.X, self.bound.x_max, self.bound.x_min)

    def make_input_constraint(self):
        return self.make_box_constraint(self.U, self.bound.u_max, self.bound.u_min)

    def make_state_violation_objective(self):
        term_upr = cp.sum(cp.pos(cp.max(self.X, axis=0) - self.bound.x_max))
        term_lwr = cp.sum(cp.pos(cp.max(-self.X, axis=0) + self.bound.x_min))
        state_violation_penalty = term_upr + term_lwr
        return state_violation_penalty

    def make_problem(self, x, method=None):
        base_objective = self.make_objective(self.stage_cost.Q, self.stage_cost.R)
        dynamics_constraint = self.make_dynamics_constraint(x, self.system.G, self.system.H)
        state_constraint = self.make_state_constraint()
        input_constraint = self.make_input_constraint()

        if method is None:
            method = 'original'

        if method == 'original':
            objective = base_objective
            constraints = dynamics_constraint + state_constraint + input_constraint
        elif method == 'unconstrain':
            objective = base_objective
            constraints = dynamics_constraint
        elif method == 'penalize_state':
            state_violation_objective = self.make_state_violation_objective()
            penalty_scale = 10.0
            objective = base_objective + penalty_scale*state_violation_objective
            constraints = dynamics_constraint + input_constraint
        else:
            raise ValueError

        return cp.Problem(cp.Minimize(objective), constraints)


class MPC:
    def __init__(self, system, stage_cost, bound, T, infeasible_behavior='penalize_state'):
        self.system = system
        self.stage_cost = stage_cost
        self.bound = bound
        self.T = T
        self.infeasible_behavior = infeasible_behavior

    def policy_cvx(self, x):
        mpc_problem = MPC_Problem_CVX(self.system, self.stage_cost, self.bound, self.T)
        problem = mpc_problem.make_problem(x, method='original')

        # Solve open-loop convex optimization problem
        failed = False
        try:
            problem.solve()
        except Exception:
            failed = True
        if problem.status in ['infeasible', 'infeasible_inaccurate']:
            failed = True

        if failed:
            warn("Problem is infeasible!")
            if self.infeasible_behavior == 'terminate':
                raise Exception("Relax constraints.")
            elif self.infeasible_behavior == 'unconstrain':
                warn("Deactivating state and input constraints for this problem.")
                problem = mpc_problem.make_problem(x, method='unconstrain')
                problem.solve()
            elif self.infeasible_behavior == 'penalize_state':
                warn("Converting state constraints to objective penalties for this problem.")
                problem = mpc_problem.make_problem(x, method='penalize_state')
                problem.solve()

        # Use only the first input in the planned control sequence
        u = mpc_problem.U[0].value
        return u

    def policy(self, x, solver='cvx'):
        if solver == 'cvx':
            return self.policy_cvx(x)
        else:
            raise ValueError


def plot_trajectory(t_hist, x_hist, idx_show=None, label_string=None, ylim=None, x_nom=None, x_upr=None, x_lwr=None):
    T, n = x_hist.shape
    if idx_show is None:
        idx_show = np.arange(n)
    if label_string is None:
        label_string = 'x'
    nrows = len(idx_show)
    fig, ax = plt.subplots(nrows=nrows, figsize=(6, nrows*2))
    for i in idx_show:
        ax[i].step(t_hist, x_hist[:, i], lw=2, color='C0')
        if x_nom is not None:
            ax[i].step(t_hist, x_nom[i]*np.ones(nsteps), lw=2, color='tab:grey', linestyle='--', alpha=0.6)
        if x_upr is not None:
            ax[i].step(t_hist, x_upr[i]*np.ones(nsteps), lw=2, color='C3', linestyle='--', alpha=0.6)
        if x_lwr is not None:
            ax[i].step(t_hist, x_lwr[i]*np.ones(nsteps), lw=2, color='C3', linestyle='--', alpha=0.6)
        ax[i].set_ylim(ylim)
        ax[i].set_ylabel(r'$%s_%d$' % (label_string, i+1), rotation=0, labelpad=20)
    ax[-1].set_xlabel('Time')
    fig.tight_layout()
    return fig, ax


if __name__ == '__main__':
    # Problem data
    seed = 1
    npr.seed(seed)

    # Discretization time
    DT = 0.2

    # Dynamics
    A, B = make_system(DT=DT)
    system = System(A, B)

    # Penalties
    Q, R, S = make_penalty()
    stage_cost = StageCost(Q, R, S)

    # Dimensions
    n, m = B.shape

    # State and control targets
    x_nom = np.zeros(n)
    u_nom = np.zeros(m)

    # State and control limits
    x_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    x_min = -x_max
    u_max = np.array([8.0, 5.0, 3.0, 1.0, 1.0, 1.0])
    u_min = -0.5*u_max
    bound = Bound(x_max, x_min, u_max, u_min)

    # Initial state
    x0 = npr.choice([-1, 1], size=n) + (npr.rand(n)-0.5)

    # MPC parameters
    T = 20  # planning time horizon

    # Allocate history matrices
    nsteps = 100
    t_hist = np.arange(nsteps)
    w_hist = make_disturbance_hist(nsteps)
    x_hist = np.zeros([nsteps, n])  # state
    u_hist = np.zeros([nsteps, m])  # input
    J_hist = np.zeros(nsteps)  # stage cost

    # Initialize state and input trajectories
    X = np.zeros([T, n])
    U = np.zeros([T, m])
    x = x0

    # Instantiate the MPC controller object
    mpc = MPC(system, stage_cost, bound, T)

    wall_clock_start = time()
    for i in range(nsteps):
        print('Timestep: %3d' % i)

        # Use MPC state-feedback policy to generate an input
        u = mpc.policy(x)

        # Record state, input, cost
        x_hist[i] = x
        u_hist[i] = u
        J_hist[i] = stage_cost.cost(x, u)

        # State update
        w = w_hist[i]
        x = system.step(x, u, w)

    wall_clock_end = time()
    wall_clock_elapsed = wall_clock_end-wall_clock_start
    print('Solve time = %.3f seconds' % wall_clock_elapsed)
    print('Solve time / timestep = %.3f seconds' % (wall_clock_elapsed/nsteps))

    # Plotting
    plt.close('all')
    plt.style.use('../conlab.mplstyle')

    plot_trajectory(t_hist, x_hist, idx_show=[0, 1, 2, 3, 4, 5], label_string='x', x_nom=x_nom, x_upr=x_max, x_lwr=x_min)
    plot_trajectory(t_hist, u_hist, idx_show=[0, 1, 2, 3, 4, 5], label_string='u', x_nom=u_nom, x_upr=u_max, x_lwr=u_min)