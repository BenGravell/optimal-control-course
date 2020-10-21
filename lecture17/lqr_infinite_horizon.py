import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
from functools import reduce
import matplotlib.pyplot as plt


def mdot(*args):
    """Multiple dot product."""
    return reduce(np.dot, args)


def sympart(A):
    """Return the symmetric part of matrix A."""
    return 0.5*(A+A.T)


def gain(P, A, B, R):
    Hux = mdot(B.T, P, A)
    Huu = R+mdot(B.T, P, B)
    return -la.solve(Huu, Hux)


def lqr(A, B, Q, R, max_iters=None, threshold=1e-6):
    """
    Solve a infinite-horizon linear-quadratic regulation problem
    minimize sum(x[k].T, @ Q @ x[k] + u[k].T @ R @ u[k])
    subject to x[k+1] = A @ x[k] + B @ u[k]
    where
    A, B are dynamics matrices
    Q, R are positive semidefinite cost matrices
    """

    P = np.copy(Q)
    t = 0
    timeout = False
    converged = False
    while not converged or timeout:
        P_prev = np.copy(P)
        Hxx = Q+mdot(A.T, P, A)
        Hux = mdot(B.T, P, A)
        Huu = R+mdot(B.T, P, B)
        K = -la.solve(Huu, Hux)
        P = sympart(Hxx+np.dot(Hux.T, K))  # Take symmetric part to avoid numerical divergence issues
        converged = la.norm(P-P_prev, ord=2) < threshold
        if max_iters is not None:
            timeout = t > max_iters
        t -= 1
    return P, K


def rollout(x0, A, B, K):
    x_hist = np.zeros([T, n])
    u_hist = np.zeros([T, m])
    w_hist = np.zeros([T, n])
    x_hist[0] = np.copy(x0)
    for t in range(T-1):
        x = x_hist[t]
        u = K.dot(x)
        w = npr.multivariate_normal(wm, W)
        x_hist[t+1] = A.dot(x) + B.dot(u) + w
        u_hist[t] = u
        w_hist[t] = w
    return x_hist, u_hist, w_hist


def plot_hist(x_hist_all, u_hist_all, w_hist_all):
    fig, ax = plt.subplots(nrows=3)
    ylabels = ['State', 'Input', 'Disturbance']
    dims = [n, m, n]
    for i, (hist, ylabel, d) in enumerate(zip([x_hist_all, u_hist_all, w_hist_all], ylabels, dims)):
        for k in range(d):
            ax[i].plot(hist[:, :, k].T, color='C%d' % k, alpha=0.8)
        ax[i].set_ylabel(ylabel)
    ax[-1].set_xlabel('Time')
    return fig, ax


if __name__ == "__main__":
    # Random seed
    npr.seed(4)

    # Problem data
    n = 5
    m = 2
    T = 30

    x0 = 5*npr.randn(n)

    A = npr.randn(n, n)
    # Normalize A to fix the spectral radius to r
    r = 2
    A = A*r/(max(abs(la.eig(A)[0])))
    B = npr.randn(n, m)

    Q = 1*np.eye(n)
    R = 1*np.eye(m)
    W = 0.1*np.eye(n)
    wm = np.zeros(n)

    # Solve DARE manually
    P, K = lqr(A, B, Q, R)

    # Solve DARE via SciPy
    Pscipy = sla.solve_discrete_are(A, B, Q, R)
    Kscipy = gain(Pscipy, A, B, R)

    # Compare DARE solutions
    print('Spectral norm of difference between P and Pscipy = %f' % la.norm(P-Pscipy, ord=2))
    print('Spectral norm of difference between K and Kscipy = %f' % la.norm(K-Kscipy, ord=2))

    # Simulate
    N = 10
    x_hist_all, u_hist_all, w_hist_all = np.zeros([N, T, n]), np.zeros([N, T, m]), np.zeros([N, T, n])
    for i in range(N):
        x_hist_all[i], u_hist_all[i], w_hist_all[i] = rollout(x0, A, B, K)

    # Plot
    plt.close('all')
    plot_hist(x_hist_all, u_hist_all, w_hist_all)
