import numpy as np
import numpy.linalg as la
import numpy.random as npr
from functools import reduce
import matplotlib.pyplot as plt


def mdot(*args):
    """Multiple dot product."""
    return reduce(np.dot, args)


def sympart(A):
    """Return the symmetric part of matrix A."""
    return 0.5*(A+A.T)


def lqr(A, B, Q, R, Qf, T):
    """
    Solve a finite-horizon linear-quadratic regulation problem
    minimize sum(x[k].T, @ Q @ x[k] + u[k].T @ R @ u[k]) + x[T].T, @ Qf @ x[T]
    subject to x[k+1] = A[k] @ x[k] + B[k] @ u[k]
    where
    A, B are dynamics matrices
    Q, R, Qf are positive semidefinite cost matrices
    """

    n, m = Q.shape[0], R.shape[0]
    P_hist = np.zeros([T+1, n, n])
    K_hist = np.zeros([T, m, n])
    P = np.copy(Qf)
    for t in range(T-1, -1, -1):
        P_hist[t+1] = P
        Hxx = Q+mdot(A.T, P, A)
        Hux = mdot(B.T, P, A)
        Huu = R+mdot(B.T, P, B)
        K = -la.solve(Huu, Hux)
        K_hist[t] = K
        P = sympart(Hxx+np.dot(Hux.T, K))  # Take symmetric part to avoid numerical divergence issues
    return P_hist, K_hist


def rollout(x0, A, B, T):
    x_hist = np.zeros([T, n])
    u_hist = np.zeros([T, m])
    w_hist = np.zeros([T, n])
    x_hist[0] = np.copy(x0)
    for t in range(T-1):
        x = x_hist[t]
        u = K_hist[t].dot(x)
        w = npr.multivariate_normal(wm, W)
        x_hist[t+1] = A.dot(x) + B.dot(u) + w
        u_hist[t] = u
        w_hist[t] = w
    return x_hist, u_hist, w_hist


def plot_gain_hist(K_hist):
    n, m = K_hist.shape[2], K_hist.shape[1]
    fig, ax = plt.subplots()
    for i in range(m):
        for j in range(n):
            ax.plot(K_hist[:, i, j], label='(%1d, %1d)' % (i, j))
    ax.legend()
    ax.set_title('Entrywise Gains')
    return fig, ax


def plot_hist(x_hist, u_hist, w_hist):
    fig, ax = plt.subplots(nrows=3)
    ylabels = ['State', 'Input', 'Disturbance']
    for i, (hist, ylabel) in enumerate(zip([x_hist, u_hist, w_hist], ylabels)):
        ax[i].plot(hist, alpha=0.8)
        ax[i].set_ylabel(ylabel)
    ax[-1].set_xlabel('Time')
    return fig, ax


if __name__ == "__main__":
    # Problem data
    n = 5
    m = 2
    T = 30

    x0 = 40*npr.randn(n)

    A = npr.randn(n, n)
    A = A/(max(abs(la.eig(A)[0])))
    B = npr.randn(n, m)

    Q = 1*np.eye(n)
    Qf = 5*np.eye(n)
    R = 1*np.eye(m)
    W = 0.1*np.eye(n)
    wm = np.zeros(n)

    P_hist, K_hist = lqr(A, B, Q, R, Qf, T)
    x_hist, u_hist, w_hist = rollout(x0, A, B, T)

    plt.close('all')
    plot_gain_hist(K_hist)
    plot_hist(x_hist, u_hist, w_hist)
