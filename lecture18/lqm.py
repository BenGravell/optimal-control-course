import numpy as np
import numpy.random as npr
from numpy import linalg as la
from functools import reduce
import warnings
from warnings import warn


def mdot(*args):
    """Multiple dot product."""
    return reduce(np.dot, args)


def specrad(A):
    """Spectral radius of matrix A."""
    try:
        return np.max(np.abs(la.eig(A)[0]))
    except np.linalg.LinAlgError:
        return np.nan


def quadratic_formula(a, b, c):
    """Solve the quadratic equation 0 = a*x**2 + b*x + c using the quadratic formula."""
    if a == 0:
        return [-c / b, np.nan]
    disc = b**2-4 * a * c
    disc_sqrt = disc**0.5
    den = 2 * a
    roots = [(-b+disc_sqrt) / den, (-b-disc_sqrt) / den]
    return roots


def dare_mult(A, B, a, Aa, b, Bb, Q, R, algo='iterative', show_warn=False):
    """
    Solve a discrete-time generalized algebraic Riccati equation
    for stochastic linear systems with multiplicative noise.
    """

    n = A.shape[1]
    m = B.shape[1]
    p = len(a)
    q = len(b)

    failed = False

    # Handle the scalar case more efficiently by solving the ARE exactly
    if n == 1 and m == 1:
        if p == 1 and q == 1:
            algo = 'scalar'
        else:
            try:
                if np.count_nonzero(a[1:]) == 0 and np.count_nonzero(b[1:]) == 0:
                    algo = 'scalar'
                elif np.sum(np.count_nonzero(Aa[1:], axis=(1, 2))) == 0 \
                        and np.sum(np.count_nonzero(Bb[1:], axis=(1, 2))) == 0:
                    algo = 'scalar'
            except:
                pass

    if algo == 'scalar':
        A2 = A[0, 0]**2
        B2 = B[0, 0]**2
        Aa2 = Aa[0, 0]**2
        Bb2 = Bb[0, 0]**2

        aAa2 = a[0] * Aa2
        bBb2 = b[0] * Bb2

        aAa2m1 = aAa2-1
        B2pbBb2 = B2+bBb2

        aa = aAa2m1 * B2pbBb2+A2 * bBb2
        bb = R[0, 0] * (A2+aAa2m1)+Q[0, 0] * B2pbBb2
        cc = Q[0, 0] * R[0, 0]

        roots = np.array(quadratic_formula(aa, bb, cc))

        if not (roots[0] > 0 or roots[1] > 0):
            failed = True
        else:
            P = roots[roots > 0][0] * np.eye(1)
            K = -B * P * A / (R+B2pbBb2 * P)

    elif algo == 'iterative':
        # Options
        max_iters = 1000
        epsilon = 1e-6
        Pelmax = 1e40

        # Initialize
        P = Q
        counter = 0
        converged = False
        stop = False

        while not stop:
            # Record previous iterate
            P_prev = np.copy(P)
            # Certain part
            APAcer = mdot(A.T, P, A)
            BPBcer = mdot(B.T, P, B)
            # Uncertain part
            APAunc = np.zeros([n, n])
            for i in range(p):
                APAunc += a[i] * mdot(Aa[i].T, P, Aa[i])
            BPBunc = np.zeros([m, m])
            for j in range(q):
                BPBunc += b[j] * mdot(Bb[j].T, P, Bb[j])
            APAsum = APAcer+APAunc
            BPBsum = BPBcer+BPBunc
            # Recurse
            P = Q+APAsum-mdot(A.T, P, B, la.solve(R+BPBsum, B.T), P, A)

            # Check for stopping condition
            if la.norm(P-P_prev, 'fro') / la.norm(P, 'fro') < epsilon:
                converged = True
            if counter >= max_iters or np.any(np.abs(P) > Pelmax):
                failed = True
            else:
                counter += 1
            stop = converged or failed

        # Compute the gains
        if not failed:
            K = -mdot(la.solve(R+BPBsum, B.T), P, A)

        if np.any(np.isnan(P)):
            failed = True

    if failed:
        if show_warn:
            warnings.simplefilter('always', UserWarning)
            warn("Recursion failed, ensure system is mean square stabilizable "
                 "or increase maximum iterations")
        P = None
        K = None

    return P, K


def rollout(K, T, x0):
    x_hist = np.zeros([T, n])
    x_hist[0] = np.copy(x0)
    for t in range(T-1):
        x = x_hist[t]
        u = K.dot(x)
        At = np.copy(A)
        for i in range(p):
            At += a[i]*Aa[i]*npr.randn()
        Bt = np.copy(B)
        for j in range(q):
            Bt += b[j]*Bb[j]*npr.randn()
        x_hist[t+1] = At.dot(x) + Bt.dot(u)
    return x_hist


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 1
    npr.seed(seed)

    # System dynamics
    n, m = 3, 2  # Number of states, inputs
    r = 1.1  # Spectral radius of A
    A = npr.randn(n, n)
    A = A*(r/specrad(A))
    B = npr.randn(n, m)

    # Penalty matrices
    Q = np.eye(3)
    R = np.eye(2)

    # Noise data
    p, q = 3, 2  # Number of state-, input-multiplicative noises
    Aa = npr.rand(p, n, n).round(1)
    Bb = npr.rand(q, n, m).round(1)
    a = 0.1*np.ones(p)
    b = 0.2*np.ones(q)

    # Solve the optimal control problem
    P, K = dare_mult(A, B, a, Aa, b, Bb, Q, R)

    if P is None or K is None:
        raise Exception('Optimal control failed to solve, check solvability conditions!')

    # Simulate
    N = 20  # Number of trials
    T = 20  # Number of timesteps
    x0 = 2*(npr.rand(n) > 0.5) + npr.rand(n)  # Initial state
    t_hist = np.arange(T)

    # Plot
    plt.close('all')
    plt.style.use('../conlab.mplstyle')
    fig, ax = plt.subplots(nrows=n, sharex=True)
    for i in range(N):
        x_hist_ol = rollout(np.zeros_like(K), T, x0)
        x_hist_cl = rollout(K, T, x0)
        label_ol = '  Open-Loop' if i == 0 else None
        label_cl = 'Closed-Loop' if i == 0 else None
        for j in range(n):
            ax[j].plot(t_hist, x_hist_ol[:, j], color='C1', alpha=0.6, lw=2, label=label_ol)
            ax[j].plot(t_hist, x_hist_cl[:, j], color='C0', alpha=0.6, lw=2, label=label_cl)
    for j in range(n):
        ax[j].set_yscale('symlog')
        ax[j].set_ylabel('State %d' % j)
        ax[j].legend()
    ax[-1].set_xlabel('Time')
    fig.tight_layout()
