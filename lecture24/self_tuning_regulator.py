import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import numpy.random as npr
from functools import reduce
import matplotlib.pyplot as plt


def vec(A):
    """Return the vectorized matrix A by stacking its columns."""
    return A.reshape(-1, order="F")


def mat(v, shape=None):
    """Return matricization i.e. the inverse operation of vec of vector v."""
    if shape is None:
        dim = int(np.sqrt(v.size))
        shape = dim, dim
    matrix = v.reshape(shape[1], shape[0]).T
    return matrix


def mdot(*args):
    """Multiple dot product."""
    return reduce(np.dot, args)


def specrad(A):
    """Spectral radius of matrix A."""
    try:
        return np.max(np.abs(la.eig(A)[0]))
    except np.linalg.LinAlgError:
        return np.nan


def lstsqb(a, b):
    """
    Return least-squares solution to a = bx.
    Similar to MATLAB / operator for rectangular matrices.
    If b is invertible then the solution is la.solve(a, b).T
    """
    return la.lstsq(b.T, a.T, rcond=None)[0].T


def gen_rand_pd(n):
    Qdiag = np.diag(npr.rand(n))
    Qvecs = la.qr(npr.randn(n, n))[0]
    Q = mdot(Qvecs, Qdiag, Qvecs.T)
    return Q


def gen_rand_AB(n=4, m=3, rho=None, seed=1, round_places=None):
    npr.seed(seed)
    if rho is None:
        rho = 0.9
    A = npr.randn(n, n)
    B = npr.rand(n, m)
    if round_places is not None:
        A = A.round(round_places)
        B = B.round(round_places)
    A = A * (rho / specrad(A))
    return A, B


def gen_rand_problem_data(n=4, m=3, rho=None, seed=1, penalty_rand=True):
    npr.seed(seed)
    A, B = gen_rand_AB(n, m, rho, seed)
    if penalty_rand:
        Q = gen_rand_pd(n)
        R = gen_rand_pd(m)
    else:
        Q = np.eye(n)
        R = np.eye(m)
    return A, B, Q, R


def gain(P, A, B, R):
    Hux = mdot(B.T, P, A)
    Huu = R+mdot(B.T, P, B)
    return -la.solve(Huu, Hux)


def steady_state_cost(A, B, Q, R, w_covr, e_covr, K):
    AK = A + np.dot(B, K)
    QK = Q + mdot(K.T, R, K)
    # Steady-state covariance of the state
    Xss = sla.solve_discrete_lyapunov(AK, w_covr+mdot(B, e_covr, B.T))
    # Steady-state cost
    css = np.trace(np.dot(QK, Xss))
    return css


if __name__ == "__main__":
    # Set the random seed for reproducibility
    seed = 1
    npr.seed(seed)

    # Generate true underlying system dynamics
    n, m = 4, 2
    rho = 1.01
    A, B, Q, R = gen_rand_problem_data(n, m, rho=rho)

    # Solve the optimal controller if we knew the dynamics a priori (for reference)
    P_are = sla.solve_discrete_are(A, B, Q, R)
    K_are = gain(P_are, A, B, R)

    # Exploration time
    t_explore = n+m
    if t_explore < n+m:
        raise ValueError('Exploration time chosen less than n+m, please increase to ensure well-posedness of LSE')

    # Total time horizon
    T = 10000
    # Time index history
    t_hist = np.arange(T)

    # Exploration noise moments
    e_mean = np.zeros(m)
    e_covr = np.eye(m)
    # Exploration schedule - use this to achieve an exploration-exploitation tradeoff
    # explore_schedule_method = 'open_loop'
    explore_schedule_method = 'closed_loop'
    if explore_schedule_method == 'open_loop':
        # Hyperparameter: Larger rate will exploit more, explore less. Zero value will never stop exploring.
        e_scale_rate = 1.0
        e_scale_hist = 1/(1 + e_scale_rate*np.sqrt(t_hist))
    elif explore_schedule_method == 'closed_loop':
        # Hyperparameter: Larger scale will exploit less, explore more
        e_scale2 = 1.0

    # Additive disturbance moments
    w_mean = np.zeros(n)
    w_covr = (1e-1)*np.eye(n)

    # Allocate history arrays
    x_hist = np.zeros([T+1, n])
    u_hist = np.zeros([T, m])
    P_hist = np.zeros([T, n, n])
    K_hist = np.zeros([T, m, n])

    # Initialize state
    x0 = np.zeros(n)
    x_hist[0] = x0
    x = np.copy(x0)

    # Initial controller used during pure exploration phase
    K0 = np.zeros([m, n])
    K = np.copy(K0)

    # Initial parameter covariance - this is a tuning hyperparameter for the RLS filter
    # Asymptotically, as time goes on, the choice of this hyperparameter becomes irrelevant,
    # it only affects the transient behavior
    V = np.eye(n*(n+m))

    # Rollout
    for t in range(T):
        # Sample an exploration input
        if explore_schedule_method == 'open_loop':
            e_scale = e_scale_hist[t]
        elif explore_schedule_method == 'closed_loop':
            e_scale = e_scale2*la.norm(V, ord=2)
        e = npr.multivariate_normal(e_mean, e_scale*e_covr)

        # Sample an additive disturbance
        w = npr.multivariate_normal(w_mean, w_covr)

        if t < t_explore:
            # Not enough data collected to form a meaningful model estimate
            pass
        else:
            if t == t_explore:
                # Use ordinary least-squares to get the first model estimate
                X = x_hist[1:t]
                Z = np.hstack([x_hist[0:t-1], u_hist[0:t-1]])
                ZTZ = np.dot(Z.T, Z)
                theta_est = lstsqb(mdot(X.T, Z), mdot(Z.T, Z))
                vec_theta_est = vec(theta_est)
                A_est, B_est = theta_est[:, 0:n], theta_est[:, n:n+m]

                # # Estimate the process noise covariance
                # w_covr_sum = np.zeros([n, n])
                # for i in range(t):
                #     x_prev = x_hist[i]
                #     u_prev = u_hist[i]
                #     x_curr = x_hist[i+1]
                #     e = x_curr - (np.dot(A_est, x_prev) + np.dot(B_est, u_prev))
                #     w_covr_sum += np.outer(e, e)
                # w_covr_est = w_covr_sum/t

            elif t > t_explore:
                # Update estimate of the process noise covariance from data using sample average of outer products
                # of estimated residuals using the current model estimate
                # e = x_curr - (np.dot(A_est, x_prev) + np.dot(B_est, u_prev))
                # w_covr_est = ((t-1)*w_covr_est + np.outer(e, e))/t

                # Update estimate of linear model parameters from latest observation of trajectory data
                # using recursive least-squares
                z = np.hstack([x_hist[t-1], u_hist[t-1]])
                H = np.kron(z, np.eye(n))
                L = np.dot(V, la.solve(mdot(H, V, H.T) + w_covr, H).T)  # TODO replace w_covr with w_covr_est
                vec_theta_est = vec_theta_est + np.dot(L, x_hist[t] - np.dot(H, vec_theta_est))
                theta_est = mat(vec_theta_est, shape=(n, n+m))
                A_est, B_est = theta_est[:, 0:n], theta_est[:, n:n+m]
                ILH = np.eye(n*(n+m)) - np.dot(L, H)
                V = mdot(ILH, V, ILH.T) + mdot(L, w_covr, L.T)
            # Design steady-state controller based on most recent model estimate
            P = sla.solve_discrete_are(A_est, B_est, Q, R)
            K = gain(P, A_est, B_est, R)
            # Record cost-to-go estimate
            P_hist[t] = P

        # Record gain used
        K_hist[t] = K
        # Generate the input
        u = np.dot(K, x) + e
        # Record the input
        u_hist[t] = u

        # Transition the state using stochastic dynamics
        x = np.dot(A, x) + np.dot(B, u) + w
        # Record the state
        x_hist[t+1] = x

    # Stage cost history
    c_hist = np.array([mdot(x_hist[t].T, Q, x_hist[t]) + mdot(u_hist[t].T, R, u_hist[t]) for t in t_hist])
    # Compute exponential moving average of the stage-cost
    alpha_list = [0.1, 0.01, 0.001]
    # alpha_list = [0.01]
    c_hist_ema_list = []
    for alpha in alpha_list:
        c_hist_ema = np.zeros(T)
        # # Initialize with the true expected steady-state cost under the initial control
        # c_hist_ema[0] = steady_state_cost(A, B, Q, R, w_covr, e_covr, K0)
        # Initialize with the simple average of the stage costs during exploration time
        c_hist_ema[0] = np.mean(c_hist[0:t_explore])
        for t in range(T-1):
            c_hist_ema[t+1] = (1-alpha)*c_hist_ema[t] + alpha*c_hist[t]
        c_hist_ema_list.append(c_hist_ema)
    # Cost-to-go matrix history
    P_err_hist = np.array([la.norm(P_are - P_hist[t], ord=2)/la.norm(P_are, ord=2) for t in t_hist])
    # Gain matrix history
    K_err_hist = np.array([la.norm(K_are - K_hist[t], ord=2)/la.norm(K_are, ord=2) for t in t_hist])

    # Plotting
    plt.close('all')
    plt.style.use('../conlab.mplstyle')
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
    ax[0].step(t_hist, c_hist, label='Stage cost')
    for alpha, c_hist_ema in zip(alpha_list, c_hist_ema_list):
        ax[0].step(t_hist, c_hist_ema, label=r'Stage cost, EMA, $\alpha=%.3f$' % alpha)
    ax[1].step(t_hist, P_err_hist, label='Relative P error')
    ax[1].step(t_hist, K_err_hist, label='Relative K error')
    # Set plot options, common items
    for a in ax:
        a.axvline(t_explore-1, color='k', alpha=0.5, linestyle='--', label='Exploration time')
        a.legend()
        a.set_xscale('log')
    ax[1].set_xlabel('Time')
    ax[0].set_ylabel('Cost')
    ax[1].set_ylabel('Error')
    fig.tight_layout()
    plt.show()
