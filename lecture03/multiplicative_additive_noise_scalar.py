# Simulation of scalar stochastic system with multiplicative and additive noise

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


def f(x, a, w):
    return a*x + w


def symlog(X,scale=1):
    """Symmetric log transform"""
    return np.multiply(np.sign(X), np.log(1+np.abs(X)/(10**scale)))


def mix(a, b, x):
    return x*a + (1-x)*b

# These problem settings have been tuned to produce nearly the same
# steady-state 10-90 percentile range of the state distribution

#  First system: driven strictly by additive noise
# Second system: driven almost entirely by multiplicative noise
# problem_settings = {'Mean-square   Stable':
#                     {'a_mean': 0.00,
#                      'a_std': 0.00,
#                      'w_mean': 0.00,
#                      'w_std': 1.00},
#                     'Mean-square Unstable':
#                     {'a_mean': 0.00,
#                      'a_std': 1.60,
#                      'w_mean': 0.00,
#                      'w_std': 0.01}}

#  First system: driven by both significant multiplicative and additive noise
# Second system: driven by both significant multiplicative and additive noise
problem_settings = {'Mean-square   Stable':
                    {'a_mean': 0.80,
                     'a_std': 0.40,
                     'w_mean': 0.00,
                     'w_std': 0.15},
                    'Mean-square Unstable':
                    {'a_mean': 0.80,
                     'a_std': 0.80,
                     'w_mean': 0.00,
                     'w_std': 0.10}}

# Problem settings
# Make sure to use many trajectories e.g. > 100000 or else, with high probability, you will not observe
# the low probability events which make variance of the state blow up as t -> inf for the ms-unstable system
num_trajectories = 1000000
num_timesteps = 100
t_hist = np.arange(num_timesteps+1)
x0_mean = 0
x0_std = 1
percentiles_list = [0, 0.1, 1, 10]
num_percentiles = len(percentiles_list)
hist_times_list = np.linspace(0, num_timesteps, 5).astype(int)

# Initialize plot
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(16, 9))
figh, axh = plt.subplots(nrows=2, ncols=len(hist_times_list), sharex=False, sharey=True, figsize=(16, 9))

for k, (key, problem_setting) in enumerate(problem_settings.items()):
    a_mean = problem_setting['a_mean']
    w_mean = problem_setting['w_mean']
    a_std = problem_setting['a_std']
    w_std = problem_setting['w_std']

    # Initialize the state histories
    x_hist = np.zeros([num_trajectories, num_timesteps+1])
    x_hist[:, 0] = x0_mean + x0_std*npr.randn(num_trajectories)
    x_mean_tru_hist = np.zeros(num_timesteps+1)
    x_mean_tru_hist[0] = x0_mean
    x_var_tru_hist = np.zeros(num_timesteps+1)
    x_var_tru_hist[0] = x0_std**2

    # Generate noise histories
    a_hist = a_mean + a_std*npr.randn(num_trajectories, num_timesteps)
    w_hist = w_mean + w_std*npr.randn(num_trajectories, num_timesteps)

    # Simulate stochastic dynamics
    for t in range(num_timesteps):
        x_hist[:, t+1] = f(x_hist[:, t], a_hist[:, t], w_hist[:, t])

    # Simulate mean dynamics
    for t in range(num_timesteps):
        x_mean_tru_hist[t+1] = a_mean*x_mean_tru_hist[t]

    # Simulate second moment dynamics
    for t in range(num_timesteps):
        x_var_tru_hist[t+1] = (a_mean**2 + a_std**2)*x_var_tru_hist[t] + w_std**2

    # Estimate first and second moments
    x_mean_est_hist = np.mean(x_hist, axis=0)
    x_var_est_hist = np.mean(x_hist**2, axis=0)

    # Plot individual trajectories
    # ax[k].plot(x_hist[0:10].T, color='k', alpha=0.1)
    # Plot percentiles of the trajectories
    shade_color1 = np.array([31, 119, 180])/255
    shade_color2 = np.array([1, 1, 1])
    for j, p in enumerate(percentiles_list):
        ax[k].fill_between(t_hist,
                           np.percentile(x_hist, p, axis=0),
                           np.percentile(x_hist, 100 - p, axis=0),
                           color=mix(shade_color1, shade_color2, 1 + 0.5*(j/(num_percentiles - 1) - 1)),
                           alpha=1.0,
                           label='Percentile %3d to %3d' % (p, 100-p))
    # Plot the true mean of the trajectories
    ax[k].plot(t_hist, x_mean_tru_hist, color='k', linestyle='--', label='True mean')

    # Plot the true standard deviation
    ax[k].plot(t_hist, x_mean_tru_hist + x_var_tru_hist**0.5, color='k', linestyle='--', label='True mean + 1 std dev')
    ax[k].plot(t_hist, x_mean_tru_hist - x_var_tru_hist**0.5, color='k', linestyle='--', label='True mean - 1 std dev')

    # Plot the empirical standard deviation
    ax[k].plot(t_hist, x_mean_est_hist + x_var_est_hist**0.5, color='tab:red', linestyle='--', label='Estimated mean + 1 std dev')
    ax[k].plot(t_hist, x_mean_est_hist - x_var_est_hist**0.5, color='tab:red', linestyle='--', label='Estimated mean - 1 std dev')

    # Set the xscale to logarithmic to compress future times
    ax[k].set_xscale('log')
    # Set the yscale to symmetric logarithmic so we can see both the "core" of the state distribution near 0
    # and the "blowup events", which grow exponentially large, at the same time
    ax[k].set_yscale('symlog')
    ax[k].set_title(key)
    ax[k].legend(loc='upper left', prop={'family': 'monospace'})

    # Plot some sample histograms as time progresses
    for i, hist_time in enumerate(hist_times_list):
        axh[k, i].hist(symlog(x_hist[:, hist_time]), bins=50, density=False)
        if i == 0:
            axh[k, i].set_ylabel(key)
        if k == 1:
            axh[k, i].set_xlabel('Time = %d' % hist_time)

    # Print a Monte Carlo estimate of the steady-state 10th and 90th percentiles
    # If they are the same for both problem settings then it is roughly an apples-to-apples comparison
    # between the distributions, and we are witnessing only the behavior in the tails of the distribution
    # The mean-square unstable system exhibits tremendous volatility, while the mean-square stable system is very benign
    print(key + ": 10th percentile %.3f  90th percentile %.3f" % (np.percentile(x_hist[:, -1], 10),
                                                                  np.percentile(x_hist[:, -1], 90)))
fig.tight_layout()
figh.tight_layout()
plt.close(figh)


# Note that we have used the symlog scale on the y-axis - this means that the mean-square unstable system
# is extraordinarily volatile as we observe rare "blowup events" which are not observed in the mean-square stable system

# Note that we have used the log scale on the x-axis - this lets us observe the long-term trends of the state
# distribution more clearly

# Notice that the mean-square unstable system looks like each percentile in (0, 100) converge to a steady-state value,
# while the 0 and 100 percentiles grow without bound, which is consistent with the tail probability increasing over time
# Since the system is mean-square unstable, this means that (a very small amount of) probability mass is gradually drawn
# out away from the mid-range

# As another thought on the same idea, notice that the estimated variance and the true variance match very closely,
# but towards the end of the simulation the estimated variance underestimates the true variance,
# and this underestimation gets worse over time. This is because, with ever increasing probability,
# we do not observe realizations of the state distribution which are far from the mean (rare events),
# so the estimated sample variance is too low.
