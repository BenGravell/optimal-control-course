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


#  First system: driven by both significant multiplicative and additive noise
# Second system: driven by both significant multiplicative and additive noise
problem_settings = {'Mean-square Stable':
                    {'a_mean': 0.80,
                     'a_std': 0.40,
                     'w_mean': 0.00,
                     'w_std': 0.20}}

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

# Plot the empirical histogram and a Gaussian pdf with identical variance
# This shows the limit distribution of states is very heavy-tailed
plt.close('all')
plt.style.use('../conlab.mplstyle')
plt.hist(x_hist[:, -1], bins=1000, density=True, label='Empirical distribution')
std = np.std(x_hist[:, -1])
from scipy.stats import norm
x = np.linspace(-4, 4, 1000)
plt.plot(x, norm.pdf(x, scale=std), label='Normal distribution with same variance')
plt.xlim([-4, 4])
plt.ylim([1e-6, 1e2])
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()
