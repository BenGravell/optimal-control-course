# Estimate a probability involving a sum of many Bernoulli random variables using the Monte Carlo method

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# Number of experiments to run
num_trials = 10

# Number of Monte Carlo samples to use
N = 100000

# Size of the Bernoulli random vector and the split point
n = 50
n2 = int(n/2)

# Success probability of the Bernoulli random vector
p = 0.4

# Threshold for the probability of np.sum(x[:, 0:25], axis=1) >= q*np.sum(x[:, 25:], axis=1)
q = 0.6

# Choose whether to use the faster simulation (slower one is slightly easier to understand)
simulation_type = 'slow'
# simulation_type = 'fast'

# Perform Monte Carlo estimation and record the estimates as more data is used
if simulation_type == 'slow':
    def sample(n, n2, p=0.4):
        x = npr.binomial(n=1, p=p, size=n)
        y1 = np.sum(x[0:n2])
        y2 = np.sum(x[n2:])
        return y1 >= q*y2

    for i in range(num_trials):
        # Initialize the history of probability estimates
        p_hist = np.zeros(N)
        # Initialize the aggregator
        agg = 0
        for j in range(N):
            # Take a new sample
            agg += sample(n, n2)
            # Estimate the probability
            p_hist[j] = agg/(j+1)
        # Plot the history of the probability estimates
        plt.plot(p_hist, color='k', alpha=0.5)

elif simulation_type == 'fast':
    for i in range(num_trials):
        # Generate data
        x = npr.rand(N, n) >= 1-p
        # Compute events
        y1 = np.sum(x[:, 0:25], axis=1)
        y2 = np.sum(x[:, 25:], axis=1)
        z = y1 >= q*y2
        p_hist = np.cumsum(z)/np.arange(1, N + 1)
        # Plot the history of the probability estimates
        plt.plot(p_hist, color='k', alpha=0.5)

plt.ylim([0.85, 1.0])
plt.xlabel('Number of Monte Carlo samples')
plt.ylabel('Estimated probability')
