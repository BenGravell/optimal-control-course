# Empirical probability density of products of n standard Gaussian random variables

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# List of number of independent standard Gaussian random variables to multiply together
n_list = 1 + np.arange(5)

# Number of Monte Carlo samples
N = 100000

# Initialize plot
fig, ax = plt.subplots(ncols=len(n_list), sharex=False, sharey=True, figsize=(12, 4))

# Take samples and plot
for i, n in enumerate(n_list):
    x = npr.randn(N, n)
    y = np.prod(x, axis=1)
    ax[i].hist(y, bins=100)
    ax[i].set_yscale('log')
    ax[i].set_title('n=%d' % (i+1))
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('Counts')
fig.tight_layout()

# By construction, each distribution plotted has the same mean of 0 and variance of 1
# See the Wikipedia article https://en.wikipedia.org/wiki/Product_distribution

# From the progression of histograms we see that the effect of multiplying together multiple Gaussian random variables
# is that the pdf loses mass in the "mid-range" and gains mass in both the center AND the tails
