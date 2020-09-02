# Sampling from a discrete distribution

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import bisect


def normalize(pmf):
    return pmf/(np.sum(np.abs(pmf)))


# This sampling function samples from a discrete distribution with probability mass function "pmf"
# and support coordinates specified in "support" by mapping a uniform random variable "u" onto the
# cumulative distribution function "cdf" and finding the associated support coordinate index
def sample(pmf, support=None):
    if support is None:
        support = np.arange(pmf.size)
    # Compute the cdf
    cdf = np.cumsum(pmf)
    # Sample from the uniform distribution
    u = npr.rand()
    # Find first index where cdf exceeds the uniform sample
    idx = bisect.bisect_left(cdf, u)
    # Map the index onto the support and return the sample
    return support[idx]


# This sampling function does the same thing as sample(), but uses the NumPy built-in function numpy.random.choice()
def sample_choice(pmf, support=None):
    return npr.choice(support, p=pmf)


# Number of points in the pmf support
n = 50
n2 = int(n/2)
# Define a support
support = np.hstack([np.linspace(-10, 0, n2), np.linspace(0+10/n2, 30, n-n2)])
# Define a pmf
a = np.linspace(0, 6, n)
pmf = normalize(np.abs(0.1*a**2 + np.sin(a)))
# Compute the pdf
d = np.diff(support)
ds = (np.hstack([d, d[-1]]) + np.hstack([d[0], d]))/2
pdf = pmf/ds

# Number of Monte Carlo samples
N = 100000
# Take Monte Carlo samples to get an empirical estimate of the pmf
samples = np.zeros(N)
for i in range(N):
    samples[i] = sample(pmf, support)

# Plot the pdf and empirical estimate
plt.plot(support, pdf, label='True pdf')
plt.hist(samples, bins=support, density=True, label='Monte Carlo pdf')
plt.legend()
