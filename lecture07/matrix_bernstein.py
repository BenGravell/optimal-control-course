import numpy as np
import matplotlib.pyplot as plt

n, m = 2, 2     # Matrix dimensions
N = 150         # Number of independent terms in the average
M = (n*m)**0.5  # Almost sure bound on the norm of each sample Y
eps = 0.3       # Small error bound

# Compute the Bernstein bound
delta_bound = (n+m)*np.exp(-(3/2)*(N*eps**2)/(3*M**2 + M*eps))

# Estimate the bound using Monte Carlo simulation
num_trials = 1000000  # 1000000 provides high fidelity, but takes several seconds to complete
Y = 2*np.random.binomial(1, 0.5, size=(num_trials, N, n, m)) - 1
A = np.mean(Y, axis=1)
Anorm = np.linalg.norm(A, axis=(1, 2))
t = np.arange(num_trials) + 1
delta_estimate_history = np.cumsum(Anorm >= eps) / t
delta_bound_history = delta_bound*np.ones(num_trials)

# Plot
plt.loglog(t, delta_bound_history, 'r--', lw=3, label='Bernstein bound')
plt.loglog(t, delta_estimate_history, lw=3, label='Monte Carlo estimate')
plt.xlabel('Number of Monte Carlo trials')
plt.ylabel('Probability')
plt.legend()
plt.show()
