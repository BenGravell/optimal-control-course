import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


n = 7
T = 100
a = np.ones(n)
b = np.ones(n-1)

P = np.diag(0.3*a, 0) + np.diag(0.4*b, -1) + np.diag(0.3*b, 1)
P[0, 0] = 1
P[0, 1] = 0
P[-1, -1] = 1
P[-1, -2] = 0

d0_list = []
for i in range(n):
    c = np.zeros(n)
    c[i] = 1
    d0_list.append(c)

plt.close('all')
plt.style.use('../conlab.mplstyle')
fig, ax = plt.subplots(ncols=n, figsize=(20, 4))
for i, d0 in enumerate(d0_list):
    d = np.zeros([T + 1, n])
    d[0] = d0
    for t in range(T):
        d[t+1] = np.dot(d[t], P)
    ax[i].plot(d)
    ax[i].set_title('Initial State %d' % i)
fig.tight_layout()
plt.show()