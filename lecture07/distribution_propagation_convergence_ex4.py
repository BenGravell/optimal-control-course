import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


n = 5
T = 100
a = np.ones(n)
b = np.ones(n-1)

P = np.array([[0.50, 0.50, 0.00, 0.00, 0.00],
              [0.25, 0.00, 0.25, 0.00, 0.50],
              [0.50, 0.00, 0.00, 0.50, 0.00],
              [0.00, 0.00, 0.00, 0.25, 0.75],
              [0.00, 0.00, 0.00, 0.50, 0.50]])





# d0_list = []
# for i in range(n):
#     c = np.zeros(n)
#     c[i] = 1
#     d0_list.append(c)
#
# fig, ax = plt.subplots(ncols=n, figsize=(20, 4))
# for i, d0 in enumerate(d0_list):
#     d = np.zeros([T + 1, n])
#     d[0] = d0
#     for t in range(T):
#         d[t+1] = np.dot(d[t], P)
#     ax[i].plot(d)
#     ax[i].set_title('Initial State %d' % i)
