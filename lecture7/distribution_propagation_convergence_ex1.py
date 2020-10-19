import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


n = 4
T = 100

P = np.array([[0.5, 0.5, 0.0, 0.0],
              [0.3, 0.4, 0.3, 0.0],
              [0.0, 0.3, 0.4, 0.3],
              [0.0, 0.0, 0.5, 0.5]])


d0_list = [np.array([1, 0, 0, 0]),
           np.array([0, 1, 0, 0]),
           np.array([0, 0, 1, 0]),
           np.array([0, 0, 0, 1])]

for d0 in d0_list:
    d = np.zeros([T + 1, n])
    d[0] = d0
    for t in range(T):
        d[t+1] = np.dot(d[t], P)
    plt.figure()
    plt.plot(d)
    plt.legend(['State %d' % (i+1) for i in range(n)])
