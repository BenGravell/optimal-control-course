import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt

P33 = np.array([[0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0]])
P44 = np.array([1])

P22 = sla.block_diag(P33, P44)

P11 = np.array([[0.4, 0.0, 0.0],
                [0.0, 0.3, 0.6],
                [0.0, 0.6, 0.2]])
P12 = np.array([[0.3, 0.0, 0.0, 0.3],
                [0.0, 0.0, 0.0, 0.1],
                [0.2, 0.0, 0.0, 0.0]])
P21 = np.zeros([4, 3])

P = np.block([[P11, P12],
              [P21, P22]])

Pss = la.matrix_power(P, 1000)
fig, ax = plt.subplots()
ax.pcolormesh(Pss, edgecolor='w')
ax.invert_yaxis()
ax.set_aspect('equal')
