import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


capacity = 6
prob_demand_mass = np.array([0.7, 0.2, 0.1])
prob_demand_supp = np.array([0, 1, 2])

T = 50
P = np.array([[0, 0, 0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2]],
              [0, 0, 0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2]],
              [prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2], 0, 0, 0, 0],
              [0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2], 0, 0, 0],
              [0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2], 0, 0],
              [0, 0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2], 0],
              [0, 0, 0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2]]])

init_state = 6  # This should not be an element of the target_states
target_states = [0, 1]
Q = np.copy(P)
for target_state in target_states:
    Q[target_state] = np.zeros(capacity+1)
    Q[target_state, target_state] = 1
t_hist = np.arange(T)
hit_prob_hist = np.zeros(T)
for t in t_hist:
    if t > 0:
        Q1 = la.matrix_power(Q, t)
        Q2 = la.matrix_power(Q, t-1)
        Qdiff = Q1 - Q2
        hit_prob_hist[t] = np.sum([Qdiff[init_state, target_state] for target_state in target_states])
plt.plot(hit_prob_hist)
plt.xlabel('Time')
plt.ylabel('Hitting probability')
