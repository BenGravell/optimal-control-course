import numpy as np
import matplotlib.pyplot as plt

capacity = 6
prob_demand_mass = np.array([0.7, 0.2, 0.1])
prob_demand_supp = np.array([0, 1, 2])


T = 5
P = np.array([[0, 0, 0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2]],
              [0, 0, 0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2]],
              [prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2], 0, 0, 0, 0],
              [0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2], 0, 0, 0],
              [0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2], 0, 0],
              [0, 0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2], 0],
              [0, 0, 0, 0, prob_demand_mass[0], prob_demand_mass[1], prob_demand_mass[2]]])

# distribution propagation
d = np.zeros([T+1, capacity+1])
d[0] = np.array([0, 0, 0, 0, 0, 0, 1])

fig, ax = plt.subplots(ncols=T)
for t in range(T):
    d[t+1] = np.dot(d[t], P)
    ax[t].bar(prob_demand_supp, d[t])
    ax[t].set_ylim([0, 1])


# dT = d(end,:)
# bar(d(end,:)); ylim([0,1])
# xlabel('state at time 100')
# ylabel('probability')
#
# reorder_probability = d(end,:)*[1, 1, 0, 0, 0, 0, 0]