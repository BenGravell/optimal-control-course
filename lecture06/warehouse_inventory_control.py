import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


def generate_initial_state(size=None):
    return capacity*np.ones(shape=size)


def generate_demand(size=None):
    return npr.choice(prob_demand_supp, p=prob_demand_mass, size=size)


def policy(state):
    action = capacity - state if state <= 1 else 0
    return action


def update(state, demand):
    action = policy(state)
    new_state = np.clip(state - demand + action, 0, capacity)
    return new_state


# Problem data
num_trajectories = 3
num_timesteps = 25
capacity = 6
prob_demand_mass = np.array([0.7, 0.2, 0.1])
prob_demand_supp = np.array([0, 1, 2])

# Initialize time, state, and demand histories
state_hist = np.zeros([num_trajectories, num_timesteps+1])
state_hist[:, 0] = generate_initial_state(size=num_trajectories)
demand_hist = generate_demand(size=(num_trajectories, num_timesteps))
time_hist = np.arange(num_timesteps+1)

# Simulate sample trajectories
for i in range(num_trajectories):
    for t in range(num_timesteps):
        state_hist[i, t+1] = update(state_hist[i, t], demand_hist[i, t])

# Plot sample trajectories
plt.close('all')
plt.style.use('../conlab.mplstyle')
plt.step(time_hist, state_hist.T, alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Inventory')
plt.tight_layout()
plt.show()
