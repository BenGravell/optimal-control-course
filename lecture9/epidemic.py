# Simulation of disease spread using a stochastic Susceptible-Infected-Deceased-Protected model

import numpy as np
import numpy.random as npr
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation


def sample_initial_state():
    # Initial state
    state_init = np.zeros([height, width], dtype=int)
    # Specify the initially infected
    for (i, j) in ((1, 1), (-2, -2)):
        state_init[i, j] = 1
    # Specify the initially protected via leaky walls
    b12_offset = int(0.8*height)
    b34_offset = int(0.6*height)
    b12_prob = 0.9
    b34_prob = 0.9
    b1, b2 = [np.ones(height-b12_offset, dtype=int)*(npr.rand(height-b12_offset) > 1-b12_prob) for i in range(2)]
    b3, b4 = [np.ones(height-b34_offset, dtype=int)*(npr.rand(height-b34_offset) > 1-b34_prob) for i in range(2)]
    state_init += 3*np.fliplr(np.diag(b1, -b12_offset))
    state_init += 3*np.fliplr(np.diag(b2, +b12_offset))
    state_init += 3*np.fliplr(np.diag(b3, -b34_offset))
    state_init += 3*np.fliplr(np.diag(b4, +b34_offset))
    return state_init


def transition(state):
    height, width = state.shape
    # Slow implementation using for loops...
    new_state = np.zeros([height, width], dtype=int)
    for i in range(height):
        for j in range(width):
            nbr_infected = False
            # Compute whether any neighbors are infected
            possible_nbrs = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            for (ii, jj) in possible_nbrs:
                if 0 <= ii < height and 0 <= jj < width:
                    if state[ii, jj] == 1:
                        nbr_infected = True
                        break
                else:
                    continue
            # Get the state-dependent transition probabilities
            P = P_nbr_infected if nbr_infected else P_nbr_noinfect
            # Get the transition probability vector for the current state
            p = P[state[i, j]]
            # Sample a new random state
            new_state[i, j] = npr.choice(state_support, p=p)
    return new_state


class Epidemic(object):
    def __init__(self, fig, ax):
        self.state = sample_initial_state()
        cmap = colors.ListedColormap(['#E0ECE4', '#66BFBF', '#FF4B5C', '#056674'])
        im = ax.imshow(self.state, cmap=cmap, vmin=0, vmax=3)
        self.im = im
        cbar = fig.colorbar(im, ax=ax, ticks=3/4*(0.5 + np.arange(4)))
        cbar.ax.set_yticklabels(state_labels)
        fig.tight_layout()

    def plot_step(self, t):
        prior_state = self.state
        self.state = transition(self.state)
        if np.all(prior_state == self.state):
            self.state = sample_initial_state()
        self.im.set_data(self.state)
        return [self.im]


def animate_rollout():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    epidemic = Epidemic(fig, ax)
    animation = FuncAnimation(fig, epidemic.plot_step, interval=1000/30, blit=True)
    plt.show()


if __name__ == "__main__":
    # State definitions
    # 0 = Susceptible
    # 1 = Infected
    # 2 = Deceased
    # 3 = Protected
    state_support = np.arange(4)
    state_labels = ('Susceptible', 'Infected', 'Deceased', 'Protected')
    # State transition probabilities (P)
    P_nbr_infected = np.array([[0.6, 0.4, 0.0, 0.0],
                               [0.2, 0.6, 0.1, 0.1],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    P_nbr_noinfect = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.2, 0.6, 0.1, 0.1],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    # Grid dimensions
    height, width = 40, 40

    # Animate a rollout
    animate_rollout()
