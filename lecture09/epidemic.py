# Simulation of disease spread using a stochastic Susceptible-Infected-Deceased-Protected model

from dataclasses import dataclass
import numpy as np
import numpy.random as npr
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation


def sample_initial_state(barrier_fracs=None, barrier_probs=None, flips=None):
    state_init = np.zeros([height, width], dtype=int)

    # Specify the initially infected in the extreme corners
    for (i, j) in ((1, 1), (-2, -2)):
        state_init[i, j] = state_infos['Infected'].support

    # Specify the initially protected via leaky barriers
    # barrier_frac: Distance from the middle diagonal to the barrier
    # barrier_prob: Probability that each cell of the barrier exists
    if barrier_fracs is None:
        barrier_fracs = [0.8, 0.5, -0.5, -0.7, 0.6, 0.4, -0.4, -0.7]
    if barrier_probs is None:
        barrier_probs = [0.9, 0.5, 0.9, 0.6, 0.9, 0.8, 0.9, 0.4]
    if flips is None:
        flips = [True, True, True, True, False, False, False, False]

    for barrier_frac, barrier_prob, flip in zip(barrier_fracs, barrier_probs, flips):
        barrier_offset = int(barrier_frac*height)
        barrier_size = height-abs(barrier_offset)

        choices = (state_infos['Susceptible'].support,
                   state_infos['Protected'].support)
        probs = (1 - barrier_prob, barrier_prob)
        barrier = npr.choice(choices, size=barrier_size, p=probs)
        d = np.diag(np.ones_like(barrier), barrier_offset)
        if flip:
            d = np.fliplr(d)
        mask = np.where(d)
        state_init[mask] = barrier
    return state_init


def transition(state):
    height, width = state.shape

    infected_state = np.zeros([height, width], dtype=int)
    mask = np.where(state == state_infos['Infected'].support)
    infected_state[mask] = state_infos['Infected'].support
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    nbr_infected = convolve2d(infected_state, kernel, boundary='fill', fillvalue=state_infos['Susceptible'].support, mode='same')

    # Slow implementation using for loops...
    new_state = np.zeros([height, width], dtype=int)
    for i in range(height):
        for j in range(width):
            # Get the state-dependent transition probabilities
            P = P_nbr_infected if nbr_infected[i, j] else P_nbr_noinfect
            # Get the transition probability vector for the current state
            p = P[state[i, j]]
            # Sample a new random state
            new_state[i, j] = npr.choice(state_support, p=p)
    return new_state


class Epidemic:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.axis('off')

        self.state = sample_initial_state()

        cmap = colors.ListedColormap(state_colors)

        im = self.ax.imshow(self.state, cmap=cmap, vmin=0, vmax=3)
        self.im = im

        cbar = self.fig.colorbar(im, ax=self.ax, ticks=3/4*(0.5 + np.arange(4)))
        cbar.ax.set_yticklabels(state_labels)

        self.fig.tight_layout()

    def plot_step(self, t):
        prior_state = self.state
        self.state = transition(self.state)
        if np.all(prior_state == self.state):
            self.state = sample_initial_state()
        self.im.set_data(self.state)
        return [self.im]


def animate_rollout():
    epidemic = Epidemic()
    animation = FuncAnimation(epidemic.fig, epidemic.plot_step, interval=1000//60, blit=True)
    plt.show()


@dataclass
class StateInfo:
    support: int
    color: str


if __name__ == "__main__":
    # State information
    state_infos = {'Susceptible': StateInfo(0, '#E0ECE4'),
                   'Infected': StateInfo(1, '#66BFBF'),
                   'Deceased': StateInfo(2, '#FF4B5C'),
                   'Protected': StateInfo(3, '#056674')}
    state_labels = []
    state_support = []
    state_colors = []
    for label, state_info in state_infos.items():
        state_labels.append(label)
        state_support.append(state_info.support)
        state_colors.append(state_info.color)

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
    height = width = 30

    # Animate a rollout
    animate_rollout()
