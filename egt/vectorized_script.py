"""
Difference to first script: Optimize code, make it run faster!
- Data in a nicer way than lists
- Function to be applied to the whole array in the best possible way
"""
import numpy as np
import matplotlib.pyplot as plt
import tqdm
# tqdm.monitor_interval = 0

import egt.visualisation as vis
# np.random.seed(0)


###############################################################################
# Parameters
###############################################################################
_plot_range = np.arange(-3, 3, 0.001)
# Discretization of the strategies
_strategy_resolution = 0.001
# Parameter of J
alpha = 2
# "Magnet"
beta = 0.1
# J rescale parameter
gamma = 0.3
# Stepsize at each iteration
delta_t = 0.1
total_steps = 2*60*60

# Szenario 1: Minimum inside
# starting_locations = [-1, 0, 1, 3]

# Szenario 2: Minimum outside
starting_locations = [1, 2, 3, 4]

# starting_locations = [-0.1, 0.3, 0.3, 0.3]


###############################################################################
# Setup
###############################################################################
# We want to minimize the following function with EGT
def f(x):
    return x ** 2 + 0.5 * np.sin(30*x)


# All available strategies:
U = np.arange(-1, 1, _strategy_resolution)

# Initial mixed strategy - continuous:
sigma = np.exp(-1/(1-(U**2)))
sigma = sigma / np.sum(sigma)

# Initial population: Now as a matrix. First col location, rest mixed strategy
population = np.concatenate(
    (np.array(starting_locations).reshape((len(starting_locations), 1)),
     np.tile(sigma, (len(starting_locations), 1))),
    axis=1)
N = population.shape[0]

# Object to save the whole process
history = []
history.append(population)


def positive(x):
    """Helper function as it's needed in the original J"""
    return (np.abs(x) + x)/2


def J_original(x, u, x2):
    """J as described by Massimo"""
    return np.exp(
        -((u - positive(np.tanh(3*(f(x) - f(x2)))) * (x2 - x))**2) /
        ((x-x2) ** alpha + (f(x) - f(x2)) ** alpha))


def J(x, u, x2):
    """Game description - not vectorized

    Used to control the vectorized function below
    """
    out = J_original(x, u, x2)
    out *= np.exp(-beta * f(x2))
    out *= gamma
    return out


def J_vectorized(points):
    """Idea: generate a whole NxNx#Strategies tensor with the values of J

    This one is actually used for computations.

    It is a #U x N x N tensor now
    axis=0 the point to evaluate
    axis=1 the point to compare to
    axis=2 are the strategies
    """
    N = len(points)
    f_vals = f(points)
    f_diffs = np.tile(f_vals, reps=(N, 1)).T - np.tile(f_vals, reps=(N, 1))
    f_diffs_tanh = np.tanh(10*f_diffs)
    f_diffs_tanh_positive = np.where(
        f_diffs_tanh > 0,
        f_diffs_tanh,
        0)
    walk_dirs = np.tile(points, reps=(N, 1)) - np.tile(points, reps=(N, 1)).T
    walk_dirs_adj = f_diffs_tanh_positive * walk_dirs
    variance = walk_dirs ** alpha + f_diffs ** alpha
    out = np.exp(
        -1 * ((U.reshape(1, 1, len(U)) - walk_dirs_adj[:, :, None])**2) /
        variance[:, :, None])

    out *= np.exp(beta * f_diffs)[:, :, None]

    # Test addon: 0 if other point higher
    # out *= (f_diffs >= 0)[:, :, None]
    # out *= f_diffs_tanh_positive[:, :, None]
    # out *= gamma

    return out


def main():
    """Computations of this script

    Separates setup and computation, enables easier testing
    """
    print('Start')
    sim_bar = tqdm.tqdm(range(total_steps))
    sim_bar.set_description('Simulation')
    for i in sim_bar:
        current_pop = history[-1]
        next_pop = current_pop.copy()

        # Strategy updates
        tot_J = J_vectorized(current_pop[:, 0])
        sum_i = tot_J.sum(axis=1)
        mean_outcome = (sum_i * population[:, 1:]).sum(axis=1)
        delta = sum_i - mean_outcome[:, None]
        delta = np.sum(
            tot_J - np.sum(
                tot_J * population[:, 1:][:, None, :], axis=2)[:, :, None],
            axis=1)
        next_pop[:, 1:] *= (1 + gamma * delta_t * delta)
        next_pop[:, 1:] /= next_pop[:, 1:].sum(axis=1)[:, None]

        # Location updates
        for j in range(len(current_pop)):
            next_pop[j, 0] += delta_t*np.random.choice(U, p=next_pop[j, 1:])

        history.append(next_pop)

        # Break condition for early stopping
        _locs = history[-1][:, 0]
        max_dist = max(_locs) - min(_locs)
        probability_to_stand = current_pop[:, 1000]
        # if max_dist < 0.01:
        if max_dist < 0.01 and probability_to_stand.sum() > N-(1e-5):
            print('Early stopping thanks to our rule!')
            break

    anim = vis.full_visualization(history, f, U)
    plt.show()


if __name__ == '__main__':
    main()
