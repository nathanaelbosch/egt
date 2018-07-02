"""
Difference to first script: Optimize code, make it run faster!
- Data in a nicer way than lists
- Function to be applied to the whole array in the best possible way
"""
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
import argparse
# tqdm.monitor_interval = 0

import egt.visualisation as vis


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
# Multiple strategy update rounds per location update
s_rounds = 1

# Szenario 1: Minimum inside
starting_locations = [-1, 0, 3, 5]

# Szenario 2: Minimum outside
# starting_locations = [1, 2, 3, 4]

# starting_locations = [-0.1, 0.3, 0.3, 0.3]


###############################################################################
# Argparse
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save', action='store_true',
        help='Save the animation')
    parser.add_argument(
        '-s', '--seed', type=int,
        help='Random seed for numpy')
    return parser.parse_args()


###############################################################################
# Setup - Universal settings that are not affected by some parameter changes
###############################################################################
# We want to minimize the following function with EGT
def f(x):
    return x ** 2 + 0.5 * np.sin(30*x)


N = len(starting_locations)

# All available strategies:
U = np.arange(-1, 1, _strategy_resolution)

# Initial mixed strategy - continuous:
with np.errstate(divide='ignore'):
    sigma = np.exp(-1/(1-(U**2)))
sigma = sigma / np.sum(sigma)


def create_initial_population(starting_locations):
    # Initial population: Now as a matrix. First col location, rest mixed strategy
    population = np.concatenate(
        (np.array(starting_locations).reshape((len(starting_locations), 1)),
         np.tile(sigma, (len(starting_locations), 1))),
        axis=1)

    # Object to save the whole process
    history = []
    history.append(population)
    return history


def positive(x):
    """Helper function as it's needed in the original J"""
    return (np.abs(x) + x)/2


def J_original(x, u, x2):
    """J as described by Massimo"""
    with np.errstate(divide='ignore'):
        out = np.exp(
            -((u - positive(np.tanh(3*(f(x) - f(x2)))) * (x2 - x))**2) /
            ((x-x2) ** alpha + (f(x) - f(x2)) ** alpha))
    return out


def J(x, u, x2):
    """Game description - not vectorized

    Used to control the vectorized function below
    """
    out = J_original(x, u, x2)
    out *= np.exp(beta * (f(x)-f(x2)))
    # out *= gamma
    return out


def J_vectorized(points):
    """Idea: generate a whole NxNx#Strategies tensor with the values of J

    This one is actually used for computations.

    axis=0 the point to evaluate
    axis=1 the point to compare to
    axis=2 are the strategies
    """
    N = len(points)
    f_vals = f(points)
    f_diffs = np.tile(f_vals, reps=(N, 1)).T - np.tile(f_vals, reps=(N, 1))

    # # take only the ones where i!=j
    # rows = [[i]*(N-1) for i in range(N)]
    # cols = [[j for j in range(N) if j!=i] for i in range(N)]
    # f_diffs = f_diffs[rows, cols]

    f_diffs = f_diffs[range(N), ]
    f_diffs_tanh = np.tanh(3*f_diffs)
    f_diffs_tanh_positive = np.where(
        f_diffs_tanh > 0,
        f_diffs_tanh,
        0)
    walk_dirs = np.tile(points, reps=(N, 1)) - np.tile(points, reps=(N, 1)).T
    # walk_dirs = walk_dirs[rows, cols]
    walk_dirs_adj = f_diffs_tanh_positive * walk_dirs
    variance = walk_dirs ** alpha + f_diffs ** alpha

    with np.errstate(divide='ignore'):
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
    args = parse_args()
    if not args.seed:
        seed = random.randint(0, 2**32-1)
        print(f'Seed used for this simulation: {seed}')
        np.random.seed(seed)
    else:
        print(f'Seed used for this simulation: {args.seed}')
        np.random.seed(args.seed)

    print('Start')
    history = create_initial_population(starting_locations)
    sim_bar = tqdm.tqdm(range(total_steps))
    sim_bar.set_description('Simulation')
    for i in sim_bar:
        current_pop = history[-1]
        next_pop = current_pop.copy()

        # Strategy updates
        for s in range(s_rounds):
            tot_J = J_vectorized(next_pop[:, 0])
            sum_i = tot_J.sum(axis=1)
            mean_outcome = (sum_i * current_pop[:, 1:]).sum(axis=1)
            delta = sum_i - mean_outcome[:, None]
            delta = np.sum(
                tot_J - np.sum(
                    tot_J * current_pop[:, 1:][:, None, :],
                    axis=2)[:, :, None],
                axis=1)
            next_pop[:, 1:] *= (1 + gamma * delta_t * delta)
            # next_pop[:, 1:] /= next_pop[:, 1:].sum(axis=1)[:, None]

        # Location updates
        for j in range(len(next_pop)):
            next_pop[j, 0] += delta_t*np.random.choice(U, p=next_pop[j, 1:])

        history.append(next_pop)

        # Break condition for early stopping
        _locs = history[-1][:, 0]
        max_dist = max(_locs) - min(_locs)
        probability_to_stand = current_pop[:, 1000]
        # if max_dist < 0.01:
        if max_dist < 0.05 and probability_to_stand.sum() > N-(1e-5):
            print('Early stopping thanks to our rule!')
            break

    anim = vis.full_visualization(history, f, U)
    plt.show()

    if args.save:
        # Need to redo the animation as closing the plot destroys it
        print('Saving animation, this might take a while')
        anim = vis.full_visualization(history, f, U)
        anim.save(f'examples/{seed}.mp4', fps=60)


if __name__ == '__main__':
    main()
