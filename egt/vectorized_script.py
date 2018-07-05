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
import logging
logging.basicConfig(level=logging.DEBUG)
# tqdm.monitor_interval = 0

import egt.visualisation as vis


###############################################################################
# Parameters
###############################################################################
_plot_range = np.arange(-5, 5, 0.001)
# Discretization of the strategies
_strategy_resolution = 0.01
DEFAULT_PARAMS = {
    'f': lambda x: x**2 + 0.5*np.sin(30*x),
    'alpha': 2,
    'beta': 100,
    'gamma': 100,
    'delta_t': 0.01,
    's_rounds': 2,
    'total_steps': int(3*60*60),
}

# Szenario 1: Minimum inside
starting_locations = [-1, 0, 1, 3, 5]

# Szenario 2: Minimum outside
# starting_locations = [1, 2, 3, 4]

# Szenario 3: N random particles
N = 2
starting_locations = np.random.uniform(-3, 10, N)
# starting_locations = [1.62245]*10 + [2]


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
    # parser.add_argument(
    #     '-f', '--function', type=str,
    #     help='Function to minimize. Write as functional python code')
    return parser.parse_args()


###############################################################################
# Setup - Universal settings that are not affected by some parameter changes
###############################################################################
N = len(starting_locations)

# All available strategies:
U = np.arange(-1, 1, _strategy_resolution)

# # Initial mixed strategy - continuous:
with np.errstate(divide='ignore'):
    sigma = np.exp(-1/(1-(U**2)))
sigma = sigma / np.sum(sigma)
# Alternative initial mixed strategy: Uniform
# sigma = np.array([1]*len(U)) / len(U)


def create_initial_population(starting_locations):
    # Initial population: Now as a matrix. First col location, rest mixed strategy
    population = np.concatenate(
        (np.array(starting_locations).reshape((len(starting_locations), 1)),
         np.tile(sigma, (len(starting_locations), 1))),
        axis=1)

    return population


def positive(x):
    """Helper function as it's needed in the original J"""
    return (np.abs(x) + x)/2


def J_original(x, u, x2, **kwargs):
    """J as described by Massimo"""
    alpha = kwargs.get('alpha', DEFAULT_PARAMS.get('alpha'))
    f = kwargs.get('f', DEFAULT_PARAMS.get('f'))
    with np.errstate(divide='ignore'):
        out = np.exp(
            -((u - positive(np.tanh(3*(f(x) - f(x2)))) * (x2 - x))**2) /
            ((x-x2) ** alpha + (f(x) - f(x2)) ** alpha))
    return out


def J(x, u, x2, **kwargs):
    """Game description - not vectorized

    Used to control the vectorized function below
    """
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    beta = params.get('beta')
    f = params.get('f')
    out = J_original(x, u, x2, **kwargs)
    # out *= np.exp(beta * (f(x)-f(x2)))
    out *= np.exp(-beta * f(x2))
    # out *= gamma
    return out


def J_vectorized(points, **kwargs):
    """Idea: generate a whole NxNx#Strategies tensor with the values of J

    This one is actually used for computations.

    axis=0 the point to evaluate
    axis=1 the point to compare to
    axis=2 are the strategies
    """
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    f = params.get('f')
    alpha = params.get('alpha')
    beta = params.get('beta')

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

    # out *= np.exp(beta * f_diffs)[:, :, None]
    out *= np.exp(-beta * f_vals)[None, :, None]

    # Test addon: 0 if other point higher
    # out *= (f_diffs >= 0)[:, :, None]
    # out *= f_diffs_tanh_positive[:, :, None]
    # out *= gamma

    return out


def simulate(initial_population, J, **kwargs):
    """Simulates the game J with the given starting population

    J is a vectorized version, such as J_vectorized

    Returns the full history of locations and strategies
    """
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    s_rounds = params.get('s_rounds')
    total_steps = params.get('total_steps')
    gamma = params.get('gamma')
    delta_t = params.get('delta_t')
    f = params.get('f')
    beta = params.get('beta')

    history = []
    history.append(initial_population)

    logging.info('Start simulation')
    sim_bar = tqdm.trange(total_steps)
    sim_bar.set_description('Simulation')
    for i in sim_bar:
        current_pop = history[-1]
        next_pop = current_pop.copy()

        # Strategy updates
        # Important to use `next_pop` if we du multiple rounds!
        for s in range(s_rounds):
            tot_J = J(next_pop[:, 0], **kwargs)
            sum_i = tot_J.sum(axis=1)
            mean_outcome = (sum_i * next_pop[:, 1:]).sum(axis=1)
            delta = sum_i - mean_outcome[:, None]
            delta = np.sum(
                tot_J - np.sum(
                    tot_J * next_pop[:, 1:][:, None, :],
                    axis=2)[:, :, None],
                axis=1)

            # Magnet Reweighting
            full_exp_eval = np.tile(
                np.exp(-beta*f(next_pop[:, 0])), (next_pop.shape[0], 1))
            np.fill_diagonal(full_exp_eval, 0)
            magnet_reweighting = full_exp_eval.sum(axis=1)
            print(magnet_reweighting)
            # My version:
            # full_diff_exp_eval = np.exp(
            #     -beta*(np.tile(f(next_pop[:, 0]), (next_pop.shape[0], 1)) - np.tile(f(next_pop[:, 0]), (next_pop.shape[0], 1)).T))
            # magnet_reweighting = full_diff_exp_eval.sum(axis=1)
            delta /= magnet_reweighting[:, None]

            next_pop[:, 1:] *= (1 + gamma * delta_t * delta)
            # next_pop[:, 1:] /= next_pop[:, 1:].sum(axis=1)[:, None]

        # Location updates
        for j in range(len(next_pop)):
            next_pop[j, 0] += delta_t*np.random.choice(U, p=next_pop[j, 1:])

        history.append(next_pop)

        # Break condition for early stopping
        _locs = history[-1][:, 0]
        max_dist = max(_locs) - min(_locs)
        max_prob_to_stay = current_pop[:, 1+len(U)//2].max()
        if max_dist < 1e-2 and max_prob_to_stay > 1 - 1e-15:
            logging.info('Early stopping thanks to our rule!')
            break

    logging.debug(f'Max distance at the end: {max_dist}')
    logging.debug(f'Max "staying-uncertainty": {max_prob_to_stay}')

    return history


def main():
    args = parse_args()
    seed = random.randint(0, 2**32-1) if not args.seed else args.seed
    logging.info(f'Seed used for this simulation: {seed}')
    np.random.seed(seed)

    population = create_initial_population(starting_locations)
    history = simulate(population, J_vectorized)

    # Create animation
    params_to_show = ['beta', 'gamma', 's_rounds']
    text = '\n'.join([f'{p}: {DEFAULT_PARAMS[p]}'for p in params_to_show])
    anim = vis.full_visualization(
        history, DEFAULT_PARAMS['f'], U,
        plot_range=_plot_range,
        parameter_text=text)
    if args.save:
        logging.info('Saving animation, this might take a while')
        text = '_'.join([f'{p}{DEFAULT_PARAMS[p]}'for p in params_to_show])
        anim.save(
            f'examples/{text}_{seed}.mp4',
            fps=60)
    else:
        plt.show()


if __name__ == '__main__':
    main()
