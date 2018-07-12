"""Run a minimization

The actual logic is in `egt/minimization.py`. This here handles parameters,
command line arguments, calls visualization, and saves the animation if
wanted.

"""
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import logging
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

import egt.visualisation as vis
from egt.original_J import OriginalJ
from egt.alternative_J import MyJ
from egt.minimization import minimize


###############################################################################
# Parameters
###############################################################################
_plot_range = np.arange(-10, 10, 0.01)
# Discretization of the strategies: U consists of n strategies in [-i, i]
n_strategies = 200
U_range = (-1, 1)
epsilon = 0

DEFAULT_PARAMS = {
    'alpha': 2,
    'beta': 1000,
    'gamma': 0.9,
    'h': 0.1,
    's_rounds': 1,
    'reweighted_delta': True,
    'total_steps': int(2*60*60),
    'MyJ': False,
}


def f(x):
    return (((x-2)**2 * (x+2)**2 + 10*x) / (x**2 + 1) +
            0.3 * (np.abs(x)+5) * np.sin(10*x))


if DEFAULT_PARAMS['reweighted_delta']:
    assert DEFAULT_PARAMS['gamma']*DEFAULT_PARAMS['h'] < 1, 'Stepsize too large!'


def get_starting_locations():
    """Need this as otherwise the seed will not be used"""

    # Szenario 1: Minimum inside
    # starting_locations = [-1, 0, 1, 3, 5]

    # Szenario 2: Minimum outside
    # starting_locations = [1, 2, 3, 4]

    # Szenario 3: N random particles
    N = 10
    starting_locations = np.random.uniform(-10, 10, N)
    # starting_locations[-3:] = [-20, -50, 3]

    return starting_locations


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
# All available strategies:
if n_strategies % 2 == 0:
    logging.warning(
        f'Use {n_strategies+1} instead of {n_strategies} strategies; ' +
        'Unpair numbers make sense here for symmetry')
    n_strategies += 1
U = np.concatenate((
    np.linspace(U_range[0], 0, n_strategies//2, endpoint=False),
    np.linspace(0, U_range[1], n_strategies//2 + 1, endpoint=True)))[:, None]
standing_index = np.where(np.isclose(U, 0))[0][0]
# Array of shape #U^d
# Ud = np.stack(np.meshgrid(*([U]*d))).reshape((3, -1)).T


# Initial mixed strategy - continuous:


def create_initial_population(points, U, strategy_distribution='uniform'):
    # Initial population: Now as a matrix. First col location, rest mixed strategy
    N = len(points)
    d = len(points[0]) if isinstance(points[0], tuple) else 1

    if strategy_distribution == 'uniform':
        sigma = np.array([1]*len(U)) / len(U)
    else:
        with np.errstate(divide='ignore'):
            sigma = np.exp(-1/(1-(U**2)))
        sigma = sigma / np.sum(sigma)

    locations = np.array(points).reshape(N, d)
    strategies = np.tile(sigma.flatten(), (len(points), 1))

    population = (locations, strategies)
    logging.debug('Starting population:')
    logging.debug(population)

    return population


def main():
    args = parse_args()
    seed = random.randint(0, 2**32-1) if not args.seed else args.seed
    logging.info(f'Seed used for this simulation: {seed}')
    np.random.seed(seed)

    starting_locations = get_starting_locations()
    population = create_initial_population(starting_locations, U)

    J_used = MyJ if DEFAULT_PARAMS['MyJ'] else OriginalJ

    history = minimize(f, J_used, population, U, DEFAULT_PARAMS)

    # J_used = confidence_J
    # J = J_used.get(f, U)
    # history = simulate(population, J)
    # import pdb; pdb.set_trace()

    # Create animation
    params_to_show = ['beta', 'gamma', 's_rounds']
    text = '\n'.join(
        [f'n_points: {len(starting_locations)}'] +
        [f'{p}: {DEFAULT_PARAMS[p]}'for p in params_to_show])
    anim = vis.full_visualization(
        history, f, U,
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
