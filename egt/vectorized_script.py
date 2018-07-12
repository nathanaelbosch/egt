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
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
# tqdm.monitor_interval = 0

import egt.visualisation as vis
from egt.original_J import OriginalJ
from egt.alternative_J import MyJ
# from egt.alternative_J import myJ_vectorized
# from egt.confidence_J import confidenceJ_vectorized


###############################################################################
# Parameters
###############################################################################
_plot_range = np.arange(-10, 10, 0.01)
# Discretization of the strategies: U consists of n strategies in [-i, i]
n_strategies = 200
U_range = (-1, 1)
epsilon = 0

DEFAULT_PARAMS = {
    # 'f': lambda x: x**2 + 0.5*np.sin(30*x),
    # 'f': lambda x: x**2,
    'f': lambda x: (
        (((x-2)**2 * (x+2)**2 + 10*x) / (x**2 + 1)) +
        0.3 * (np.abs(x)+5) * np.sin(10*x)),
    'alpha': 2,
    'beta': 1000,
    # 'gamma': 0.9,
    'gamma': 0.9,
    'h': 0.1,
    's_rounds': 1,
    'reweighted_delta': True,
    'total_steps': int(2*60*60),
    'MyJ': True,
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
    # parser.add_argument(
    #     '-f', '--function', type=str,
    #     help='Function to minimize. Write as functional python code')
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
with np.errstate(divide='ignore'):
    sigma = np.exp(-1/(1-(U**2)))
sigma = sigma / np.sum(sigma)
# Alternative initial mixed strategy: Uniform
sigma = np.array([1]*len(U)) / len(U)


def create_initial_population(points):
    # Initial population: Now as a matrix. First col location, rest mixed strategy
    N = len(points)
    d = len(points[0]) if isinstance(points[0], tuple) else 1
    locations = np.array(points).reshape(N, d)
    strategies = np.tile(sigma.flatten(), (len(points), 1))

    # population = np.concatenate(
    #     (np.array(starting_locations).reshape((len(starting_locations), 1)),
    #      np.tile(sigma, (len(starting_locations), 1))),
    #     axis=1)

    population = (locations, strategies)
    logging.debug('Starting population:')
    logging.debug(population)

    return population


def default_magnet(tot_J, locations, **kwargs):
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    f = params.get('f')
    beta = params.get('beta')

    N, d = locations.shape

    # Reweighting
    f_vals = f(locations).flatten()
    f_min_index = f_vals.argmin()
    f_second_min_index = np.delete(f_vals, f_min_index).argmin()
    if f_second_min_index >= f_min_index:
        f_second_min_index += 1

    f_value_matrix = np.tile(f_vals, (N, 1))
    mins = np.array(
        [f_vals[f_min_index]] * f_min_index +
        [f_vals[f_second_min_index]] +
        [f_vals[f_min_index]] * (N-f_min_index-1))

    f_value_matrix = np.tile(f_vals, (N, 1))
    f_value_matrix -= mins[:, None]
    # f_value_matrix -= mins[:, None]
    np.fill_diagonal(f_value_matrix, 0)
    weights = np.exp(-beta*f_value_matrix)
    np.fill_diagonal(weights, 0)
    return weights


def replicator_dynamics(locations, strategies, J_vectorized, **kwargs):
    N, d = locations.shape

    tot_J = J_vectorized(locations, **kwargs)
    tot_J[range(N), range(N)] = 0  # Don't compare points to themselves

    weights = default_magnet(tot_J, locations, **kwargs)

    mean_outcomes = np.sum(tot_J * strategies[:, None, :], axis=2)
    delta = np.sum(
        weights[:, :, None] * (
            tot_J - mean_outcomes[:, :, None]),
        axis=1) / np.sum(weights, axis=1)[:, None]

    return delta


def simulate(initial_population, J_vectorized, **kwargs):
    """Simulates the game J_vectorized with the given starting population

    J_vectorized is a vectorized version, such as J_vectorized

    Returns the full history of locations and strategies
    """
    # Get parameters
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    s_rounds = params.get('s_rounds')
    total_steps = params.get('total_steps')
    gamma = params.get('gamma')
    reweighted_delta = params.get('reweighted_delta')

    h = params.get('h')

    locations, strategies = initial_population
    N, d = locations.shape

    history = []
    history.append((locations.copy(), strategies.copy()))

    logging.info('Start simulation')
    sim_bar = tqdm.trange(total_steps)
    for i in sim_bar:
        # Strategy updates
        for s in range(s_rounds):
            # Formula: sigma = (1 + h * gamma * delta) * sigma

            # All possible calls of J, in a single array, but without the diag
            delta = replicator_dynamics(
                locations, strategies, J_vectorized, **kwargs)

            if reweighted_delta:
                delta = - delta / delta.min(axis=1)[:, None]

            strategies *= (1 + h * gamma * delta)
            # import pdb; pdb.set_trace()

            prob_sums = strategies.sum(axis=1)
            if np.any(prob_sums != 1) and np.all(np.isclose(prob_sums, 1)):
                # Numerical problems, but otherwise should be fine - Reweight
                strategies /= prob_sums[:, None]

        # Location updates
        for j in range(N):
            random_u_index = np.random.choice(
                len(U), p=strategies[j].flatten())
            locations[j] += h*U[random_u_index]

        history.append((locations.copy(), strategies.copy()))

        # Break condition for early stopping
        max_dist = (max(locations) - min(locations))[0]
        max_staying_uncertainty = 1 - strategies[:, standing_index].min()
        # logging.debug(f'Max distance: {max_dist}')
        sim_bar.set_description(
            '[Simulation] max_dist={:.3f} staying_uncertainty={:.2E}'.format(
                max_dist, max_staying_uncertainty))
        if max_dist < 1e-2 and max_staying_uncertainty < 1e-10:
            logging.info('Early stopping thanks to our rule!')
            break
        if max_staying_uncertainty == 0.0:
            logging.info('Early stopping! No point wants to move anymore')
            break

    logging.info(f'Max distance at the end: {max_dist}')
    logging.info(f'Max "staying-uncertainty": {max_staying_uncertainty}')

    return history


def main():
    args = parse_args()
    seed = random.randint(0, 2**32-1) if not args.seed else args.seed
    logging.info(f'Seed used for this simulation: {seed}')
    np.random.seed(seed)

    starting_locations = get_starting_locations()
    population = create_initial_population(starting_locations)

    J_used = MyJ if DEFAULT_PARAMS['MyJ'] else OriginalJ
    # J_used = confidence_J
    J = J_used.get(f, U)
    history = simulate(population, J)
    # import pdb; pdb.set_trace()

    # Create animation
    params_to_show = ['beta', 'gamma', 's_rounds']
    text = '\n'.join(
        [f'n_points: {len(starting_locations)}'] +
        [f'{p}: {DEFAULT_PARAMS[p]}'for p in params_to_show])
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
