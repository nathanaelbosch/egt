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
from egt.tools import positive
from egt.alternative_J import myJ_vectorized


###############################################################################
# Parameters
###############################################################################
_plot_range = np.arange(-7, 7, 0.01)
# Discretization of the strategies
_strategy_resolution = 0.01
epsilon = 0

DEFAULT_PARAMS = {
    # 'f': lambda x: x**2 + 0.5*np.sin(30*x),
    # 'f': lambda x: x**2,
    'f': lambda x: (
        (((x-2)**2 * (x+2)**2 + 10*x) / (x**2 + 1)) +
        0.3 * (np.abs(x)+5) * np.sin(10*x)),
    'alpha': 2,
    'beta': 10000,
    'gamma': 'adaptive',
    'gamma_max': 0.6,
    # 'gamma': 10,
    'delta_t': 1,
    's_rounds': 20,
    'total_steps': int(20*60),
}


def get_starting_locations():
    """Need this as otherwise the seed will not be used"""

    # Szenario 1: Minimum inside
    # starting_locations = [-1, 0, 1, 3, 5]

    # Szenario 2: Minimum outside
    # starting_locations = [1, 2, 3, 4]

    # Szenario 3: N random particles
    N = 5
    starting_locations = np.random.uniform(0, 20, N)
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
U = np.arange(-1, 1, _strategy_resolution)[:, None]
standing_index = np.where(np.isclose(U, 0))[0][0]
U[standing_index] = 0
# Array of shape #U^d
# Ud = np.stack(np.meshgrid(*([U]*d))).reshape((3, -1)).T


# # Initial mixed strategy - continuous:
# epsilon = 1e-10
with np.errstate(divide='ignore'):
    sigma = np.exp(-1/(1-(U**2)))
    # sigma = np.exp(-1/(1+epsilon-(U**2)))
sigma = sigma / np.sum(sigma)
# Alternative initial mixed strategy: Uniform
# sigma = np.array([1]*len(U)) / len(U)


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


def J(x, u, x2, **kwargs):
    """Game description - not vectorized

    Used to control the vectorized function below
    """
    if x==x2:
        if u==0:
            return 1
        else:
            return 0
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    alpha = kwargs.get('alpha', DEFAULT_PARAMS.get('alpha'))
    f = kwargs.get('f', DEFAULT_PARAMS.get('f'))

    # variance = (np.abs(x-x2) ** alpha + np.abs(f(x)-f(x2)) ** alpha)
    variance = (np.abs(x-x2) ** alpha)

    with np.errstate(divide='ignore'):
        out = np.exp(
            -((u - positive(np.tanh(3*(f(x) - f(x2)))) * (x2 - x))**2) /
            variance)

    return out


def J_vectorized(locations, **kwargs):
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

    N, d = locations.shape

    f_vals = f(locations)
    assert len(f_vals) == N
    f_vals = f_vals.flatten()
    f_diffs = np.tile(f_vals, reps=(N, 1)).T - np.tile(f_vals, reps=(N, 1))
    # f_diffs = f_diffs[range(N), ]
    f_diffs_tanh = np.tanh(3*f_diffs)
    f_diffs_tanh_positive = np.where(
        f_diffs_tanh > 0,
        f_diffs_tanh,
        0)

    # Walk dirs should be a NxNxd array, containing xj-xi at location [i, j]
    walk_dirs = (np.tile(locations[None, :, :], (N, 1, 1)) -
                 np.tile(locations[:, None, :], (1, N, 1)))

    # walk_dirs = np.tile(locations, reps=(N, 1)) - np.tile(locations, reps=(N, 1)).T
    walk_dirs_adj = f_diffs_tanh_positive[:, :, None] * walk_dirs

    # Multi dimensionality does not work here anymore!!!
    walk_dirs_adj = walk_dirs_adj.reshape(N, N)
    walk_dirs = walk_dirs.reshape(N, N)
    # variance = (np.abs(walk_dirs) ** alpha +
    #             np.abs(f_diffs) ** alpha)
    variance = (np.abs(walk_dirs) ** alpha)

    # Make things stable: for x=x2, depending on if u==0
    problems = np.array([
        [np.all(locations[i] == locations[j]) for j in range(N)]
        for i in range(N)])
    variance[problems] = 1

    upper_side = (U.flatten()[None, None, :] - walk_dirs_adj[:, :, None]) ** 2

    out = np.exp(-1 * upper_side / variance[:, :, None])

    # with np.errstate(divide='ignore'):
    #     out = np.exp(
    #         -1 * ((U.reshape(1, 1, len(U)) - walk_dirs_adj[:, :, None])**2) /
    #         variance[:, :, None])

    # Points should not be compared to themselves!
    out[problems] = 0
    out[problems, standing_index] = 1

    # out[range(N), range(N)] = 0

    # out *= np.exp(-beta * f_vals)[None, :, None]

    return out


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


def my_magnet(tot_J, locations, **kwargs):
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


def replicator_dynamics(locations, strategies, **kwargs):
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
    _gamma = params.get('gamma')
    gamma_max = params.get('gamma_max')

    delta_t = params.get('delta_t')

    locations, strategies = initial_population
    N, d = locations.shape

    history = []
    history.append((locations.copy(), strategies.copy()))

    logging.info('Start simulation')
    sim_bar = tqdm.trange(total_steps)
    for i in sim_bar:
        # Strategy updates
        for s in range(s_rounds):
            # Formula: sigma = (1 + delta_t * gamma * delta) * sigma

            # All possible calls of J, in a single array, but without the diag
            delta = replicator_dynamics(locations, strategies, **kwargs)

            if _gamma == 'adaptive':
                min_delta = np.min(delta, axis=1)
                gamma = -gamma_max/delta_t/min_delta
                gamma = gamma[:, None]
            else:
                gamma = _gamma

            strategies *= (1 + gamma * delta_t * delta)
            # import pdb; pdb.set_trace()

            prob_sums = strategies.sum(axis=1)
            if np.any(prob_sums != 1) and np.all(np.isclose(prob_sums, 1)):
                # Numerical problems, but otherwise should be fine - Reweight
                strategies /= prob_sums[:, None]

        # Location updates
        for j in range(N):
            random_u_index = np.random.choice(
                len(U), p=strategies[j].flatten())
            locations[j] += delta_t*U[random_u_index]

        history.append((locations.copy(), strategies.copy()))

        # Break condition for early stopping
        max_dist = (max(locations) - min(locations))[0]
        max_prob_to_stay = strategies[:, standing_index].max()
        # logging.debug(f'Max distance: {max_dist}')
        sim_bar.set_description(
            '[Simulation] max_dist={:.3f}'.format(max_dist))
        if max_dist < 1e-2 and max_prob_to_stay > 1 - 1e-15:
            logging.info('Early stopping thanks to our rule!')
            break

    logging.info(f'Max distance at the end: {max_dist}')
    logging.info(f'Max "staying-uncertainty": {max_prob_to_stay}')

    return history


def main():
    args = parse_args()
    seed = random.randint(0, 2**32-1) if not args.seed else args.seed
    logging.info(f'Seed used for this simulation: {seed}')
    np.random.seed(seed)

    starting_locations = get_starting_locations()
    population = create_initial_population(starting_locations)
    history = simulate(population, J_vectorized)
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
