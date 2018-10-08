"""Run a minimization

The actual logic is in `egt/minimization.py`. This here handles parameters,
command line arguments, calls visualization, and saves the animation if
wanted.

Job:
    - Setup
    - Run `minimize`
    - Present results
    - Save if wanted
"""
import datetime as dt
import random
import argparse
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import matplotlib.pyplot as plt

import egt.visualisation as vis
from egt.game_functions.original_J import OriginalJ
from egt.game_functions.alternative_J import MyJ
# from egt.game_functions.confidence_J import ConfidenceJ
from egt.minimization import minimize
import egt.carillo as carillo
import egt.convergence_analysis as convergence_analysis
from egt.test_functions import *
from egt.tools import save_data


def parse_args(*args, **kwargs):
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument(
        '--minimizer', default='egt',
        help='Choice of minimization technique: `egt` or `carillo`')
    parser.add_argument(
        '--test-function', default='ackley',
        help='Choice of test-function: `simple`, `ackley`, `double_ackley`')
    parser.add_argument(
        '-s', '--seed', type=int, default=random.randint(0, 2**32-1),
        help='Random seed for numpy')
    parser.add_argument(
        '--plot-range', type=float, default=(-10, 10), nargs=2,
        help='Interval in which to plot the function')
    parser.add_argument(
        '-M', '--max-iterations', type=int, default=1000000,
        help='Number of iterations to perform (max, might be less)')

    # Strategies U
    parser.add_argument(
        '--U-max', type=float, default=1,
        help='Max movement; U will be chosen symmetric around 0')
    parser.add_argument(
        '--U-min', type=float, default=5e-3,
        help='Min movement')
    parser.add_argument(
        '--n-strategies', type=int, default=50,
        help='Number of total strategies to use')

    # Initial population
    parser.add_argument(
        '-N', '--n-points', type=int, default=10,
        help='Number of points to use')
    parser.add_argument(
        '--point-interval', type=float, default=(-10, 10), nargs=2,
        help='Interval in which to uniformly put points')
    parser.add_argument(
        '--initial-strategy', type=str, default='standard',
        help='Initial strategy distribution for all points - default uniform')

    # Simulation parameters
    parser.add_argument(
        '--stepsize', type=float, default=0.1,
        help='Stepsize - Influences both strategy and location update')
    parser.add_argument(
        '--beta', type=float, default=1e4,
        help='beta parameter to use for the magnet')
    parser.add_argument(
        '--gamma', type=float, default=0.05,
        help='Stepsize-multiplicator to use for the strategy update')
    parser.add_argument(
        '--normalize-delta', type=bool, default=True,
        help='aka "adaptive stepsize" - highly recommended!')
    parser.add_argument(
        '--s-rounds', type=int, default=1,
        help='Strategy update rounds per location update')

    # J
    parser.add_argument(
        '--alpha', type=float, default=2,
        help='alpha parameter to use for the J')
    parser.add_argument(
        '--epsilon', type=float, default=1e-4,
        help='epsilon in J to assure Lipschitz continuity')
    parser.add_argument(
        '--my-j', action='store_true',
        help='Use my WIP J - default is the well-tested original J')

    # Misc
    parser.add_argument(
        '--max-animation-seconds', type=int, default=10,
        help='Maximum time of the resulting animation, in seconds')
    parser.add_argument(
        '--save-prefix', type=str,
        help='Prefix to use when saving the data to disk')
    parser.add_argument(
        '--smooth-movement', action='store_true',
        help='Move according to the exact ODE, not sampled')

    args = parser.parse_args(*args, **kwargs)

    assert args.gamma*args.stepsize <= 1, 'Gamma too large! Should be <= {}'.format(1/args.stepsize)

    return args


funcs = {
    'simple': simple_nonconvex_function,
    'ackley': ackley,
    'double_ackley': double_ackley,
    'easom': easom,
    'rastrigin': rastrigin,
    'cosine': cosine,
    'double_global_ackley': double_global_ackley,
}


def make_strategies(n_U=100, U_max=1, U_min=1e-2):
    assert n_U//2 == n_U/2
    # v_min, v_max = np.log(U_min+1), np.log(U_max+1)
    # v = np.arange(v_min, v_max+v_min, v_min)
    # u = np.exp(v) - 1

    # Variant 1: U^2
    v = np.linspace(np.sqrt(U_min), np.sqrt(U_max), n_U//2)
    u = v**2

    # Variant 2: More or less U + U^2
    # v = np.linspace(
    #     (-1+np.sqrt(1+8*U_min))/2,
    #     (-1+np.sqrt(1+8*U_max))/2,
    #     n_U//2)
    # u = (v + (v ** 2))/2

    # Variant 3: Linear
    # u = np.linspace(U_min, U_max, n_U//2+1)

    logging.info(f'n_U={n_U}, U_max={U_max}')
    logging.info(f'Minimum movement with current setup: {u[0]}')
    logging.info(f'Maximum movement with current setup: {u[-1]}')
    u = np.concatenate((-u[::-1], [0], u))[:, None]
    logging.info(f'Total number of strategies: {len(u)}')
    return u


def make_initial_population(initial_strategy, point_interval, n_points, U):
    if initial_strategy == 'uniform':
        sigma = np.array([1]*len(U)) / len(U)
    else:
        with np.errstate(divide='ignore'):
            sigma = np.exp(-1/(1-(U**2)))
        sigma = sigma / np.sum(sigma)

    # Initial population
    points = np.random.uniform(
        point_interval[0], point_interval[1], n_points)
    N = len(points)
    d = len(points[0]) if isinstance(points[0], tuple) else 1
    locations = np.array(points).reshape(N, d)
    strategies = np.tile(sigma.flatten(), (len(points), 1))
    population = (locations, strategies)
    return population


def main():
    ###########################################################################
    # 1. Setup
    args = parse_args()
    logging.info(f'Seed used for this simulation: {args.seed}')
    np.random.seed(args.seed)

    # f:
    f = funcs[args.test_function]

    # U:
    U = make_strategies(args.n_strategies, args.U_max, args.U_min)

    # Initial population:
    population = make_initial_population(
        args.initial_strategy, args.point_interval, args.n_points, U)

    # J:
    J_used = MyJ if args.my_j else OriginalJ

    ###########################################################################
    # 2. Run Minimize
    if args.minimizer == "egt":
        history = minimize(f, J_used, population, U, vars(args))
    elif args.minimizer == "carillo":
        history = carillo.minimize(f, population, vars(args))

    ###########################################################################
    # 3. Save Data
    now = dt.datetime.now()
    if args.save_prefix:
        name = args.save_prefix
    else:
        name = input(f'Filename prefix to save the data? Defaults to just "{now}"')
    if name != '':
        name = name+'_'
    filename = f'examples/{name}{dt.datetime.now()}'
    data = {
        'history': history,
        'f': f,
        'args': args,
        'U': U,
    }
    path = filename+'.pickle'
    save_data(data, path)
    logging.info(f'Saved data to {path}')

    ###########################################################################
    # 3. Visualize
    SHOW = input('Show animation? [y/N]').lower().startswith('y')
    params_to_show = ['beta', 'gamma', 's_rounds', 'stepsize']
    # params_to_show = ['beta', 'gamma', 'stepsize']
    parameter_text = '\n'.join(
        [f'n_points: {args.n_points}'] +
        [f'{p}: {vars(args)[p]}'for p in params_to_show])
    plot_range = np.linspace(args.plot_range[0], args.plot_range[1], 1000)
    def get_animation():
        return vis.full_visualization(
            history, f, U,
            plot_range=plot_range,
            parameter_text=parameter_text,
            max_len=args.max_animation_seconds*60)

    if SHOW:
        # Show animation
        anim = get_animation()
        plt.show()

        # Show convergence
        ax = convergence_analysis.visualize(history, f)
        fig = ax.get_figure()
        # g = convergence_analysis.visualize(history, f)
        plt.show()

        # Saving animation
        anim = get_animation()
        # text = input('Name?')
        logging.info('Saving animation, this might take a while')
        # text = '_'.join([f'{p}{vars(args)[p]}'for p in params_to_show])
        # path = f'examples/{dt.datetime.now()}.mp4'
        path = filename+'.mp4'
        anim.save(path,
                  fps=60)

        # Save convergence
        path = filename+'_fvalue.png'
        fig.savefig(path)
        # g.savefig(path)


if __name__ == '__main__':
    main()
