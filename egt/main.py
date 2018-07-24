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
import numpy as np
import random
import argparse
import logging
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

import egt.visualisation as vis
from egt.game_functions.original_J import OriginalJ
from egt.game_functions.alternative_J import MyJ
# from egt.game_functions.confidence_J import ConfidenceJ
from egt.minimization import minimize
import egt.carillo as carillo
import egt.convergence_analysis as convergence_analysis
from egt.test_functions import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save', action='store_true',
        help='Save the animation')
    parser.add_argument(
        '--carillo', action='store_true',
        help='Use the carillo minimization technique')
    parser.add_argument(
        '-s', '--seed', type=int,
        help='Random seed for numpy')
    parser.add_argument(
        '-n', '--max-iterations', type=int,
        help='Number of iterations to perform (max, might be less)')
    parser.add_argument(
        '--U-interval', type=float, default=(-1, 1), nargs=2,
        help='Min and max value for the U interval')
    parser.add_argument(
        '--plot-range', type=float, default=(-10, 10), nargs=2,
        help='Interval in which to plot the function')
    parser.add_argument(
        '--n-strategies', type=int, default=200,
        help='Number of total strategies to use')
    parser.add_argument(
        '--s-rounds', type=int, default=1,
        help='Strategy update rounds per location update')
    parser.add_argument(
        '-N', '--n-points', type=int, default=10,
        help='Number of points to use')
    parser.add_argument(
        '--point-interval', type=float, default=(-10, 10), nargs=2,
        help='Interval in which to uniformly put points')
    parser.add_argument(
        '--beta', type=float, default=1000,
        help='beta parameter to use for the magnet')
    parser.add_argument(
        '--alpha', type=float, default=2,
        help='alpha parameter to use for the J')
    parser.add_argument(
        '--gamma', type=float, default=0.9,
        help='Stepsize-multiplicator to use for the strategy update')
    parser.add_argument(
        '--stepsize', type=float, default=0.1,
        help='Stepsize - Influences both strategy and location update')
    parser.add_argument(
        '--max-animation-seconds', type=int, default=10,
        help='Maximum time of the resulting animation, in seconds')
    parser.add_argument(
        '--normalize-delta', action='store_true',
        help='aka "adaptive stepsize" - highly recommended!')
    parser.add_argument(
        '--my-j', action='store_true',
        help='Use my WIP J - default is the well-tested original J')
    parser.add_argument(
        '--initial-strategy', type=str, default='uniform',
        help='Initial strategy distribution for all points - default uniform')

    args = parser.parse_args()

    assert args.gamma*args.stepsize <= 1, 'Stepsize too large!'

    return args



f = two_wells
f = simple_nonconvex_function
# f = ackley
# f = easom





def main():
    ###########################################################################
    # 1. Setup
    args = parse_args()
    seed = random.randint(0, 2**32-1) if not args.seed else args.seed
    logging.info(f'Seed used for this simulation: {seed}')
    np.random.seed(seed)

    # U:
    if args.n_strategies % 2 == 0:
        logging.warning(
            f'Use {args.n_strategies+1} instead of {args.n_strategies} strategies; ' +
            'Unpair numbers make sense here for symmetry')
        args.n_strategies += 1
    U = np.concatenate((
        np.linspace(
            args.U_interval[0], 0, args.n_strategies//2, endpoint=False),
        np.linspace(
            0, args.U_interval[1], args.n_strategies//2 + 1, endpoint=True))
    )[:, None]
    # Array of shape #U^d
    # Ud = np.stack(np.meshgrid(*([U]*d))).reshape((3, -1)).T

    # Sigma:
    if args.initial_strategy == 'uniform':
        sigma = np.array([1]*len(U)) / len(U)
    else:
        with np.errstate(divide='ignore'):
            sigma = np.exp(-1/(1-(U**2)))
        sigma = sigma / np.sum(sigma)

    # Initial population
    points = np.random.uniform(
        args.point_interval[0], args.point_interval[1], args.n_points)
    N = len(points)
    d = len(points[0]) if isinstance(points[0], tuple) else 1
    locations = np.array(points).reshape(N, d)
    strategies = np.tile(sigma.flatten(), (len(points), 1))
    population = (locations, strategies)

    # J
    J_used = MyJ if args.my_j else OriginalJ

    ###########################################################################
    # 2. Run Minimize
    if args.carillo:
        history = carillo.minimize(f, population, vars(args))
    else:
        history = minimize(f, J_used, population, U, vars(args))

    ###########################################################################
    # 3. Visualize
    params_to_show = ['beta', 'gamma', 's_rounds', 'stepsize']
    # params_to_show = ['beta', 'gamma', 'stepsize']
    parameter_text = '\n'.join(
        [f'n_points: {N}'] +
        [f'{p}: {vars(args)[p]}'for p in params_to_show])
    plot_range = np.linspace(args.plot_range[0], args.plot_range[1], 1000)

    # Need this weird construct
    for i in range(2):
        anim = vis.full_visualization(
            history, f, U,
            plot_range=plot_range,
            parameter_text=parameter_text,
            max_len=args.max_animation_seconds*60)
        if i==0:
            plt.show()

            response = input('Save plot? [y/N]')
            if response.lower().startswith('y'):
                continue
            else:
                break
        elif i==1:
            text = input('Name?')
            logging.info('Saving animation, this might take a while')
            # text = '_'.join([f'{p}{vars(args)[p]}'for p in params_to_show])
            anim.save(
                f'examples/{text}_{seed}.mp4',
                fps=60)

    ###########################################################################
    # 4. Analyze Convergence behaviour
    if not (f == simple_nonconvex_function):
        return
    ax = convergence_analysis.visualize(history, f)
    fig = ax.get_figure()
    # g = convergence_analysis.visualize(history, f)
    plt.show()
    if 'text' in locals():
        path = f'examples/{text}_{seed}_fvalue.png'
        fig.savefig(path)
        # g.savefig(path)


if __name__ == '__main__':
    main()
