"""Monte Carlo Simulation

Run N, e.g. 100, simulations and save the result to some HDF5.
"""
import logging
logging.basicConfig(level=logging.WARNING)

from tqdm import trange
import pandas as pd
import h5py

from egt.main import *


N_SIMULATIONS = 20
args_string = [
    '--gamma', '0.05',
    '--test-function', 'ackley',
    '--max-iterations', '1000000',
    '--gamma', '0.05',
    '--stepsize', '0.5',
    '--n-points', '50',
    '--plot-range', '-100', '100',
    '--point-interval', '-100', '100',
    '--initial-strategy', 'standard',
    '--s-rounds', '1',
    '--max-animation-seconds', '60',
    '--epsilon', '1e-4',
    '--normalize-delta',
    '--U-interval', '-10', '10',
    '--beta', '30',
    '--n-strategies', '100',
    # '--save-prefix', 'ackley',
]
args = parse_args(args_string)

# SETUP
f = funcs[args.test_function]
U = make_strategies(args.n_strategies, args.U_interval[1])
J_used = MyJ if args.my_j else OriginalJ


# Find the first empty index so that we can use it to store the data
try:
    for first_empty_index in range(10000):
        with h5py.File('examples/ackley_montecarlo_data.h5', 'r') as file:
            locs = file[f'locations_{first_empty_index}']
except Exception as e:
    print('Stopped testing indices after the following exception:', e)
    print('Seems like the first empty index is', first_empty_index)


mean_val_df = pd.DataFrame()
mean_dist_df = pd.DataFrame()
for sim in trange(N_SIMULATIONS):
    population = make_initial_population(
        args.initial_strategy, args.point_interval, args.n_points, U)

    history = minimize(f, J_used, population, U, vars(args))

    # We need only the locations
    location_hist, _ = zip(*history)
    location_hist = list(location_hist)
    locs = np.array(location_hist).reshape(len(location_hist), -1)

    # Now save the locations. They contain all info!ä
    index = first_empty_index + sim
    with h5py.File('examples/ackley_montecarlo_data.h5', 'a') as file:
        print('')
        print('Saving location history to index', index)
        print('')
        file.create_dataset(f'locations_{index}', data=locs)
