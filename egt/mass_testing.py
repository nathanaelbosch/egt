import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
import numpy as np
import tqdm
import logging
logging.basicConfig(level=logging.WARNING)

# import egt.vectorized_script as script
from egt.vectorized_script import *

###############################################################################
# Parameters
###############################################################################
PARAMS = {
    'n_sims': 200,
    'starting_locations': [-1, 0, 1],
    # 'f': lambda x: x**2 + 0.5*np.sin(30*x),
    'alpha': 2,
    'beta': 0.1,
    'gamma': 0.1,
    'delta_t': 0.1,
    'total_steps': 5*60*60,
    's_rounds': 1,
}

VISUAL = False
FULL_HISTORY = False


###############################################################################
# Simulation and Visualization
###############################################################################
if VISUAL:
    plot_range=np.arange(-3, 3, 0.001)
    _xmin, _xmax = plot_range.min(), plot_range.max()
    _ymin, _ymax = f(plot_range).min(), f(plot_range).max()
    ax = plt.axes(
        xlim=(_xmin, _xmax),
        ylim=(_ymin, _ymax))
    ax.plot(
        plot_range, f(plot_range))
    plt.draw()
    plt.pause(.01)

# Simulation
full_data = []
sim_bar = tqdm.trange(PARAMS['n_sims'])
sim_bar.set_description('Mass Testing')
for i in sim_bar:
    seed = random.randint(0, 2**32-1)
    logging.info(f'Seed used for this simulation: {seed}')
    np.random.seed(seed)

    population = create_initial_population(starting_locations)
    history = simulate(population, J_vectorized, **PARAMS)
    if not FULL_HISTORY:
        history = [history[-1]]
    full_data.append((seed, history))
    final_locations = history[-1][:, 0]
    if VISUAL:
        ax.plot(
            final_locations, f(final_locations),
            'x',
            color='black',
            alpha=0.2)
        plt.draw()
        plt.pause(.01)


if VISUAL:
    plt.show()
# plt.savefig('examples/')


###############################################################################
# Quantitative analysis
###############################################################################
# Goal here was to find suitable boundaries to break the simulation loop
# Result: distance < 1e-2, max_uncertainty < 1e-15
full_final_locations = np.array(
    [list(history[-1][:, 0]) for seed, history in full_data])
max_distances = np.max(
    full_final_locations, axis=1) - np.min(full_final_locations, axis=1)


probs_to_stay = np.stack(
    (history[-1][:, 1+len(U)//2] for _, history in full_data))
max_uncertainty = 1-probs_to_stay.min(axis=1)

# Visualize this:
# 1. Histogram of the distances afterwards
plt.cla()
ax = sns.distplot(np.log10(max_distances), bins=100, kde=False, rug=True)
ax.set_title('Max distance between points - log scale')
plt.savefig('examples/log_distance.png')
plt.cla()
ax = sns.distplot(max_distances, bins=100, kde=False, rug=True)
ax.set_title('Max distance between points')
plt.savefig('examples/distance.png')
plt.cla()
# 2. Histogram of the uncertainty at the end
ax = sns.distplot(np.log10(max_uncertainty), bins=25, kde=False, rug=True)
ax.set_title('1-min("Probability to stay in place") - log scale')
plt.savefig('examples/log_uncertainty.png')
plt.cla()
ax = sns.distplot(max_uncertainty, bins=25, kde=False, rug=True)
ax.set_title('1-min("Probability to stay in place")')
plt.savefig('examples/uncertainty.png')

# Save Data
DATA = {
    'PARAMS': PARAMS,
    'full_data': full_data,
}
with open('examples/mass_testing_data.pickle', 'wb') as f:
    pickle.dump(DATA, f, pickle.HIGHEST_PROTOCOL)
