import matplotlib.pyplot as plt
import random
import numpy as np
import tqdm
import logging
logging.basicConfig(level=logging.WARNING)

import egt.vectorized_script as script


starting_locations = [-1, 0]
n_sims = 2
full_data = []

sim_bar = tqdm.tqdm(range(n_sims))
sim_bar.set_description('Mass Testing')
for i in sim_bar:
    seed = random.randint(0, 2**32-1)
    logging.info(f'Seed used for this simulation: {seed}')
    np.random.seed(seed)

    population = script.create_initial_population(starting_locations)
    history = script.simulate(population, script.J_vectorized)
    full_data.append((seed, history))


# full_data is a list of tuples (seed, history)
# history is a list of np.arrays containing the full states of the simulation
final_locations = np.array(
    [[seed] + list(history[-1][:, 0]) for seed, history in full_data])

ax = plt.plot(script._plot_range, script.f(script._plot_range))
ax.plot(final_locations, script.f(final_locations), 'ro')
plt.show()
