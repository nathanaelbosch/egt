"""Minimization as in the Paper from Carillo
"""
import numpy as np
import logging
import tqdm


def minimize(f, initial_population, parameters):
    """

    No more strategies etc, only locations
    """
    max_iterations = parameters['max_iterations']
    beta = parameters['beta']
    gamma = parameters['gamma']
    stepsize = parameters['stepsize']

    locations, _ = initial_population
    N, d = locations.shape

    assert d == 1, 'Not yet implemented for higher dimensions'
    locations.flatten()

    history = []
    history.append((locations.copy(), None))

    logging.info('Start simulation')
    sim_bar = tqdm.trange(max_iterations)
    for i in sim_bar:
        f_values = f(locations)
        f_values_adjusted = f_values - np.min(f_values)
        weights = np.exp(-beta * f_values_adjusted)
        mean_loc = np.sum(locations * weights) / np.sum(weights)

        directions = mean_loc - locations

        u = np.random.normal(0, 1, size=locations.shape)
        locations = (locations +
                     stepsize*directions +
                     gamma * np.sqrt(stepsize) * np.abs(directions) * u)

        history.append((locations.copy(), None))

        # Break condition for early stopping
        max_dist_from_mean = np.max(np.abs(directions))
        mean_value = np.mean(f(locations))
        sim_bar.set_description((
            '[Simulation] max_dist={:.2E} ' +
            'mean_value={:.2f}').format(
                max_dist_from_mean, mean_value))
        if max_dist_from_mean < 1e-3:
            break

    return history
