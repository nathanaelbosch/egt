"""I really want to clean up my script

Passing kwargs and parameters around leads to quite some errors.
"""
import numpy as np
import logging
import tqdm

import cython
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from cython.parallel import prange

from libc.math cimport exp

from egt.test_functions cimport simple_nonconvex_function_double
from egt.game_functions.original_J_cy cimport cython_naive


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def minimize(
        f, J_class,
        tuple initial_population,
        np.ndarray[DTYPE_t] U,
        dict parameters):
    """Run the WHOLE simulation


    Parameters
    ----------
    f : function
        Function to minimize
    J_class : subclass of egt.game_template.J_template
        The game J to use for minimization
    initial_population : tuple
        Tuple of (locations, strategies) describing the starting population
    U : np.array
        Set of available strategies for the individuals
    """
    # 1. Init
    #   Initialize population, strategies, parameters
    # 2. Run the iteration, consisting of
    #   a. Strategy updates: replicator dynamics
    #   b. Location updates
    cdef double beta = parameters['beta']
    cdef double gamma = parameters['gamma']
    cdef double stepsize = parameters['stepsize']
    cdef int max_iterations = parameters['max_iterations']
    cdef int s_rounds = parameters['s_rounds']
    cdef bint normalize_delta = parameters['normalize_delta']

    cdef int standing_index = np.where(np.isclose(U, 0))[0][0]

    J_vectorized = J_class.get(f, U)

    cdef np.ndarray[DTYPE_t, ndim=1] locations
    cdef np.ndarray[DTYPE_t, ndim=2] strategies
    locations, strategies = initial_population
    cdef np.ndarray[DTYPE_t, ndim=2] new_strategies = np.copy(strategies)
    # N, d = locations.shape
    cdef int N = locations.shape[0]
    cdef int U_num = U.shape[0]

    ###########################################################################
    # Start the iterative part here
    ###########################################################################
    cdef DTYPE_t delta
    cdef DTYPE_t xi, ui, xj, v

    cdef list history = [None]*max_iterations
    history[0] = (locations.copy(), strategies.copy())

    logging.info('Start simulation')

    cdef int _current_iteration, _current_strategyupdate_iteration
    cdef int i, k, _j, _k
    cdef double probsum
    cdef int random_u_index
    cdef np.ndarray[DTYPE_t, ndim=1] single_mixed_strategy = np.zeros_like(U)
    cdef double max_staying_uncertainty, mean_value

    sim_bar = tqdm.trange(max_iterations)
    for _current_iteration in sim_bar:

        # 1: Strategy updates, possibly multiple iterations
        for _current_strategyupdate_iteration in range(s_rounds):
            for i in range(N):
                xi = locations[i]
                probsum = 0
                for k in range(U_num):
                    ui = U[k]

                    # Calculate delta
                    delta = 0
                    for _j in range(N):
                        xj = locations[_j]
                        delta += cython_naive(xi, ui, xj)
                        for _k in range(U_num):
                            v = U[_k]
                            delta -= strategies[i, _k] * cython_naive(
                                xi, v, xj)

                    # Update the current strategy for the current guy
                    new_strategies[i, k] *= (1 + gamma * stepsize * delta)

                    # Sanity check: Strategies should sum to 1
                    probsum += new_strategies[i, k]

                # if probsum != 1 and np.isclose(probsum, 1):
                #     # Numerical problems, but otherwise should be fine
                #     print("Strategies do not sum to 1 but almost do!")

            # Now put the `new_strategies` into `strategies`
            for i in range(N):
                for k in range(U_num):
                    strategies[i, k] = new_strategies[i, k]


        # Location updates
        for i in range(N):
            for k in range(U_num):
                single_mixed_strategy[k] = strategies[i, k]
            random_u_index = np.random.choice(
                U_num, p=single_mixed_strategy)
            locations[i] += stepsize*U[random_u_index]

        history[_current_iteration] = (locations.copy(), strategies.copy())

        # Break condition for early stopping
        # max_dist = (max(locations) - min(locations))
        for i in range(N):
            max_staying_uncertainty = 0
            if 1-strategies[i, standing_index] > max_staying_uncertainty:
                max_staying_uncertainty = 1-strategies[i, standing_index]

            mean_value += simple_nonconvex_function_double(locations[i])/N

        sim_bar.set_description(
            ('[Simulation] ' +
             'staying_uncertainty={:.2E} ' +
             'mean_value={:.2f}').format(
                max_staying_uncertainty, mean_value))
        if max_staying_uncertainty == 0.0:
            logging.info('Early stopping! No point wants to move anymore')
            break

    # logging.info(f'Max distance at the end: {max_dist}')
    logging.info(f'Max "staying-uncertainty": {max_staying_uncertainty}')

    return history
