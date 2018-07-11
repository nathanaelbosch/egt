"""Test file

Single file, as there will probably not be much testing necessary"""
import numpy as np
from egt.vectorized_script import *
from egt.alternative_J import myJ, myJ_vectorized


def test_J():
    starting_locations = get_starting_locations()
    N = len(starting_locations)
    locations, strategies = create_initial_population(starting_locations)
    out = J_vectorized(locations)
    old = out.copy()
    for i in range(N):
        for j in range(N):
            for k in range(len(U)):
                old[i, j, k] = J(
                    locations[i], U[k], locations[j])
    assert np.all(old == out)


def test_myJ():
    starting_locations = get_starting_locations()
    N = len(starting_locations)
    locations, strategies = create_initial_population(starting_locations)
    out = myJ_vectorized(locations)
    old = out.copy()
    for i in range(N):
        for j in range(N):
            for k in range(len(U)):
                old[i, j, k] = myJ(
                    locations[i], U[k], locations[j])
    assert np.all(old == out)
