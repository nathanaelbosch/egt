"""Test file

Single file, as there will probably not be much testing necessary"""
import numpy as np

from egt.vectorized_script import f, U, create_initial_population
from egt.original_J import OriginalJ
from egt.alternative_J import MyJ


starting_locations = np.linspace(-10, 10, 5)


def test_J():
    locations, strategies = create_initial_population(starting_locations)
    J = OriginalJ(f, U)
    out = J._vectorized(locations)
    old = J._badly_vectorized(locations)
    assert np.all(old == out)


def test_MyJ():
    locations, strategies = create_initial_population(starting_locations)
    J = MyJ(f, U)
    out = J._vectorized(locations)
    old = J._badly_vectorized(locations)
    assert np.all(old == out)
