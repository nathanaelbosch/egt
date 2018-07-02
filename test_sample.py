"""Test file

Single file, as there will probably not be much testing necessary"""
import numpy as np
from egt.vectorized_script import *


def test_J():
    population = create_initial_population(starting_locations)
    points = population[:, 0]
    out = J_vectorized(population[:, 0])
    old = out.copy()
    for i in range(N):
        for j in range(N-1):
            # if i<=j:
            #     for k in range(len(U)):
            #         old[i, j, k] = J(
            #             points[i], U[k], points[j+1])
            # else:
            for k in range(len(U)):
                old[i, j, k] = J(
                    points[i], U[k], points[j])
    assert np.all(old == out)
