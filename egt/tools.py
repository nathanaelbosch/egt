import numpy as np


def positive(x):
    """Helper function as it's needed in the original J"""
    return (np.abs(x) + x)/2


def sigmoid(x):
    with np.errstate(over='ignore'):
        return 1 / (1 + np.exp(-x))
