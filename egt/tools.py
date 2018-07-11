import numpy as np


def positive(x):
    """Helper function as it's needed in the original J"""
    return (np.abs(x) + x)/2


def my_norm(x, ord=2, axis=2):
    """Norm to the power of ord"""
    return np.sum(np.abs(x) ** ord, axis=axis)


def J_original(x, u, x2, **kwargs):
    """J as described by Massimo"""
    alpha = kwargs.get('alpha', DEFAULT_PARAMS.get('alpha'))
    f = kwargs.get('f', DEFAULT_PARAMS.get('f'))
    with np.errstate(divide='ignore'):
        out = np.exp(
            -((u - positive(np.tanh(3*(f(x) - f(x2)))) * (x2 - x))**2) /
            ((x-x2) ** alpha + (f(x) - f(x2)) ** alpha))
    return out
