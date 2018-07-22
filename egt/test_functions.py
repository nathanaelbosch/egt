import numpy as np


def ackley(x):
    out = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x) + 1)) + np.exp(1) + 20
    return out


def two_wells(x):
    return (((x-2)**2 * (x+2)**2 + 10*x) / (x**2 + 1) +
            0.3 * (np.abs(x)+5) * np.sin(10*x))


def easom(x):
    return np.cos(x) * np.exp(-((x-np.pi)**2))
