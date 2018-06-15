import numpy as np
import matplotlib.pyplot as plt


_range = np.arange(-10, 10, 0.1)


def f(x):
    return x ** 2 + 0.5 * np.sin(10*x)


# All available strategies:
_stepsize = 0.01
U = np.arange(-1+_stepsize, 1-_stepsize, _stepsize)

# Initial mixed strategy - continuous:
sigma = np.exp(-1/(1- (U**2)))
sigma = sigma / np.sum(sigma)


# Simple case: Two points, {0, 1}
x1, sigma1 = 0, sigma
x2, sigma2 = 1, sigma


# J as described by Massimo
def positive(x):
    return (np.abs(x) + x)/2
alpha = 2
def J(x, u, x2):
    return np.exp(
        -((u - positive(np.tanh(3*(f(x) - f(x2)))) * (x2 - x))**2) /
        ((x-x2) ** alpha + (f(x) - f(x2)) ** alpha))


for i in range(10000):
    # No sum here as there are only two particles
    delta1 = 1/1 * (J(x1, U, x2) - np.sum(J(x1, U, x2) * sigma1))
    delta2 = 1/1 * (J(x2, U, x1) - np.sum(J(x2, U, x1) * sigma2))

    gamma = 1
    delta_t = 0.1
    # Update sigmas
    sigma1 = (1 + gamma * delta_t * delta1) * sigma1
    sigma2 = (1 + gamma * delta_t * delta2) * sigma2

    # Update Locations
    x1 = x1 + delta_t * np.random.choice(U, p=sigma1)
    x2 = x2 + delta_t * np.random.choice(U, p=sigma2)

