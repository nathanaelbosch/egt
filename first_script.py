"""
"""
import numpy as np
import matplotlib.pyplot as plt


_range = np.arange(-10, 10, 0.1)

###############################################################################
# Setup: We want to minimize the following function with EGT
###############################################################################
def f(x):
    return x ** 2 #+ 0.5 * np.sin(10*x)
# To plot:
# plt.plot(_range, f(_range))


# All available strategies:
_stepsize = 0.01
U = np.arange(-1+_stepsize, 1-_stepsize, _stepsize)


# Initial mixed strategy - continuous:
sigma = np.exp(-1/(1- (U**2)))
sigma = sigma / np.sum(sigma)


# Now with more than two points
population = []
population.append([0, sigma])
population.append([1, sigma])
population.append([-1, sigma])
population.append([2, sigma])


# J as described by Massimo
def positive(x):
    return (np.abs(x) + x)/2
alpha = 2
def J(x, u, x2):
    return np.exp(
        -((u - positive(np.tanh(3*(f(x) - f(x2)))) * (x2 - x))**2) /
        ((x-x2) ** alpha + (f(x) - f(x2)) ** alpha))


###############################################################################
# Here the actual simulation starts
###############################################################################
print('Start')
print([y[0] for y in population])
beta = 1
gamma = 0.5
delta_t = 0.1
for i in range(10000):
    if i % 1000 == 0:
        print([y[0] for y in population])
    # No sum here as there are only two particles
    deltas = []
    for j, (x1, sigma1) in enumerate(population):
        rescale_factor = np.sum([np.exp(- beta * f(x2)) for k, (x2, _) in enumerate(population) if k != j])
        deltas.append(1/rescale_factor * np.sum(
            [(J(x1, U, x2)-np.sum(J(x1, U, x2) * sigma1)) * np.exp(-beta*f(x2))
             for k, (x2, _) in enumerate(population) if k != j], axis=0))

    # Update sigmas
    for j in range(len(population)):
        population[j][1] *= (1 + gamma * delta_t * deltas[j])
        population[j][1] /= np.sum(population[j][1])
        population[j][0] += delta_t * np.random.choice(U, p=population[j][1])


# Locations:
print([y[0] for y in population])
