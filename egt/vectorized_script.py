"""
Difference to first script: Optimize code, make it run faster!
- Data in a nicer way than lists
- Function to be applied to the whole array in the best possible way
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import tqdm
# tqdm.monitor_interval = 0
np.random.seed(0)


###############################################################################
# Parameters
###############################################################################
_plot_range = np.arange(-3, 3, 0.001)
_strategy_resolution = 0.001           # Discretization of the strategies
alpha = 2                   # Parameter of J
beta = 1                    # Reweighting paramenter
gamma = 1                   # Additional update paremeter
delta_t = 0.1              # Stepsize at each iteration
total_steps = 60*60
# N = 4
starting_locations = [3, 0, 1, -1]
# starting_locations = [np.random.uniform(-2, 2) for i in range(N)]


###############################################################################
# Setup
###############################################################################
# We want to minimize the following function with EGT
def f(x):
    return x ** 2 + 0.5 * np.sin(30*x)


# All available strategies:
U = np.arange(-1, 1, _strategy_resolution)

# Initial mixed strategy - continuous:
sigma = np.exp(-1/(1-(U**2)))
sigma = sigma / np.sum(sigma)

# Initial population: Now as a matrix. First col location, rest mixed strategy
population = np.concatenate(
    (np.array(starting_locations).reshape((len(starting_locations), 1)),
     np.tile(sigma, (len(starting_locations), 1))),
    axis=1)
N = population.shape[0]

# Object to save the whole process
history = []
history.append(population)


# Game description:
# J as described by Massimo
def positive(x):
    return (np.abs(x) + x)/2
def J_original(x, u, x2):
    return np.exp(
        -((u - positive(np.tanh(3*(f(x) - f(x2)))) * (x2 - x))**2) /
        ((x-x2) ** alpha + (f(x) - f(x2)) ** alpha))

# My playground now:
def J(x, u, x2):
    out = J_original(x, u, x2)
    out *= np.exp(-beta * f(x2))
    # out = out * delta
    return out

def J_vectorized(points):
    """Idea: generate a whole NxNx#Strategies tensor with the values of J

    It is a #U x N x N tensor now
    axis=0 the point to evaluate
    axis=1 the point to compare to
    axis=2 are the strategies
    """
    N = len(points)
    f_vals = f(points)
    f_diffs = np.tile(f_vals, reps=(N, 1)).T - np.tile(f_vals, reps=(N, 1))
    f_diffs_tanh = np.tanh(3*f_diffs)
    f_diffs_positive = np.where(
        f_diffs_tanh > 0,
        f_diffs_tanh,
        0)
    walk_dirs = np.tile(points, reps=(N, 1)) - np.tile(points, reps=(N, 1)).T
    walk_dirs_adj = f_diffs_positive * walk_dirs
    variance = walk_dirs ** alpha + f_diffs ** alpha
    out = np.exp(
        -1 * ((U.reshape(1, 1, len(U)) - walk_dirs_adj[:, :, None])**2) /
        variance[:, :, None])
    out *= np.exp(-beta * f(points))[None, :, None]
    return out


# Test if the new function is the same as the old!
points = population[:, 0]
out = J_vectorized(population[:, 0])
old = out.copy()
for i in range(N):
    for j in range(N):
        for k in range(len(U)):
            old[i, j, k] = J(
                points[i], U[k], points[j])
assert np.all(old == out)


###############################################################################
# Here the actual simulation starts
###############################################################################
print('Start')
sim_bar = tqdm.tqdm(range(total_steps))
sim_bar.set_description('Simulation')
for i in sim_bar:
    current_pop = history[-1]
    next_pop = current_pop.copy()

    # Strategy updates
    tot_J = J_vectorized(current_pop[:, 0])
    sum_i = tot_J.sum(axis=1)
    mean_outcome = (sum_i * population[:, 1:]).sum(axis=1)
    delta = sum_i - mean_outcome[:, None]
    delta = np.sum(
        tot_J - np.sum(
            tot_J * population[:, 1:][:, None, :], axis=2)[:, :, None],
        axis=1)
    next_pop[:, 1:] *= (1 + gamma * delta_t * delta)
    next_pop[:, 1:] /= next_pop[:, 1:].sum(axis=1)[:, None]

    # Location updates
    for j in range(len(current_pop)):
        next_pop[j, 0] += delta_t*np.random.choice(U, p=next_pop[j, 1:])

    history.append(next_pop)

    # Break condition for early stopping
    _locs = history[-1][:, 0]
    max_dist = max(_locs) - min(_locs)
    probability_to_stand = current_pop[:, 1000]
    # if max_dist < 0.01:
    if max_dist < 0.01 and probability_to_stand.sum() > N-(1e-5):
        print('Early stopping thanks to our rule!')
        break


###############################################################################
# Visualisation
###############################################################################
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
_xmin, _xmax = _plot_range.min(), _plot_range.max()
_ymin, _ymax = f(_plot_range).min(), f(_plot_range).max()
ax = plt.axes(
    xlim=(_xmin, _xmax),
    ylim=(_ymin, _ymax))
dots, = ax.plot([], [], 'ro')
base_function, = ax.plot([], [], lw=2)
base_function.set_data(_plot_range, f(_plot_range))
line2, = ax.plot([], [])


# initialization function: plot the background of each frame
def init():
    dots.set_data([], [])
    line2.set_data([], [])
    return dots, line2


# animation function.  This is called sequentially
def animate(i):
    current_pop = history[i]
    point_locations_x = np.array([y[0] for y in current_pop])
    point_locations_x = current_pop[:, 0]
    point_locations_y = f(point_locations_x)
    dots.set_data(point_locations_x, point_locations_y)

    # Density for one of the points:
    y1 = current_pop[0, :]
    line2.set_data(y1[0] + U, y1[1:]/np.max(y1[1:]))
    return dots, line2


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(history), interval=1, blit=False,
                               repeat=False)
plt.show()
