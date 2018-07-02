import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm
import random
import argparse
import itertools

# import vectorized_script
%pylab

def f(x):
    return x ** 2 + 0.5 * np.sin(30*x)

def _f2(x, y):
    return f(x)+f(y)


def f2(point):
    # Assumes x is a Nx2 array of N points with dimension 2
    x_coord = point[:, 0]
    y_coord = point[:, 1]
    return f(x_coord) + f(y_coord)


_plot_range = np.arange(-3, 3, 0.1)
M = len(_plot_range)
X = np.tile(_plot_range, (M, 1))
Y = X.T
grid = np.stack((X.flatten(), Y.flatten()), axis=1)
Z = f2(grid).reshape(M, M)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z)
# ax.plot_trisurf(
#     X.flatten(),
#     Y.flatten(),
#     Z.flatten(),
#     linewidth=0.2,
#     antialiased=True)

points = np.array([[0, 1], [1, 1], [-1, -1]])
ax.scatter(
    points[:, 0], points[:, 1], f2(points),
    c='red',
    s=100,
    depthshade=False)
