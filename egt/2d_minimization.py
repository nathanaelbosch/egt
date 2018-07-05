import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm
import random
import argparse
import itertools

import egt.visualisation as vis


# %pylab
def f(x):
    return x ** 2 + 0.5 * np.sin(30*x)


def f2(point):
    # Assumes x is a Nx2 array of N points with dimension 2
    x_coord = point[:, 0]
    y_coord = point[:, 1]
    return f(x_coord) + f(y_coord)


ax = vis.plot_2d(f2, alpha=50)

points = np.array([[0, 1], [1, 1], [-1, -1]])
ax = vis.plot_points_2d(f2, points, ax=ax)
plt.show()
