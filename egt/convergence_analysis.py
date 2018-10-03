"""Visualize convergence"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .test_functions import simple_nonconvex_function, ackley
from .visualisation import FIGSIZE


def max_distances(history, f):
    data = {'max_distance': max_distances}
    stats = pd.DataFrame(data)

    plt.gcf().clear()
    stats.max_distance.plot(
        logy=True)
    plt.show()


def visualize(history, f):
    data = {}
    plot_range = np.arange(-10, 10, 0.0001)
    values = f(plot_range)
    min_val, min_loc = values.min(), values.argmin()

    location_hist, _ = zip(*history)
    location_hist = list(location_hist)
    while np.all(location_hist[-1] == location_hist[-2]):
        location_hist.pop(-1)
    locs = np.array(location_hist).reshape(len(location_hist), -1)

    # Norm the values to have value 0 at the min
    mean_vals = f(locs).mean(axis=1) - min_val
    data['mean_values'] = mean_vals

    # if f == ackley or f == simple_nonconvex_function:
    #     if f == simple_nonconvex_function:
    #         def convex_f(x):
    #             return x**2
    #         mean_vals_convex = convex_f(locs).mean(axis=1) - min_val
    #         data['mean_values_convex'] = mean_vals_convex
    #     elif f == ackley:
    #         # Create convex hull on compact set
    #         left_x, right_x = np.floor(locs.min()), np.ceil(locs.max())
    #         left_x = min(-20, left_x)
    #         right_x = max(20, right_x)
    #         left_y, right_y = f(left_x), f(right_x)
    #         min_y = f(0)

    #         def convex_f(x):
    #             return np.where(
    #                 x > 0,
    #                 x * (right_y / right_x) + min_y,
    #                 x * (left_y / left_x) + min_y)
    #     mean_vals_convex = convex_f(locs).mean(axis=1) - min_val
    #     data['mean_values_convex'] = mean_vals_convex

    # max_distances = locs.max(axis=1) - locs.min(axis=1)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    df = pd.DataFrame(data)
    ax = df.plot(
        ax=ax,
        logy=True
    )
    plt.tight_layout()

    # win_size = 100
    # df.mean_values.rolling(win_size, center=True).mean().plot(
    #     ax=ax,
    #     color='blue',
    #     alpha=0.5,
    #     linestyle='dashed')
    # df.mean_values_convex.rolling(win_size, center=True).mean().plot(
    #     ax=ax,
    #     color='orange',
    #     alpha=0.5,
    #     linestyle='dashed')

    return ax

    if False:
        data = {
            'iteration': np.concatenate((
                np.arange(locs.shape[0]), np.arange(locs.shape[0]))),
            'mean_values': np.concatenate((mean_vals, mean_vals_convex)),
            'convex_hull': ([False for _ in mean_vals] +
                            [True for _ in mean_vals_convex]),
        }

        df = pd.DataFrame(data)
        df['log_mean_values'] = np.log(df.mean_values)
        # df.iteration = df.iteration // 10 * 10
        sns.set()
        g = sns.lmplot(
            data=df,
            x='iteration',
            y='log_mean_values',
            hue='convex_hull',
        )
        # ax.set(yscale='log')

        return g
