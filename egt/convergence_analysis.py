"""Visualize convergence"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .test_functions import convex_hull


def max_distances(history, f):
    data = {'max_distance': max_distances}
    stats = pd.DataFrame(data)

    plt.gcf().clear()
    stats.max_distance.plot(
        logy=True)
    plt.show()


def visualize(history, f):
    range = np.arange(-10, 10, 0.0001)
    values = f(range)
    min_val, min_loc = values.min(), values.argmin()

    location_hist, _ = zip(*history)
    locs = np.array(location_hist).reshape(len(location_hist), -1)

    # Norm the values to have value 0 at the min
    mean_vals = f(locs).mean(axis=1) - min_val
    convex_f = convex_hull(
        f, plot_range=np.arange(locs.min(), locs.max(), 0.01))
    mean_vals_convex = convex_f(locs).mean(axis=1) - min_val
    # max_distances = locs.max(axis=1) - locs.min(axis=1)

    data = {
        'mean_values': mean_vals,
        'mean_values_convex': mean_vals_convex
    }

    df = pd.DataFrame(data)
    ax = df.plot(
        y=['mean_values', 'mean_values_convex'],
        logy=True
    )

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
