import numpy as np
import matplotlib.pyplot as plt


top = 0.5

_strategy_resolution = 0.01
U = np.arange(-1, 1, _strategy_resolution)[:, None]
standing_index = np.where(np.isclose(U, 0))[0][0]
U[standing_index] = 0


def f(x):
    """Function to minimize"""
    # return(
    #     ((x-2)**2 * (x+2)**2 + 10*x) / (x**2 + 1) +
    #     0.3 * (np.abs(x)+5) * np.sin(10*x))
    return x**2


def J_idea(x, top):
    if top > 1:
        a = 1 / ((1-top)**2 - (1+top)**2)
        b = -a * ((1+top)**2)
    elif top < -1:
        a = 1 / ((1+top)**2 - (1-top)**2)
        b = -a * ((1-top)**2)
    else:
        if top > 0:
            a = -1/((-1-top)**2)
        else:
            a = -1/((1-top)**2)
        b = 1
    return a * ((x-top)**2) + b


def myJ(x, u, x2):
    if f(x2) < f(x):
        return J_idea(u, x2-x)
    else:
        return J_idea(u, 0)


def myJ_vectorized(locations):
    N, d = locations.shape

    f_vals = f(locations)
    assert len(f_vals) == N
    f_vals = f_vals.flatten()
    # f_diffs should contain xi-xj at location ij; >0 if j is better
    f_diffs = (np.tile(f_vals, reps=(N, 1)).T -
               np.tile(f_vals, reps=(N, 1)))

    walk_dirs = (np.tile(locations[None, :, :], (N, 1, 1)) -
                 np.tile(locations[:, None, :], (1, N, 1)))

    walk_dirs_ifgood = np.where(
        f_diffs[:, :, None] > 0,
        walk_dirs,
        0)

    # Starting here its not higher-dimensional
    walk_dirs_ifgood = walk_dirs_ifgood.reshape(N, N)
    U_adj = (U.flatten()[None, None, :] - walk_dirs_ifgood[:, :, None]) ** 2
    with np.errstate(divide='ignore'):
        a = np.where(
            walk_dirs_ifgood > 1,
            1 / ((1-walk_dirs_ifgood)**2 - (1+walk_dirs_ifgood)**2),
            np.where(
                walk_dirs_ifgood < -1,
                1 / (((1+walk_dirs_ifgood)**2) - ((1-walk_dirs_ifgood)**2)),
                np.where(
                    walk_dirs_ifgood > 0,
                    -1/((-1-walk_dirs_ifgood)**2),
                    -1/((1-walk_dirs_ifgood)**2)
                )
            )
        )
        b = np.where(
            walk_dirs_ifgood > 1,
            -a * ((1+walk_dirs_ifgood)**2),
            np.where(
                walk_dirs_ifgood < -1,
                -a * ((1-walk_dirs_ifgood)**2),
                1
            )
        )

    # b = 1
    out = a[:, :, None] * U_adj + b[:, :, None]
    # out = a[:, :, None] * U_adj + b
    return out


if __name__ == '__main__':
    starting_points = [-1, -1, 0, 0.5]
    x1, x2, x3, x4 = starting_points
    locations, strategies = create_initial_population(starting_points)
    stepsize = 0.01
    n=100
    plotrange = np.arange(-12, 2, 0.01)
    ax = plt.subplot(111)
    ax.set_xlim([-2, 1])
    ax.set_ylim([0, 1.1])
    # ax.plot(
    #     U, [myJ(x1, u, x2) for u in U])
    # ax.plot(
    #     U, [myJ(x1, u, x3) for u in U])
    # ax.plot(
    #     U, [myJ(x1, u, x4) for u in U])
    fullJ = myJ_vectorized(locations)
    ax.plot(
        x1+U, fullJ[0, 1, :])
    ax.plot(
        x1+U, fullJ[0, 2, :])
    ax.plot(
        x1+U, fullJ[0, 3, :])
    x1_J = fullJ[0, :, :]
    weights = np.exp(-100*f(locations))
    weights[0] = 0
    x1_J = x1_J * weights / weights.sum()
    ax.plot(
        x1+U, x1_J.sum(axis=0))
    ax.plot(
        plotrange, f(plotrange))
    ax.plot(
        x1, f(x1), 'x',
        x2, f(x2), 'x',
        x3, f(x3), 'x',
        x4, f(x4), 'x', color='black')
    ax.plot(
        x1+U, fullJ[0, 1, :]**n)
    plt.show()
