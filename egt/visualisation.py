"""Visualisation"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec


TUM_COLORS = {
    'blue': '#005293',
    'accent_light_blue': '#98c6ea',
    'dark_blue': '#003359',
    'accent_dark_blue': '#64a0c8',
    'accent_ivory': '#dad7cb',
    'accent_orange': '#e37222',
    'accent_green': '#a2ad00'
}
FIGSIZE = (11.692, 8.267)


def graph_visualization(
        history,
        function,
        U,
        plot_range=np.arange(-3, 3, 0.001)):
    """Animation of the graph of the function and dots moving over it

    Parameters
    ----------
    history : list
        Contains locations and strategies of individuals at each timestep
    function : function
        Function to minimize
    U : np.array
        Set of all possible strategies
    plot_range : np.array, optional
        Range in which we plot the function

    Returns
    -------
    np.matplotlib.animation.Animation
        Animation to show at the end
    """
    fig = plt.figure(figsize=FIGSIZE)
    _xmin, _xmax = plot_range.min(), plot_range.max()
    _ymin, _ymax = function(plot_range).min(), function(plot_range).max()
    ax = plt.axes(
        xlim=(_xmin, _xmax),
        ylim=(_ymin, _ymax))
    dots, = ax.plot([], [], 'ro')
    base_function, = ax.plot([], [], lw=2)
    base_function.set_data(plot_range, function(plot_range))
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
        point_locations_y = function(point_locations_x)
        dots.set_data(point_locations_x, point_locations_y)

        y1 = current_pop[0]
        line2.set_data(y1[0] + U, y1[1:]/np.max(y1[1:]))
        return dots, line2

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(history), interval=1, blit=False,
                                   repeat=False)
    return anim


def full_visualization(
        history,
        function,
        U,
        plot_range=np.arange(-3, 3, 0.001),
        parameter_text=''):
    """Animation

    Complete visualization of the process:
        - Graph of the function and dots moving over it
        - Graph of the mixed strategies

    Parameters
    ----------
    history : list
        Contains locations and strategies of individuals at each timestep
    function : function
        Function to minimize
    U : np.array
        Set of all possible strategies
    plot_range : np.array, optional
        Range in which we plot the function

    Returns
    -------
    np.matplotlib.animation.Animation
        Animation to show at the end
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    n_points = history[0].shape[0]
    if n_points > len(colors):
        return graph_visualization(history, function, U)

    # Setup layout
    fig = plt.figure(figsize=FIGSIZE)
    gs = gridspec.GridSpec(
        3, n_points)
    ax_function_graph = fig.add_subplot(gs[0:2, :])
    ax_strategies = [
        fig.add_subplot(gs[2, i]) for i in range(n_points)]

    # Initialize function plot
    _xmin, _xmax = plot_range.min(), plot_range.max()
    _ymin, _ymax = function(plot_range).min(), function(plot_range).max()
    ax_function_graph.set_xlim((_xmin, _xmax))
    ax_function_graph.set_ylim((_ymin, _ymax))

    # Text to specify the parameters
    ax_function_graph.text(
        0.5, 0.8, parameter_text,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax_function_graph.transAxes,
        fontsize=12,
        # animated=True,
        bbox=dict(facecolor=TUM_COLORS['blue'], alpha=0.5),
    )

    dotplots = []
    for i in range(n_points):
        dots, = ax_function_graph.plot(
            [], [],
            color=colors[i],
            marker='o')
        dotplots.append(dots)
    base_function, = ax_function_graph.plot(
        [], [], lw=2, color=TUM_COLORS['blue'], antialiased=True)
    base_function.set_data(plot_range, function(plot_range))

    # Initialize strategy plots
    linearr = []
    for i, ax in enumerate(ax_strategies):
        ax.set_ylim([0, 1.1])
        ax.set_xlim([U.min(), U.max()])
        line, = ax.plot(
            [], [],
            color=colors[i])
        linearr.append(line)

    plt.tight_layout()

    def init():
        for dots in dotplots:
            dots.set_data([], [])
        for line in linearr:
            line.set_data([], [])
        return dots, linearr

    def animate(i):
        current_pop = history[i]
        point_locations_x = current_pop[:, 0]
        point_locations_y = function(point_locations_x)
        for j, dots in enumerate(dotplots):
            dots.set_data(point_locations_x[j], point_locations_y[j])

        for j in range(n_points):
            linearr[j].set_data(
                U, current_pop[j, 1:]/np.max(current_pop[j, 1:]))
        return dots, linearr

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(history), interval=1, blit=False,
                                   repeat=False)
    return anim


def plot_2d(
        f,
        ax=None,
        _plot_range=np.arange(-3, 3, 0.1),
        type='wireframe',
        alpha=30):
    M = len(_plot_range)
    X = np.tile(_plot_range, (M, 1))
    Y = X.T
    grid = np.stack((X.flatten(), Y.flatten()), axis=1)
    Z = f(grid).reshape(M, M)

    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111, projection='3d')

    transparency_suffix = hex(int(alpha/100*255))[2:]
    assert len(transparency_suffix) == 2
    if type=='wireframe':
        ax.plot_wireframe(
            X, Y, Z,
            colors=TUM_COLORS['blue']+transparency_suffix)
    elif type=='trisurf':
        ax.plot_trisurf(
            X.flatten(),
            Y.flatten(),
            Z.flatten(),
            linewidth=0.2,
            antialiased=True,
            color=TUM_COLORS['blue']+transparency_suffix)

    return ax


def plot_points_2d(f, points, ax=None, **kwargs):
    """Scatter Plot of 2d points

    kwargs will be passed directly to ax.scatter
    """
    kwargs.setdefault('s', 100)
    kwargs.setdefault('c', TUM_COLORS['accent_orange'])
    kwargs.setdefault('depthshade', False)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        points[:, 0], points[:, 1], f(points),
        **kwargs)

    return ax
