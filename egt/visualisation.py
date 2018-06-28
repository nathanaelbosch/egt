"""Visualisation"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec


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
    fig = plt.figure()
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
        strategy_plot_number=4):
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
    colors = ['red', 'blue', 'green', 'orange']
    n_points = history[0].shape[0]
    if n_points > len(colors):
        return graph_visualization(history, function, U)

    # Setup layout
    fig = plt.figure()
    gs = gridspec.GridSpec(
        3, strategy_plot_number)
    ax_function_graph = fig.add_subplot(gs[0:2, :])
    ax_strategies = [
        fig.add_subplot(gs[2, i]) for i in range(strategy_plot_number)]

    # Initialize function plot
    _xmin, _xmax = plot_range.min(), plot_range.max()
    _ymin, _ymax = function(plot_range).min(), function(plot_range).max()
    ax_function_graph.set_xlim((_xmin, _xmax))
    ax_function_graph.set_ylim((_ymin, _ymax))

    dotplots = []
    for i in range(n_points):
        dots, = ax_function_graph.plot(
            [], [],
            color=colors[i],
            marker='o')
        dotplots.append(dots)
    base_function, = ax_function_graph.plot([], [], lw=2)
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
        for i, dots in enumerate(dotplots):
            dots.set_data(point_locations_x[i], point_locations_y[i])

        for i in range(n_points):
            linearr[i].set_data(
                U, current_pop[i, 1:]/np.max(current_pop[i, 1:]))
        return dots, linearr

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(history), interval=1, blit=False,
                                   repeat=False)
    return anim
