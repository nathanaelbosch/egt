"""Visualisation"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from abc import ABC, abstractmethod


from egt.animation import (
    Animation, DotsAnimation, StrategyAnimation, FrameCounter,
    StrategyOnDotsAnimation)


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


def plot_function_graph(ax, f, plot_range):
    ax.plot(
        plot_range, f(plot_range),
        lw=2,
        color=TUM_COLORS['blue'])


def plot_parameter_text(ax, text):
    ax.text(
        0.99, 0.02, text,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        fontsize=12,
        # animated=True,
        bbox=dict(facecolor=TUM_COLORS['blue'], alpha=0.5),
    )


def graph_visualization(
        history,
        f,
        U,
        plot_range=np.arange(-3, 3, 0.001),
        parameter_text='',
        max_len=30*60):
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
    # Characteristic thing of this function: Layout, and choice of components
    # Layout: Upper side with the graph, bottom with the strategies
    fig = plt.figure(figsize=FIGSIZE)
    _xmin, _xmax = plot_range.min(), plot_range.max()
    _ymin, _ymax = f(plot_range).min(), f(plot_range).max()
    ax = plt.axes(
        xlim=(_xmin, _xmax),
        ylim=(_ymin, _ymax))

    # Static Components:
    plot_function_graph(ax, f, plot_range)
    plot_parameter_text(ax, parameter_text)

    # Animated Components
    anim_handler = Animation(fig)
    anim_handler.register(DotsAnimation(
        ax=ax,
        f=f,
        x_locations=[loc for loc, strat in history],
        color='red'))
    anim_handler.register(FrameCounter(ax=ax, max_frames=len(history)))
    last_locs, last_strats = history[-1]
    strat_index = np.argmin(last_locs)
    anim_handler.register(StrategyOnDotsAnimation(
        ax=ax,
        U=U,
        loc_history=[loc[strat_index] for loc, strat in history],
        strat_history=[strat[strat_index] for loc, strat in history],
        max_val=20))
    plt.tight_layout()

    if len(history) > max_len:
        frames = range(0, len(history), len(history)//max_len)
    else:
        frames = len(history)
    return anim_handler.generate(frames=frames)


def full_visualization(
        history,
        f,
        U,
        plot_range=np.arange(-3, 3, 0.001),
        parameter_text='',
        max_len=30*60):
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
    # Only use this if we have few enough points for it to be useful
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
              'black', 'magenta', 'darkblue', 'darkgreen']
    start_locs, start_strats = history[0]
    n_points = start_locs.shape[0]
    if n_points > len(colors):
        return graph_visualization(history, f, U, plot_range, parameter_text)

    # Layout: Upper side with the graph, bottom with the strategies
    fig = plt.figure(figsize=FIGSIZE)
    gs = gridspec.GridSpec(3, n_points)
    ax_function_graph = fig.add_subplot(gs[0:2, :])
    ax_strategies = [fig.add_subplot(gs[2, i]) for i in range(n_points)]

    # Initialize function plot
    _xmin, _xmax = plot_range.min(), plot_range.max()
    _ymin, _ymax = f(plot_range).min(), f(plot_range).max()
    ax_function_graph.set_xlim((_xmin, _xmax))
    ax_function_graph.set_ylim((_ymin, _ymax))

    # Text to specify the parameters
    plot_parameter_text(ax_function_graph, parameter_text)

    # Static Components
    plot_function_graph(ax_function_graph, f, plot_range)

    ###########################################################################
    # Animated Components
    anim_handler = Animation(fig)

    # Dots: Each with a different color, therefore different component
    for i in range(n_points):
        anim_handler.register(
            DotsAnimation(
                ax=ax_function_graph,
                f=f,
                x_locations=[loc[i] for loc, strat in history],
                color=colors[i]))

    # Strategy Plots
    for i, ax in enumerate(ax_strategies):
        ax.set_ylim([0, 1.1])
        ax.set_xlim([U.min(), U.max()])
        anim_handler.register(
            StrategyAnimation(
                ax=ax,
                U=U,
                strat_history=[strat[i] for pop, strat in history],
                color=colors[i])
        )

    anim_handler.register(FrameCounter(
        ax=ax_function_graph, max_frames=len(history)))

    # A E S T H E T I C S
    plt.tight_layout()

    if len(history) > max_len:
        frames = range(0, len(history), len(history)//max_len)
    else:
        frames = len(history)
    return anim_handler.generate(frames=frames)


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
