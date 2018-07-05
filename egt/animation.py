import numpy as np
from abc import ABC, abstractmethod
from matplotlib import animation


class AnimationComponent(ABC):
    """Wrapper for parts of animations"""
    def __init__(self):
        super(AnimationComponent, self).__init__()

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def animate(self, i):
        pass


class DotsAnimation(AnimationComponent):
    """Wrapper for the animated dots"""
    def __init__(self, ax, f, x_locations, *args, **kwargs):
        super(DotsAnimation, self).__init__()
        kwargs.setdefault('color', 'red')
        self.dots, = ax.plot([], [], 'o', **kwargs)
        # Get locations as list of x coordinates
        self.x_locations = x_locations
        # We can calculate the y coordinates
        self.y_locations = [f(x) for x in x_locations]

    def init(self):
        self.dots.set_data([], [])

    def animate(self, i):
        self.dots.set_data(
            self.x_locations[i], self.y_locations[i])


class StrategyAnimation(AnimationComponent):
    """Wrapper for the animated dots"""
    def __init__(self, ax, U, strat_history, *args, **kwargs):
        super(StrategyAnimation, self).__init__()
        kwargs.setdefault('color', 'red')
        self.line, = ax.plot([], [], **kwargs)
        self.U = U
        self.strat_history = strat_history

    def init(self):
        self.line.set_data([], [])

    def animate(self, i):
        self.line.set_data(
            self.U, self.strat_history[i]/np.max(self.strat_history[i]))


class FrameCounter(AnimationComponent):
    """Wrapper for the animated dots"""
    def __init__(self, ax, max_frames=None, *args, **kwargs):
        super(FrameCounter, self).__init__()
        self.text = ax.text(
            0.0, 0.0, '',
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom',
        )
        self.max_frames = max_frames

    def init(self):
        self.text.set_text('')

    def animate(self, i):
        text = f'{i}'
        if self.max_frames:
            text += f'/{self.max_frames-1}'
        self.text.set_text(text)


class Animation(object):
    """Animates"""
    def __init__(self, fig):
        super(Animation, self).__init__()
        self.components = []
        self.fig = fig

    def register(self, component):
        self.components.append(component)

    def generate(self, **kwargs):
        def init():
            for c in self.components:
                c.init()
            return self.components

        # animation function.  This is called sequentially
        def animate(i):
            for c in self.components:
                c.animate(i)
            return self.components

        anim = animation.FuncAnimation(
            self.fig, animate,
            init_func=init,
            interval=1,
            blit=False,
            repeat=False,
            **kwargs)
        return anim
