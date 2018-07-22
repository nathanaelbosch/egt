import numpy as np

from .game_template import J_template


class ConfidenceJ(J_template):
    """My first own J - quadratic functions"""
    def __init__(self, *args, **kwargs):
        super(ConfidenceJ, self).__init__(*args, **kwargs)

    def _naive(self, x, u, x2):
        epsilon = 1e-10
        alpha = 1/(epsilon + np.abs(x2-x))

        if self.f(x2) <= self.f(x):
            top = np.clip(x2-x)
            if x2-x >= 0:
                bot = -1
                if u > top:
                    return self._naive(x, top-(u-top), x2)
            elif x2-x < 0:
                bot = 1
                if u < top:
                    return self._naive(x, top-(u-top), x2)
            a = 1/(np.abs(top-bot)**alpha)
            return a * (np.abs(u-bot)**alpha)
        elif self.f(x2) > self.f(x):
            top = 0
            if u > top:
                bot = 1
            else:
                bot = -1
            a = 1/(np.abs(top-bot)**alpha)
            return a * (np.abs(u-bot)**alpha)

    def _vectorized(self, locations):
        N, d = locations.shape
        # locations = locations.flatten()

        f_vals = self.f(locations)      # f(x)
        assert len(f_vals) == N
        f_vals = f_vals.flatten()

        fx = np.tile(f_vals, reps=(N, 1)).T
        fx2 = np.tile(f_vals, reps=(N, 1))

        x2_minus_x = (np.tile(locations[None, :, :], (N, 1, 1)) -
                      np.tile(locations[:, None, :], (1, N, 1)))
        x2_minus_x = x2_minus_x.reshape(N, N)

        x_bar = x2_minus_x.clip(-1, 1)
        x_bar = np.where(fx >= fx2, x_bar, 0)
        x0 = -1 * np.sign(x_bar)
        x0[x0==0] = 1

        epsilon = 0
        alpha = 1 / (epsilon + np.abs(x2_minus_x[:, :, None]))
        # alpha = 0.1

        out = -1 * (
            (np.abs(self.U.flatten()[None, None, :] -
                    x_bar[:, :, None])) ** alpha /
            (np.abs(x0[:, :, None] - x_bar[:, :, None])) ** alpha) + 1

        out[range(N), range(N), :] = 0

        return out
