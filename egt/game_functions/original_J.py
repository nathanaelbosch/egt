import logging

import numpy as np

from .game_template import J_template


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class OriginalJ(J_template):
    """Original J - As described by Massimo"""
    def __init__(self, *args, **kwargs):
        super(OriginalJ, self).__init__(*args, **kwargs)
        self.alpha = kwargs.get('alpha', 2)
        self.b = kwargs.get('b', 100)
        self.epsilon = kwargs.get('epsilon', 2)
        logging.info(f'J parameters: alpha={self.alpha} epsilon={self.epsilon}')

    def _naive(self, x, u, x2):
        if x==x2:
            if u==0:
                return 1
            else:
                return 0

        with np.errstate(divide='ignore'):
            out = np.exp(
                -((u - (max(0, np.tanh(3*(self.f(x) - self.f(x2)))) *
                        (x2 - x)))**2) /
                (np.abs(x-x2) ** self.alpha) + self.epsilon)

        return out

    def _vectorized(self, locations):
        """Idea: generate a whole NxNx#Strategies tensor with the values of J

        This one is actually used for computations.

        axis=0 the point to evaluate
        axis=1 the point to compare to
        axis=2 are the strategies
        """
        N, d = locations.shape

        f_vals = self.f(locations)
        assert len(f_vals) == N, 'f is not suited for higher-dimensional stuff'
        f_vals = f_vals.flatten()
        f_diffs = np.tile(f_vals, reps=(N, 1)).T - np.tile(f_vals, reps=(N, 1))
        f_diffs_tanh = np.tanh(3*f_diffs)
        f_diffs_tanh_positive = np.where(
            f_diffs_tanh > 0,
            f_diffs_tanh,
            0)
        # f_diffs_tanh_positive = sigmoid(100*f_diffs)

        # Walk dirs should be a NxNxd array, containing xj-xi at location [i, j, :]
        walk_dirs = (np.tile(locations[None, :, :], (N, 1, 1)) -
                     np.tile(locations[:, None, :], (1, N, 1)))

        walk_dirs_adj = f_diffs_tanh_positive[:, :, None] * walk_dirs
        # walk_dirs_adj = np.clip(walk_dirs_adj, -1, 1)

        ##############################################################
        # Multi dimensionality does not yet work, starting from here #
        ##############################################################
        walk_dirs_adj = walk_dirs_adj.reshape(N, N)
        walk_dirs = walk_dirs.reshape(N, N)
        # variance = (np.abs(walk_dirs) ** self.alpha +
        #             np.abs(f_diffs) ** self.alpha)
        # variance = 2 * ((np.abs(walk_dirs) + self.epsilon) ** self.alpha)
        variance = 2 * (np.abs(walk_dirs) ** self.alpha)
        variance += self.epsilon

        # Make things stable for x=x2
        # problems = np.array([
        #     [np.all(locations[i] == locations[j]) for j in range(N)]
        #     for i in range(N)])
        # variance[problems] = 1

        upper_side = (self.U.flatten()[None, None, :] -
                      walk_dirs_adj[:, :, None]) ** 2

        out = np.exp(-1 * upper_side / variance[:, :, None])

        # Set the "problems" to the values they should contain (by analysis of J)
        # out[problems] = 0
        # standing_index = np.where(self.U==0)[0][0]
        # out[problems, standing_index] = 1

        return out
