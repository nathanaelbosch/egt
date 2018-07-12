import numpy as np

from egt.game_template import J_template


class MyJ(J_template):
    """My first own J - quadratic functions"""
    def __init__(self, *args, **kwargs):
        super(MyJ, self).__init__(*args, **kwargs)

    def _naive(self, x, u, x2):
        if self.f(x2) <= self.f(x):
            top = x2-x
        else:
            top = 0
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
        return a * ((u-top)**2) + b

    def _vectorized(self, locations):
        N, d = locations.shape

        f_vals = self.f(locations)
        assert len(f_vals) == N
        f_vals = f_vals.flatten()
        # f_diffs should contain xi-xj at location ij; >0 if j is better
        f_diffs = (np.tile(f_vals, reps=(N, 1)).T -
                   np.tile(f_vals, reps=(N, 1)))

        walk_dirs = (np.tile(locations[None, :, :], (N, 1, 1)) -
                     np.tile(locations[:, None, :], (1, N, 1)))

        walk_dirs_ifgood = np.where(
            f_diffs[:, :, None] >= 0,
            walk_dirs,
            0)

        # Starting here its not higher-dimensional
        walk_dirs_ifgood = walk_dirs_ifgood.reshape(N, N)
        U_adj = (self.U.flatten()[None, None, :] -
                 walk_dirs_ifgood[:, :, None]) ** 2
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
