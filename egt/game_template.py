import numpy as np
from abc import ABC, abstractmethod


class J_template(ABC):
    """All Js have the same structure

    This class secures the namespace as there have been bugs before.
    Also implementing the functionality twice makes for easier testing.
    """
    def __init__(self, f, U):
        """Initialization

        f: function to minimize
        U: set of all strategies
        """
        super(J_template, self).__init__()
        self.f = f
        self.U = U

    @abstractmethod
    def _naive(self, x, u, x2):
        """Entry-wise function

        Easier to implement, and useful for testing"""
        pass

    def _badly_vectorized(self, locations):
        N, d = locations.shape
        out = np.ones((N, N, len(self.U)))

        for i in range(N):
            for j in range(N):
                for k in range(len(self.U)):
                    out[i, j, k] = self._naive(
                        locations[i], self.U[k], locations[j])
        return out

    @abstractmethod
    def _vectorized(self, locations):
        """Vectorized function

        Harder to implement, but wayyyyy faster.
        """
        pass

    def run(self, locations):
        return self._vectorized(locations)

    @classmethod
    def get(cls, f, U, *args, **kwargs):
        J = cls(f, U, *args, **kwargs)
        return J.run
