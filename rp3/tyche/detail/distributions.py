import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .utils import numpy_preprocessor, row, col


@numpy_preprocessor
@dataclass(frozen=True)
class Normal:
    """1-D Normal distribution for multiple components.

    Parameters
    ----------
    mu: np.ndarray
        array of component means
    tau: np.ndarray
        array of component precisions (1 / variances)
    """

    mu: np.ndarray
    tau: np.ndarray

    def __post_init__(self):
        if self.mu.size != self.tau.size:
            raise ValueError("mu and tau must be the same size.")

    @property
    def K(self):
        """Number of components."""
        return self.mu.size

    @property
    def var(self):
        """Component variances."""
        return self.tau**-1

    @property
    def sigma(self):
        """Component standard deviations."""
        return np.sqrt(self.var)

    def pdf(self, x):
        """Calculate the probability distributions functions for each
        component given observation vector `x` of length `T`.

        Parameters
        ----------
        x: np.ndarray
            array of observations; `x.size := T`

        Returns
        -------
        np.ndarray
            probability distribution functions evaluated at `x`; `shape = (K, T)`
        """
        x, m, t = row(x), col(self.mu), col(self.tau)
        ln_p = -0.5 * np.log(2 * np.pi / t) - t / 2 * (x - m) ** 2
        return np.exp(ln_p)

    def plot(self, x):
        for component in self.pdf(x):
            plt.plot(x, component)
