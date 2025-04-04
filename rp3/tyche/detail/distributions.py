import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .utils import numpy_preprocessor, row, col


@numpy_preprocessor
@dataclass(frozen=True)
class Normal:
    mu: np.ndarray
    tau: np.ndarray

    @property
    def K(self):
        return self.mu.size

    @property
    def var(self):
        return self.tau**-1

    @property
    def sigma(self):
        return np.sqrt(self.var)

    def pdf(self, x):
        x, m, t = row(x), col(self.mu), col(self.tau)
        ln_p = -0.5 * np.log(2 * np.pi / t) - t / 2 * (x-m)**2
        return np.exp(ln_p)

    def plot(self, x):
        for component in self.pdf(x):
            plt.plot(x, component)
