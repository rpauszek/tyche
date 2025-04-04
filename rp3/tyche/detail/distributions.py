import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

def row(x):
    return x[np.newaxis, :]


def col(x):
    return x[:, np.newaxis]


@dataclass
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
        X = row(x) - col(self.mu)
        lnP = -0.5 * np.log(2*np.pi/col(self.tau)) - col(self.tau)/2 * X**2
        return np.exp(lnP)

    def plot(self, x):
        for component in self.pdf(x):
            plt.plot(x, component)
