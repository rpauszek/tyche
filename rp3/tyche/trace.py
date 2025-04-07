import numpy as np


class Trace:

    def __init__(self, data, *, dt=1):
        self._data = np.array(data)
        self._frames = np.arange(len(data))
        self._dt = dt

    @property
    def data(self):
        return self._data

    @property
    def frames(self):
        return self._frames

    @property
    def time(self):
        return self._frames * self._dt
