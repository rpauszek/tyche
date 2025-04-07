import numpy as np


class Trace:

    def __init__(self, data, *, dt=1, start=0):
        self._data = np.array(data, ndmin=1)
        self._start = start if len(self) > 0 else np.nan
        self._dt = dt if len(self) > 1 else np.nan

    def __len__(self):
        return self._data.size

    def __getitem__(self, indices):
        data = self.data[indices]
        time = np.atleast_1d(self.time[indices])
        dt = np.unique(np.diff(time))
        return Trace(
            data,
            dt=dt.item() if len(dt) else np.nan,
            start=time[0] if len(time) else np.nan,
        )

    @property
    def data(self):
        return self._data

    @property
    def start(self):
        return self._start

    @property
    def dt(self):
        return self._dt

    @property
    def time(self):
        return (
            np.arange(len(self)) * self.dt + self.start
            if len(self) != 1
            else np.array(self.start, ndmin=1)
        )
