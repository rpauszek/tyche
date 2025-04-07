import numpy as np
from rp3.tyche.trace import Trace

def test_trace():
    trace = Trace([1, 2, 3])
    assert isinstance(trace.data, np.ndarray)
    assert isinstance(trace.frames, np.ndarray)
    assert isinstance(trace.time, np.ndarray)
    np.testing.assert_equal(trace.data, np.array([1, 2, 3]))
    np.testing.assert_equal(trace.frames, np.array([0, 1, 2]))
    np.testing.assert_equal(trace.time, np.array([0, 1, 2]))

    trace = Trace([1, 2, 3], dt=0.5)
    np.testing.assert_equal(trace.data, np.array([1, 2, 3]))
    np.testing.assert_equal(trace.frames, np.array([0, 1, 2]))
    np.testing.assert_equal(trace.time, np.array([0, 0.5, 1]))
