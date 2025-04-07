import pytest
import numpy as np
import re
from rp3.tyche.trace import Trace


def _assert_array(trace):
    assert isinstance(trace.data, np.ndarray)
    assert isinstance(trace.time, np.ndarray)


def test_trace():
    trace = Trace([1, 2, 3])
    _assert_array(trace)
    np.testing.assert_equal(trace.data, np.array([1, 2, 3]))
    np.testing.assert_equal(trace.time, np.array([0, 1, 2]))
    assert len(trace) == 3
    np.testing.assert_equal(trace.dt, 1)
    np.testing.assert_equal(trace.start, 0)

    trace = Trace([1, 2, 3], dt=0.5)
    _assert_array(trace)
    np.testing.assert_equal(trace.data, np.array([1, 2, 3]))
    np.testing.assert_equal(trace.time, np.array([0, 0.5, 1]))
    assert len(trace) == 3
    np.testing.assert_equal(trace.dt, 0.5)
    np.testing.assert_equal(trace.start, 0)


def test_trace_one_point():
    trace = Trace(5)
    _assert_array(trace)
    np.testing.assert_equal(trace.data, 5)
    np.testing.assert_equal(trace.time, 0)
    assert len(trace) == 1
    np.testing.assert_equal(trace.dt, np.nan)
    np.testing.assert_equal(trace.start, 0)

    trace = Trace(5, start=15)
    _assert_array(trace)
    np.testing.assert_equal(trace.data, 5)
    np.testing.assert_equal(trace.time, 15)
    assert len(trace) == 1
    np.testing.assert_equal(trace.dt, np.nan)
    np.testing.assert_equal(trace.start, 15)


def test_empty_trace():
    trace = Trace([])
    _assert_array(trace)
    np.testing.assert_equal(trace.data, [])
    np.testing.assert_equal(trace.time, [])
    assert len(trace) == 0
    np.testing.assert_equal(trace.dt, np.nan)
    np.testing.assert_equal(trace.start, np.nan)

    trace = Trace([], dt=0.5)
    _assert_array(trace)
    np.testing.assert_equal(trace.data, [])
    np.testing.assert_equal(trace.time, [])
    assert len(trace) == 0
    np.testing.assert_equal(trace.dt, np.nan)
    np.testing.assert_equal(trace.start, np.nan)

    trace = Trace([], start=15)
    _assert_array(trace)
    np.testing.assert_equal(trace.data, [])
    np.testing.assert_equal(trace.time, [])
    assert len(trace) == 0
    np.testing.assert_equal(trace.dt, np.nan)
    np.testing.assert_equal(trace.start, np.nan)


def _assert_slice(ref_trace, sliced, expected_data, expected_time, expected_dt, expected_start):
    assert id(ref_trace) != id(sliced)
    np.testing.assert_equal(sliced.data, expected_data)
    np.testing.assert_equal(sliced.time, expected_time)
    np.testing.assert_equal(sliced.dt, expected_dt)
    np.testing.assert_equal(sliced.start, expected_start)


def test_slicing():
    trace = Trace(np.arange(10), dt=0.5)

    _assert_slice(trace, trace[:], trace.data, trace.time, trace.dt, trace.time[0])
    _assert_slice(trace, trace[2:5], [2, 3, 4], [1, 1.5, 2], trace.dt, 1)
    _assert_slice(trace, trace[:3], [0, 1, 2], [0, 0.5, 1], trace.dt, 0)
    _assert_slice(trace, trace[7:], [7, 8, 9], [3.5, 4, 4.5], trace.dt, 3.5)
    _assert_slice(trace, trace[5], [5], [2.5], np.nan, 2.5)


def test_reverse_slicing():
    trace = Trace(np.arange(10), dt=0.5)

    _assert_slice(trace, trace[5:2:-1], [5, 4, 3], [2.5, 2, 1.5], -trace.dt, 2.5)
    _assert_slice(trace, trace[::-1], trace.data[::-1], trace.time[::-1], -trace.dt, trace.time[-1])


def test_multiple_slicing():
    trace = Trace(np.arange(10), dt=0.5)
    sliced = trace[2:9]
    _assert_slice(sliced, sliced[2:4], [4, 5], [2, 2.5], trace.dt, 2)
    _assert_slice(trace[2:9], trace[2:9][2:4], [4, 5], [2, 2.5], trace.dt, 2)


def test_empty_slice():
    trace = Trace(np.arange(10), dt=0.5)
    _assert_slice(trace, trace[11:15], [], [], np.nan, np.nan)
    _assert_slice(trace, trace[2:4][3:5], [], [], np.nan, np.nan)


def test_bad_slicing():
    trace = Trace(np.arange(10), dt=0.5)

    # unsupported slice types
    with pytest.raises(
        IndexError,
        match="too many indices for array: array is 1-dimensional, but 2 were indexed",
    ):
        trace[(1, 2)]

    with pytest.raises(
        IndexError,
        match=re.escape(
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer "
            "or boolean arrays are valid indices"
        ),
    ):
        trace["abc"]

    # TODO: will be supported to slice directly comparing time axis
    with pytest.raises(
        TypeError,
        match="slice indices must be integers or None or have an __index__ method",
    ):
        trace["1.5":"4"]
