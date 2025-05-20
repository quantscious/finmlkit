import numpy as np
import pytest
import os
# disable jit
os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.tbm import triple_barrier

def test_upper_barrier_hit():
    timestamps = np.array([1e9, 2e9, 3e9, 4e9, 5e9], dtype=np.int64)
    close = np.array([100, 102, 104, 106, 108], dtype=np.float64)
    events = np.array([0], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)
    min_ret = 0.01
    horizontal_barriers = (1.0, 1.0)
    vertical_barrier = 3

    labels, barrier_touch_idxs, rets, max_dd_ratio = triple_barrier(
        timestamps, close, events, targets, min_ret, horizontal_barriers, vertical_barrier
    )

    assert labels[0] == 1
    assert barrier_touch_idxs[0] == 3  # Barrier hit at index 3
    np.testing.assert_almost_equal(rets[0], np.log(106 / 100), decimal=5)
    np.testing.assert_almost_equal(max_dd_ratio[0], (np.log(106 / 100) / np.log(1.05)), decimal=5)

def test_lower_barrier_hit():
    timestamps = np.array([1e9, 2e9, 3e9], dtype=np.int64)
    close = np.array([100, 97, 95], dtype=np.float64)
    events = np.array([0], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)
    min_ret = 0.01
    horizontal_barriers = (1.0, 1.0)
    vertical_barrier = 2

    labels, barrier_touch_idxs, rets, max_dd_ratio = triple_barrier(
        timestamps, close, events, targets, min_ret, horizontal_barriers, vertical_barrier
    )

    assert labels[0] == -1
    assert barrier_touch_idxs[0] == 2
    np.testing.assert_almost_equal(rets[0], np.log(95 / 100), decimal=5)
    np.testing.assert_almost_equal(max_dd_ratio[0], (np.log(95 / 100) / -np.log(1.05)), decimal=5)

def test_vertical_barrier_hit():
    timestamps = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.int64)
    close = np.array([100, 101, 102, 103], dtype=np.float64)
    events = np.array([0], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)
    min_ret = 0.05
    horizontal_barriers = (1.0, 1.0)
    vertical_barrier = 2

    labels, barrier_touch_idxs, rets, max_dd_ratio = triple_barrier(
        timestamps, close, events, targets, min_ret, horizontal_barriers, vertical_barrier
    )

    assert labels[0] == 0  # Vertical barrier hit without reaching min_ret
    assert barrier_touch_idxs[0] == 2
    np.testing.assert_almost_equal(rets[0], np.log(102 / 100), decimal=5)

def test_exact_barrier_hit():
    timestamps = np.array([1e9, 2e9], dtype=np.int64)
    close = np.array([100, 105], dtype=np.float64)
    events = np.array([0], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)
    min_ret = 0.01
    horizontal_barriers = (1.0, 1.0)
    vertical_barrier = 1

    labels, barrier_touch_idxs, rets, max_dd_ratio = triple_barrier(
        timestamps, close, events, targets, min_ret, horizontal_barriers, vertical_barrier
    )

    assert labels[0] == 1
    assert barrier_touch_idxs[0] == 1
    np.testing.assert_almost_equal(rets[0], np.log(105 / 100), decimal=5)

def test_min_ret_threshold():
    timestamps = np.array([1e9, 2e9], dtype=np.int64)
    close = np.array([100, 101], dtype=np.float64)
    events = np.array([0], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)
    min_ret = 0.02  # Higher than actual return
    horizontal_barriers = (1.0, 1.0)
    vertical_barrier = 1

    labels, barrier_touch_idxs, rets, max_dd_ratio = triple_barrier(
        timestamps, close, events, targets, min_ret, horizontal_barriers, vertical_barrier
    )

    assert labels[0] == 0  # Return does not meet min_ret
    assert barrier_touch_idxs[0] == 1

def test_asymmetric_barriers():
    timestamps = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.int64)
    close = np.array([100, 102, 98, 95], dtype=np.float64)
    events = np.array([0, 1], dtype=np.int64)
    targets = np.array([0.05, 0.05], dtype=np.float64)
    min_ret = 0.001
    horizontal_barriers = (1.0, 2.0)  # Set bottom barrier multiplier to 1.0
    vertical_barrier = 2

    labels, barrier_touch_idxs, rets, max_dd_ratio = triple_barrier(
        timestamps, close, events, targets, min_ret, horizontal_barriers, vertical_barrier
    )

    assert labels[0] == 0  # Still no barrier hit for first event
    assert labels[1] == -1  # Lower barrier hit for second event

def test_max_dd_ratio():
    timestamps = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.int64)
    close = np.array([100, 103, 99, 105], dtype=np.float64)
    events = np.array([0], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)
    min_ret = 0.01
    horizontal_barriers = (1.0, 1.0)
    vertical_barrier = 3

    labels, barrier_touch_idxs, rets, max_dd_ratio = triple_barrier(
        timestamps, close, events, targets, min_ret, horizontal_barriers, vertical_barrier
    )

    # Compute expected max_dd_ratio
    upper_barrier = np.log(1 + targets[0] * horizontal_barriers[1])
    lower_barrier = -np.log(1 + targets[0] * horizontal_barriers[0])
    ret1 = np.log(103 / 100)
    ret2 = np.log(99 / 100)
    ratio1 = ret1 / upper_barrier
    ratio2 = ret2 / lower_barrier
    expected_max_dd = 1.0  # Barrier hit at j=3
    np.testing.assert_almost_equal(max_dd_ratio[0], expected_max_dd, decimal=5)

def test_multiple_events():
    timestamps = np.array([1e9, 2e9, 3e9, 4e9, 5e9], dtype=np.int64)
    close = np.array([100, 102, 98, 104, 96], dtype=np.float64)
    events = np.array([0, 2], dtype=np.int64)
    targets = np.array([0.05, 0.05], dtype=np.float64)
    min_ret = 0.01
    horizontal_barriers = (1.0, 1.0)
    vertical_barrier = 3

    labels, barrier_touch_idxs, rets, max_dd_ratio = triple_barrier(
        timestamps, close, events, targets, min_ret, horizontal_barriers, vertical_barrier
    )

    assert labels[0] == 0  # No barrier hit for first event
    assert labels[1] == 0  # Upper barrier hit for second event

def test_nan_handling():
    timestamps = np.array([1e9, 2e9], dtype=np.int64)
    close = np.array([100, 102], dtype=np.float64)
    events = np.array([1], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)
    min_ret = 0.01
    horizontal_barriers = (1.0, 1.0)
    vertical_barrier = 5  # Vertical barrier extends beyond available data

    labels, barrier_touch_idxs, rets, max_dd_ratio = triple_barrier(
        timestamps, close, events, targets, min_ret, horizontal_barriers, vertical_barrier
    )

    assert np.isnan(rets[0])
    assert np.isnan(max_dd_ratio[0])
    assert labels[0] == 0