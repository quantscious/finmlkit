import numpy as np
import pytest
import os
# disable jit
os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.weights import average_uniqueness


def test_label_concurrency_weights_basic():
    # Timestamps from 0 to 9
    timestamps = np.arange(10, dtype=np.int64)

    # Two events
    event_idxs = np.array([2, 4], dtype=np.int64)
    lookahead_idxs = np.array([5, 7], dtype=np.int64)

    expected_weights = np.array([0.75, 0.625], dtype=np.float64)
    weights = average_uniqueness(timestamps, event_idxs, lookahead_idxs)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)

def test_label_concurrency_weights_no_overlap():
    # Timestamps from 0 to 9
    timestamps = np.arange(10, dtype=np.int64)

    # Non-overlapping events
    event_idxs = np.array([1, 5], dtype=np.int64)
    lookahead_idxs = np.array([3, 7], dtype=np.int64)
    expected_weights = np.array([1.0, 1.0], dtype=np.float64)
    weights = average_uniqueness(timestamps, event_idxs, lookahead_idxs)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)

def test_label_concurrency_weights_full_overlap():
    # Timestamps from 0 to 9
    timestamps = np.arange(10, dtype=np.int64)

    # Fully overlapping events
    event_idxs = np.array([2, 2], dtype=np.int64)
    lookahead_idxs = np.array([5, 5], dtype=np.int64)

    expected_weights = np.array([0.5, 0.5], dtype=np.float64)
    weights = average_uniqueness(timestamps, event_idxs, lookahead_idxs)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)

def test_label_concurrency_weights_edge_cases():
    # Timestamps from 0 to 5
    timestamps = np.arange(6, dtype=np.int64)

    # Event that starts at the first timestamp
    event_idxs = np.array([0], dtype=np.int64)
    lookahead_idxs = np.array([2], dtype=np.int64)

    expected_weights = np.array([1.0], dtype=np.float64)
    weights = average_uniqueness(timestamps, event_idxs, lookahead_idxs)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)

def test_label_concurrency_weights_invalid_input():
    timestamps = np.arange(10, dtype=np.int64)
    event_idxs = np.array([2, 4], dtype=np.int64)
    lookahead_idxs = np.array([5], dtype=np.int64)  # Mismatched length

    with pytest.raises(ValueError):
        average_uniqueness(timestamps, event_idxs, lookahead_idxs)
