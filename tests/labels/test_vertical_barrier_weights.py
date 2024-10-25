import numpy as np
import pytest
import os
# disable jit
os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.labels.tbm import vertical_barrier_weights


def test_vertical_barrier_weights_normal():
    max_return_ratio = np.array([0.2, 0.5, 0.75, 1.0])
    expected_weights = np.array([1.0, 1.0, 0.5, 0.0])
    weights = vertical_barrier_weights(max_return_ratio)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)

def test_vertical_barrier_weights_nan():
    max_return_ratio = np.array([0.2, 0.5, 0.75, np.nan])
    expected_weights = np.array([1.0, 1.0, 0.5, np.nan])
    weights = vertical_barrier_weights(max_return_ratio)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)

def test_vertical_barrier_weights_lower_bound():
    max_return_ratio = np.array([0.0, 0.1, 0.49])
    expected_weights = np.array([1.0, 1.0, 1.0])
    weights = vertical_barrier_weights(max_return_ratio)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)

def test_vertical_barrier_weights_upper_bound():
    max_return_ratio = np.array([0.5, 0.75, 0.9, 1.0])
    expected_weights = np.array([1.0, 0.5, 0.2, 0.0])
    weights = vertical_barrier_weights(max_return_ratio)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)

def test_vertical_barrier_weights_edge_cases():
    # Test values exactly at the thresholds
    max_return_ratio = np.array([0.5, 1.0])
    expected_weights = np.array([1.0, 0.0])
    weights = vertical_barrier_weights(max_return_ratio)
    np.testing.assert_almost_equal(weights, expected_weights, decimal=5)

def test_vertical_barrier_weights_invalid_input_negative():
    max_return_ratio = np.array([-0.1, 0.2])
    with pytest.raises(ValueError):
        vertical_barrier_weights(max_return_ratio)

def test_vertical_barrier_weights_invalid_input_over_one():
    max_return_ratio = np.array([0.8, 1.1])
    with pytest.raises(ValueError):
        vertical_barrier_weights(max_return_ratio)