import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)
import pytest
import numpy as np
from finmlkit.sampling.filters import cusum_filter

def test_cusum_filter_no_events():
    """
    Test the cusum_filter function on a time series where no events are expected.
    """
    # Constant price series
    prices = np.array([100, 100, 100, 100, 100], dtype=np.float64)
    threshold = np.array([0.5], dtype=np.float64)  # High threshold to prevent events

    # Expected no events
    expected_event_indices = np.array([], dtype=np.int64)

    # Call the cusum_filter function
    event_indices = cusum_filter(prices, threshold)

    # Check that no events are detected
    np.testing.assert_array_equal(event_indices, expected_event_indices)

def test_cusum_filter_single_price():
    """
    Test the cusum_filter function with a single price point.
    Expect a ValueError due to insufficient data.
    """
    prices = np.array([100], dtype=np.float64)
    threshold = np.array([0.01], dtype=np.float64)

    with pytest.raises(ValueError):
        cusum_filter(prices, threshold)

def test_cusum_filter_empty_series():
    """
    Test the cusum_filter function with an empty price series.
    Expect a ValueError due to insufficient data.
    """
    prices = np.array([], dtype=np.float64)
    threshold = np.array([0.01], dtype=np.float64)

    with pytest.raises(ValueError):
        cusum_filter(prices, threshold)

def test_cusum_filter_invalid_threshold_length():
    """
    Test the cusum_filter function with a mismatched threshold array length.
    Expect a ValueError due to incorrect threshold length.
    """
    prices = np.array([100, 101, 102], dtype=np.float64)
    threshold = np.array([0.01, 0.02], dtype=np.float64)  # Incorrect length

    with pytest.raises(ValueError):
        cusum_filter(prices, threshold)

def test_cusum_filter_large_series():
    """
    Test the cusum_filter function with a large time series.
    """
    # Generate a large random walk series
    np.random.seed(42)
    n = 10000
    prices = np.cumsum(np.random.randn(n)) + 1000  # Ensure prices remain positive
    threshold = np.array([0.5], dtype=np.float64)

    # Call the cusum_filter function
    event_indices = cusum_filter(prices, threshold)

    # Check that the function returns an array of indices
    assert isinstance(event_indices, np.ndarray)
    assert event_indices.dtype == np.int64
    assert len(event_indices) <= n  # Cannot have more events than data points
    assert np.all(event_indices >= 1)
    assert np.all(event_indices < n)

def test_cusum_filter_all_events():
    """
    Test the cusum_filter function with a very low threshold, causing events at each time step.
    """
    prices = np.array([100, 100.01, 100.02, 100.03, 100.04, 100.05], dtype=np.float64)
    threshold = np.array([1e-5], dtype=np.float64)  # Very low threshold

    # Expected events at every time step after the first
    expected_event_indices = np.array([1, 2, 3, 4, 5])

    # Call the cusum_filter function
    event_indices = cusum_filter(prices, threshold)

    # Check that events are detected at every time step
    np.testing.assert_array_equal(event_indices, expected_event_indices)