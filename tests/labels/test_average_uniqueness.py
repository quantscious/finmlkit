import pytest
import numpy as np
from finmlkit.label.weights import average_uniqueness


def test_no_overlap():
    """Test when there are no overlapping labels."""
    # Setup: 3 non-overlapping events
    timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    event_idxs = np.array([0, 4, 8], dtype=np.int64)
    touch_idxs = np.array([2, 6, 9], dtype=np.int64)

    # When there's no overlap, each weight should be 1.0
    expected_weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    expected_concurrency = np.zeros(len(timestamps), dtype=np.int16)
    expected_concurrency[0:3] = 1
    expected_concurrency[4:7] = 1
    expected_concurrency[8:10] = 1

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    np.testing.assert_array_almost_equal(weights, expected_weights)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_complete_overlap():
    """Test when all labels completely overlap."""
    timestamps = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    event_idxs = np.array([0, 0, 0], dtype=np.int64)
    touch_idxs = np.array([4, 4, 4], dtype=np.int64)

    # When there's complete overlap, each weight should be 1/3
    expected_weights = np.array([1/3, 1/3, 1/3], dtype=np.float64)
    expected_concurrency = np.array([3, 3, 3, 3, 3], dtype=np.int16)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    np.testing.assert_array_almost_equal(weights, expected_weights)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_partial_overlap():
    """Test when labels partially overlap."""
    timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    event_idxs = np.array([0, 2, 4], dtype=np.int64)
    touch_idxs = np.array([3, 5, 7], dtype=np.int64)

    # Calculate expected concurrency based on how the function works
    expected_concurrency = np.array([1, 1, 2, 2, 2, 2, 1, 1], dtype=np.int16)

    # Expected weights based on the mean of inverse concurrency for each event
    # Event 0: [1,1,2,2] -> mean(1/concurrency) = (1 + 1 + 1/2 + 1/2)/4 = 0.75
    # Event 1: [2,2,2,2] -> mean(1/concurrency) = (1/2 + 1/2 + 1/2 + 1/2)/4 = 0.5
    # Event 2: [2,2,1,1] -> mean(1/concurrency) = (1/2 + 1/2 + 1 + 1)/4 = 0.75
    expected_weights = np.array([0.75, 0.5, 0.75], dtype=np.float64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    np.testing.assert_array_almost_equal(weights, expected_weights)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_complex_overlap():
    """Test a more complex overlap scenario."""
    timestamps = np.array(list(range(10)), dtype=np.int64)
    event_idxs = np.array([0, 1, 3, 6], dtype=np.int64)
    touch_idxs = np.array([4, 5, 8, 9], dtype=np.int64)

    # Calculate expected concurrency
    expected_concurrency = np.zeros(10, dtype=np.int16)
    expected_concurrency[0:5] = 1  # Event 0
    expected_concurrency[1:6] = expected_concurrency[1:6] + 1  # Event 1
    expected_concurrency[3:9] = expected_concurrency[3:9] + 1  # Event 2
    expected_concurrency[6:10] = expected_concurrency[6:10] + 1  # Event 3

    # Calculate expected weights based on actual implementation
    # Event 0: [1,2,2,3,3] -> mean(1/concurrency) = (1 + 1/2 + 1/2 + 1/3 + 1/3)/5 = 0.533
    # Event 1: [2,2,3,3,2,1] -> mean(1/concurrency) = (1/2 + 1/2 + 1/3 + 1/3 + 1/2 + 1)/5 = 0.433
    # Event 2: [3,3,2,2,2,1] -> mean(1/concurrency) = (1/3 + 1/3 + 1/2 + 1/2 + 1/2 + 1)/6 = 0.444
    # Event 3: [3,2,2,1] -> mean(1/concurrency) = (1/3 + 1/2 + 1/2 + 1)/4 = 0.625
    expected_weights = np.array([0.533, 0.433, 0.444, 0.625], dtype=np.float64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=3)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_error_different_lengths():
    """Test that an error is raised when event_idxs and touch_idxs have different lengths."""
    timestamps = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    event_idxs = np.array([0, 1, 2], dtype=np.int64)
    touch_idxs = np.array([3, 4], dtype=np.int64)

    with pytest.raises(ValueError):
        average_uniqueness(timestamps, event_idxs, touch_idxs)


def test_empty_arrays():
    """Test with empty event arrays."""
    timestamps = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    event_idxs = np.array([], dtype=np.int64)
    touch_idxs = np.array([], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    assert len(weights) == 0
    np.testing.assert_array_equal(concurrency, np.zeros(len(timestamps), dtype=np.int16))


# Parametrized tests for additional test cases
@pytest.mark.parametrize(
    "timestamps,event_idxs,touch_idxs,expected_weights,expected_concurrency",
    [
        # Edge case: Single event
        (
            np.array([1, 2, 3, 4, 5], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            np.array([0, 1, 1, 1, 0], dtype=np.int16)
        ),
        # Edge case: Adjacent but non-overlapping events
        (
            np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
            np.array([0, 3], dtype=np.int64),
            np.array([2, 5], dtype=np.int64),
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([1, 1, 1, 1, 1, 1], dtype=np.int16)
        ),
        # Edge case: Nested events (one event completely within another)
        (
            np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            np.array([5, 3], dtype=np.int64),
            np.array([0.83333333, 0.5], dtype=np.float64),
            np.array([1, 1, 2, 2, 1, 1], dtype=np.int16)
        ),
    ]
)
def test_parametrized_cases(timestamps, event_idxs, touch_idxs, expected_weights, expected_concurrency):
    """Test various edge cases using parametrization."""
    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    np.testing.assert_array_almost_equal(weights, expected_weights)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


if __name__ == "__main__":
    pytest.main()