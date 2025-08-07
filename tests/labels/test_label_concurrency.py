import numpy as np
import pytest
import os
# disable jit
# os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.weights import average_uniqueness


def test_average_uniqueness_basic():
    """Test basic functionality with overlapping events."""
    timestamps = np.arange(10, dtype=np.int64)
    event_idxs = np.array([2, 4], dtype=np.int64)
    touch_idxs = np.array([5, 7], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Manual calculation:
    # Event 1: indices 2-5 (timestamps [2,3,4,5])
    # Event 2: indices 4-7 (timestamps [4,5,6,7])
    # Overlap at indices 4,5
    #
    # Concurrency: [0,0,1,1,2,2,1,1,0,0]
    # Event 1 weight: mean(1/[1,1,2,2]) = mean([1,1,0.5,0.5]) = 0.75
    # Event 2 weight: mean(1/[2,2,1,1]) = mean([0.5,0.5,1,1]) = 0.75

    expected_weights = np.array([0.75, 0.75], dtype=np.float64)
    expected_concurrency = np.array([0, 0, 1, 1, 2, 2, 1, 1, 0, 0], dtype=np.int16)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_average_uniqueness_no_overlap():
    """Test with non-overlapping events."""
    timestamps = np.arange(10, dtype=np.int64)
    event_idxs = np.array([1, 6], dtype=np.int64)
    touch_idxs = np.array([3, 8], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Event 1: indices 1-3, Event 2: indices 6-8 (no overlap)
    # Concurrency: [0,1,1,1,0,0,1,1,1,0]
    # Both events have concurrency = 1 throughout, so weights = 1.0

    expected_weights = np.array([1.0, 1.0], dtype=np.float64)
    expected_concurrency = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=np.int16)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_average_uniqueness_full_overlap():
    """Test with fully overlapping events."""
    timestamps = np.arange(8, dtype=np.int64)
    event_idxs = np.array([2, 2], dtype=np.int64)
    touch_idxs = np.array([5, 5], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Both events span indices 2-5, complete overlap
    # Concurrency: [0,0,2,2,2,2,0,0]
    # Both events: mean(1/[2,2,2,2]) = mean([0.5,0.5,0.5,0.5]) = 0.5

    expected_weights = np.array([0.5, 0.5], dtype=np.float64)
    expected_concurrency = np.array([0, 0, 2, 2, 2, 2, 0, 0], dtype=np.int16)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_average_uniqueness_single_event():
    """Test with single event."""
    timestamps = np.arange(6, dtype=np.int64)
    event_idxs = np.array([1], dtype=np.int64)
    touch_idxs = np.array([4], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Single event from index 1-4
    # Concurrency: [0,1,1,1,1,0]
    # Weight: mean(1/[1,1,1,1]) = 1.0

    expected_weights = np.array([1.0], dtype=np.float64)
    expected_concurrency = np.array([0, 1, 1, 1, 1, 0], dtype=np.int16)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_average_uniqueness_edge_boundary():
    """Test events at array boundaries."""
    timestamps = np.arange(6, dtype=np.int64)

    # Event starting at first index
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([2], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    expected_weights = np.array([1.0], dtype=np.float64)
    expected_concurrency = np.array([1, 1, 1, 0, 0, 0], dtype=np.int16)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_equal(concurrency, expected_concurrency)

    # Event ending at last index
    event_idxs = np.array([3], dtype=np.int64)
    touch_idxs = np.array([5], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    expected_weights = np.array([1.0], dtype=np.float64)
    expected_concurrency = np.array([0, 0, 0, 1, 1, 1], dtype=np.int16)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_average_uniqueness_partial_overlap():
    """Test with partial overlaps between multiple events."""
    timestamps = np.arange(12, dtype=np.int64)
    event_idxs = np.array([1, 3, 7], dtype=np.int64)
    touch_idxs = np.array([5, 9, 10], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Event 1: indices 1-5 (timestamps [1,2,3,4,5])
    # Event 2: indices 3-9 (timestamps [3,4,5,6,7,8,9])
    # Event 3: indices 7-10 (timestamps [7,8,9,10])
    #
    # Overlaps:
    # - Event 1 & 2: indices 3,4,5
    # - Event 2 & 3: indices 7,8,9
    #
    # Concurrency: [0,1,1,2,2,2,1,2,2,2,1,0]

    expected_concurrency = np.array([0, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 0], dtype=np.int16)

    # Event 1: concurrency [1,1,2,2,2] -> weight = mean([1,1,0.5,0.5,0.5]) = 0.7
    # Event 2: concurrency [2,2,2,1,2,2,2] -> weight = mean([0.5,0.5,0.5,1,0.5,0.5,0.5]) ≈ 0.571
    # Event 3: concurrency [2,2,2,1] -> weight = mean([0.5,0.5,0.5,1]) = 0.625

    expected_weights = np.array([0.7, 0.5714285714285714, 0.625], dtype=np.float64)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


# Alternative test that demonstrates the overlap issue more clearly
def test_average_uniqueness_edge_overlap():
    """Test events that overlap at boundary indices."""
    timestamps = np.arange(15, dtype=np.int64)

    # Events that share a boundary
    event_idxs = np.array([2, 5], dtype=np.int64)
    touch_idxs = np.array([5, 8], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Event 1: indices 2-5, Event 2: indices 5-8
    # They overlap at index 5, so both should have weight < 1.0

    expected_concurrency = np.zeros(15, dtype=np.int16)
    expected_concurrency[2:6] = [1, 1, 1, 2]  # indices 2,3,4 have concurrency 1, index 5 has concurrency 2
    expected_concurrency[6:9] = [1, 1, 1]  # indices 6,7,8 have concurrency 1

    # Event 1: concurrency [1,1,1,2] -> weight = mean([1,1,1,0.5]) = 0.875
    # Event 2: concurrency [2,1,1,1] -> weight = mean([0.5,1,1,1]) = 0.875
    expected_weights = np.array([0.875, 0.875], dtype=np.float64)

    np.testing.assert_array_equal(concurrency, expected_concurrency)
    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)


def test_average_uniqueness_adjacent_events():
    """Test with adjacent (touching) events."""
    timestamps = np.arange(8, dtype=np.int64)
    event_idxs = np.array([1, 4], dtype=np.int64)
    touch_idxs = np.array([3, 6], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Event 1: indices 1-3, Event 2: indices 4-6 (adjacent, no overlap)
    # Concurrency: [0,1,1,1,1,1,1,0]

    expected_weights = np.array([1.0, 1.0], dtype=np.float64)
    expected_concurrency = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.int16)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_average_uniqueness_single_timestamp_events():
    """Test with events spanning single timestamps."""
    timestamps = np.arange(5, dtype=np.int64)
    event_idxs = np.array([1, 3], dtype=np.int64)
    touch_idxs = np.array([1, 3], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Each event spans only one timestamp
    # Concurrency: [0,1,0,1,0]

    expected_weights = np.array([1.0, 1.0], dtype=np.float64)
    expected_concurrency = np.array([0, 1, 0, 1, 0], dtype=np.int16)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_average_uniqueness_complex_overlap_pattern():
    """Test with complex overlapping pattern."""
    timestamps = np.arange(10, dtype=np.int64)
    event_idxs = np.array([0, 2, 4, 6], dtype=np.int64)
    touch_idxs = np.array([3, 5, 7, 9], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Event 1: 0-3, Event 2: 2-5, Event 3: 4-7, Event 4: 6-9
    # Multiple overlaps creating different concurrency levels

    expected_concurrency = np.array([1, 1, 2, 2, 2, 2, 2, 2, 1, 1], dtype=np.int16)

    np.testing.assert_array_equal(concurrency, expected_concurrency)

    # CORRECTED: Weights are NOT all the same due to different overlap patterns
    # Event 1: concurrency [1,1,2,2] -> weight = 0.75
    # Event 2: concurrency [2,2,2,2] -> weight = 0.5 (completely overlapped)
    # Event 3: concurrency [2,2,2,2] -> weight = 0.5 (completely overlapped)
    # Event 4: concurrency [2,2,1,1] -> weight = 0.75
    expected_weights = np.array([0.75, 0.5, 0.5, 0.75], dtype=np.float64)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)


def test_average_uniqueness_mismatched_lengths():
    """Test error handling for mismatched input lengths."""
    timestamps = np.arange(10, dtype=np.int64)
    event_idxs = np.array([2, 4], dtype=np.int64)
    touch_idxs = np.array([5], dtype=np.int64)  # Mismatched length

    with pytest.raises(ValueError, match="Timestamps and lookahead indices must have the same length"):
        average_uniqueness(timestamps, event_idxs, touch_idxs)


def test_average_uniqueness_empty_inputs():
    """Test with empty event arrays."""
    timestamps = np.arange(5, dtype=np.int64)
    event_idxs = np.array([], dtype=np.int64)
    touch_idxs = np.array([], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # No events, so all weights empty and concurrency all zeros
    expected_weights = np.array([], dtype=np.float64)
    expected_concurrency = np.zeros(5, dtype=np.int16)

    np.testing.assert_array_equal(weights, expected_weights)
    np.testing.assert_array_equal(concurrency, expected_concurrency)


def test_average_uniqueness_mathematical_properties():
    """Test mathematical properties of uniqueness weights."""
    timestamps = np.arange(20, dtype=np.int64)

    # CORRECTED: Create scenario where Event 4 is truly isolated
    event_idxs = np.array([2, 3, 4, 12], dtype=np.int64)  # Changed from 10 to 12
    touch_idxs = np.array([8, 9, 10, 15], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Events with more overlap should have lower weights
    # Event 4 (index 3) is now truly isolated (indices 12-15), should have weight = 1.0
    assert weights[3] == 1.0

    # Other events overlap, should have weights < 1.0
    assert all(w < 1.0 for w in weights[:3])

    # All weights should be positive
    assert all(w > 0 for w in weights)

    # Weights should be <= 1.0 (uniqueness constraint)
    assert all(w <= 1.0 for w in weights)


def test_average_uniqueness_return_types():
    """Test correct return types."""
    timestamps = np.arange(5, dtype=np.int64)
    event_idxs = np.array([1], dtype=np.int64)
    touch_idxs = np.array([3], dtype=np.int64)

    weights, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)

    # Verify return types
    assert isinstance(weights, np.ndarray)
    assert isinstance(concurrency, np.ndarray)
    assert weights.dtype == np.float64
    assert concurrency.dtype == np.int16


def test_average_uniqueness_large_dataset():
    """Test with larger dataset for performance validation."""
    np.random.seed(42)
    n_timestamps = 10000
    n_events = 1000

    timestamps = np.arange(n_timestamps, dtype=np.int64)

    # Generate random events with some overlap
    event_starts = np.random.randint(0, n_timestamps - 100, n_events)
    event_starts = np.sort(event_starts)  # Keep sorted for realistic scenario
    event_ends = event_starts + np.random.randint(10, 100, n_events)
    event_ends = np.minimum(event_ends, n_timestamps - 1)  # Clamp to valid range

    weights, concurrency = average_uniqueness(timestamps, event_starts, event_ends)

    # Should complete without error
    assert len(weights) == n_events
    assert len(concurrency) == n_timestamps

    # All weights should be valid
    assert np.all(weights > 0)
    assert np.all(weights <= 1.0)
    assert np.all(np.isfinite(weights))

    # Concurrency should be non-negative integers
    assert np.all(concurrency >= 0)
    assert concurrency.dtype == np.int16


if __name__ == "__main__":
    pytest.main([__file__])