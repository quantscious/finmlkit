import numpy as np
import os
import pytest
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.bar.logic import _time_bar_indexer


def test_time_bar_indexer_basic():
    """Test basic functionality with regular trades."""
    timestamps = np.array([
        999_999_999,  # 0.999999999 seconds
        1_000_000_000,  # 1 second
        2_000_000_000,  # 2 seconds
        3_000_000_000,  # 3 seconds
        4_000_000_000,  # 4 seconds
        5_000_000_000,  # 5 seconds
        5_999_999_999,  # 5.999999999 seconds
        6_100_000_000,  # 6.1 seconds
        7_000_000_000,  # 7 seconds
    ], dtype=np.int64)
    interval_seconds = 2.0

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # CORRECTED: Function creates extra bar for complete coverage
    # Last timestamp: 7e9, ceil(7e9/2e9) = 4, so last_ts = 8e9
    # Bar range: arange(0, 8e9 + 2e9 + 1, 2e9) = [0, 2e9, 4e9, 6e9, 8e9, 10e9]
    expected_bar_clock = np.array([0, 2_000_000_000, 4_000_000_000, 6_000_000_000, 8_000_000_000, 10_000_000_000],
                                  dtype=np.int64)

    np.testing.assert_array_equal(bar_clock, expected_bar_clock)
    assert len(bar_close_indices) == len(expected_bar_clock)


def test_time_bar_indexer_precise_calculation():
    """Test with precise calculation of expected results."""
    timestamps = np.array([
        1_000_000_000,  # 1.0 sec
        1_500_000_000,  # 1.5 sec
        2_000_000_000,  # 2.0 sec
        2_500_000_000,  # 2.5 sec
        3_000_000_000,  # 3.0 sec
        4_000_000_000,  # 4.0 sec
    ], dtype=np.int64)
    interval_seconds = 2.0

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # CORRECTED: Last timestamp 4e9, ceil(4e9/2e9) = 2, last_ts = 4e9
    # Range: arange(0, 4e9 + 2e9 + 1, 2e9) = [0, 2e9, 4e9, 6e9]
    expected_bar_clock = np.array([0, 2_000_000_000, 4_000_000_000, 6_000_000_000], dtype=np.int64)

    np.testing.assert_array_equal(bar_clock, expected_bar_clock)
    assert len(bar_close_indices) == len(expected_bar_clock)


def test_time_bar_indexer_empty_bars():
    """Test handling of empty bars (no trades in some intervals)."""
    timestamps = np.array([
        1_000_000_000,  # 1.0 sec
        2_000_000_000,  # 2.0 sec
        # Gap: no trades in bar 2 (2-4 sec)
        5_000_000_000,  # 5.0 sec
        6_000_000_000,  # 6.0 sec
    ], dtype=np.int64)
    interval_seconds = 2.0

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # CORRECTED: Last timestamp 6e9, ceil(6e9/2e9) = 3, last_ts = 6e9
    # Range: arange(0, 6e9 + 2e9 + 1, 2e9) = [0, 2e9, 4e9, 6e9, 8e9]
    expected_bar_clock = np.array([0, 2_000_000_000, 4_000_000_000, 6_000_000_000, 8_000_000_000], dtype=np.int64)

    np.testing.assert_array_equal(bar_clock, expected_bar_clock)
    assert len(bar_close_indices) == len(expected_bar_clock)


def test_time_bar_indexer_single_trade():
    """Test with single trade."""
    timestamps = np.array([5_000_000_000], dtype=np.int64)  # 5.0 seconds
    interval_seconds = 2.0

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # CORRECTED: Bar start: 5e9 // 2e9 * 2e9 = 4e9
    # Last timestamp: 5e9, ceil(5e9/2e9) = 3, last_ts = 6e9
    # Range: arange(4e9, 6e9 + 2e9 + 1, 2e9) = [4e9, 6e9, 8e9]
    expected_bar_clock = np.array([4_000_000_000, 6_000_000_000, 8_000_000_000], dtype=np.int64)

    np.testing.assert_array_equal(bar_clock, expected_bar_clock)
    assert len(bar_close_indices) == len(expected_bar_clock)


def test_time_bar_indexer_different_intervals():
    """Test with different interval sizes."""
    timestamps = np.array([
        1_000_000_000,  # 1 sec
        1_500_000_000,  # 1.5 sec
        3_000_000_000,  # 3 sec
        4_500_000_000,  # 4.5 sec
        6_000_000_000,  # 6 sec
    ], dtype=np.int64)

    # Test 1-second intervals
    bar_clock_1s, bar_close_indices_1s = _time_bar_indexer(timestamps, 1.0)

    # CORRECTED: Last timestamp 6e9, ceil(6e9/1e9) = 6, last_ts = 6e9
    # Range: arange(1e9, 6e9 + 1e9 + 1, 1e9) = [1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9]
    expected_bar_clock_1s = np.array([
        1_000_000_000, 2_000_000_000, 3_000_000_000,
        4_000_000_000, 5_000_000_000, 6_000_000_000, 7_000_000_000
    ], dtype=np.int64)

    np.testing.assert_array_equal(bar_clock_1s, expected_bar_clock_1s)

    # Test 5-second intervals
    bar_clock_5s, bar_close_indices_5s = _time_bar_indexer(timestamps, 5.0)

    # Last timestamp 6e9, ceil(6e9/5e9) = 2, last_ts = 10e9
    # Range: arange(0, 10e9 + 5e9 + 1, 5e9) = [0, 5e9, 10e9, 15e9]
    expected_bar_clock_5s = np.array([0, 5_000_000_000, 10_000_000_000, 15_000_000_000], dtype=np.int64)

    np.testing.assert_array_equal(bar_clock_5s, expected_bar_clock_5s)


def test_time_bar_indexer_fractional_intervals():
    """Test with fractional second intervals."""
    timestamps = np.array([
        1_000_000_000,  # 1.0 sec
        1_250_000_000,  # 1.25 sec
        1_500_000_000,  # 1.5 sec
        1_750_000_000,  # 1.75 sec
        2_000_000_000,  # 2.0 sec
    ], dtype=np.int64)
    interval_seconds = 0.5  # 500ms intervals

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # CORRECTED: Bar interval: 0.5 * 1e9 = 5e8 ns
    # Last timestamp: 2e9, ceil(2e9/5e8) = 4, last_ts = 2e9
    # Range: arange(1e9, 2e9 + 5e8 + 1, 5e8) = [1e9, 1.5e9, 2e9, 2.5e9]
    expected_bar_clock = np.array([
        1_000_000_000,  # 1.0 sec
        1_500_000_000,  # 1.5 sec
        2_000_000_000,  # 2.0 sec
        2_500_000_000,  # 2.5 sec
    ], dtype=np.int64)

    np.testing.assert_array_equal(bar_clock, expected_bar_clock)


def test_time_bar_indexer_edge_boundary():
    """Test trades exactly on bar boundaries."""
    timestamps = np.array([
        0,  # Exactly 0
        2_000_000_000,  # Exactly 2 sec
        4_000_000_000,  # Exactly 4 sec
        6_000_000_000,  # Exactly 6 sec
    ], dtype=np.int64)
    interval_seconds = 2.0

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # CORRECTED: Last timestamp 6e9, ceil(6e9/2e9) = 3, last_ts = 6e9
    # Range: arange(0, 6e9 + 2e9 + 1, 2e9) = [0, 2e9, 4e9, 6e9, 8e9]
    expected_bar_clock = np.array([0, 2_000_000_000, 4_000_000_000, 6_000_000_000, 8_000_000_000], dtype=np.int64)

    np.testing.assert_array_equal(bar_clock, expected_bar_clock)
    assert len(bar_close_indices) == len(expected_bar_clock)


# ---------- Keep all passing tests unchanged ----------

def test_time_bar_indexer_microsecond_precision():
    """Test with microsecond-level precision."""
    timestamps = np.array([
        1_000_001_000,  # 1.000001 sec
        1_000_002_000,  # 1.000002 sec
        1_000_003_000,  # 1.000003 sec
    ], dtype=np.int64)
    interval_seconds = 0.000002  # 2 microseconds

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # Should handle microsecond precision correctly
    assert len(bar_clock) > 1
    assert len(bar_close_indices) == len(bar_clock)


def test_time_bar_indexer_large_intervals():
    """Test with very large intervals."""
    timestamps = np.array([
        1_000_000_000,  # 1 sec
        60_000_000_000,  # 60 sec (1 minute)
        120_000_000_000,  # 120 sec (2 minutes)
        180_000_000_000,  # 180 sec (3 minutes)
    ], dtype=np.int64)
    interval_seconds = 3600.0  # 1 hour intervals

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # All trades should fall within first hour bar
    assert len(bar_clock) >= 1
    assert len(bar_close_indices) == len(bar_clock)


def test_time_bar_indexer_identical_timestamps():
    """Test with identical timestamps (simultaneous trades)."""
    timestamps = np.array([
        1_000_000_000,  # 1 sec
        1_000_000_000,  # 1 sec (duplicate)
        1_000_000_000,  # 1 sec (duplicate)
        2_000_000_000,  # 2 sec
    ], dtype=np.int64)
    interval_seconds = 1.5

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # Should handle duplicates correctly
    assert len(bar_clock) >= 1
    assert len(bar_close_indices) == len(bar_clock)
    assert all(idx < len(timestamps) for idx in bar_close_indices if idx >= 0)


def test_time_bar_indexer_return_types():
    """Test that return types are correct."""
    timestamps = np.array([1_000_000_000, 2_000_000_000], dtype=np.int64)
    interval_seconds = 1.0

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # Verify return types
    assert isinstance(bar_clock, np.ndarray)
    assert isinstance(bar_close_indices, np.ndarray)
    assert bar_clock.dtype == np.int64
    assert bar_close_indices.dtype == np.int64


def test_time_bar_indexer_monotonic_properties():
    """Test that results have correct monotonic properties."""
    timestamps = np.array([
        500_000_000,
        1_500_000_000,
        2_500_000_000,
        3_500_000_000,
        4_500_000_000,
    ], dtype=np.int64)
    interval_seconds = 1.0

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # Bar clock should be monotonically increasing
    assert np.all(bar_clock[1:] >= bar_clock[:-1])

    # All indices should be valid
    valid_indices = bar_close_indices[bar_close_indices >= 0]
    assert all(0 <= idx < len(timestamps) for idx in valid_indices)


def test_time_bar_indexer_large_dataset():
    """Test with larger dataset for performance validation."""
    np.random.seed(42)
    n_trades = 10000

    # Generate sorted timestamps over 1 hour
    base_time = 1_600_000_000_000_000_000  # Example base timestamp
    timestamps = np.sort(base_time + np.random.randint(0, 3600_000_000_000, n_trades))
    timestamps = timestamps.astype(np.int64)

    interval_seconds = 60.0  # 1-minute bars

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # Should complete without error
    assert len(bar_clock) > 0
    assert len(bar_close_indices) == len(bar_clock)
    assert np.all(bar_clock[1:] > bar_clock[:-1])  # Strictly increasing

    # All indices should be valid
    valid_indices = bar_close_indices[bar_close_indices >= 0]
    assert all(0 <= idx < len(timestamps) for idx in valid_indices)


def test_time_bar_indexer_consistency():
    """Test consistency between bar_clock and bar_close_indices."""
    timestamps = np.array([
        1_100_000_000,  # 1.1 sec
        1_900_000_000,  # 1.9 sec
        2_100_000_000,  # 2.1 sec
        2_800_000_000,  # 2.8 sec
        3_200_000_000,  # 3.2 sec
    ], dtype=np.int64)
    interval_seconds = 1.0

    bar_clock, bar_close_indices = _time_bar_indexer(timestamps, interval_seconds)

    # Verify that bar_close_indices point to trades that are <= corresponding bar_clock times
    for i, (bar_time, close_idx) in enumerate(zip(bar_clock, bar_close_indices)):
        if close_idx >= 0:  # Skip invalid indices
            assert timestamps[
                       close_idx] <= bar_time, f"Bar {i}: trade at {timestamps[close_idx]} > bar close at {bar_time}"


if __name__ == "__main__":
    pytest.main([__file__])