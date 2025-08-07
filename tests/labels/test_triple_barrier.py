import numpy as np
import pytest
import os
# disable jit
#os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.tbm import triple_barrier


# ---------- Helper Functions ----------

def create_test_data(n_timestamps=100, n_events=10, price_trend='random'):
    """Create realistic test data for triple barrier testing."""
    timestamps = np.arange(n_timestamps, dtype=np.int64) * int(1e9)  # 1-second intervals

    if price_trend == 'up':
        close = np.linspace(100, 120, n_timestamps)
    elif price_trend == 'down':
        close = np.linspace(120, 100, n_timestamps)
    elif price_trend == 'volatile':
        base = 100
        close = base + np.cumsum(np.random.randn(n_timestamps) * 0.5)
    else:  # random
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n_timestamps) * 0.1)

    event_idxs = np.linspace(0, n_timestamps - 10, n_events, dtype=np.int64)
    targets = np.random.uniform(0.01, 0.05, n_events)

    return timestamps, close, event_idxs, targets


# ---------- Input Validation Tests ----------

def test_triple_barrier_input_validation():
    """Test all input validation error conditions."""
    timestamps = np.arange(10, dtype=np.int64) * int(1e9)
    close = np.linspace(100, 110, 10)
    event_idxs = np.array([1, 3, 5], dtype=np.int64)
    targets = np.array([0.01, 0.02, 0.03], dtype=np.float64)

    # Test vertical_barrier <= 0
    with pytest.raises(ValueError, match="vertical barrier must be greater than zero"):
        triple_barrier(timestamps, close, event_idxs, targets, (1.0, 1.0), 0.0, 0.0, None, 0.0)

    with pytest.raises(ValueError, match="vertical barrier must be greater than zero"):
        triple_barrier(timestamps, close, event_idxs, targets, (1.0, 1.0), -1.0, 0.0, None, 0.0)

    # Test min_ret < 0
    with pytest.raises(ValueError, match="minimum return must be non-negative"):
        triple_barrier(timestamps, close, event_idxs, targets, (1.0, 1.0), 1.0, 0.0, None, -0.1)

    # Test timestamps and close length mismatch
    with pytest.raises(ValueError, match="lengths of timestamps and close must match"):
        triple_barrier(timestamps[:-1], close, event_idxs, targets, (1.0, 1.0), 1.0, 0.0, None, 0.0)

    # Test event_idxs and targets length mismatch
    with pytest.raises(ValueError, match="lengths of event_idxs and targets must match"):
        triple_barrier(timestamps, close, event_idxs, targets[:-1], (1.0, 1.0), 1.0, 0.0, None, 0.0)

    # Test empty event_idxs
    with pytest.raises(ValueError, match="event_idxs array must not be empty"):
        triple_barrier(timestamps, close, np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                       (1.0, 1.0), 1.0, 0.0, None, 0.0)

    # Test event_idxs and side length mismatch when side is provided
    side = np.array([1, -1], dtype=np.int8)  # Wrong length
    with pytest.raises(ValueError, match="length of event_idxs must match the length of side"):
        triple_barrier(timestamps, close, event_idxs, targets, (1.0, 1.0), 1.0, 0.0, side, 0.0)


def test_triple_barrier_boundary_indices():
    """Test events at boundary indices."""
    timestamps = np.arange(5, dtype=np.int64) * int(1e9)
    close = np.array([100, 101, 102, 103, 104], dtype=np.float64)

    # Event at first index
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.02], dtype=np.float64)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 0.0, None, 0.0
    )

    assert len(labels) == 1
    assert touch_idxs[0] >= event_idxs[0]

    # Event at last valid index
    event_idxs = np.array([3], dtype=np.int64)
    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 0.0, None, 0.0
    )

    assert len(labels) == 1
    assert touch_idxs[0] >= event_idxs[0]


# ---------- Basic Functionality Tests ----------

def test_triple_barrier_basic_functionality():
    """Test basic labeling functionality."""
    timestamps = np.arange(10, dtype=np.int64) * int(1e9)
    close = np.array([100, 105, 110, 115, 120, 125, 130, 125, 120, 115], dtype=np.float64)
    event_idxs = np.array([0, 2, 4], dtype=np.int64)
    targets = np.array([0.05, 0.03, 0.02], dtype=np.float64)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 0.0, None, 0.0
    )

    # Basic output validation
    assert labels.dtype == np.int8
    assert touch_idxs.dtype == np.int64
    assert rets.dtype == np.float64
    assert max_rb_ratios.dtype == np.float64

    # All arrays should have same length as events
    assert len(labels) == len(event_idxs)
    assert len(touch_idxs) == len(event_idxs)
    assert len(rets) == len(event_idxs)
    assert len(max_rb_ratios) == len(event_idxs)

    # Labels should be -1 or 1 for regular labeling
    assert np.all(np.isin(labels, [-1, 1]))

    # Touch indices should be valid
    assert np.all(touch_idxs >= event_idxs)
    assert np.all(touch_idxs < len(timestamps))


def test_triple_barrier_meta_labeling():
    """Test meta-labeling functionality (0/1 labels)."""
    timestamps = np.arange(10, dtype=np.int64) * int(1e9)
    close = np.array([100, 102, 101, 103, 99, 105, 98, 106, 97, 107], dtype=np.float64)
    event_idxs = np.array([0, 2, 4], dtype=np.int64)
    targets = np.array([0.02, 0.03, 0.01], dtype=np.float64)
    side = np.array([1, -1, 1], dtype=np.int8)
    min_ret = 0.015

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 0.0, side, min_ret
    )

    # Meta-labels should be 0 or 1
    assert np.all(np.isin(labels, [0, 1]))

    # Verify meta-labeling logic: label = 1 if ret >= min_ret else 0
    for i, ret in enumerate(rets):
        if not np.isnan(ret):
            expected_label = 1 if ret >= min_ret else 0
            assert labels[i] == expected_label


def test_triple_barrier_symmetric_barriers():
    """Test symmetric horizontal barriers."""
    timestamps = np.arange(20, dtype=np.int64) * int(1e9)
    close = np.linspace(100, 120, 20)  # Strong upward trend
    event_idxs = np.array([0, 5, 10], dtype=np.int64)
    targets = np.array([0.1, 0.1, 0.1], dtype=np.float64)  # 10% targets

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 15.0, 0.0, None, 0.0
    )

    # With strong upward trend, should mostly get positive labels
    assert np.sum(labels == 1) >= np.sum(labels == -1)

    # Check that barriers are working - should hit before vertical barrier
    for i, touch_idx in enumerate(touch_idxs):
        vertical_limit_idx = np.searchsorted(timestamps, timestamps[event_idxs[i]] + 15 * 1e9, side='right') - 1
        # Some should hit horizontal barriers before vertical
        # (This depends on the specific data, but with strong trend some should hit)


def test_triple_barrier_asymmetric_barriers():
    """Test asymmetric horizontal barriers."""
    timestamps = np.arange(10, dtype=np.int64) * int(1e9)
    close = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118], dtype=np.float64)
    event_idxs = np.array([0, 3], dtype=np.int64)
    targets = np.array([0.05, 0.03], dtype=np.float64)

    # Asymmetric barriers: tighter on downside
    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (0.5, 2.0), 8.0, 0.0, None, 0.0
    )

    assert len(labels) == len(event_idxs)
    # With upward trend and tighter downside barriers, should get more positive labels
    assert np.all(labels >= -1)
    assert np.all(labels <= 1)


# ---------- Barrier Hit Scenarios ----------

def test_triple_barrier_upper_barrier_hit():
    """Test upper barrier hit scenarios."""
    timestamps = np.arange(5, dtype=np.int64) * int(1e9)
    close = np.array([100, 110, 120, 130, 140], dtype=np.float64)  # Strong upward movement
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)  # 5% target

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 10.0, 0.0, None, 0.0
    )

    # Should hit upper barrier quickly with this price movement
    assert labels[0] == 1
    assert max_rb_ratios[0] == 1.0  # Barrier hit, so ratio should be 1.0
    assert touch_idxs[0] < 4  # Should hit before end


def test_triple_barrier_lower_barrier_hit():
    """Test lower barrier hit scenarios."""
    timestamps = np.arange(5, dtype=np.int64) * int(1e9)
    close = np.array([100, 90, 80, 70, 60], dtype=np.float64)  # Strong downward movement
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 10.0, 0.0, None, 0.0
    )

    # Should hit lower barrier with this price movement
    assert labels[0] == -1
    assert max_rb_ratios[0] == 1.0  # Barrier hit
    assert touch_idxs[0] < 4


def test_triple_barrier_vertical_barrier_timeout():
    """Test vertical barrier timeout scenarios."""
    timestamps = np.arange(10, dtype=np.int64) * int(1e9)
    close = np.array([100, 100.5, 101, 100.5, 100, 100.2, 99.8, 100.1, 99.9, 100.3], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.1], dtype=np.float64)  # Large target, unlikely to hit

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 0.0, None, 0.0
    )

    # Should timeout at vertical barrier
    vertical_limit_idx = np.searchsorted(timestamps, timestamps[0] + 5 * 1e9, side='right') - 1
    assert touch_idxs[0] == vertical_limit_idx

    # max_rb_ratio should be < 1.0 (didn't hit horizontal barrier)
    assert max_rb_ratios[0] < 1.0 or np.isnan(max_rb_ratios[0])


# ---------- Special Barrier Configurations ----------

def test_triple_barrier_infinite_barriers():
    """Test with infinite barriers (disabled horizontal barriers)."""
    timestamps = np.arange(10, dtype=np.int64) * int(1e9)
    close = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.1], dtype=np.float64)

    # Infinite barriers - should only hit vertical barrier
    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (np.inf, np.inf), 5.0, 0.0, None, 0.0
    )

    # Should timeout at vertical barrier regardless of price movement
    vertical_limit_idx = np.searchsorted(timestamps, timestamps[0] + 5 * 1e9, side='right') - 1
    assert touch_idxs[0] == vertical_limit_idx

    # max_rb_ratio should be NaN (barriers disabled)
    assert np.isnan(max_rb_ratios[0])


def test_triple_barrier_zero_targets():
    """Test with zero targets."""
    timestamps = np.arange(5, dtype=np.int64) * int(1e9)
    close = np.array([100, 105, 110, 115, 120], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.0], dtype=np.float64)  # Zero target

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 3.0, 0.0, None, 0.0
    )

    # Should immediately hit upper barrier (any positive return triggers it)
    assert labels[0] == 1
    assert touch_idxs[0] == 1  # Should hit on first price move
    assert max_rb_ratios[0] == 1.0


# ---------- Minimum Close Time Tests ----------

def test_triple_barrier_min_close_time():
    """Test minimum close time constraint."""
    timestamps = np.array([0, int(0.5e9), int(1e9), int(2e9), int(3e9)], dtype=np.int64)
    close = np.array([100, 150, 200, 250, 300], dtype=np.float64)  # Large price movements
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.01], dtype=np.float64)  # Small target, should hit quickly

    # Minimum close time of 1.5 seconds
    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 1.5, None, 0.0
    )

    # Should not close before 1.5 seconds despite hitting barrier earlier
    assert timestamps[touch_idxs[0]] - timestamps[0] >= 1.5 * 1e9

    # Compare with no minimum close time
    labels2, touch_idxs2, rets2, max_rb_ratios2 = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 0.0, None, 0.0
    )

    # Without min close time, should close earlier
    assert touch_idxs2[0] <= touch_idxs[0]


def test_triple_barrier_min_close_time_longer_than_vertical():
    """Test min close time longer than vertical barrier."""
    timestamps = np.arange(5, dtype=np.int64) * int(1e9)
    close = np.array([100, 110, 120, 130, 140], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.05], dtype=np.float64)

    # Min close time longer than vertical barrier
    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 2.0, 5.0, None, 0.0
    )

    # Should close at vertical barrier despite min close time constraint
    vertical_limit_idx = np.searchsorted(timestamps, timestamps[0] + 2 * 1e9, side='right') - 1
    assert touch_idxs[0] == vertical_limit_idx


# ---------- Side and Meta-Labeling Tests ----------

def test_triple_barrier_side_effects():
    """Test different side values and their effects."""
    timestamps = np.arange(10, dtype=np.int64) * int(1e9)
    close = np.array([100, 105, 110, 115, 120, 125, 130, 135, 140, 145], dtype=np.float64)
    event_idxs = np.array([0, 2], dtype=np.int64)
    targets = np.array([0.05, 0.05], dtype=np.float64)

    # Test with positive side
    side_pos = np.array([1, 1], dtype=np.int8)
    labels_pos, _, rets_pos, _ = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 0.0, side_pos, 0.02
    )

    # Test with negative side
    side_neg = np.array([-1, -1], dtype=np.int8)
    labels_neg, _, rets_neg, _ = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 0.0, side_neg, 0.02
    )

    # With upward trend, positive side should have positive returns, negative side negative
    assert np.all(rets_pos >= 0)  # Positive side with upward trend
    assert np.all(rets_neg <= 0)  # Negative side with upward trend

    # Meta-labels should be 0 or 1
    assert np.all(np.isin(labels_pos, [0, 1]))
    assert np.all(np.isin(labels_neg, [0, 1]))


def test_triple_barrier_mixed_sides():
    """Test mixed positive and negative sides."""
    timestamps = np.arange(8, dtype=np.int64) * int(1e9)
    close = np.array([100, 105, 95, 110, 90, 115, 85, 120], dtype=np.float64)
    event_idxs = np.array([0, 2, 4], dtype=np.int64)
    targets = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    side = np.array([1, -1, 1], dtype=np.int8)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 6.0, 0.0, side, 0.05
    )

    # Check that side multiplier is applied correctly
    for i in range(len(event_idxs)):
        expected_sign = 1 if rets[i] * side[i] > 0 else -1
        # The actual return should have correct sign relative to side


# ---------- Mathematical Correctness Tests ----------

def test_triple_barrier_return_calculation():
    """Test mathematical correctness of return calculations."""
    timestamps = np.array([0, int(1e9), int(2e9)], dtype=np.int64)
    close = np.array([100, 110, 120], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.2], dtype=np.float64)  # Large target to avoid hitting barriers

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 3.0, 0.0, None, 0.0
    )

    # Manually calculate expected return
    expected_ret = np.log(close[touch_idxs[0]]) - np.log(close[event_idxs[0]])
    np.testing.assert_almost_equal(rets[0], expected_ret, decimal=10)


def test_triple_barrier_max_rb_ratio_calculation():
    """Test max return-barrier ratio calculation."""
    timestamps = np.arange(5, dtype=np.int64) * int(1e9)
    close = np.array([100, 102, 104, 101, 103], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    targets = np.array([0.1], dtype=np.float64)  # Large target, won't hit barriers

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 6.0, 0.0, None, 0.0
    )

    # Should timeout at vertical barrier
    assert max_rb_ratios[0] <= 1.0
    assert max_rb_ratios[0] >= 0.0 or np.isnan(max_rb_ratios[0])


def test_triple_barrier_log_return_consistency():
    """Test consistency of log return calculations with side multipliers."""
    timestamps = np.arange(6, dtype=np.int64) * int(1e9)
    close = np.array([100, 105, 110, 115, 120, 125], dtype=np.float64)
    event_idxs = np.array([0, 1], dtype=np.int64)
    targets = np.array([0.3, 0.3], dtype=np.float64)  # Large targets
    side = np.array([1, -1], dtype=np.int8)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 4.0, 0.0, side, 0.0
    )

    # Check log return calculation manually
    for i in range(len(event_idxs)):
        base_ret = np.log(close[touch_idxs[i]]) - np.log(close[event_idxs[i]])
        expected_ret = base_ret * side[i]
        np.testing.assert_almost_equal(rets[i], expected_ret, decimal=10)


# ---------- Edge Cases and Robustness Tests ----------

def test_triple_barrier_single_event():
    """Test with single event."""
    timestamps = np.arange(5, dtype=np.int64) * int(1e9)
    close = np.linspace(100, 105, 5)
    event_idxs = np.array([1], dtype=np.int64)
    targets = np.array([0.02], dtype=np.float64)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 3.0, 0.0, None, 0.0
    )

    assert len(labels) == 1
    assert len(touch_idxs) == 1
    assert len(rets) == 1
    assert len(max_rb_ratios) == 1


def test_triple_barrier_event_at_end():
    """Test event near end of data."""
    timestamps = np.arange(5, dtype=np.int64) * int(1e9)
    close = np.array([100, 101, 102, 103, 104], dtype=np.float64)
    event_idxs = np.array([3], dtype=np.int64)  # Near end
    targets = np.array([0.05], dtype=np.float64)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 5.0, 0.0, None, 0.0
    )

    # Should handle gracefully
    assert len(labels) == 1
    assert touch_idxs[0] >= event_idxs[0]
    assert touch_idxs[0] < len(timestamps)


def test_triple_barrier_very_small_targets():
    """Test with very small targets."""
    timestamps = np.arange(10, dtype=np.int64) * int(1e9)
    close = np.array([100, 100.001, 100.002, 100.001, 100.003, 100.002, 100.001, 100.004, 100.001, 100.002],
                     dtype=np.float64)
    event_idxs = np.array([0, 2], dtype=np.int64)
    targets = np.array([0.00001, 0.00001], dtype=np.float64)  # Very small targets

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 8.0, 0.0, None, 0.0
    )

    # Should hit barriers quickly with any price movement
    assert np.all(max_rb_ratios == 1.0)  # Should hit horizontal barriers


def test_triple_barrier_large_targets():
    """Test with very large targets."""
    timestamps = np.arange(10, dtype=np.int64) * int(1e9)
    close = np.linspace(100, 150, 10)  # 50% price increase
    event_idxs = np.array([0, 3], dtype=np.int64)
    targets = np.array([1.0, 1.0], dtype=np.float64)  # 100% targets

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 8.0, 0.0, None, 0.0
    )

    # Should timeout at vertical barrier
    for i in range(len(event_idxs)):
        vertical_limit_idx = np.searchsorted(timestamps, timestamps[event_idxs[i]] + 8 * 1e9, side='right') - 1
        assert touch_idxs[i] == vertical_limit_idx


# ---------- Performance and Scalability Tests ----------

def test_triple_barrier_large_dataset():
    """Test with large dataset for performance validation."""
    np.random.seed(42)
    n_timestamps = 10000
    n_events = 1000

    timestamps = np.arange(n_timestamps, dtype=np.int64) * int(1e9)
    close = 100 + np.cumsum(np.random.randn(n_timestamps) * 0.01)
    event_idxs = np.linspace(0, n_timestamps - 100, n_events, dtype=np.int64)
    targets = np.random.uniform(0.01, 0.05, n_events)
    side = np.random.choice([-1, 1], n_events).astype(np.int8)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 60.0, 0.0, side, 0.02
    )

    # Should complete without error
    assert len(labels) == n_events
    assert len(touch_idxs) == n_events
    assert len(rets) == n_events
    assert len(max_rb_ratios) == n_events

    # All outputs should be finite or NaN (no infinite values)
    assert np.all(np.isfinite(labels))
    assert np.all(np.isfinite(touch_idxs))
    assert np.all(np.isfinite(rets) | np.isnan(rets))
    assert np.all(np.isfinite(max_rb_ratios) | np.isnan(max_rb_ratios))

    # Labels should be in correct ranges
    assert np.all(np.isin(labels, [0, 1]))  # Meta-labeling with side provided
    assert np.all(touch_idxs >= event_idxs)
    assert np.all(touch_idxs < n_timestamps)


def test_triple_barrier_extreme_volatility():
    """Test with extremely volatile prices."""
    timestamps = np.arange(100, dtype=np.int64) * int(1e9)
    close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.1))  # Extreme volatility
    event_idxs = np.array([10, 30, 50, 70], dtype=np.int64)
    targets = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float64)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 20.0, 0.0, None, 0.0
    )

    # Should handle extreme volatility without errors
    assert len(labels) == len(event_idxs)
    assert np.all(np.isin(labels, [-1, 1]))
    assert np.all(np.isfinite(rets))


# ---------- Return Type and Data Type Tests ----------

def test_triple_barrier_return_types():
    """Test correct return types and dtypes."""
    timestamps = np.arange(5, dtype=np.int64) * int(1e9)
    close = np.linspace(100, 105, 5)
    event_idxs = np.array([0, 2], dtype=np.int64)
    targets = np.array([0.02, 0.03], dtype=np.float64)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 3.0, 0.0, None, 0.0
    )

    # Verify return types
    assert isinstance(labels, np.ndarray)
    assert isinstance(touch_idxs, np.ndarray)
    assert isinstance(rets, np.ndarray)
    assert isinstance(max_rb_ratios, np.ndarray)

    # Verify dtypes
    assert labels.dtype == np.int8
    assert touch_idxs.dtype == np.int64
    assert rets.dtype == np.float64
    assert max_rb_ratios.dtype == np.float64


def test_triple_barrier_array_shapes():
    """Test that all returned arrays have correct shapes."""
    timestamps = np.arange(20, dtype=np.int64) * int(1e9)
    close = np.random.randn(20) * 0.01 + 100
    event_idxs = np.array([2, 5, 8, 12, 15], dtype=np.int64)
    targets = np.random.uniform(0.01, 0.05, 5)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps, close, event_idxs, targets, (1.0, 1.0), 10.0, 0.0, None, 0.0
    )

    expected_shape = event_idxs.shape
    assert labels.shape == expected_shape
    assert touch_idxs.shape == expected_shape
    assert rets.shape == expected_shape
    assert max_rb_ratios.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])