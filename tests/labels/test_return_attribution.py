import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

# disable jit
#os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.weights import return_attribution


# ---------- Input Validation Tests ----------
def test_return_attribution_empty_inputs():
    """Test with empty event arrays."""
    close = np.array([100, 101, 102], dtype=np.float64)
    event_idxs = np.array([], dtype=np.int64)
    touch_idxs = np.array([], dtype=np.int64)
    concurrency = np.array([1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, False)

    assert len(weights) == 0
    assert weights.dtype == np.float64


def test_return_attribution_normalization_zero_weights():
    """Test normalization when sum of weights is zero."""
    close = np.array([100, 100, 100, 100], dtype=np.float64)  # No price changes
    event_idxs = np.array([0, 2], dtype=np.int64)
    touch_idxs = np.array([1, 3], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1], dtype=np.int16)

    with pytest.raises(ValueError, match="Sum of weights is zero or negative, cannot normalize"):
        return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=True)


# ---------- Basic Functionality Tests ----------

def test_return_attribution_basic_functionality():
    """Test basic return attribution calculation."""
    close = np.array([100, 105, 110, 115, 120], dtype=np.float64)
    event_idxs = np.array([0, 2], dtype=np.int64)
    touch_idxs = np.array([2, 4], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Basic output validation
    assert weights.dtype == np.float64
    assert len(weights) == len(event_idxs)
    assert np.all(weights >= 0)  # Should be non-negative due to abs()
    assert np.all(np.isfinite(weights))


def test_return_attribution_mathematical_correctness():
    """Test mathematical correctness of return attribution calculation."""
    close = np.array([100, 102, 104, 106], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([3], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Manual calculation:
    # log_rets = [NaN, log(102/100), log(104/102), log(106/104)]
    # Event spans indices 0-3, but log_rets[0] is NaN
    # weight = |log(102/100)/1 + log(104/102)/1 + log(106/104)/1|
    expected_log_rets = [
        np.log(102 / 100),  # ≈ 0.0198
        np.log(104 / 102),  # ≈ 0.0194
        np.log(106 / 104)  # ≈ 0.0190
    ]
    expected_weight = abs(sum(expected_log_rets))

    assert_array_almost_equal([weights[0]], [expected_weight], decimal=6)


def test_return_attribution_single_event():
    """Test with single event."""
    close = np.array([100, 105, 110], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([2], dtype=np.int64)
    concurrency = np.array([1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    assert len(weights) == 1
    assert weights[0] > 0  # Should be positive with price increases


def test_return_attribution_multiple_events():
    """Test with multiple non-overlapping events."""
    close = np.array([100, 102, 104, 106, 108, 110], dtype=np.float64)
    event_idxs = np.array([0, 3], dtype=np.int64)
    touch_idxs = np.array([2, 5], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    assert len(weights) == 2
    assert np.all(weights > 0)  # Both should be positive
    assert np.all(np.isfinite(weights))


# ---------- Concurrency Tests ----------

def test_return_attribution_with_concurrency():
    """Test return attribution with overlapping events (concurrency > 1)."""
    close = np.array([100, 105, 110, 115, 120], dtype=np.float64)
    event_idxs = np.array([0, 1], dtype=np.int64)
    touch_idxs = np.array([3, 4], dtype=np.int64)
    concurrency = np.array([1, 2, 2, 2, 1], dtype=np.int16)  # Events overlap at indices 1,2,3

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Manual calculation for first event (indices 0-3):
    # log_rets[0] = NaN (skipped)
    # log_rets[1] = log(105/100), concurrency[1] = 2 → contrib = log(105/100)/2
    # log_rets[2] = log(110/105), concurrency[2] = 2 → contrib = log(110/105)/2
    # log_rets[3] = log(115/110), concurrency[3] = 2 → contrib = log(115/110)/2

    expected_weight_0 = abs(np.log(105 / 100) / 2 + np.log(110 / 105) / 2 + np.log(115 / 110) / 2)

    # Second event (indices 1-4):
    # log_rets[1] = log(105/100)/2, log_rets[2] = log(110/105)/2,
    # log_rets[3] = log(115/110)/2, log_rets[4] = log(120/115)/1
    expected_weight_1 = abs(
        np.log(105 / 100) / 2 + np.log(110 / 105) / 2 + np.log(115 / 110) / 2 + np.log(120 / 115) / 1)

    assert_array_almost_equal(weights, [expected_weight_0, expected_weight_1], decimal=6)


def test_return_attribution_zero_concurrency():
    """Test handling of zero concurrency values."""
    close = np.array([100, 105, 110, 115], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([3], dtype=np.int64)
    concurrency = np.array([1, 0, 1, 1], dtype=np.int16)  # Zero concurrency at index 1

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Should skip the zero concurrency timestamp
    # weight = |log_rets[1]/0 + log_rets[2]/1 + log_rets[3]/1|
    # But log_rets[1]/0 is skipped due to concurrency[1] == 0
    # So weight = |log(110/105) + log(115/110)|
    expected_weight = abs(np.log(110 / 105) + np.log(115 / 110))

    assert_array_almost_equal([weights[0]], [expected_weight], decimal=6)


# ---------- Edge Cases ----------
def test_return_attribution_identical_prices():
    """Test with identical prices (zero returns)."""
    close = np.array([100, 100, 100, 100], dtype=np.float64)
    event_idxs = np.array([0, 1], dtype=np.int64)
    touch_idxs = np.array([2, 3], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # All log returns should be 0, so weights should be 0
    assert_array_almost_equal(weights, [0.0, 0.0], decimal=10)


def test_return_attribution_single_price_event():
    """Test event that spans only one timestamp."""
    close = np.array([100, 105, 110, 115], dtype=np.float64)
    event_idxs = np.array([1], dtype=np.int64)
    touch_idxs = np.array([1], dtype=np.int64)  # Event spans only index 1
    concurrency = np.array([1, 1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Event at single timestamp: only log_rets[1] = log(105/100)
    expected_weight = abs(np.log(105 / 100))
    assert_array_almost_equal([weights[0]], [expected_weight], decimal=6)


# ---------- Normalization Tests ----------

def test_return_attribution_normalization_basic():
    """Test basic normalization functionality."""
    close = np.array([100, 105, 110, 115, 120], dtype=np.float64)
    event_idxs = np.array([0, 2], dtype=np.int64)
    touch_idxs = np.array([2, 4], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1, 1], dtype=np.int16)

    weights_unnorm = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)
    weights_norm = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=True)

    # Normalized weights should sum to n_events
    assert_array_almost_equal([np.sum(weights_norm)], [len(event_idxs)], decimal=10)

    # Ratio should be preserved
    if np.sum(weights_unnorm) > 0:
        scaling_factor = len(event_idxs) / np.sum(weights_unnorm)
        expected_norm = weights_unnorm * scaling_factor
        assert_array_almost_equal(weights_norm, expected_norm, decimal=10)


def test_return_attribution_normalization_single_event():
    """Test normalization with single event."""
    close = np.array([100, 110, 120], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([2], dtype=np.int64)
    concurrency = np.array([1, 1, 1], dtype=np.int16)

    weights_norm = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=True)

    # Single event normalized should sum to 1
    assert_array_almost_equal([np.sum(weights_norm)], [1.0], decimal=10)
    assert weights_norm[0] == 1.0


def test_return_attribution_normalization_proportions():
    """Test that normalization preserves proportions."""
    close = np.array([100, 105, 110, 115, 120, 125], dtype=np.float64)
    event_idxs = np.array([0, 2, 4], dtype=np.int64)
    touch_idxs = np.array([1, 3, 5], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1, 1, 1], dtype=np.int16)

    weights_unnorm = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)
    weights_norm = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=True)

    if np.sum(weights_unnorm) > 0:
        # Proportions should be preserved
        proportions_unnorm = weights_unnorm / np.sum(weights_unnorm)
        proportions_norm = weights_norm / np.sum(weights_norm)
        assert_array_almost_equal(proportions_unnorm, proportions_norm, decimal=10)


# ---------- Mathematical Properties Tests ----------

def test_return_attribution_absolute_value_property():
    """Test that weights are always non-negative due to abs()."""
    # Create scenario with negative returns
    close = np.array([100, 95, 90, 85], dtype=np.float64)  # Decreasing prices
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([3], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Should be non-negative despite negative log returns
    assert np.all(weights >= 0)
    assert weights[0] > 0  # Should be positive due to absolute value


def test_return_attribution_log_return_accumulation():
    """Test correct accumulation of log returns over event duration."""
    close = np.array([100, 110, 121, 133.1], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([3], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Manual calculation
    log_ret_1 = np.log(110 / 100)  # ≈ 0.0953
    log_ret_2 = np.log(121 / 110)  # ≈ 0.0953
    log_ret_3 = np.log(133.1 / 121)  # ≈ 0.0953
    expected_weight = abs(log_ret_1 + log_ret_2 + log_ret_3)

    assert_array_almost_equal([weights[0]], [expected_weight], decimal=4)


def test_return_attribution_concurrency_scaling():
    """Test that concurrency properly scales return contributions."""
    close = np.array([100, 110, 121], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([2], dtype=np.int64)

    # Test with different concurrency levels
    concurrency_1 = np.array([1, 1, 1], dtype=np.int16)
    concurrency_2 = np.array([1, 2, 2], dtype=np.int16)

    weights_1 = return_attribution(event_idxs, touch_idxs, close, concurrency_1, normalize=False)
    weights_2 = return_attribution(event_idxs, touch_idxs, close, concurrency_2, normalize=False)

    # Higher concurrency should result in lower attribution
    assert weights_2[0] < weights_1[0]

    # Event spans indices 0-2, but log_rets[0] is NaN (first element)
    # So only indices 1 and 2 contribute:
    # log_rets[1] = log(110/100), concurrency[1] = 2
    # log_rets[2] = log(121/110), concurrency[2] = 2
    expected_weight_2 = abs(np.log(110 / 100) / 2 + np.log(121 / 110) / 2)
    assert_array_almost_equal([weights_2[0]], [expected_weight_2], decimal=6)


# ---------- Performance and Robustness Tests ----------

def test_return_attribution_large_dataset():
    """Test with large dataset for performance validation."""
    np.random.seed(42)
    n_prices = 10000
    n_events = 1000

    # Generate realistic price series
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_prices - 1)  # Small daily returns
    close = np.zeros(n_prices)
    close[0] = base_price
    for i in range(1, n_prices):
        close[i] = close[i - 1] * np.exp(returns[i - 1])

    # Generate random events
    event_idxs = np.sort(np.random.choice(range(n_prices - 10), n_events, replace=False))
    touch_idxs = event_idxs + np.random.randint(1, 10, n_events)
    touch_idxs = np.minimum(touch_idxs, n_prices - 1)

    concurrency = np.random.randint(1, 5, n_prices).astype(np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Should complete without error
    assert len(weights) == n_events
    assert weights.dtype == np.float64
    assert np.all(weights >= 0)
    assert np.all(np.isfinite(weights) | np.isnan(weights))


def test_return_attribution_extreme_values():
    """Test with extreme price values."""
    # Test with very large prices
    close_large = np.array([1e6, 1.1e6, 1.2e6], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([2], dtype=np.int64)
    concurrency = np.array([1, 1, 1], dtype=np.int16)

    weights_large = return_attribution(event_idxs, touch_idxs, close_large, concurrency, normalize=False)

    assert len(weights_large) == 1
    assert np.isfinite(weights_large[0])
    assert weights_large[0] > 0

    # Test with very small prices
    close_small = np.array([1e-6, 1.1e-6, 1.2e-6], dtype=np.float64)
    weights_small = return_attribution(event_idxs, touch_idxs, close_small, concurrency, normalize=False)

    assert len(weights_small) == 1
    assert np.isfinite(weights_small[0])


def test_return_attribution_high_volatility():
    """Test with highly volatile prices."""
    close = np.array([100, 200, 50, 300, 25, 400], dtype=np.float64)  # Very volatile
    event_idxs = np.array([0, 2], dtype=np.int64)
    touch_idxs = np.array([2, 5], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Should handle high volatility gracefully
    assert len(weights) == 2
    assert np.all(weights >= 0)
    assert np.all(np.isfinite(weights))


# ---------- Return Types and Data Consistency ----------

def test_return_attribution_return_types():
    """Test correct return types and dtypes."""
    close = np.array([100, 105, 110], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([2], dtype=np.int64)
    concurrency = np.array([1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Verify return type and dtype
    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    assert weights.shape == event_idxs.shape

# ---------- Integration with Financial ML Pipeline ----------

def test_return_attribution_realistic_scenario():
    """Test with realistic financial ML scenario."""
    # Simulate daily price data for 100 days
    np.random.seed(42)
    n_days = 100
    base_price = 100.0
    daily_returns = np.random.normal(0.001, 0.02, n_days - 1)

    close = np.zeros(n_days)
    close[0] = base_price
    for i in range(1, n_days):
        close[i] = close[i - 1] * np.exp(daily_returns[i - 1])

    # Create events representing trading signals
    event_idxs = np.array([10, 25, 40, 60, 80], dtype=np.int64)
    touch_idxs = np.array([20, 35, 50, 70, 90], dtype=np.int64)

    # Realistic concurrency (some overlap between events)
    concurrency = np.ones(n_days, dtype=np.int16)
    concurrency[15:30] = 2  # Overlap between events 1 and 2
    concurrency[45:65] = 2  # Overlap between events 3 and 4

    # Test both normalized and unnormalized
    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)
    weights_norm = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=True)

    # Validate results
    assert len(weights) == len(event_idxs)
    assert np.all(weights >= 0)
    assert np.all(np.isfinite(weights))

    # Normalized should sum to n_events
    assert_array_almost_equal([np.sum(weights_norm)], [len(event_idxs)], decimal=10)

    # Events with higher absolute returns should generally get higher weights
    # (though concurrency affects this)


def test_return_attribution_mathematical_edge_cases():
    """Test mathematical edge cases that could cause issues."""
    # Test with prices that would cause very large/small log returns
    close = np.array([100, 1, 1000, 0.001], dtype=np.float64)
    event_idxs = np.array([0], dtype=np.int64)
    touch_idxs = np.array([3], dtype=np.int64)
    concurrency = np.array([1, 1, 1, 1], dtype=np.int16)

    weights = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=False)

    # Should handle extreme returns gracefully
    assert len(weights) == 1
    assert isinstance(weights[0], (float, np.floating))
    # Result might be large but should be a valid float
    assert not np.isnan(weights[0]) or np.isinf(weights[0])  # Allow inf but not NaN from calculation errors


if __name__ == "__main__":
    pytest.main([__file__])