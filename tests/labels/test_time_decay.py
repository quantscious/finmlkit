import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

# disable jit
#os.environ['NUMBA_DISABLE_JIT'] = "1"

from finmlkit.label.weights import time_decay


# ---------- Input Validation Tests ----------
def test_time_decay_zero_sum_uniqueness():
    """Test error when sum of uniqueness weights is zero."""
    avg_uniqueness = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="sum of all average uniqueness weights must be grater than 0"):
        time_decay(avg_uniqueness, 0.5)


# ---------- Basic Functionality Tests ----------

def test_time_decay_no_decay():
    """Test with last_weight = 1.0 (no decay)."""
    avg_uniqueness = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float64)

    weights = time_decay(avg_uniqueness, 1.0)

    # FIXED: With last_weight = 1.0, slope = 0, so all weights = const = 1.0
    expected_weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    assert_array_almost_equal(weights, expected_weights, decimal=10)


def test_time_decay_linear_positive():
    """Test linear decay with positive last_weight."""
    avg_uniqueness = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    last_weight = 0.4

    weights = time_decay(avg_uniqueness, last_weight)

    # CORRECTED: Manual calculation using actual algorithm
    # cum_avg_uniqueness = [0.5, 1.0, 1.5, 2.0]
    # slope = (1 - 0.4) / 2.0 = 0.3
    # const = 1 - 0.3 * 2.0 = 0.4
    # weights = 0.4 + 0.3 * [0.5, 1.0, 1.5, 2.0] = [0.55, 0.7, 0.85, 1.0]

    expected_weights = np.array([0.55, 0.7, 0.85, 1.0], dtype=np.float64)
    assert_array_almost_equal(weights, expected_weights, decimal=10)


def test_time_decay_linear_decay_mathematical_verification():
    """Test mathematical properties of linear decay."""
    avg_uniqueness = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    last_weight = 0.6

    weights = time_decay(avg_uniqueness, last_weight)

    # CORRECTED: The newest observation (last element) always gets weight 1.0
    assert abs(weights[-1] - 1.0) < 1e-10

    # Verify linear relationship holds
    cum_avg = np.cumsum(avg_uniqueness)
    slope = (1.0 - last_weight) / cum_avg[-1]
    const = 1.0 - slope * cum_avg[-1]
    expected_weights = const + slope * cum_avg

    assert_array_almost_equal(weights, expected_weights, decimal=10)


def test_time_decay_zero_last_weight():
    """Test with last_weight = 0.0."""
    avg_uniqueness = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    weights = time_decay(avg_uniqueness, 0.0)

    # CORRECTED: Last element should be 1.0, first calculated from formula
    assert abs(weights[-1] - 1.0) < 1e-10

    # With last_weight = 0.0, slope = 1.0 / total, const = 0.0
    # weights[0] = 0.0 + (1.0/1.0) * 0.25 = 0.25
    assert abs(weights[0] - 0.25) < 1e-10

    # Should be monotonically increasing
    assert np.all(np.diff(weights) >= -1e-10)


# ---------- Negative Last Weight Tests ----------

def test_time_decay_negative_last_weight_basic():
    """Test basic functionality with negative last_weight."""
    avg_uniqueness = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    last_weight = -0.3

    weights = time_decay(avg_uniqueness, last_weight)

    # Last weight should still be 1.0 (newest observation)
    assert abs(weights[-1] - 1.0) < 1e-10

    # No negative weights should remain after clipping
    assert np.all(weights >= 0.0)


def test_time_decay_negative_last_weight_near_boundary():
    """Test with negative last_weight close to -1.0 but not exactly -1.0."""
    avg_uniqueness = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
    last_weight = -0.99  # Very close to -1.0 but avoids division by zero

    weights = time_decay(avg_uniqueness, last_weight)

    # Should not raise error and produce valid weights
    assert len(weights) == len(avg_uniqueness)
    assert weights[-1] == 1.0  # Newest should always be 1.0
    assert np.all(weights >= 0.0)  # All weights non-negative after clipping


# ---------- Edge Cases ----------

def test_time_decay_single_element():
    """Test with single element array."""
    avg_uniqueness = np.array([0.8], dtype=np.float64)

    # Test positive last_weight
    weights = time_decay(avg_uniqueness, 0.5)
    assert len(weights) == 1
    # Single element: weights[0] = const + slope * cum[0]
    # With cum[0] = cum[-1] = 0.8, slope = 0.5/0.8, const = 1-0.5 = 0.5
    # weights[0] = 0.5 + (0.5/0.8) * 0.8 = 0.5 + 0.5 = 1.0
    assert abs(weights[0] - 1.0) < 1e-10


def test_time_decay_uniform_uniqueness():
    """Test with uniform uniqueness values."""
    avg_uniqueness = np.ones(5, dtype=np.float64)
    last_weight = 0.2

    weights = time_decay(avg_uniqueness, last_weight)

    # CORRECTED: Last element should be 1.0, not first
    assert abs(weights[-1] - 1.0) < 1e-10

    # Should be monotonically increasing (oldest to newest)
    assert np.all(np.diff(weights) >= -1e-10)


def test_time_decay_increasing_uniqueness():
    """Test with increasing uniqueness values."""
    avg_uniqueness = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    last_weight = 0.4

    weights = time_decay(avg_uniqueness, last_weight)

    # CORRECTED: Verify newest (last) gets weight 1.0
    assert abs(weights[-1] - 1.0) < 1e-10

    # All weights should be positive for positive last_weight
    assert np.all(weights > 0)


def test_time_decay_decreasing_uniqueness():
    """Test with decreasing uniqueness values."""
    avg_uniqueness = np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=np.float64)
    last_weight = 0.6

    weights = time_decay(avg_uniqueness, last_weight)

    # CORRECTED: Verify boundary conditions
    assert abs(weights[-1] - 1.0) < 1e-10


def test_time_decay_very_small_uniqueness():
    """Test with very small uniqueness values."""
    avg_uniqueness = np.array([1e-10, 1e-9, 1e-8, 1e-7], dtype=np.float64)
    last_weight = 0.5

    weights = time_decay(avg_uniqueness, last_weight)

    # Should still work with very small values
    assert len(weights) == len(avg_uniqueness)
    assert abs(weights[-1] - 1.0) < 1e-10


# ---------- Mathematical Properties Tests ----------

def test_time_decay_slope_calculation_positive():
    """Test slope calculation for positive last_weight."""
    avg_uniqueness = np.array([0.3, 0.4, 0.3], dtype=np.float64)
    last_weight = 0.7

    weights = time_decay(avg_uniqueness, last_weight)

    # Manual verification of slope calculation
    cum_sum = np.sum(avg_uniqueness)
    expected_slope = (1.0 - last_weight) / cum_sum

    # Verify using the linear relationship
    cum_avg_uniqueness = np.cumsum(avg_uniqueness)
    const = 1.0 - expected_slope * cum_sum
    expected_weights = const + expected_slope * cum_avg_uniqueness

    assert_array_almost_equal(weights, expected_weights, decimal=10)


def test_time_decay_slope_calculation_negative():
    """Test slope calculation for negative last_weight."""
    avg_uniqueness = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    last_weight = -0.4

    weights = time_decay(avg_uniqueness, last_weight)

    # Manual verification
    cum_sum = np.sum(avg_uniqueness)
    expected_slope = 1.0 / ((last_weight + 1.0) * cum_sum)

    cum_avg_uniqueness = np.cumsum(avg_uniqueness)
    const = 1.0 - expected_slope * cum_sum
    expected_weights = const + expected_slope * cum_avg_uniqueness
    expected_weights = np.maximum(expected_weights, 0.0)  # Clip negative values

    assert_array_almost_equal(weights, expected_weights, decimal=10)


# ---------- Consistency Tests ----------

def test_time_decay_consistency_across_scales():
    """Test consistency when scaling uniqueness values."""
    avg_uniqueness_1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    avg_uniqueness_2 = avg_uniqueness_1 * 10  # Scaled version

    last_weight = 0.3

    weights_1 = time_decay(avg_uniqueness_1, last_weight)
    weights_2 = time_decay(avg_uniqueness_2, last_weight)

    # Results should be identical (scale-invariant)
    assert_array_almost_equal(weights_1, weights_2, decimal=10)


def test_time_decay_boundary_values():
    """Test boundary values for last_weight."""
    avg_uniqueness = np.array([0.5, 0.5, 0.5], dtype=np.float64)

    # Test last_weight = 1.0 (no decay)
    weights_1 = time_decay(avg_uniqueness, 1.0)
    assert_array_almost_equal(weights_1, np.ones_like(avg_uniqueness))

    # Test last_weight = 0.0 (maximum positive decay)
    weights_0 = time_decay(avg_uniqueness, 0.0)
    assert abs(weights_0[-1] - 1.0) < 1e-10  # Newest always 1.0

    # Test last_weight close to -1.0 (but not exactly to avoid division by zero)
    weights_neg = time_decay(avg_uniqueness, -0.999)
    assert weights_neg[-1] == 1.0
    assert np.all(weights_neg >= 0.0)


# ---------- Large Dataset Tests ----------

def test_time_decay_large_dataset():
    """Test with large dataset for performance validation."""
    np.random.seed(42)
    n_samples = 10000
    avg_uniqueness = np.random.uniform(0.01, 1.0, n_samples)

    weights = time_decay(avg_uniqueness, 0.3)

    # Should complete without error
    assert len(weights) == n_samples
    assert weights.dtype == np.float64

    # All weights should be finite and non-negative
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)

    # CORRECTED: Last element should be 1.0 (newest observation)
    assert abs(weights[-1] - 1.0) < 1e-10


def test_time_decay_extreme_values():
    """Test with extreme uniqueness values."""
    # Test with very large values
    avg_uniqueness_large = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
    weights_large = time_decay(avg_uniqueness_large, 0.2)

    assert len(weights_large) == 3
    assert abs(weights_large[-1] - 1.0) < 1e-10  # CORRECTED: newest = 1.0

    # Test with very small values
    avg_uniqueness_small = np.array([1e-6, 2e-6, 3e-6], dtype=np.float64)
    weights_small = time_decay(avg_uniqueness_small, 0.8)

    assert len(weights_small) == 3
    assert abs(weights_small[-1] - 1.0) < 1e-10  # CORRECTED: newest = 1.0


def test_time_decay_numerical_stability():
    """Test numerical stability with challenging inputs."""
    # Test with inputs that could cause numerical issues
    avg_uniqueness = np.array([1e-15, 1e-14, 1e-13, 1.0], dtype=np.float64)

    weights = time_decay(avg_uniqueness, 0.1)

    # Should handle without numerical issues
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert abs(weights[-1] - 1.0) < 1e-10  # CORRECTED: newest = 1.0


# ---------- Return Types Tests ----------

def test_time_decay_return_types():
    """Test correct return types and dtypes."""
    avg_uniqueness = np.array([0.2, 0.4, 0.6], dtype=np.float64)

    weights = time_decay(avg_uniqueness, 0.5)

    # Verify return type and dtype
    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    assert weights.shape == avg_uniqueness.shape


if __name__ == "__main__":
    pytest.main([__file__])