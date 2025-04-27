import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.bar.logic import _adaptive_cusum_bar_indexer


@pytest.fixture
def sample_data():
    """Create sample timestamp and price data for testing."""
    # Create 1000 timestamps at 1-second intervals (in nanoseconds)
    timestamps = np.arange(1000) * 1_000_000_000

    # Create prices with some patterns
    prices = np.ones(1000, dtype=np.float64) * 100.0

    # Add a price trend and some volatility
    prices = prices + np.linspace(0, 10, 1000)
    prices += np.sin(np.linspace(0, 10, 1000)) * 2  # Add some waves

    return timestamps, prices


def test_basic_functionality(sample_data):
    """Test basic functionality with default parameters."""
    timestamps, prices = sample_data
    warmup_ticks = 100

    open_indices = _adaptive_cusum_bar_indexer(
        timestamps, prices, lambda_mult=2.0, half_life_sec=300.0,
        warmup_ticks=warmup_ticks, sigma_floor=1e-6
    )
    open_indices = np.array(open_indices, dtype=np.int64)
    open_timestamps = timestamps[open_indices]

    # Check that the function returns numpy arrays
    assert isinstance(open_timestamps, np.ndarray)
    assert isinstance(open_indices, np.ndarray)

    # Check that the arrays are of the same length
    assert len(open_timestamps) == len(open_indices)

    # First index should be the warmup_ticks index
    assert open_indices[0] == warmup_ticks

    # Check that indices are monotonically increasing
    assert np.all(np.diff(open_indices) > 0)

    # Check that timestamps correspond to indices
    assert_array_equal(open_timestamps, timestamps[open_indices])


def test_insufficient_ticks():
    """Test that an exception is raised when there are too few ticks."""
    timestamps = np.arange(50) * 1_000_000_000
    prices = np.ones(50, dtype=np.float64) * 100.0

    with pytest.raises(ValueError):
        _adaptive_cusum_bar_indexer(
            timestamps, prices, lambda_mult=2.0, half_life_sec=300.0,
            warmup_ticks=100, sigma_floor=1e-6
        )


def test_constant_prices():
    """Test with constant prices, which should result in minimal bar openings."""
    timestamps = np.arange(1000) * 1_000_000_000
    prices = np.ones(1000, dtype=np.float64) * 100.0
    warmup_ticks = 100

    open_timestamps, open_indices = _adaptive_cusum_bar_indexer(
        timestamps, prices, lambda_mult=2.0, half_life_sec=300.0,
        warmup_ticks=warmup_ticks, sigma_floor=1e-6
    )

    # Should only have the initial warmup index
    assert len(open_indices) == 1
    assert open_indices[0] == warmup_ticks


def test_high_volatility():
    """Test with highly volatile prices."""
    timestamps = np.arange(1000) * 1_000_000_000

    # Create a volatile price series with jumps
    prices = np.ones(1000, dtype=np.float64) * 100.0
    for i in range(1, 1000):
        if i % 50 == 0:
            prices[i] = prices[i - 1] * 1.05  # 5% jump

    warmup_ticks = 100

    open_timestamps, open_indices = _adaptive_cusum_bar_indexer(
        timestamps, prices, lambda_mult=2.0, half_life_sec=300.0,
        warmup_ticks=warmup_ticks, sigma_floor=1e-6
    )

    # Should have more bar openings due to jumps
    assert len(open_indices) > 1

    # Check that jumps tend to create new bars
    jump_indices = np.where(np.diff(prices) > 1.0)[0] + 1
    jump_indices = jump_indices[jump_indices > warmup_ticks]

    # Not all jumps will create bars due to adaptive threshold,
    # but at least some should
    assert any(index in open_indices for index in jump_indices)


def test_parameter_sensitivity():
    """Test sensitivity to different parameter values."""
    timestamps = np.arange(1000) * 1_000_000_000

    # Create a price series with a trend
    prices = 100.0 + np.linspace(0, 10, 1000) + np.random.normal(0, 0.1, 1000)
    warmup_ticks = 100

    # Test with different lambda multipliers
    high_lambda_results = _adaptive_cusum_bar_indexer(
        timestamps, prices, lambda_mult=4.0, half_life_sec=300.0,
        warmup_ticks=warmup_ticks, sigma_floor=1e-6
    )

    low_lambda_results = _adaptive_cusum_bar_indexer(
        timestamps, prices, lambda_mult=1.0, half_life_sec=300.0,
        warmup_ticks=warmup_ticks, sigma_floor=1e-6
    )

    # Lower lambda should produce more bars
    assert len(low_lambda_results[1]) >= len(high_lambda_results[1])


def test_stable_with_outliers():
    """Test stability with outliers in the price data."""
    timestamps = np.arange(1000) * 1_000_000_000
    prices = np.ones(1000, dtype=np.float64) * 100.0

    # Add a significant outlier
    prices[500] = 200.0  # 100% price spike

    warmup_ticks = 100

    open_timestamps, open_indices = _adaptive_cusum_bar_indexer(
        timestamps, prices, lambda_mult=2.0, half_life_sec=300.0,
        warmup_ticks=warmup_ticks, sigma_floor=1e-6
    )

    # The outlier should trigger a bar opening
    assert 500 in open_indices or 501 in open_indices