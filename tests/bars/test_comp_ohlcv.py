import numpy as np
import pytest
import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.bar.base import comp_bar_ohlcv

def test_comp_ohlcv():
    # Sample input data
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0, 150.0, 100.0, 50.0, 25.0], dtype=np.float64)
    # Bar close indices: first bar closes at index 0, second at index 3, third at index 6
    bar_close_indices = np.array([0, 3, 5], dtype=np.int64)  # Defines two bars

    # Expected outputs
    expected_bar_open = np.array([11.0, 14.0], dtype=np.float64)  # First trade after close
    expected_bar_high = np.array([13.0, 15.0], dtype=np.float64)
    expected_bar_low = np.array([11.0, 14.0], dtype=np.float64)
    expected_bar_close = np.array([13.0, 15.0], dtype=np.float64)
    expected_bar_volume = np.array([450.0, 75.0], dtype=np.float64)
    expected_bar_vwap = np.array([(11*200+12*150+13*100)/450., (14*50+15*25)/75.], dtype=np.float32)
    expected_bar_trades = np.array([3, 2], dtype=np.int64)  # Number of trades in each bar
    expected_bar_median_trade_size = np.array([np.median([200., 150., 100.]), np.median([50., 25.])], dtype=np.float64)  # Median trade size in each bar

    # Call the function
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, bar_trades, bar_median_trade_size = comp_bar_ohlcv(
        prices, volumes, bar_close_indices
    )

    # Assertions to verify the output
    np.testing.assert_array_equal(bar_open, expected_bar_open)
    np.testing.assert_array_equal(bar_high, expected_bar_high)
    np.testing.assert_array_equal(bar_low, expected_bar_low)
    np.testing.assert_array_equal(bar_close, expected_bar_close)
    np.testing.assert_array_equal(bar_volume, expected_bar_volume)
    np.testing.assert_allclose(bar_vwap, expected_bar_vwap, rtol=1e-6)
    np.testing.assert_array_equal(bar_trades, expected_bar_trades)
    np.testing.assert_array_equal(bar_median_trade_size, expected_bar_median_trade_size)

    print("All tests passed successfully!")

def test_single_trade_per_bar():
    # Each bar has exactly one trade
    prices = np.array([10.0, 12.0, 14.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    bar_close_indices = np.array([0, 1, 2], dtype=np.int64)  # Three bars with closing indices

    expected_bar_open = np.array([12.0, 14.0], dtype=np.float64)  # Last one is out of bounds
    expected_bar_high = np.array([12.0, 14.0], dtype=np.float64)
    expected_bar_low = np.array([12.0, 14.0], dtype=np.float64)
    expected_bar_close = np.array([12.0, 14.0], dtype=np.float64)
    expected_bar_volume = np.array([200.0, 300.0], dtype=np.float64)
    expected_bar_vwap = np.array([12.0, 14.0], dtype=np.float32)
    expected_bar_trades = np.array([1, 1], dtype=np.int64)  # One trade per bar, except last
    expected_bar_median_trade_size = np.array([200.0, 300.0], dtype=np.float64)

    # Call the function
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, bar_trades, bar_median_trade_size = comp_bar_ohlcv(
        prices, volumes, bar_close_indices
    )

    # Assertions
    np.testing.assert_array_equal(bar_open, expected_bar_open)
    np.testing.assert_array_equal(bar_high, expected_bar_high)
    np.testing.assert_array_equal(bar_low, expected_bar_low)
    np.testing.assert_array_equal(bar_close, expected_bar_close)
    np.testing.assert_array_equal(bar_volume, expected_bar_volume)
    np.testing.assert_allclose(bar_vwap, expected_bar_vwap, rtol=1e-6)
    np.testing.assert_array_equal(bar_trades, expected_bar_trades)
    np.testing.assert_array_equal(bar_median_trade_size, expected_bar_median_trade_size)

def test_zero_volume():
    # Test with zero volume but with correct bar close indices
    prices = np.array([10.0, 12.0], dtype=np.float64)
    volumes = np.array([100.0, 0.0], dtype=np.float64)  # Second trade has zero volume
    bar_close_indices = np.array([0, 1], dtype=np.int64)  # One bar closing at index 2

    # Expected outputs
    expected_bar_open = np.array([12.0], dtype=np.float64)  # First trade after close (index 1)
    expected_bar_high = np.array([12.0], dtype=np.float64)
    expected_bar_low = np.array([12.0], dtype=np.float64)
    expected_bar_close = np.array([12.0], dtype=np.float64)
    expected_bar_volume = np.array([0.0], dtype=np.float64)
    expected_bar_vwap = np.array([0.0], dtype=np.float32)  # VWAP should be 0 due to 0 volume
    expected_bar_trades = np.array([1], dtype=np.int64)  # One trade in the bar
    expected_bar_median_trade_size = np.array([0.0], dtype=np.float64)  # Median trade size is 0

    # Call the function
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, bar_trades, bar_median_trade_size = comp_bar_ohlcv(
        prices, volumes, bar_close_indices
    )

    # Assertions
    np.testing.assert_array_equal(bar_open, expected_bar_open)
    np.testing.assert_array_equal(bar_high, expected_bar_high)
    np.testing.assert_array_equal(bar_low, expected_bar_low)
    np.testing.assert_array_equal(bar_close, expected_bar_close)
    np.testing.assert_array_equal(bar_volume, expected_bar_volume)
    np.testing.assert_array_equal(bar_vwap, expected_bar_vwap)
    np.testing.assert_array_equal(bar_trades, expected_bar_trades)
    np.testing.assert_array_equal(bar_median_trade_size, expected_bar_median_trade_size)

def test_empty_bars():
    """
    In case of time bar, there can be empty bar with no trades.
    """
    # Gap between bar open indices, indicating an empty bar
    prices = np.array([10.0, 12.0, 14.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    bar_close_indices = np.array([0, 1, 1, 2], dtype=np.int64)  # One empty bar in the middle

    # Expected outputs
    expected_bar_open = np.array([12.0, 12.0, 14.0], dtype=np.float64)
    expected_bar_high = np.array([12.0, 12.0, 14.0], dtype=np.float64)
    expected_bar_low = np.array([12.0, 12.0, 14.0], dtype=np.float64)
    expected_bar_close = np.array([12.0, 12.0, 14.0], dtype=np.float64)
    expected_bar_volume = np.array([200.0, 0.0, 300.0], dtype=np.float64)
    expected_bar_vwap = np.array([12.0, 0.0, 14.], dtype=np.float32)  # The middle bar is empty
    expected_bar_trades = np.array([1, 0, 1], dtype=np.int64)  # 1 trade, 0 trades (empty bar), 2 trades
    expected_bar_median_trade_size = np.array([200.0, 0.0, 300.], dtype=np.float64)  # Median sizes

    # Call the function
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, bar_trades, bar_median_trade_size = comp_bar_ohlcv(
        prices, volumes, bar_close_indices
    )

    # Assertions
    np.testing.assert_array_equal(bar_open, expected_bar_open)
    np.testing.assert_array_equal(bar_high, expected_bar_high)
    np.testing.assert_array_equal(bar_low, expected_bar_low)
    np.testing.assert_array_equal(bar_close, expected_bar_close)
    np.testing.assert_array_equal(bar_volume, expected_bar_volume)
    np.testing.assert_allclose(bar_vwap, expected_bar_vwap, rtol=1e-6)
    np.testing.assert_array_equal(bar_trades, expected_bar_trades)
    np.testing.assert_array_equal(bar_median_trade_size, expected_bar_median_trade_size)


def test_comp_ohlcv_input_validation():
    """Test error conditions and input validation."""

    # Test mismatched array lengths
    prices = np.array([10.0, 11.0], dtype=np.float64)
    volumes = np.array([100.0], dtype=np.float64)  # Different length
    bar_close_indices = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="Prices and volumes arrays must have the same length"):
        comp_bar_ohlcv(prices, volumes, bar_close_indices)

    # Test insufficient bar_close_indices
    prices = np.array([10.0], dtype=np.float64)
    volumes = np.array([100.0], dtype=np.float64)
    bar_close_indices = np.array([0], dtype=np.int64)  # Only one element

    with pytest.raises(ValueError, match="Bar close indices must contain at least two elements"):
        comp_bar_ohlcv(prices, volumes, bar_close_indices)


def test_comp_ohlcv_empty_data():
    """Test with completely empty data."""
    prices = np.array([], dtype=np.float64)
    volumes = np.array([], dtype=np.float64)
    bar_close_indices = np.array([0], dtype=np.int64)  # No bars possible

    # This should handle gracefully or raise appropriate error
    try:
        result = comp_bar_ohlcv(prices, volumes, bar_close_indices)
        # If it succeeds, verify empty results
        assert all(len(arr) == 0 for arr in result)
    except (ValueError, IndexError):
        # Expected behavior for empty data
        pass


def test_comp_ohlcv_ascending_prices():
    """Test with strictly ascending prices."""
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)
    volumes = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    bar_close_indices = np.array([0, 2, 5], dtype=np.int64)

    bar_open, bar_high, bar_low, bar_close, *_ = comp_bar_ohlcv(prices, volumes, bar_close_indices)

    # First bar: trades 1,2 (11.0, 12.0)
    assert bar_open[0] == 11.0
    assert bar_high[0] == 12.0
    assert bar_low[0] == 11.0
    assert bar_close[0] == 12.0

    # Second bar: trades 3,4,5 (13.0, 14.0, 15.0)
    assert bar_open[1] == 13.0
    assert bar_high[1] == 15.0
    assert bar_low[1] == 13.0
    assert bar_close[1] == 15.0


def test_comp_ohlcv_descending_prices():
    """Test with strictly descending prices."""
    prices = np.array([15.0, 14.0, 13.0, 12.0, 11.0, 10.0], dtype=np.float64)
    volumes = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    bar_close_indices = np.array([0, 2, 5], dtype=np.int64)

    bar_open, bar_high, bar_low, bar_close, *_ = comp_bar_ohlcv(prices, volumes, bar_close_indices)

    # First bar: trades 1,2 (14.0, 13.0)
    assert bar_open[0] == 14.0
    assert bar_high[0] == 14.0
    assert bar_low[0] == 13.0
    assert bar_close[0] == 13.0


def test_comp_ohlcv_volatile_prices():
    """Test with highly volatile price movements."""
    prices = np.array([100.0, 150.0, 50.0, 200.0, 25.0, 175.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0, 300.0, 150.0, 250.0, 100.0], dtype=np.float64)
    bar_close_indices = np.array([0, 3, 5], dtype=np.int64)

    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, *_ = comp_bar_ohlcv(prices, volumes,
                                                                                      bar_close_indices)

    # First bar: trades 1,2,3 (150, 50, 200)
    assert bar_high[0] == 200.0
    assert bar_low[0] == 50.0
    assert bar_open[0] == 150.0
    assert bar_close[0] == 200.0

    # Verify VWAP calculation for volatile prices
    expected_vwap_0 = (150 * 200 + 50 * 300 + 200 * 150) / (200 + 300 + 150)
    np.testing.assert_allclose([bar_vwap[0]], [expected_vwap_0], rtol=1e-6)


def test_comp_ohlcv_extreme_volumes():
    """Test with extreme volume values."""
    prices = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    volumes = np.array([0.001, 1000000.0, 0.0, 999999.999], dtype=np.float64)
    bar_close_indices = np.array([0, 1, 3], dtype=np.int64)

    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, bar_trades, bar_median = comp_bar_ohlcv(prices,
                                                                                                          volumes,
                                                                                                          bar_close_indices)

    # First bar: trade 1 (volume 1000000.0)
    assert bar_volume[0] == 1000000.0
    assert bar_vwap[0] == 11.0  # Single trade VWAP

    # Second bar: trades 2,3 (volumes 0.0, 999999.999)
    assert bar_volume[1] == 999999.999
    expected_vwap_1 = (12 * 0.0 + 13 * 999999.999) / 999999.999
    np.testing.assert_allclose([bar_vwap[1]], [expected_vwap_1], rtol=1e-6)


def test_comp_ohlcv_all_zero_volumes_bar():
    """Test bar with all zero volumes."""
    prices = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    volumes = np.array([100.0, 0.0, 0.0, 200.0], dtype=np.float64)
    bar_close_indices = np.array([0, 2, 3], dtype=np.int64)

    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, *_ = comp_bar_ohlcv(prices, volumes,
                                                                                      bar_close_indices)

    # First bar: trades 1,2 (volumes 0.0, 0.0)
    assert bar_volume[0] == 0.0
    assert bar_vwap[0] == 0.0  # Should be 0 when total volume is 0


def test_comp_ohlcv_median_edge_cases():
    """Test median calculation with various trade count scenarios."""

    # Even number of trades
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64)
    volumes = np.array([100.0, 50.0, 200.0, 150.0, 75.0], dtype=np.float64)
    bar_close_indices = np.array([0, 3], dtype=np.int64)  # 3 trades in bar

    *_, bar_median = comp_bar_ohlcv(prices, volumes, bar_close_indices)

    # Bar: trades 1,2,3 (volumes 50, 200, 150) → median = 150
    assert bar_median[0] == 150.0

    # Single trade median
    prices_single = np.array([10.0, 11.0], dtype=np.float64)
    volumes_single = np.array([100.0, 250.0], dtype=np.float64)
    bar_close_indices_single = np.array([0, 1], dtype=np.int64)

    *_, bar_median_single = comp_bar_ohlcv(prices_single, volumes_single, bar_close_indices_single)
    assert bar_median_single[0] == 250.0  # Single value median


def test_comp_ohlcv_identical_trade_sizes():
    """Test with identical trade sizes."""
    prices = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    volumes = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    bar_close_indices = np.array([0, 2], dtype=np.int64)

    *_, bar_median = comp_bar_ohlcv(prices, volumes, bar_close_indices)
    assert bar_median[0] == 100.0  # All identical → median = 100


def test_comp_ohlcv_single_bar():
    """Test with only one bar."""
    prices = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0, 150.0], dtype=np.float64)
    bar_close_indices = np.array([0, 2], dtype=np.int64)  # Single bar

    result = comp_bar_ohlcv(prices, volumes, bar_close_indices)

    # Should return arrays of length 1
    for arr in result:
        assert len(arr) == 1

    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, bar_trades, bar_median = result
    assert bar_open[0] == 11.0
    assert bar_close[0] == 12.0
    assert bar_trades[0] == 2


def test_comp_ohlcv_consecutive_empty_bars():
    """Test multiple consecutive empty bars."""
    prices = np.array([10.0, 15.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0], dtype=np.float64)
    bar_close_indices = np.array([0, 0, 0, 1], dtype=np.int64)  # Two empty bars, then one with data

    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, bar_trades, bar_median = comp_bar_ohlcv(prices,
                                                                                                          volumes,
                                                                                                          bar_close_indices)

    # First two bars should be empty (use price at end index)
    assert bar_volume[0] == 0.0
    assert bar_volume[1] == 0.0
    assert bar_trades[0] == 0
    assert bar_trades[1] == 0

    # Third bar should have data
    assert bar_volume[2] == 200.0
    assert bar_trades[2] == 1


def test_comp_ohlcv_floating_point_precision():
    """Test floating point precision edge cases."""
    # Very small price differences
    prices = np.array([100.0, 100.0000001, 99.9999999, 100.0000002], dtype=np.float64)
    volumes = np.array([1000.0, 2000.0, 1500.0, 2500.0], dtype=np.float64)
    bar_close_indices = np.array([0, 3], dtype=np.int64)

    bar_open, bar_high, bar_low, bar_close, *_ = comp_bar_ohlcv(prices, volumes, bar_close_indices)

    # Verify precision is maintained
    assert bar_high[0] == 100.0000002
    assert bar_low[0] == 99.9999999

    # Very large numbers
    prices_large = np.array([1e10, 1e10 + 1, 1e10 - 1], dtype=np.float64)
    volumes_large = np.array([1e6, 2e6, 1.5e6], dtype=np.float64)
    bar_close_indices_large = np.array([0, 2], dtype=np.int64)

    result = comp_bar_ohlcv(prices_large, volumes_large, bar_close_indices_large)
    # Should handle large numbers without overflow
    assert all(np.isfinite(arr).all() for arr in result)


def test_comp_ohlcv_data_types():
    """Verify correct output data types."""
    prices = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0, 150.0], dtype=np.float64)
    bar_close_indices = np.array([0, 2], dtype=np.int64)

    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap, bar_trades, bar_median = comp_bar_ohlcv(prices,
                                                                                                          volumes,
                                                                                                          bar_close_indices)

    # Verify correct dtypes
    assert bar_open.dtype == np.float64
    assert bar_high.dtype == np.float64
    assert bar_low.dtype == np.float64
    assert bar_close.dtype == np.float64
    assert bar_volume.dtype == np.float32
    assert bar_vwap.dtype == np.float64
    assert bar_trades.dtype == np.int64
    assert bar_median.dtype == np.float64


def test_comp_ohlcv_large_dataset():
    """Test with larger dataset for performance validation."""
    n_trades = 10000
    n_bars = 100

    np.random.seed(42)  # For reproducible results
    prices = np.cumsum(np.random.randn(n_trades) * 0.01) + 100.0
    volumes = np.random.exponential(100.0, n_trades)

    # Create bar boundaries
    bar_close_indices = np.linspace(0, n_trades - 1, n_bars + 1, dtype=np.int64)

    result = comp_bar_ohlcv(prices, volumes, bar_close_indices)

    # Verify all outputs have correct length
    for arr in result:
        assert len(arr) == n_bars

    # Verify no NaN or infinite values
    for arr in result:
        assert np.isfinite(arr).all()

if __name__ == "__main__":
    pytest.main([__file__])