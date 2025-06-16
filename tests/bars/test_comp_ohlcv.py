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


if __name__ == "__main__":
    pytest.main([__file__])