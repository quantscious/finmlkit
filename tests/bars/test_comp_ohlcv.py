import numpy as np
import pytest
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.bars.build import comp_bar_ohlcv

def test_comp_ohlcv():
    # Sample input data
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0, 150.0, 100.0, 50.0, 25.0], dtype=np.float64)
    bar_open_indices = np.array([0, 3, 6], dtype=np.int64)  # Defines two bars

    # Expected outputs
    expected_bar_open = np.array([10.0, 13.0], dtype=np.float64)
    expected_bar_high = np.array([12.0, 15.0], dtype=np.float64)
    expected_bar_low = np.array([10.0, 13.0], dtype=np.float64)
    expected_bar_close = np.array([12.0, 15.0], dtype=np.float64)
    expected_bar_volume = np.array([450.0, 175.0], dtype=np.float64)
    expected_bar_vwap = np.array([11.111111, 13.571428], dtype=np.float32)

    # Call the function
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap = comp_bar_ohlcv(
        prices, volumes, bar_open_indices
    )

    # Assertions to verify the output
    np.testing.assert_array_equal(bar_open, expected_bar_open)
    np.testing.assert_array_equal(bar_high, expected_bar_high)
    np.testing.assert_array_equal(bar_low, expected_bar_low)
    np.testing.assert_array_equal(bar_close, expected_bar_close)
    np.testing.assert_array_equal(bar_volume, expected_bar_volume)
    np.testing.assert_allclose(bar_vwap, expected_bar_vwap, rtol=1e-6)

    print("All tests passed successfully!")

def test_single_trade_per_bar():
    # Each bar has exactly one trade
    prices = np.array([10.0, 12.0, 14.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    bar_open_indices = np.array([0, 1, 2, 3], dtype=np.int64)  # Three bars, each with one trade

    # Expected outputs: for a single trade, OHLCV should all be the trade itself
    expected_bar_open = prices
    expected_bar_high = prices
    expected_bar_low = prices
    expected_bar_close = prices
    expected_bar_volume = volumes
    expected_bar_vwap = prices.astype(np.float32)

    # Call the function
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap = comp_bar_ohlcv(
        prices, volumes, bar_open_indices
    )

    # Assertions
    np.testing.assert_array_equal(bar_open, expected_bar_open)
    np.testing.assert_array_equal(bar_high, expected_bar_high)
    np.testing.assert_array_equal(bar_low, expected_bar_low)
    np.testing.assert_array_equal(bar_close, expected_bar_close)
    np.testing.assert_array_equal(bar_volume, expected_bar_volume)
    np.testing.assert_allclose(bar_vwap, expected_bar_vwap, rtol=1e-6)

def test_zero_volume():
    # One trade with zero volume
    prices = np.array([10.0], dtype=np.float64)
    volumes = np.array([0.0], dtype=np.float64)  # Zero volume
    bar_open_indices = np.array([0, 1], dtype=np.int64)  # One bar

    # Expected outputs: VWAP should handle zero volume carefully (division by zero)
    expected_bar_open = np.array([10.0], dtype=np.float64)
    expected_bar_high = np.array([10.0], dtype=np.float64)
    expected_bar_low = np.array([10.0], dtype=np.float64)
    expected_bar_close = np.array([10.0], dtype=np.float64)
    expected_bar_volume = np.array([0.0], dtype=np.float64)
    expected_bar_vwap = np.array([0.0], dtype=np.float32)  # VWAP should be 0 due to 0 volume

    # Call the function
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap = comp_bar_ohlcv(
        prices, volumes, bar_open_indices
    )

    # Assertions
    np.testing.assert_array_equal(bar_open, expected_bar_open)
    np.testing.assert_array_equal(bar_high, expected_bar_high)
    np.testing.assert_array_equal(bar_low, expected_bar_low)
    np.testing.assert_array_equal(bar_close, expected_bar_close)
    np.testing.assert_array_equal(bar_volume, expected_bar_volume)
    np.testing.assert_array_equal(bar_vwap, expected_bar_vwap)

def test_empty_bars():
    """
    In case of time bars, there can be empty bars with no trades.
    Returns
    -------

    """
    # Gap between bar open indices, indicating an empty bar
    prices = np.array([10.0, 12.0, 14.0], dtype=np.float64)
    volumes = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    bar_open_indices = np.array([0, 1, 1, 3], dtype=np.int64)  # One empty bar in the middle

    # Expected outputs
    expected_bar_open = np.array([10.0, 10.0, 12.0], dtype=np.float64)
    expected_bar_high = np.array([10.0, 10.0, 14.0], dtype=np.float64)
    expected_bar_low = np.array([10.0, 10.0, 12.0], dtype=np.float64)
    expected_bar_close = np.array([10.0, 10.0, 14.0], dtype=np.float64)
    expected_bar_volume = np.array([100.0, 0.0, 500.0], dtype=np.float64)
    expected_bar_vwap = np.array([10.0, 0.0, 13.2], dtype=np.float32)  # The middle bar is empty

    # Call the function
    bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap = comp_bar_ohlcv(
        prices, volumes, bar_open_indices
    )

    # Assertions
    np.testing.assert_array_equal(bar_open, expected_bar_open)
    np.testing.assert_array_equal(bar_high, expected_bar_high)
    np.testing.assert_array_equal(bar_low, expected_bar_low)
    np.testing.assert_array_equal(bar_close, expected_bar_close)
    np.testing.assert_array_equal(bar_volume, expected_bar_volume)
    np.testing.assert_allclose(bar_vwap, expected_bar_vwap, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])