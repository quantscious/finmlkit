import pytest
import numpy as np
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.bars.base import comp_bar_directional_features

def test_comp_bar_directional_features_simple_case():
    prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    volumes = np.array([10.0, 15.0, 20.0], dtype=np.float64)
    bar_open_indices = np.array([0, 3], dtype=np.int64)  # Single bar

    # Expected trade sides:
    # trade_sides[0] = 0 (initial value)
    # trade_sides[1] = 1 (101 > 100)
    # trade_sides[2] = 1 (102 > 101)

    # Manually compute expected outputs
    ticks_buy_expected = np.array([2], dtype=np.int64)
    ticks_sell_expected = np.array([0], dtype=np.int64)
    volume_buy_expected = np.array([15.0 + 20.0], dtype=np.float32)
    volume_sell_expected = np.array([0.0], dtype=np.float32)
    dollars_buy_expected = np.array([101.0 * 15.0 + 102.0 * 20.0], dtype=np.float32)
    dollars_sell_expected = np.array([0.0], dtype=np.float32)
    max_spread_expected = np.array([1.0], dtype=np.float32)  # 101.0 - 100.0

    cum_ticks_min_expected = np.array([1.0])  # After first buy
    cum_ticks_max_expected = np.array([2.0])  # After second buy
    cum_volumes_min_expected = np.array([15.0])  # After first buy
    cum_volumes_max_expected = np.array([35.0])  # Sum of both buys
    cum_dollars_min_expected = np.array([101.0 * 15.0])  # After first buy
    cum_dollars_max_expected = np.array([101.0 * 15.0 + 102.0 * 20.0])  # Sum of both buys

    # Call the function
    result = comp_bar_directional_features(prices, volumes, bar_open_indices)
    (ticks_buy, ticks_sell,
     volume_buy, volume_sell,
     dollars_buy, dollars_sell,
     max_spread,
     cum_ticks_min, cum_ticks_max,
     cum_volumes_min, cum_volumes_max,
     cum_dollars_min, cum_dollars_max) = result

    # Assertions
    np.testing.assert_array_equal(ticks_buy, ticks_buy_expected)
    np.testing.assert_array_equal(ticks_sell, ticks_sell_expected)
    np.testing.assert_array_almost_equal(volume_buy, volume_buy_expected)
    np.testing.assert_array_almost_equal(volume_sell, volume_sell_expected)
    np.testing.assert_array_almost_equal(dollars_buy, dollars_buy_expected)
    np.testing.assert_array_almost_equal(dollars_sell, dollars_sell_expected)
    np.testing.assert_array_almost_equal(max_spread, max_spread_expected)
    np.testing.assert_array_almost_equal(cum_ticks_min, cum_ticks_min_expected)
    np.testing.assert_array_almost_equal(cum_ticks_max, cum_ticks_max_expected)
    np.testing.assert_array_almost_equal(cum_volumes_min, cum_volumes_min_expected)
    np.testing.assert_array_almost_equal(cum_volumes_max, cum_volumes_max_expected)
    np.testing.assert_array_almost_equal(cum_dollars_min, cum_dollars_min_expected)
    np.testing.assert_array_almost_equal(cum_dollars_max, cum_dollars_max_expected)


def test_comp_bar_directional_features_mixed_trades():
    prices = np.array([100.0, 99.0, 100.0, 101.0], dtype=np.float64)
    volumes = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    bar_open_indices = np.array([0, 4], dtype=np.int64)  # Single bar

    # Expected trade sides:
    # trade_sides[0] = 0
    # trade_sides[1] = -1 (99 < 100)
    # trade_sides[2] = 1 (100 > 99)
    # trade_sides[3] = 1 (101 > 100)

    # Manually compute expected outputs
    ticks_buy_expected = np.array([2], dtype=np.int64)   # trades at indices 2 and 3
    ticks_sell_expected = np.array([1], dtype=np.int64)  # trade at index 1
    volume_buy_expected = np.array([30.0 + 40.0], dtype=np.float32)
    volume_sell_expected = np.array([20.0], dtype=np.float32)
    dollars_buy_expected = np.array([100.0 * 30.0 + 101.0 * 40.0], dtype=np.float32)
    dollars_sell_expected = np.array([99.0 * 20.0], dtype=np.float32)
    max_spread_expected = np.array([2.0], dtype=np.float32)  # 101.0 - 99.0

    cum_ticks_min_expected = np.array([-1.0])
    cum_ticks_max_expected = np.array([1.0])
    cum_volumes_min_expected = np.array([-20.0])
    cum_volumes_max_expected = np.array([70.0])
    cum_dollars_min_expected = np.array([-1980.0])  # -99.0 * 20.0
    cum_dollars_max_expected = np.array([100.0 * 30.0 + 101.0 * 40.0])

    # Call the function
    result = comp_bar_directional_features(prices, volumes, bar_open_indices)
    (ticks_buy, ticks_sell,
     volume_buy, volume_sell,
     dollars_buy, dollars_sell,
     max_spread,
     cum_ticks_min, cum_ticks_max,
     cum_volumes_min, cum_volumes_max,
     cum_dollars_min, cum_dollars_max) = result

    # Assertions
    np.testing.assert_array_equal(ticks_buy, ticks_buy_expected)
    np.testing.assert_array_equal(ticks_sell, ticks_sell_expected)
    np.testing.assert_array_almost_equal(volume_buy, volume_buy_expected)
    np.testing.assert_array_almost_equal(volume_sell, volume_sell_expected)
    np.testing.assert_array_almost_equal(dollars_buy, dollars_buy_expected)
    np.testing.assert_array_almost_equal(dollars_sell, dollars_sell_expected)
    np.testing.assert_array_almost_equal(max_spread, max_spread_expected)
    np.testing.assert_array_almost_equal(cum_ticks_min, cum_ticks_min_expected)
    np.testing.assert_array_almost_equal(cum_ticks_max, cum_ticks_max_expected)
    np.testing.assert_array_almost_equal(cum_volumes_min, cum_volumes_min_expected)
    np.testing.assert_array_almost_equal(cum_volumes_max, cum_volumes_max_expected)
    np.testing.assert_array_almost_equal(cum_dollars_min, cum_dollars_min_expected)
    np.testing.assert_array_almost_equal(cum_dollars_max, cum_dollars_max_expected)


def test_comp_bar_directional_features_empty_bar():
    prices = np.array([100.0, 101.0], dtype=np.float64)
    volumes = np.array([10.0, 15.0], dtype=np.float64)
    bar_open_indices = np.array([0, 1, 1, 2], dtype=np.int64)  # Second bar is empty

    # Call the function
    result = comp_bar_directional_features(prices, volumes, bar_open_indices)
    (ticks_buy, ticks_sell,
     volume_buy, volume_sell,
     dollars_buy, dollars_sell,
     max_spread,
     cum_ticks_min, cum_ticks_max,
     cum_volumes_min, cum_volumes_max,
     cum_dollars_min, cum_dollars_max) = result

    # Expected outputs
    # First bar has trades from index 0 to 1
    # Second bar is empty (start == end)
    # Third bar has trades from index 1 to 2

    ticks_buy_expected = np.array([0, 0, 1], dtype=np.int64)
    ticks_sell_expected = np.array([0, 0, 0], dtype=np.int64)
    volume_buy_expected = np.array([0.0, 0.0, 15.0], dtype=np.float32)
    volume_sell_expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dollars_buy_expected = np.array([0.0, 0.0, 101.0 * 15.0], dtype=np.float32)
    dollars_sell_expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    max_spread_expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cum_ticks_min_expected = np.array([1e9, 1e9, 1.0])  # First two bars have no cumulative ticks
    cum_ticks_max_expected = np.array([-1e9, -1e9, 1.0])
    cum_volumes_min_expected = np.array([1e9, 1e9, 15.0])
    cum_volumes_max_expected = np.array([-1e9, -1e9, 15.0])
    cum_dollars_min_expected = np.array([1e9, 1e9, 101.0 * 15.0])
    cum_dollars_max_expected = np.array([-1e9, -1e9, 101.0 * 15.0])

    # Assertions
    np.testing.assert_array_equal(ticks_buy, ticks_buy_expected)
    np.testing.assert_array_equal(ticks_sell, ticks_sell_expected)
    np.testing.assert_array_almost_equal(volume_buy, volume_buy_expected)
    np.testing.assert_array_almost_equal(volume_sell, volume_sell_expected)
    np.testing.assert_array_almost_equal(dollars_buy, dollars_buy_expected)
    np.testing.assert_array_almost_equal(dollars_sell, dollars_sell_expected)
    np.testing.assert_array_almost_equal(max_spread, max_spread_expected)
    np.testing.assert_array_equal(cum_ticks_min, cum_ticks_min_expected)
    np.testing.assert_array_equal(cum_ticks_max, cum_ticks_max_expected)
    np.testing.assert_array_almost_equal(cum_volumes_min, cum_volumes_min_expected)
    np.testing.assert_array_almost_equal(cum_volumes_max, cum_volumes_max_expected)
    np.testing.assert_array_almost_equal(cum_dollars_min, cum_dollars_min_expected)
    np.testing.assert_array_almost_equal(cum_dollars_max, cum_dollars_max_expected)


def test_comp_bar_directional_features_all_sells():
    prices = np.array([100.0, 99.0, 98.0], dtype=np.float64)
    volumes = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    bar_open_indices = np.array([0, 3], dtype=np.int64)  # Single bar

    # Expected trade sides:
    # trade_sides[0] = 0
    # trade_sides[1] = -1 (99.0 < 100.0)
    # trade_sides[2] = -1 (98.0 < 99.0)

    # Manually compute expected outputs
    ticks_buy_expected = np.array([0], dtype=np.int64)
    ticks_sell_expected = np.array([2], dtype=np.int64)
    volume_buy_expected = np.array([0.0], dtype=np.float32)
    volume_sell_expected = np.array([20.0 + 30.0], dtype=np.float32)
    dollars_buy_expected = np.array([0.0], dtype=np.float32)
    dollars_sell_expected = np.array([99.0 * 20.0 + 98.0 * 30.0], dtype=np.float32)
    max_spread_expected = np.array([0.0], dtype=np.float32)  # No spread as all trades are sells

    cum_ticks_min_expected = np.array([-2.0])
    cum_ticks_max_expected = np.array([-1.0])
    cum_volumes_min_expected = np.array([-50.0])
    cum_volumes_max_expected = np.array([-20.0])
    cum_dollars_min_expected = np.array([-99.0 * 20.0 - 98.0 * 30.0])
    cum_dollars_max_expected = np.array([-99.0 * 20.0])

    # Call the function
    result = comp_bar_directional_features(prices, volumes, bar_open_indices)
    (ticks_buy, ticks_sell,
     volume_buy, volume_sell,
     dollars_buy, dollars_sell,
     max_spread,
     cum_ticks_min, cum_ticks_max,
     cum_volumes_min, cum_volumes_max,
     cum_dollars_min, cum_dollars_max) = result

    # Assertions
    np.testing.assert_array_equal(ticks_buy, ticks_buy_expected)
    np.testing.assert_array_equal(ticks_sell, ticks_sell_expected)
    np.testing.assert_array_almost_equal(volume_buy, volume_buy_expected)
    np.testing.assert_array_almost_equal(volume_sell, volume_sell_expected)
    np.testing.assert_array_almost_equal(dollars_buy, dollars_buy_expected)
    np.testing.assert_array_almost_equal(dollars_sell, dollars_sell_expected)
    np.testing.assert_array_almost_equal(max_spread, max_spread_expected)
    np.testing.assert_array_almost_equal(cum_ticks_min, cum_ticks_min_expected)
    np.testing.assert_array_almost_equal(cum_ticks_max, cum_ticks_max_expected)
    np.testing.assert_array_almost_equal(cum_volumes_min, cum_volumes_min_expected)
    np.testing.assert_array_almost_equal(cum_volumes_max, cum_volumes_max_expected)
    np.testing.assert_array_almost_equal(cum_dollars_min, cum_dollars_min_expected)
    np.testing.assert_array_almost_equal(cum_dollars_max, cum_dollars_max_expected)


def test_comp_bar_directional_features_constant_price():
    prices = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    volumes = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    bar_open_indices = np.array([0, 3], dtype=np.int64)  # Single bar

    # Expected trade sides:
    # trade_sides[0] = 0
    # trade_sides[1] = 0 (price unchanged)
    # trade_sides[2] = 0 (price unchanged)

    # Manually compute expected outputs
    ticks_buy_expected = np.array([0], dtype=np.int64)
    ticks_sell_expected = np.array([0], dtype=np.int64)
    volume_buy_expected = np.array([0.0], dtype=np.float32)
    volume_sell_expected = np.array([0.0], dtype=np.float32)
    dollars_buy_expected = np.array([0.0], dtype=np.float32)
    dollars_sell_expected = np.array([0.0], dtype=np.float32)
    max_spread_expected = np.array([0.0], dtype=np.float32)

    # Since trade sides are zero, cumulative values remain at zero
    cum_ticks_min_expected = np.array([1e9])    # No updates, remains at initial value
    cum_ticks_max_expected = np.array([-1e9])
    cum_volumes_min_expected = np.array([1e9])
    cum_volumes_max_expected = np.array([-1e9])
    cum_dollars_min_expected = np.array([1e9])
    cum_dollars_max_expected = np.array([-1e9])

    # Call the function
    result = comp_bar_directional_features(prices, volumes, bar_open_indices)
    (ticks_buy, ticks_sell,
     volume_buy, volume_sell,
     dollars_buy, dollars_sell,
     max_spread,
     cum_ticks_min, cum_ticks_max,
     cum_volumes_min, cum_volumes_max,
     cum_dollars_min, cum_dollars_max) = result

    # Assertions
    np.testing.assert_array_equal(ticks_buy, ticks_buy_expected)
    np.testing.assert_array_equal(ticks_sell, ticks_sell_expected)
    np.testing.assert_array_almost_equal(volume_buy, volume_buy_expected)
    np.testing.assert_array_almost_equal(volume_sell, volume_sell_expected)
    np.testing.assert_array_almost_equal(dollars_buy, dollars_buy_expected)
    np.testing.assert_array_almost_equal(dollars_sell, dollars_sell_expected)
    np.testing.assert_array_almost_equal(max_spread, max_spread_expected)
    np.testing.assert_array_equal(cum_ticks_min, cum_ticks_min_expected)
    np.testing.assert_array_equal(cum_ticks_max, cum_ticks_max_expected)
    np.testing.assert_array_equal(cum_volumes_min, cum_volumes_min_expected)
    np.testing.assert_array_equal(cum_volumes_max, cum_volumes_max_expected)
    np.testing.assert_array_equal(cum_dollars_min, cum_dollars_min_expected)
    np.testing.assert_array_equal(cum_dollars_max, cum_dollars_max_expected)