import numpy as np
import os
import pytest
from numba.typed import List as NumbaList
from finmlkit.bar.base import comp_bar_footprints


def test_comp_bar_footprints_basic():
    """
    Test the comp_bar_footprints function with basic functionality.
    """
    # Sample data with proper trade sides
    prices = np.array([100.0, 100.5, 101.0, 100.5, 100.0], dtype=np.float64)
    amounts = np.array([1.0, 2.0, 1.5, 1.0, 2.0], dtype=np.float64)
    bar_close_indices = np.array([0, 3, 4], dtype=np.int64)  # Bar boundaries (inclusive end indices)
    trade_sides = np.array([0, 1, 1, -1, -1], dtype=np.int8)  # dummy, buy, buy, sell, sell
    price_tick_size = 0.5
    bar_lows = np.array([100.0, 100.0], dtype=np.float64)
    bar_highs = np.array([101.0, 100.0], dtype=np.float64)  # Fixed: bar 1 only has 100.0
    imbalance_factor = 1.5

    # Call the function
    (
        price_levels, buy_volumes, sell_volumes, buy_ticks, sell_ticks,
        buy_imbalances, sell_imbalances, buy_imbalances_sum, sell_imbalances_sum,
        cot_price_levels, imb_max_run_signed, vp_skew, vp_gini
    ) = comp_bar_footprints(
        prices, amounts, bar_close_indices, trade_sides, price_tick_size,
        bar_lows, bar_highs, imbalance_factor
    )

    # Basic length assertions
    n_bars = len(bar_close_indices) - 1
    assert len(price_levels) == n_bars
    assert len(buy_volumes) == n_bars
    assert len(sell_volumes) == n_bars
    assert len(buy_ticks) == n_bars
    assert len(sell_ticks) == n_bars
    assert len(buy_imbalances) == n_bars
    assert len(sell_imbalances) == n_bars
    assert len(buy_imbalances_sum) == n_bars
    assert len(sell_imbalances_sum) == n_bars
    assert len(cot_price_levels) == n_bars
    assert len(imb_max_run_signed) == n_bars
    assert len(vp_skew) == n_bars
    assert len(vp_gini) == n_bars

    # Bar 0: trades indices 1, 2, 3 (prices: 100.5 buy, 101.0 buy, 100.5 sell)
    expected_price_levels_bar0 = np.array([200, 201, 202], dtype=np.int32)  # 100.0, 100.5, 101.0 in ticks
    expected_buy_volumes_bar0 = np.array([0.0, 2.0, 1.5], dtype=np.float32)
    expected_sell_volumes_bar0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    expected_buy_ticks_bar0 = np.array([0, 1, 1], dtype=np.int32)
    expected_sell_ticks_bar0 = np.array([0, 1, 0], dtype=np.int32)

    np.testing.assert_array_equal(price_levels[0], expected_price_levels_bar0)
    np.testing.assert_array_almost_equal(buy_volumes[0], expected_buy_volumes_bar0)
    np.testing.assert_array_almost_equal(sell_volumes[0], expected_sell_volumes_bar0)
    np.testing.assert_array_equal(buy_ticks[0], expected_buy_ticks_bar0)
    np.testing.assert_array_equal(sell_ticks[0], expected_sell_ticks_bar0)

    # Bar 1: trade index 4 (price: 100.0 sell)
    expected_price_levels_bar1 = np.array([200], dtype=np.int32)  # 100.0 in ticks
    expected_buy_volumes_bar1 = np.array([0.0], dtype=np.float32)
    expected_sell_volumes_bar1 = np.array([2.0], dtype=np.float32)
    expected_buy_ticks_bar1 = np.array([0], dtype=np.int32)
    expected_sell_ticks_bar1 = np.array([1], dtype=np.int32)

    np.testing.assert_array_equal(price_levels[1], expected_price_levels_bar1)
    np.testing.assert_array_almost_equal(buy_volumes[1], expected_buy_volumes_bar1)
    np.testing.assert_array_almost_equal(sell_volumes[1], expected_sell_volumes_bar1)
    np.testing.assert_array_equal(buy_ticks[1], expected_buy_ticks_bar1)
    np.testing.assert_array_equal(sell_ticks[1], expected_sell_ticks_bar1)

    # Check COT price levels (highest total volume)
    assert cot_price_levels[0] == 201  # Price level 100.5 has highest volume (3.0 total)
    assert cot_price_levels[1] == 200  # Price level 100.0 has only volume (2.0 total)

    # Check imbalance arrays are boolean type
    assert buy_imbalances[0].dtype == np.bool_
    assert sell_imbalances[0].dtype == np.bool_
    assert buy_imbalances[1].dtype == np.bool_
    assert sell_imbalances[1].dtype == np.bool_

    # Check imbalance sums match
    assert buy_imbalances_sum[0] == np.sum(buy_imbalances[0])
    assert sell_imbalances_sum[0] == np.sum(sell_imbalances[0])
    assert buy_imbalances_sum[1] == np.sum(buy_imbalances[1])
    assert sell_imbalances_sum[1] == np.sum(sell_imbalances[1])


def test_comp_bar_footprints_single_price_level():
    """
    Test with all trades at the same price level.
    """
    prices = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    amounts = np.array([1.0, 2.0, 1.5, 3.0], dtype=np.float64)
    bar_close_indices = np.array([0, 3], dtype=np.int64)
    trade_sides = np.array([0, 1, -1, 1], dtype=np.int8)
    price_tick_size = 0.5
    bar_lows = np.array([100.0], dtype=np.float64)
    bar_highs = np.array([100.0], dtype=np.float64)
    imbalance_factor = 2.0

    result = comp_bar_footprints(
        prices, amounts, bar_close_indices, trade_sides, price_tick_size,
        bar_lows, bar_highs, imbalance_factor
    )

    price_levels, buy_volumes, sell_volumes, buy_ticks, sell_ticks = result[:5]

    # Should have one price level
    expected_price_levels = np.array([200], dtype=np.int32)
    expected_buy_volumes = np.array([5.0], dtype=np.float32)  # 2.0 + 3.0
    expected_sell_volumes = np.array([1.5], dtype=np.float32)
    expected_buy_ticks = np.array([2], dtype=np.int32)
    expected_sell_ticks = np.array([1], dtype=np.int32)

    np.testing.assert_array_equal(price_levels[0], expected_price_levels)
    np.testing.assert_array_almost_equal(buy_volumes[0], expected_buy_volumes)
    np.testing.assert_array_almost_equal(sell_volumes[0], expected_sell_volumes)
    np.testing.assert_array_equal(buy_ticks[0], expected_buy_ticks)
    np.testing.assert_array_equal(sell_ticks[0], expected_sell_ticks)


def test_comp_bar_footprints_multiple_bars():
    """
    Test with multiple bars and varying price ranges.
    """
    prices = np.array([100.0, 100.5, 101.0, 101.5, 99.5, 99.0], dtype=np.float64)
    amounts = np.array([1.0, 2.0, 1.5, 2.5, 1.0, 3.0], dtype=np.float64)
    bar_close_indices = np.array([0, 2, 5], dtype=np.int64)  # Two bars
    trade_sides = np.array([0, 1, 1, -1, -1, 1], dtype=np.int8)
    price_tick_size = 0.5
    bar_lows = np.array([100.0, 99.0], dtype=np.float64)
    bar_highs = np.array([101.0, 101.5], dtype=np.float64)
    imbalance_factor = 1.5

    result = comp_bar_footprints(
        prices, amounts, bar_close_indices, trade_sides, price_tick_size,
        bar_lows, bar_highs, imbalance_factor
    )

    price_levels, buy_volumes, sell_volumes = result[:3]
    cot_price_levels, imb_max_run_signed, vp_skew, vp_gini = result[9:13]

    # Bar 0: trades 1, 2 (100.5 buy, 101.0 buy)
    expected_price_levels_bar0 = np.array([200, 201, 202], dtype=np.int32)
    expected_buy_volumes_bar0 = np.array([0.0, 2.0, 1.5], dtype=np.float32)
    expected_sell_volumes_bar0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    np.testing.assert_array_equal(price_levels[0], expected_price_levels_bar0)
    np.testing.assert_array_almost_equal(buy_volumes[0], expected_buy_volumes_bar0)
    np.testing.assert_array_almost_equal(sell_volumes[0], expected_sell_volumes_bar0)

    # Bar 1: trades 3, 4, 5 (101.5 sell, 99.5 sell, 99.0 buy)
    expected_price_levels_bar1 = np.array([198, 199, 200, 201, 202, 203], dtype=np.int32)
    expected_buy_volumes_bar1 = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    expected_sell_volumes_bar1 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 2.5], dtype=np.float32)

    np.testing.assert_array_equal(price_levels[1], expected_price_levels_bar1)
    np.testing.assert_array_almost_equal(buy_volumes[1], expected_buy_volumes_bar1)
    np.testing.assert_array_almost_equal(sell_volumes[1], expected_sell_volumes_bar1)

    # Check types of additional metrics
    assert isinstance(cot_price_levels[0], (int, np.integer))
    assert isinstance(cot_price_levels[1], (int, np.integer))
    assert isinstance(imb_max_run_signed[0], (int, np.integer))
    assert isinstance(imb_max_run_signed[1], (int, np.integer))
    assert isinstance(vp_skew[0], (float, np.floating))
    assert isinstance(vp_skew[1], (float, np.floating))
    assert isinstance(vp_gini[0], (float, np.floating))
    assert isinstance(vp_gini[1], (float, np.floating))


def test_comp_bar_footprints_edge_cases():
    """
    Test edge cases including zero volumes and mixed scenarios.
    """
    # Test with all neutral sides (should result in zero volumes)
    prices = np.array([100.0, 100.5], dtype=np.float64)
    amounts = np.array([1.0, 2.0], dtype=np.float64)
    bar_close_indices = np.array([0, 1], dtype=np.int64)
    trade_sides = np.array([0, 0], dtype=np.int8)  # All neutral
    price_tick_size = 0.5
    bar_lows = np.array([100.0], dtype=np.float64)
    bar_highs = np.array([100.5], dtype=np.float64)
    imbalance_factor = 1.5

    result = comp_bar_footprints(
        prices, amounts, bar_close_indices, trade_sides, price_tick_size,
        bar_lows, bar_highs, imbalance_factor
    )

    buy_volumes, sell_volumes = result[1:3]

    # All volumes should be zero for neutral trades
    np.testing.assert_array_almost_equal(buy_volumes[0], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(sell_volumes[0], np.array([0.0, 0.0], dtype=np.float32))


def test_comp_bar_footprints_small_tick_size():
    """
    Test with smaller tick size for precision.
    """
    prices = np.array([100.00, 100.01, 100.02], dtype=np.float64)
    amounts = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    bar_close_indices = np.array([0, 2], dtype=np.int64)
    trade_sides = np.array([0, 1, -1], dtype=np.int8)
    price_tick_size = 0.01
    bar_lows = np.array([100.00], dtype=np.float64)
    bar_highs = np.array([100.02], dtype=np.float64)
    imbalance_factor = 1.5

    result = comp_bar_footprints(
        prices, amounts, bar_close_indices, trade_sides, price_tick_size,
        bar_lows, bar_highs, imbalance_factor
    )

    price_levels, buy_volumes, sell_volumes = result[:3]

    # Price levels should be 10000, 10001, 10002 (100.00, 100.01, 100.02 in 0.01 ticks)
    expected_price_levels = np.array([10000, 10001, 10002], dtype=np.int32)
    expected_buy_volumes = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    expected_sell_volumes = np.array([0.0, 0.0, 1.5], dtype=np.float32)

    np.testing.assert_array_equal(price_levels[0], expected_price_levels)
    np.testing.assert_array_almost_equal(buy_volumes[0], expected_buy_volumes)
    np.testing.assert_array_almost_equal(sell_volumes[0], expected_sell_volumes)


def test_comp_bar_footprints_empty_bar():
    """
    Test with empty bars (no trades).
    """
    prices = np.array([], dtype=np.float64)
    amounts = np.array([], dtype=np.float64)
    bar_close_indices = np.array([0], dtype=np.int64)  # Single boundary, no bars
    trade_sides = np.array([], dtype=np.int8)
    price_tick_size = 0.5
    bar_lows = np.array([], dtype=np.float64)
    bar_highs = np.array([], dtype=np.float64)
    imbalance_factor = 1.5

    result = comp_bar_footprints(
        prices, amounts, bar_close_indices, trade_sides, price_tick_size,
        bar_lows, bar_highs, imbalance_factor
    )

    price_levels = result[0]
    buy_imbalances_sum, sell_imbalances_sum = result[7:9]
    cot_price_levels, imb_max_run_signed, vp_skew, vp_gini = result[9:13]

    # All should be empty
    assert len(price_levels) == 0
    assert len(buy_imbalances_sum) == 0
    assert len(sell_imbalances_sum) == 0
    assert len(cot_price_levels) == 0
    assert len(imb_max_run_signed) == 0
    assert len(vp_skew) == 0
    assert len(vp_gini) == 0


def test_comp_bar_footprints_large_imbalance():
    """
    Test with large imbalances across multiple price levels.
    """
    prices = np.array([100.0, 100.5, 100.0, 100.5], dtype=np.float64)
    amounts = np.array([1.0, 10.0, 1.0, 1.0], dtype=np.float64)
    bar_close_indices = np.array([0, 3], dtype=np.int64)
    trade_sides = np.array([0, 1, -1, -1], dtype=np.int8)
    price_tick_size = 0.5
    bar_lows = np.array([100.0], dtype=np.float64)
    bar_highs = np.array([100.5], dtype=np.float64)
    imbalance_factor = 2.0

    result = comp_bar_footprints(
        prices, amounts, bar_close_indices, trade_sides, price_tick_size,
        bar_lows, bar_highs, imbalance_factor
    )

    buy_volumes, sell_volumes = result[1:3]
    buy_imbalances, sell_imbalances = result[5:7]

    # Expected volumes at two price levels [200, 201] = [100.0, 100.5]
    expected_buy_volumes = np.array([0.0, 10.0], dtype=np.float32)  # Level 200: 0, Level 201: 10.0
    expected_sell_volumes = np.array([1.0, 1.0], dtype=np.float32)  # Level 200: 1.0, Level 201: 1.0

    np.testing.assert_array_almost_equal(buy_volumes[0], expected_buy_volumes)
    np.testing.assert_array_almost_equal(sell_volumes[0], expected_sell_volumes)

    # Imbalance calculations:
    # sell_imbalances[0] = sell_volumes[0] > buy_volumes[1] * 2.0 → 1.0 > 10.0 * 2.0 = False
    # buy_imbalances[1] = buy_volumes[1] > sell_volumes[0] * 2.0 → 10.0 > 1.0 * 2.0 = True

    expected_buy_imbalances = np.array([False, True], dtype=np.bool_)
    expected_sell_imbalances = np.array([False, False], dtype=np.bool_)

    np.testing.assert_array_equal(buy_imbalances[0], expected_buy_imbalances)
    np.testing.assert_array_equal(sell_imbalances[0], expected_sell_imbalances)


if __name__ == "__main__":
    pytest.main([__file__])