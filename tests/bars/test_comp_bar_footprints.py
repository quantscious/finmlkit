import numpy as np
import os

os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)
from numba.typed import List as NumbaList
from finmlkit.bars.base import comp_bar_footprints


def test_comp_bar_footprints():
    """
    Test the comp_bar_footprints function for correct functioning.
    """
    # Sample data
    prices = np.array([100.0, 100.5, 101.0, 100.5, 100.0], dtype=np.float64)
    amounts = np.array([1.0, 2.0, 1.5, 1.0, 2.0], dtype=np.float64)
    bar_open_indices = np.array([0, 3, 5], dtype=np.int64)  # Two bars: indices 0-3, 3-5
    price_tick_size = 0.5
    bar_lows = np.array([100.0, 100.0], dtype=np.float64)
    bar_highs = np.array([101.0, 100.5], dtype=np.float64)
    imbalance_factor = 1.5

    # Call the function
    (
        price_levels,
        buy_volumes, sell_volumes,
        buy_ticks, sell_ticks,
        buy_imbalances, sell_imbalances,
        buy_imbalances_sum, sell_imbalances_sum, cot_price_levels
    ) = comp_bar_footprints(
        prices, amounts, bar_open_indices, price_tick_size, bar_lows, bar_highs, imbalance_factor
    )

    # Assertions to check lengths
    n_bars = len(bar_open_indices) - 1
    assert len(price_levels) == n_bars, "Length of price_levels should match number of bars"
    assert len(buy_volumes) == n_bars, "Length of buy_volumes should match number of bars"
    assert len(sell_volumes) == n_bars, "Length of sell_volumes should match number of bars"
    assert len(buy_ticks) == n_bars, "Length of buy_ticks should match number of bars"
    assert len(sell_ticks) == n_bars, "Length of sell_ticks should match number of bars"
    assert len(buy_imbalances) == n_bars, "Length of buy_imbalances should match number of bars"
    assert len(sell_imbalances) == n_bars, "Length of sell_imbalances should match number of bars"
    assert len(buy_imbalances_sum) == n_bars, "Length of buy_imbalances_sum should match number of bars"
    assert len(sell_imbalances_sum) == n_bars, "Length of sell_imbalances_sum should match number of bars"
    assert len(cot_price_levels) == n_bars, "Length of cot_price_levels should match number of bars"

    # Detailed checks for bar 0
    # Expected price levels for bar 0 (price in ticks)
    expected_price_levels_bar0 = np.array([200, 201, 202], dtype=np.int32)  # Prices: 100.0, 100.5, 101.0
    np.testing.assert_array_equal(
        price_levels[0], expected_price_levels_bar0,
        err_msg="Price levels for bar 0 do not match expected values"
    )

    # Expected volumes and ticks for bar 0
    expected_buy_volumes_bar0 = np.array([0.0, 2.0, 1.5], dtype=np.float32)
    expected_sell_volumes_bar0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    expected_buy_ticks_bar0 = np.array([0, 1, 1], dtype=np.int32)
    expected_sell_ticks_bar0 = np.array([0, 0, 0], dtype=np.int32)

    np.testing.assert_array_almost_equal(
        buy_volumes[0], expected_buy_volumes_bar0,
        err_msg="Buy volumes for bar 0 do not match expected values"
    )
    np.testing.assert_array_almost_equal(
        sell_volumes[0], expected_sell_volumes_bar0,
        err_msg="Sell volumes for bar 0 do not match expected values"
    )
    np.testing.assert_array_equal(
        buy_ticks[0], expected_buy_ticks_bar0,
        err_msg="Buy ticks for bar 0 do not match expected values"
    )
    np.testing.assert_array_equal(
        sell_ticks[0], expected_sell_ticks_bar0,
        err_msg="Sell ticks for bar 0 do not match expected values"
    )

    # Check COT price level for bar 0
    expected_cot_price_level_bar0 = 201  # Corresponds to price level 100.5
    assert cot_price_levels[0] == expected_cot_price_level_bar0, (
        f"COT price level for bar 0 should be {expected_cot_price_level_bar0}"
    )

    # Detailed checks for bar 1
    expected_price_levels_bar1 = np.array([200, 201], dtype=np.int32)  # Prices: 100.0, 100.5
    np.testing.assert_array_equal(
        price_levels[1], expected_price_levels_bar1,
        err_msg="Price levels for bar 1 do not match expected values"
    )

    expected_buy_volumes_bar1 = np.array([0.0, 0.0], dtype=np.float32)
    expected_sell_volumes_bar1 = np.array([2.0, 1.0], dtype=np.float32)
    expected_buy_ticks_bar1 = np.array([0, 0], dtype=np.int32)
    expected_sell_ticks_bar1 = np.array([1, 1], dtype=np.int32)

    np.testing.assert_array_almost_equal(
        buy_volumes[1], expected_buy_volumes_bar1,
        err_msg="Buy volumes for bar 1 do not match expected values"
    )
    np.testing.assert_array_almost_equal(
        sell_volumes[1], expected_sell_volumes_bar1,
        err_msg="Sell volumes for bar 1 do not match expected values"
    )
    np.testing.assert_array_equal(
        buy_ticks[1], expected_buy_ticks_bar1,
        err_msg="Buy ticks for bar 1 do not match expected values"
    )
    np.testing.assert_array_equal(
        sell_ticks[1], expected_sell_ticks_bar1,
        err_msg="Sell ticks for bar 1 do not match expected values"
    )

    # Check COT price level for bar 1
    expected_cot_price_level_bar1 = 200  # Corresponds to price level 100.0
    assert cot_price_levels[1] == expected_cot_price_level_bar1, (
        f"COT price level for bar 1 should be {expected_cot_price_level_bar1}"
    )

    # Check imbalances types
    assert isinstance(buy_imbalances[0], np.ndarray), "Buy imbalances for bar 0 should be a numpy array"
    assert isinstance(sell_imbalances[0], np.ndarray), "Sell imbalances for bar 0 should be a numpy array"
    assert isinstance(buy_imbalances[1], np.ndarray), "Buy imbalances for bar 1 should be a numpy array"
    assert isinstance(sell_imbalances[1], np.ndarray), "Sell imbalances for bar 1 should be a numpy array"

    # Ensure the sums match
    assert buy_imbalances_sum[0] == np.sum(buy_imbalances[0]), "Buy imbalance sum for bar 0 does not match"
    assert sell_imbalances_sum[0] == np.sum(sell_imbalances[0]), "Sell imbalance sum for bar 0 does not match"
    assert buy_imbalances_sum[1] == np.sum(buy_imbalances[1]), "Buy imbalance sum for bar 1 does not match"
    assert sell_imbalances_sum[1] == np.sum(sell_imbalances[1]), "Sell imbalance sum for bar 1 does not match"

    # Edge case: test with empty data
    prices_empty = np.array([], dtype=np.float64)
    amounts_empty = np.array([], dtype=np.float64)
    bar_open_indices_empty = np.array([0], dtype=np.int64)
    bar_lows_empty = np.array([], dtype=np.float64)
    bar_highs_empty = np.array([], dtype=np.float64)

    (
        price_levels_empty,
        buy_volumes_empty, sell_volumes_empty,
        buy_ticks_empty, sell_ticks_empty,
        buy_imbalances_empty, sell_imbalances_empty,
        buy_imbalances_sum_empty, sell_imbalances_sum_empty, cot_price_levels_empty
    ) = comp_bar_footprints(
        prices_empty, amounts_empty, bar_open_indices_empty, price_tick_size,
        bar_lows_empty, bar_highs_empty, imbalance_factor
    )

    # Assertions for empty data
    assert len(price_levels_empty) == 0, "Price levels should be empty"
    assert len(buy_volumes_empty) == 0, "Buy volumes should be empty"
    assert len(sell_volumes_empty) == 0, "Sell volumes should be empty"
    assert len(buy_ticks_empty) == 0, "Buy ticks should be empty"
    assert len(sell_ticks_empty) == 0, "Sell ticks should be empty"
    assert len(buy_imbalances_empty) == 0, "Buy imbalances should be empty"
    assert len(sell_imbalances_empty) == 0, "Sell imbalances should be empty"
    assert len(buy_imbalances_sum_empty) == 0, "Buy imbalance sums should be empty"
    assert len(sell_imbalances_sum_empty) == 0, "Sell imbalance sums should be empty"
    assert len(cot_price_levels_empty) == 0, "COT price levels should be empty"