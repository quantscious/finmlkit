import pytest
import numpy as np
import pandas as pd
import os
from numba import njit
from numpy.typing import NDArray
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)
from finmlkit.bars.base import comp_footprint_features


@njit(nopython=True, nogil=True)
def comp_footprint_features_old(price_levels: NDArray[np.int32],
                            buy_volumes: NDArray[np.float32], sell_volumes: NDArray[np.float32],
                            imbalance_multiplier: float):
    """
    Calculate bar's footprint features: COT price level, buy imbalances, sell imbalances.
    Note that it is assumed footprint price levels are sorted in ascending order and increments by price_tick_size.

    :param price_levels: price levels of a bar (1D numpy array)
    :param buy_volumes: buy volumes of a bar (1D numpy array)
    :param sell_volumes: sell volumes of a bar (1D numpy array)
    :param imbalance_multiplier: imbalance multiplier

    Returns: buy imbalances (1D bool array), sell imbalances (1D bool array), COT price level (float)

    """
    n_levels = len(price_levels)
    buy_imbalances = np.zeros((n_levels,), dtype=np.bool_)
    sell_imbalances = np.zeros((n_levels,), dtype=np.bool_)
    highest_cot_vol = 0.
    highest_cot_idx = 0

    # Ascending Bid-Ask Table         Descending Bid-Ask Table (Conventional)
    # -----------------                 -----------------
    # | L | Sell| Buy |                 | L | Sell| Buy |
    # |---|-----|-----|                 |---|-----|-----|
    # | 0 |   0 |   1 |                 | 3 |  10 |   0 |
    # |---|-----|-----|                 |---|-----|-----|
    # | 1 |   2 |  56 |                 | 2 | 181 |  15 |
    # |---|-----|-----|                 |---|-----|-----|
    # | 2 | 181 |  15 |                 | 1 |   2 |  56 |
    # |---|-----|-----|                 |---|-----|-----|
    # | 3 |  10 |   0 |                 | 0 |   0 |   1 |
    # -----------------                 -----------------
    # In the conventional table the l-th sell (bid) level corresponds to the (l-1)-th buy (ask) level;
    # the l-th buy (ask) level corresponds to the (l+1)-th sell (bid) level
    #
    # In the ascending table it is reversed and the l-th sell (bid) level corresponds to the (l+1)-th buy (ask) level;
    # the l-th buy (ask) level corresponds to the (l-1)-th sell (bid) level

    for level in range(n_levels):
        # Find sell (bid) imbalances (ascending price levels)
        if level < n_levels - 1:
            sell_imbalances[level] = 1 if sell_volumes[level] > (buy_volumes[level + 1] * imbalance_multiplier) else 0

        # Find buy (ask) imbalances (ascending price levels)
        if level > 0:
            buy_imbalances[level] = 1 if buy_volumes[level] > (sell_volumes[level - 1] * imbalance_multiplier) else 0

        # Find Commitment of Traders (COT) price level
        sum_level_volume = buy_volumes[level] + sell_volumes[level]
        if sum_level_volume > highest_cot_vol:
            highest_cot_vol = sum_level_volume
            highest_cot_idx = level

    cot_price_level = price_levels[highest_cot_idx]

    return buy_imbalances, sell_imbalances, cot_price_level


# Test function for correct functioning
def test_comp_footprint_features_correctness():
    """
    Test the comp_footprint_features_old function for correct functioning.
    """
    price_levels = np.array([100, 101, 102, 103], dtype=np.int32)
    buy_volumes = np.array([10, 50, 20, 5], dtype=np.float32)
    sell_volumes = np.array([5, 25, 30, 10], dtype=np.float32)
    imbalance_multiplier = 1.5

    # Expected buy imbalances
    expected_buy_imbalances = np.array([False, True, False, False], dtype=np.bool_)

    # Expected sell imbalances
    expected_sell_imbalances = np.array([False, False, True, False], dtype=np.bool_)

    # Expected COT price level
    sum_volumes = buy_volumes + sell_volumes
    highest_cot_idx = np.argmax(sum_volumes)
    expected_cot_price_level = price_levels[highest_cot_idx]

    # Call the function
    buy_imbalances, sell_imbalances, cot_price_level = comp_footprint_features_old(
        price_levels, buy_volumes, sell_volumes, imbalance_multiplier
    )

    # Assertions
    np.testing.assert_array_equal(
        buy_imbalances, expected_buy_imbalances,
        err_msg="Buy imbalances do not match expected values"
    )
    np.testing.assert_array_equal(
        sell_imbalances, expected_sell_imbalances,
        err_msg="Sell imbalances do not match expected values"
    )
    assert cot_price_level == expected_cot_price_level, (
        f"Expected COT price level {expected_cot_price_level}, got {cot_price_level}"
    )


# Test function for equivalence
def test_comp_footprint_features_equivalence():
    """
    Test that comp_footprint_features_old and comp_footprint_features produce the same outputs.
    """
    # Define test cases
    test_cases = [
        # Each test case is a tuple of (price_levels, buy_volumes, sell_volumes, imbalance_multiplier)
        (
            np.array([100, 101, 102, 103], dtype=np.int32),
            np.array([10, 50, 20, 5], dtype=np.float32),
            np.array([5, 25, 30, 10], dtype=np.float32),
            1.5
        ),
        # Edge case: n_levels = 1
        (
            np.array([100], dtype=np.int32),
            np.array([10], dtype=np.float32),
            np.array([5], dtype=np.float32),
            1.5
        ),
        # Edge case: zero volumes
        (
            np.array([100, 101, 102], dtype=np.int32),
            np.array([0, 0, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            1.5
        ),
        # Random test case
        (
            np.array([99, 100, 101, 102, 103, 104], dtype=np.int32),
            np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
            np.array([60, 50, 40, 30, 20, 10], dtype=np.float32),
            2.0
        ),
        # High imbalance multiplier
        (
            np.array([100, 101, 102, 103], dtype=np.int32),
            np.array([10, 20, 30, 40], dtype=np.float32),
            np.array([40, 30, 20, 10], dtype=np.float32),
            10.0
        ),
        # Low imbalance multiplier
        (
            np.array([100, 101, 102, 103], dtype=np.int32),
            np.array([10, 20, 30, 40], dtype=np.float32),
            np.array([40, 30, 20, 10], dtype=np.float32),
            0.1
        ),
        # Negative volumes (should not happen but testing robustness)
        (
            np.array([100, 101, 102, 103], dtype=np.int32),
            np.array([-10, 20, -30, 40], dtype=np.float32),
            np.array([40, -30, 20, -10], dtype=np.float32),
            1.5
        ),
        # Large arrays
        (
            np.arange(1000, 2000, dtype=np.int32),
            np.random.rand(1000).astype(np.float32) * 1000,
            np.random.rand(1000).astype(np.float32) * 1000,
            1.5
        ),
    ]

    for i, (price_levels, buy_volumes, sell_volumes, imbalance_multiplier) in enumerate(test_cases):
        # Call both functions
        buy_imbalances_old, sell_imbalances_old, cot_price_level_old = comp_footprint_features_old(
            price_levels, buy_volumes, sell_volumes, imbalance_multiplier
        )
        buy_imbalances_new, sell_imbalances_new, cot_price_level_new = comp_footprint_features(
            price_levels, buy_volumes, sell_volumes, imbalance_multiplier
        )

        # Compare the outputs
        try:
            np.testing.assert_array_equal(
                buy_imbalances_old, buy_imbalances_new,
                err_msg=f"Buy imbalances do not match for test case {i}"
            )
            np.testing.assert_array_equal(
                sell_imbalances_old, sell_imbalances_new,
                err_msg=f"Sell imbalances do not match for test case {i}"
            )
            assert cot_price_level_old == cot_price_level_new, (
                f"COT price levels do not match for test case {i}: "
                f"{cot_price_level_old} vs {cot_price_level_new}"
            )
        except AssertionError as e:
            print(f"Test case {i} failed.")
            raise