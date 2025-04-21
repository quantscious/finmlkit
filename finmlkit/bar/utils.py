from typing import Literal
from numba import njit
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from math import gcd


@njit(nogil=True, fastmath=True)
def comp_trade_side(price: float, prev_price: float, prev_tick: int) -> Literal[-1, 1]:
    """
    Implements the tick rule defined on page 29 of "Advances in Financial Machine Learning". It classifies the
    trade as an aggressive buy or sell based on the direction of price change.

    Parameters
    ----------
    price : float
        The current price.
    prev_price : float
        The previous price.
    prev_tick : int
        The previous tick rule.

    Returns
    -------
    int
        1 for an upward change (buy), -1 for a downward change (sell)
    """
    epsilon = 1e-12  # Small value to avoid floating point error
    dp = price - prev_price

    return np.sign(dp) if abs(dp) > epsilon else prev_tick


@njit(nogil=True)
def comp_trade_side_vector(prices: NDArray[np.float64]) -> NDArray[np.int8]:
    """
    Calculate the trade side for a sequence of raw trades prices.

    Parameters
    ----------
    prices : np.array(np.float64)
        The raw trades data prices sequence.

    Returns
    -------
    np.ndarray(np.int8)
        The sequence of trade sides (1 for buy, -1 for sell).
    """
    n = len(prices)
    trade_sides = np.zeros(n, dtype=np.int8)

    prev_tick_sign = 0
    prev_price = np.float64(prices[0])  # Explicit cast to avoid warnings

    for i in range(1, n):
        price = np.float64(prices[i])  # Explicit cast to avoid warnings
        prev_tick_sign = comp_trade_side(price, prev_price, prev_tick_sign)
        trade_sides[i] = prev_tick_sign
        prev_price = price

    return trade_sides


@njit(nogil=True)
def comp_price_tick_size(prices: NDArray[np.float64]) -> float:
    """
    Calculate the asset's tick size from a sequence of raw trades prices.

    Parameters
    ----------
    prices : np.array(np.float64)

    Returns
    -------
    tick_size: float

    """
    if len(prices) == 0:
        raise ValueError("Empty prices array")

    # Limit sample size and round to mitigate floating-point errors
    price_sample = np.round(prices[:min(10000, len(prices))], decimals=12)
    unique_prices = np.unique(price_sample)

    if len(unique_prices) <= 1:
        return 0.0

    diffs = np.diff(unique_prices)
    scale = 10.0 ** (-np.floor(np.log10(np.min(diffs[diffs > 0]))))
    int_px = np.round(unique_prices * scale).astype(np.int64)

    # Calculate greatest common divisor
    # tick_int = np.gcd.reduce(np.diff(int_px))  # numba does not support it
    tick_int = 0
    for diff_int in np.diff(int_px):
        if diff_int > 0:
            tick_int = diff_int if tick_int == 0 else gcd(tick_int, diff_int)
            if tick_int == 1:
                break

    return tick_int / scale


@njit(nogil=True)
def comp_price_tick_size_old(prices: NDArray[np.float64]) -> float:
    """
    Compute the price tick size from raw trades prices data.

    Parameters
    ----------
    prices : np.array(np.float64)
        Raw trades prices.

    Returns
    -------
    float
        The price tick size. Returns 0.0 if not determinable from the input prices.

    """
    # Select first 10000 trades to calculate the price tick size
    n_samples = len(prices)
    price_sample = prices[:min(10000, n_samples)]

    # raise value error if prices is empty
    if len(price_sample) == 0:
        raise ValueError("Empty prices array")

    # Round the prices to mitigate floating-point errors
    price_sample = np.round(price_sample, decimals=12)

    # Get the sorted unique prices
    unique_prices = np.unique(price_sample)
    if len(unique_prices) <= 1:
        # No variation in prices; tick size is zero
        return 0.0

    # Calculate the median price tick size
    price_tick_size = np.median(np.diff(unique_prices))

    if price_tick_size == 0.0:
        # Avoid computing log10(0)
        return 0.0

    # Determine the exponent for adaptive rounding
    exponent = np.floor(np.log10(abs(price_tick_size)))
    # Specify the desired number of significant digits
    desired_significant_digits = 2
    # Calculate the number of decimal places to round to
    ndigits = int(desired_significant_digits - 1 - exponent)
    # Round the tick size adaptively based on the exponent
    rounded_tick_size = round(price_tick_size, ndigits)

    return rounded_tick_size


def footprint_to_dataframe(bar_timestamps, price_levels, buy_volumes, sell_volumes, buy_ticks, sell_ticks,
                           buy_imbalance, sell_imbalance, price_tick):
    """
    Convert footprint data to a pandas DataFrame format.
    The output filters out NaN price levels and sorts them in descending order by price level and ascending order by bar datetime.

    Args:
        bar_timestamps: bar timestamps (1d numpy array)
        price_levels: price levels (list of 1d numpy arrays) in ascending order
        buy_volumes: aligned buy volumes (list of 1d numpy arrays)
        sell_volumes: aligned sell volumes (list of 1d numpy arrays)
        buy_ticks: aligned buy ticks (list of 1d numpy arrays)
        sell_ticks: aligned sell ticks (list of 1d numpy arrays)
        buy_imbalance: aligned buy imbalance (list of 1d numpy arrays)
        sell_imbalance: aligned sell imbalance (list of 1d numpy arrays)
        price_tick: price tick size (float)

    Returns: footprint dataframe
    """
    # Convert bar_timestamps to datetime
    bar_dt = pd.to_datetime(bar_timestamps)

    # Create lists to hold the data for the DataFrame
    bar_dt_repeated = []
    bar_ids_repeated = []
    price_levels_flat = []
    buy_volumes_flat = []
    sell_volumes_flat = []
    buy_ticks_flat = []
    sell_ticks_flat = []
    buy_imbalance_flat = []
    sell_imbalance_flat = []

    # Process each bar's data
    for bar_idx, bar_time in enumerate(bar_dt):
        n_levels = len(price_levels[bar_idx])

        bar_dt_repeated.extend([bar_time] * n_levels)
        bar_ids_repeated.extend([bar_idx] * n_levels)
        price_levels_flat.extend(price_levels[bar_idx])
        buy_volumes_flat.extend(buy_volumes[bar_idx])
        sell_volumes_flat.extend(sell_volumes[bar_idx])
        buy_ticks_flat.extend(buy_ticks[bar_idx])
        sell_ticks_flat.extend(sell_ticks[bar_idx])
        buy_imbalance_flat.extend(buy_imbalance[bar_idx])
        sell_imbalance_flat.extend(sell_imbalance[bar_idx])

    # Convert lists to numpy arrays
    bar_dt_repeated = np.array(bar_dt_repeated)
    bar_ids_repeated = np.array(bar_ids_repeated)
    price_levels_flat = np.array(price_levels_flat)
    buy_volumes_flat = np.array(buy_volumes_flat)
    sell_volumes_flat = np.array(sell_volumes_flat)
    buy_ticks_flat = np.array(buy_ticks_flat)
    sell_ticks_flat = np.array(sell_ticks_flat)
    buy_imbalance_flat = np.array(buy_imbalance_flat)
    sell_imbalance_flat = np.array(sell_imbalance_flat)

    data = {
        'price_level': price_levels_flat,
        'sell_ticks': sell_ticks_flat,
        'buy_ticks': buy_ticks_flat,
        'sell_volume': sell_volumes_flat,
        'buy_volume': buy_volumes_flat,
        'sell_imbalance': sell_imbalance_flat,
        'buy_imbalance': buy_imbalance_flat
    }

    # Create MultiIndex from the valid row indices, bar timestamps, and price levels
    multi_index = pd.MultiIndex.from_arrays([
        bar_ids_repeated,
        bar_dt_repeated,
    ], names=['bar_idx', 'bar_datetime_idx'])

    # Create the DataFrame
    df = pd.DataFrame(data, index=multi_index)

    # Convert price levels to actual price unit
    df['price_level'] = df['price_level'] * price_tick

    # Descending order by price level and ascending order by bar datetime
    df = df.sort_values(by=['bar_datetime_idx', 'price_level'], ascending=[True, False])

    return df
