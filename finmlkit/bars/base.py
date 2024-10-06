import numpy as np
from numba import njit
from numpy.typing import NDArray
from numba import prange
from .utils import comp_trade_side_vector
from finmlkit.utils.log import logger


@njit(nopython=True, nogil=True, parallel=True)
def comp_bar_ohlcv(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    bar_open_indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float32], NDArray[np.float64]]:
    """
    Build the candlestick bars from raw trades data based in bar open indices.

    Parameters
    ----------
    prices : np.array(np.float64)
        Raw sorted **trades data** prices.
    volumes : np.array(np.float64)
        Raw sorted **trades data** volumes.
    bar_open_indices : np.array(np.int64)
        Bar open indices calculated with a bar indexer function.

    Returns
    -------
    tuple (np.array(np.float64), np.array(np.float64), np.array(np.float64), np.array(np.float64), np.array(np.float32), np.array(np.float64))
        A tuple containing the bars' open, high, low, close prices, volumes and vwap. OHLCV + VWAP.
    """
    n_bars = len(bar_open_indices) - 1  # The last open index determines the last bars' close
    bar_high = np.zeros(n_bars, dtype=np.float64)
    bar_low = np.zeros(n_bars, dtype=np.float64)
    bar_open = np.zeros(n_bars, dtype=np.float64)
    bar_close = np.zeros(n_bars, dtype=np.float64)
    bar_volume = np.zeros(n_bars, dtype=np.float64)
    bar_vwap = np.zeros(n_bars, dtype=np.float32)
    bar_dollar = np.zeros(n_bars, dtype=np.float64)

    for i in prange(n_bars):
        start = bar_open_indices[i]
        end = bar_open_indices[i + 1]

        # Handle empty bars
        if start == end:
            last_price = prices[start - 1]
            bar_open[i] = last_price
            bar_close[i] = last_price
            bar_high[i] = last_price
            bar_low[i] = last_price
            bar_volume[i] = 0.0
            bar_dollar[i] = 0.0
            bar_vwap[i] = 0.0
            continue

        # Initialize variables for this bar
        high_price = prices[start]
        low_price = prices[start]
        total_volume = 0.0
        total_dollar = 0.0

        # Iterate over trades in the current bar
        for j in range(start, end):
            price = prices[j]
            volume = volumes[j]

            if price > high_price:
                high_price = price
            if price < low_price:
                low_price = price

            total_volume += volume
            total_dollar += price * volume

        bar_open[i] = prices[start]
        bar_close[i] = prices[end - 1]
        bar_high[i] = high_price
        bar_low[i] = low_price
        bar_volume[i] = total_volume
        bar_dollar[i] = total_dollar
        bar_vwap[i] = total_dollar / total_volume if total_volume > 0 else 0.0

    return bar_open, bar_high, bar_low, bar_close, bar_volume, bar_vwap



@njit(nopython=True, nogil=True, parallel=True)
def comp_bar_directional_features(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    bar_open_indices: NDArray[np.int64]
) -> tuple[
    NDArray[np.int64], NDArray[np.int64],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float64], NDArray[np.float64],
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32], NDArray[np.float32]
]:
    """
    Compute directional bar features like sell/buy volumes etc... for each bar.

    Parameters
    ----------
    prices : np.array(np.float64)
        Raw trades prices.
    volumes : np.array(np.float64)
        Raw trades volumes.
    bar_open_indices : np.array(np.int64)
        Bar open indices in the raw trades timestamps.

    Returns
    -------
    tuple
        A tuple containing the bars' ticks_buy, ticks_sell, volume_buy, volume_sell, dollars_buy, dollars_sell, max_spread, cum_volumes_min, cum_volumes_max, cum_dollars_min, cum_dollars_max.
    """
    n_bars = len(bar_open_indices) - 1
    ticks_buy = np.zeros(n_bars, dtype=np.int64)
    ticks_sell = np.zeros(n_bars, dtype=np.int64)
    volume_buy = np.zeros(n_bars, dtype=np.float32)
    volume_sell = np.zeros(n_bars, dtype=np.float32)
    dollars_buy = np.zeros(n_bars, dtype=np.float32)
    dollars_sell = np.zeros(n_bars, dtype=np.float32)
    max_spread = np.zeros(n_bars, dtype=np.float32)

    # Initialize cumulative min and max arrays with appropriate values
    cum_ticks_min = np.full(n_bars, 1e9, dtype=np.int64)            # inf (large value)
    cum_ticks_max = np.full(n_bars, -1e9, dtype=np.int64)                   # -inf (small value)
    cum_volumes_min = np.full(n_bars, 1e9, dtype=np.float32)
    cum_volumes_max = np.full(n_bars, -1e9, dtype=np.float32)
    cum_dollars_min = np.full(n_bars, 1e9, dtype=np.float32)
    cum_dollars_max = np.full(n_bars, -1e9, dtype=np.float32)

    # 1.) Obtaining direction information
    # Calculate trade side for each trade
    trade_sides = comp_trade_side_vector(prices)

    # 2.) Compute the bar directional features
    for i in prange(n_bars):
        start = bar_open_indices[i]
        end = bar_open_indices[i + 1]

        current_tics_buy = 0
        current_tics_sell = 0
        current_volume_buy = 0.0
        current_volume_sell = 0.0
        current_dollars_buy = 0.0
        current_dollars_sell = 0.0
        current_cum_ticks = 0.0
        current_cum_volumes = 0.0
        current_cum_dollars = 0.0
        current_max_spread = 0.0

        # Initialize previous tick sign for spread calculation
        if end > start:
            prev_tick_sign = trade_sides[start]
        else:
            prev_tick_sign = 0  # Default value if no trades in bar

        for j in range(start, end):
            current_tick_sign = trade_sides[j]

            # Calculate the spread between buy and sell prices
            if current_tick_sign != prev_tick_sign:
                spread = prices[j] - prices[j - 1]
                if spread > current_max_spread:
                    current_max_spread = spread
            prev_tick_sign = current_tick_sign

            if current_tick_sign == 1:
                current_tics_buy += 1
                current_volume_buy += volumes[j]
                current_dollars_buy += prices[j] * volumes[j]
                # Cumulative values
                current_cum_ticks += 1
                current_cum_volumes += volumes[j]
                current_cum_dollars += prices[j] * volumes[j]
            elif current_tick_sign == -1:
                current_tics_sell += 1
                current_volume_sell += volumes[j]
                current_dollars_sell += prices[j] * volumes[j]
                # Cumulative values
                current_cum_ticks -= 1
                current_cum_volumes -= volumes[j]
                current_cum_dollars -= prices[j] * volumes[j]

            # Update the cumulative min and max values
            cum_ticks_max[i] = max(cum_ticks_max[i], current_cum_ticks)
            cum_ticks_min[i] = min(cum_ticks_min[i], current_cum_ticks)
            cum_volumes_max[i] = max(cum_volumes_max[i], current_cum_volumes)
            cum_volumes_min[i] = min(cum_volumes_min[i], current_cum_volumes)
            cum_dollars_max[i] = max(cum_dollars_max[i], current_cum_dollars)
            cum_dollars_min[i] = min(cum_dollars_min[i], current_cum_dollars)

        ticks_buy[i] = current_tics_buy
        ticks_sell[i] = current_tics_sell
        volume_buy[i] = current_volume_buy
        volume_sell[i] = current_volume_sell
        dollars_buy[i] = current_dollars_buy
        dollars_sell[i] = current_dollars_sell
        max_spread[i] = current_max_spread

    return (
        ticks_buy, ticks_sell,
        volume_buy, volume_sell,
        dollars_buy, dollars_sell,
        max_spread,
        cum_ticks_min, cum_ticks_max,
        cum_volumes_min, cum_volumes_max,
        cum_dollars_min, cum_dollars_max
    )


@njit(nopython=True, nogil=True)
def comp_bar_footprints(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    bar_open_indices: NDArray[np.int64]
):
    pass
