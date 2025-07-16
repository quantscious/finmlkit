"""
This module contains the logic for generating time, tick, volume, and dollar bar.
These functions return the open indices of the bar in the raw trades data.
"""
from typing import Tuple
import numpy as np
from numba import njit
from numba.typed import List as NumbaList
from numpy.typing import NDArray


@njit(nogil=True)
def _time_bar_indexer(
        timestamps: NDArray[np.int64],
        interval_seconds: float
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine the time bar open indices in the raw trades timestamp array.

    :param timestamps: Raw sorted trade timestamps in nanoseconds.
    :param interval_seconds: Length of the time bar in seconds.
    :returns: A tuple of:
        - bar_close_ts: Timestamps at which each bar closes.
        - bar_close_indices: Indices in the trade data corresponding to bar closings.

    .. note::
        The first bar is aligned to the ceiling of the first timestamp, ensuring consistent bar boundaries.
        Duplicate indices may occur if a bar interval contains no trades (empty bars).
    """
    bar_interval_ns = interval_seconds * 1e9

    # determine the first bar start time (closest integer multiple of bar_interval_ns)
    bar_start_ts = timestamps[0] // bar_interval_ns * bar_interval_ns

    # last tick timestamp
    last_ts = np.ceil(timestamps[-1] / bar_interval_ns) * bar_interval_ns

    # create the array of bar close timestamps
    bar_clock = np.arange(bar_start_ts, last_ts + bar_interval_ns + 1, bar_interval_ns, dtype=np.int64)

    # find the indices of the bar close timestamps in the raw trades timestamps
    bar_close_indices = (np.searchsorted(timestamps, bar_clock, side='right') - 1).astype(np.int64)

    # open times and raw trades samples:
    # |----1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|     -> bar_clock
    # ..  . . .... . .. . . ... .. . . ... ... ... ...   . .  .    .  .... .            .  ... ....   -> raw trades timestamps
    # ^         ^         ^        ^         ^       ^        ^            ^                    ^     -> close indices
    #                                                                      ^                          -> close indices (empty bar)
    # 0         6         12         19        26        32        35                   41,41   46    -> raw trades indices

    return bar_clock, bar_close_indices


@njit(nogil=True)
def _tick_bar_indexer(
        timestamps: NDArray[np.int64],
        threshold: int
) -> NumbaList:
    """
    Determine the tick bar open indices in the raw trades timestamp array.

    :param timestamps: Raw trade timestamps.
    :param threshold: The tick count threshold for opening a new bar.
    :returns: close_indices: Timestamps at which each tick bar opens.

    .. note::
        The first trade is always the start of a bar.
        A new bar is opened every time the tick count reaches the specified threshold.
    """
    n = len(timestamps)

    # Initialize a Numba typed list to store indices
    tick_bar_indices = NumbaList()
    tick_bar_indices.append(0)  # First tick is always marked

    cum_ticks = 1  # Start counting from the first tick

    for i in range(1, n):
        cum_ticks += 1
        if cum_ticks >= threshold:
            tick_bar_indices.append(i)
            cum_ticks = 0  # Reset the counter after reaching the threshold

    return tick_bar_indices


@njit(nogil=True)
def _volume_bar_indexer(
        volumes: NDArray[np.float64],
        threshold: float
) -> NumbaList:
    """
    Determine the volume bar open indices using cumulative volume.
    :param volumes: Trade volumes.
    :param threshold: Volume bucket threshold for opening a new bar.
    :returns: close_indices: Timestamps at which each volume bar opens.

    .. note::
        The first trade is always the start of a bar.
        A new bar is opened when the cumulative trade volume meets or exceeds the threshold.
    """
    n = len(volumes)

    # Initialize a Numba typed list to store indices
    volume_bar_indices = NumbaList()
    volume_bar_indices.append(0)

    cum_volume = volumes[0]  # Start counting from the first tick
    for i in range(1, n):
        cum_volume += volumes[i]
        if cum_volume >= threshold:
            volume_bar_indices.append(i)
            cum_volume = 0.  # No carry-over volume, if target is hit, reset the counter

    return volume_bar_indices


@njit(nogil=True)
def _dollar_bar_indexer(
        prices: NDArray[np.int64],
        volumes: NDArray[np.float64],
        threshold: float
) -> NumbaList:
    """
    Determine the dollar bar open indices using cumulative dollar value.

    :param prices: Trade prices.
    :param volumes: Trade volumes.
    :param threshold: Dollar value threshold for opening a new bar.
    :returns: close_indices: Timestamps at which each dollar bar opens.

    .. note::
        The first trade is always the start of a bar.
        A new bar is opened when the cumulative dollar value (price × volume) meets or exceeds the threshold.
    """
    n = len(prices)

    # Initialize a Numba typed list to store indices
    dollar_bar_indices = NumbaList()
    dollar_bar_indices.append(0)

    cum_dollar = prices[0] * volumes[0]
    for i in range(1, n):
        cum_dollar += prices[i] * volumes[i]
        if cum_dollar >= threshold:
            dollar_bar_indices.append(i)
            cum_dollar = cum_dollar - threshold

    return dollar_bar_indices


@njit(nogil=True)
def _cusum_bar_indexer(
        timestamps: NDArray[np.int64],
        prices: NDArray[np.float64],
        sigma: NDArray[np.float64],
        sigma_floor: float,
        sigma_mult: float
) -> NumbaList:
    """
    Determine CUSUM bar open indices using a symmetric CUSUM filter
    on successive price changes (López de Prado, 2018).

    A new bar starts whenever the cumulative sum of price changes
    exceeds +sigma*lambda or –sigma*lambda.

    :param timestamps: timestamps of the trades.
    :param prices: Trade prices.
    :param sigma: Threshold vector for CUSUM (e.g. calculated EWMS volatility or constant).
    :param sigma_floor: Minimum value for sigma to avoid division by zero.
    :param sigma_mult: sigma multiplier for the CUSUM filter (threshold will be lambda_mult*sigma).
    :returns: close_indices
    """
    if len(prices) != len(sigma) != len(timestamps):
        raise ValueError("Prices, timestamps, and sigma arrays must have the same length.")

    n = len(prices)

    # Find first non-NaN index in sigma
    first_non_nan_idx = 0
    for i in range(len(sigma)):
        if not np.isnan(sigma[i]):
            first_non_nan_idx = i
            break

    # Fill NaN values with previous non-NaN value
    for i in range(first_non_nan_idx, n):
        if np.isnan(sigma[i]):
            sigma[i] = sigma[i-1]

    # store bar–closing indices
    cusum_bar_indices = NumbaList()
    cusum_bar_indices.append(first_non_nan_idx)         # first trade starts bar

    s_pos = 0.0                         # positive cum-sum
    s_neg = 0.0                         # negative cum-sum

    i = first_non_nan_idx + 1
    while i < n:
        ret = np.log(prices[i]/prices[i - 1])

        # update symmetric CUSUMs
        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        # If we are within a ms print block, we cannot close the bar
        if i + 1 < n and timestamps[i] == timestamps[i + 1]:
            i += 1
            continue

        lam = max(float(sigma_mult * sigma[i]), sigma_floor)
        # open a new bar if either side hits the threshold
        if s_pos >= lam:
            cusum_bar_indices.append(i)
            s_pos = 0.0
        elif s_neg <= -lam:
            cusum_bar_indices.append(i)
            s_neg = 0.0
        i += 1

    return cusum_bar_indices


@njit(nogil=True)
def _imbalance_bar_indexer(
        timestamps: NDArray[np.int64],
        prices: NDArray[np.float64],
        volumes: NDArray[np.float64],
        threshold: float
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine the imbalance bar open indices based on cumulative imbalance.

    :param timestamps: Raw trade timestamps.
    :param prices: Trade prices.
    :param volumes: Trade volumes.
    :param threshold: Imbalance threshold for opening a new bar.
    :returns: A tuple of open timestamps and indices for imbalance bars.
    :raises NotImplementedError: Always raised as this function is not yet implemented.
    """
    raise NotImplementedError("Imbalance bar indexer is not implemented yet.")


@njit(nogil=True)
def _run_bar_indexer(
        timestamps: NDArray[np.int64],
        prices: NDArray[np.float64],
        volumes: NDArray[np.float64],
        threshold: float
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine the run bar open indices using cumulative run activity.

    :param timestamps: Raw trade timestamps.
    :param prices: Trade prices.
    :param volumes: Trade volumes.
    :param threshold: Run threshold for opening a new bar.
    :returns: A tuple of open timestamps and indices for run bars.
    :raises NotImplementedError: Always raised as this function is not yet implemented.
    """
    raise NotImplementedError("Run bar indexer is not implemented yet.")
