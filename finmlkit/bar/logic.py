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
        interval_seconds: int
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine the time bar open indices in the raw trades timestamp array.

    :param timestamps: Raw sorted trade timestamps in nanoseconds.
    :param interval_seconds: Length of the time bar in seconds.
    :returns: A tuple of:
        - bar_open_ts: Timestamps at which each bar opens.
        - bar_open_indices: Indices in the trade data corresponding to bar openings.

    .. note::
        The first bar is aligned to the ceiling of the first timestamp, ensuring consistent bar boundaries.
        Duplicate indices may occur if a bar interval contains no trades (empty bars).
    """
    bar_interval_ns = interval_seconds * 1e9

    # determine the first bar start time (closest integer multiple of bar_interval_ns)
    bar_start_ts = np.ceil(timestamps[0] / bar_interval_ns) * bar_interval_ns

    # last tick timestamp
    last_ts = timestamps[-1]

    # create the array of bar open timestamps
    bar_open_ts = np.arange(bar_start_ts, last_ts + 1, bar_interval_ns, dtype=np.int64)

    # find the indices of the bar open timestamps in the raw trades timestamps
    # side='left' ensures that the index returned is the first index where the value is greater than or equal to the bar open timestamp
    bar_open_indices = np.searchsorted(timestamps, bar_open_ts, side='left')

    # open times and raw trades samples:
    # |----1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|     -> bar_open_ts
    # ..  . . .... . .. . . ... .. . . ... ... ... ...   . .  .    .  .... .            .  ... ....   -> raw trades timestamps
    # ^         ^         ^          ^         ^         ^         ^                    ^       ^     -> open indices
    #                                                                                   ^             -> open indices (empty bar)
    # 0         6         12         19        26        32        35                   41,41   46    -> raw trades indices

    return bar_open_ts, bar_open_indices


@njit(nogil=True)
def _tick_bar_indexer(
        timestamps: NDArray[np.int64],
        threshold: int
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine the tick bar open indices in the raw trades timestamp array.

    :param timestamps: Raw trade timestamps.
    :param threshold: The tick count threshold for opening a new bar.
    :returns: A tuple of:
        - open_timestamps: Timestamps at which each bar opens.
        - result: Indices in the trade data corresponding to bar openings.

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

    # Convert the typed list directly to a NumPy array
    result = np.array(tick_bar_indices, dtype=np.int64)
    open_timestamps = timestamps[result]

    return open_timestamps, result


@njit(nogil=True)
def _volume_bar_indexer(
        timestamps: NDArray[np.int64],
        volumes: NDArray[np.float64],
        threshold: float
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine the volume bar open indices using cumulative volume.

    :param timestamps: Raw trade timestamps.
    :param volumes: Trade volumes.
    :param threshold: Volume threshold for opening a new bar.
    :returns: A tuple of:
        - open_timestamps: Timestamps at which each volume bar opens.
        - result: Indices in the trade data corresponding to volume bar openings.

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
            cum_volume = 0

    # Convert the typed list directly to a NumPy array
    result = np.array(volume_bar_indices, dtype=np.int64)
    open_timestamps = timestamps[result]

    return open_timestamps, result


@njit(nogil=True)
def _dollar_bar_indexer(
        timestamps: NDArray[np.int64],
        prices: NDArray[np.int64],
        volumes: NDArray[np.float64],
        threshold: float
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine the dollar bar open indices using cumulative dollar value.

    :param timestamps: Raw trade timestamps.
    :param prices: Trade prices.
    :param volumes: Trade volumes.
    :param threshold: Dollar value threshold for opening a new bar.
    :returns: A tuple of:
        - open_timestamps: Timestamps at which each dollar bar opens.
        - result: Indices in the trade data corresponding to dollar bar openings.

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
            cum_dollar = 0

    # Convert the typed list directly to a NumPy array
    result = np.array(dollar_bar_indices, dtype=np.int64)
    open_timestamps = timestamps[result]

    return open_timestamps, result


@njit(nogil=True)
def _cusum_bar_indexer(
        timestamps: NDArray[np.int64],
        prices: NDArray[np.float64],
        threshold: float
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine CUSUM bar open indices using a symmetric CUSUM filter
    on successive price changes (López de Prado, 2018).

    A new bar starts whenever the cumulative sum of price changes
    exceeds +threshold or –threshold.

    :param timestamps: Raw trade timestamps (ns).
    :param prices: Trade prices.
    :param threshold: Absolute CUSUM threshold.
    :returns: (open_timestamps, open_indices)
    """
    n = len(prices)

    # store bar–opening indices
    cusum_bar_indices = NumbaList()
    cusum_bar_indices.append(0)         # first trade starts bar

    s_pos = 0.0                         # positive cum-sum
    s_neg = 0.0                         # negative cum-sum

    for i in range(1, n):
        price_change = prices[i] - prices[i - 1]

        # update symmetric CUSUMs
        s_pos = max(0.0, s_pos + price_change)
        s_neg = min(0.0, s_neg + price_change)

        # open a new bar if either side hits the threshold
        if s_pos >= threshold or s_neg <= -threshold:
            cusum_bar_indices.append(i)
            s_pos = 0.0
            s_neg = 0.0

    # convert to NumPy arrays
    result = np.array(cusum_bar_indices, dtype=np.int64)
    open_timestamps = timestamps[result]

    return open_timestamps, result


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
