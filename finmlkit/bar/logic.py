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


    Parameters
    ----------
    timestamps : np.array(np.int64)
        Raw sorted **trades data** timestamps in nanoseconds.
    interval_seconds : int
        Length of the time bar in seconds.

    Returns
    -------
    tuple(np.array(np.int64), np.array(np.int64))
        Time bar open timestamps and corresponding open indices in the raw trades timestamps.
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
    Determine the tick bar open indices in the raw trades price array.

    Parameters
    ----------
    timestamps : np.ndarray
        Raw trades timestamps.
    threshold : int
        The tick threshold at which to sample.

    Returns
    -------
    tuple(np.array(np.int64), np.array(np.int64))
        Tick bar open timestamps and corresponding open indices in the raw trades timestamps.
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


@njit(nogi=True)
def _volume_bar_indexer(
        timestamps: NDArray[np.int64],
        volumes: NDArray[np.float64],
        threshold: float
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine the volume bar open indices in the raw trades price array.

    Parameters
    ----------
    timestamps : np.array(np.int64)
        Raw trades timestamps.
    volumes : np.array(np.float64)
        Raw trades volumes.
    threshold : float
        The volume threshold at which to sample.

    Returns
    -------
    tuple(np.array(np.int64), np.array(np.int64))
        Volume bar open timestamps and corresponding open indices in the raw trades timestamps.

    Notes
    -----
    The first tick is the first volume bar open index. The function uses a cumulative volume counter to determine the volume bar open indices.
    When the cumulative volume counter reaches the threshold a new volume bar is opened.
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
    Determine the dollar bar open indices in the raw trades price array.

    Parameters
    ----------
    timestamps : np.array(np.int64)
        Raw trades timestamps.
    prices : np.array(np.float64)
        Raw trades prices.
    volumes : np.array(np.float64)
        Raw trades volumes.
    threshold : float
        The dollar threshold at which to sample.

    Returns
    -------
    tuple(np.array(np.int64), np.array(np.int64))
        Dollar bar open timestamps and corresponding open indices in the raw trades timestamps

    Notes
    -----
    The first tick is the first dollar bar open index. The function uses a cumulative dollar counter to determine the dollar bar open indices.
    When the cumulative dollar counter reaches the threshold a new dollar bar is opened.
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
def _imbalance_bar_indexer(
        timestamps: NDArray[np.int64],
        prices: NDArray[np.float64],
        volumes: NDArray[np.float64],
        threshold: float
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Determine the imbalance bar open indices in the raw trades price array.

    Parameters
    ----------
    timestamps: np.array(np.int64)
        Raw trades timestamps.
    prices : np.array(np.float64)
        Raw trades prices.
    volumes : np.array(np.float64)
        Raw trades volumes.
    threshold : float
        The imbalance threshold at which to sample.

    Returns
    -------
    tuple(np.array(np.int64), np.array(np.int64))
        Imbalance bar open timestamps and corresponding open indices in the raw trades timestamps.

    Notes
    -----
    The first tick is the first imbalance bar open index. The function uses a cumulative imbalance counter to determine the imbalance bar open indices.
    When the cumulative imbalance counter reaches the threshold a new imbalance bar is opened.
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
    Determine the run bar open indices in the raw trades price array.

    Parameters
    ----------
    timestamps : np.array(np.int64)
        Raw trades timestamps.
    prices : np.array(np.float64)
        Raw trades prices.
    volumes : np.array(np.float64)
        Raw trades volumes.
    threshold : float
        The run threshold at which to sample.

    Returns
    -------
    tuple(np.array(np.int64), np.array(np.int64))
        Run bar open timestamps and corresponding open indices in the raw trades timestamps.

    Notes
    -----
    The first tick is the first run bar open index. The function uses a cumulative run counter to determine the run bar open indices.
    When the cumulative run counter reaches the threshold a new run bar is opened.
    """
    raise NotImplementedError("Run bar indexer is not implemented yet.")