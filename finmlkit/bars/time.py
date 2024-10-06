import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(nopython=True, nogil=True)
def time_bar_indexer(timestamps: NDArray[np.int64], interval_seconds: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
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
    bar_open_ts = np.arange(bar_start_ts, last_ts, bar_interval_ns, dtype=np.int64)

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

