import numpy as np
from numba import njit
from numpy.typing import NDArray
from .utils import tick_rule


@njit(nopython=True, nogil=True)
def time_bar_indexer(timestamps: NDArray[np.int64], interval_seconds: int) -> NDArray[np.int64]:
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
    np.array(np.int64)
        Time bar open indices.
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

    return bar_open_indices

