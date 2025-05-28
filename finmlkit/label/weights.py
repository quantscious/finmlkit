import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Tuple


@njit(nogil=True, parallel=True)
def label_average_uniqueness(
        timestamps: NDArray[np.int64],
        event_idxs: NDArray[np.int64],
        touch_idxs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Calculate the uniqueness weights for the overlapping label.
    Based on Advances in Financial Machine Learning, Chapter 4. page 61.

    :param timestamps: The timestamps in nanoseconds for the close prices series.
    :param event_idxs: The indices of the labeled events, e.g. acquired from the cusum filter. (subset of timestamps)
    :param touch_idxs: The touch indices for the given events.
    :returns: The uniqueness weights [0, 1] for the label.
    :raises ValueError: If timestamps and touch indices are of different lengths.
    """
    if len(event_idxs) != len(touch_idxs):
        raise ValueError("Timestamps and lookahead indices must have the same length.")

    n = len(timestamps)
    n_events = len(event_idxs)

    concurrency = np.zeros(n, dtype=np.int32)
    weights = np.zeros(n_events, dtype=np.float64)

    # 1.) Calculate the concurrency for each timestamp
    for i in range(n_events):
        start_idx = event_idxs[i]
        end_idx = touch_idxs[i]
        concurrency[start_idx:end_idx+1] += 1  # return overlaps within [t-1, t]

    # 2.) Calculate the weight for each label
    for i in prange(n_events):
        start_idx = event_idxs[i]
        end_idx = touch_idxs[i]
        concurrency_slice = concurrency[start_idx:end_idx+1]

        # The weights are calculated as per the arithmetic mean of the inverse concurrency over the label’s duration.
        weights[i] = np.mean(1.0 / concurrency_slice)

    return weights
