"""
Implements the Triple Barrier Method (TBM) for labeling financial data based on
Advances in Financial Machine Learning, Chapter 3.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Tuple


@njit(nogil=True, parallel=True)
def triple_barrier(
        timestamps: NDArray[np.int64],
        close: NDArray[np.float64],
        event_idxs: NDArray[np.int64],
        targets: NDArray[np.float64],
        min_ret: float,
        horizontal_barriers: Tuple[float, float],
        vertical_barrier: int
) -> Tuple[NDArray[np.int8], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Implements the Triple Barrier Method (TBM) for labeling financial data based on
    Advances in Financial Machine Learning, Chapter 3.

    :param timestamps: The timestamps in nanoseconds for the close prices series.
    :param close: The close prices of the asset.
    :param event_idxs: The indices of the events, e.g. acquired from the cusum filter. (subset of timestamps)
    :param targets: The target returns for the events, e.g. acquired from a moving volatility estimator.
    :param min_ret: The minimum target return required for running the triple barrier search.
    :param horizontal_barriers: The bottom and top horizontal barrier multipliers for the triple barrier search by which the target is multiplied.
        This setup determines the width of the horizontal barriers. If you want to disable one of the barriers, set it to np.inf.
    :param vertical_barrier: The temporal barrier in seconds.

    :returns: A tuple of 4 elements containing:
        - The labels (-1, 0, 1),
        - The first barrier touch index,
        - The return,
        - Maximum return to target ratio during the search describing how close the path came to a horizontal barrier.
        This can be used later to calculate weights for 0 labels.
    """

    n_samples = len(close)  # Number of samples in the close price array
    n_events = len(event_idxs)  # Number of events (subset of samples)
    bottom_barrier, top_barrier = horizontal_barriers
    vertical_barrier_ns = vertical_barrier * 1e9  # Convert to nanoseconds

    labels = np.zeros(n_events, dtype=np.int8)               # The labels (-1, 0, 1)
    barrier_touch_idxs = np.zeros(n_events, dtype=np.int32)  # Index of the first barrier touch
    rets = np.zeros(n_events, dtype=np.float64)              # The return corresponding to the given label
    max_return_ratio = np.zeros(n_events, dtype=np.float64)      # Maximum return/target ratio during the search

    # Loop over the events parallelized
    for i in prange(n_events):
        t0_idx = event_idxs[i]
        t0 = timestamps[t0_idx]         # Start timestamp
        t1 = t0 + vertical_barrier_ns   # End timestamp
        # Find index to closest to vertical barrier
        t1_idx = np.searchsorted(timestamps, t1)

        # Skip if the vertical barrier is out of bounds
        if t1_idx >= n_samples:
            # Set float64 containers to nan
            rets[i] = np.nan
            max_return_ratio[i] = np.nan
            continue

        # Evaluate the path
        upper_barrier = np.log(1 + targets[i] * top_barrier)
        lower_barrier = -np.log(1 + targets[i] * bottom_barrier)
        current_label = 0
        touch_idx = t1_idx
        max_rr = 0.

        for j in range(t0_idx + 1, t1_idx + 1):
            ret = np.log(close[j]/close[t0_idx])

            # Calculate how close we get to the barrier
            if ret > 0:
                ratio = ret / upper_barrier
            else:
                ratio = ret / lower_barrier
            max_rr = max(max_rr, ratio)

            # Check if we touch the barrier
            if ret >= upper_barrier:
                current_label = 1
                touch_idx = j
                break
            elif ret <= lower_barrier:
                current_label = -1
                touch_idx = j
                break

        # Assign the label and other values
        final_ret = np.log(close[touch_idx] / close[t0_idx])
        labels[i] = current_label if abs(final_ret) >= min_ret else 0
        barrier_touch_idxs[i] = touch_idx
        rets[i] = final_ret
        max_return_ratio[i] = max_rr

    return labels, barrier_touch_idxs, rets, max_return_ratio
