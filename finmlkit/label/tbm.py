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
    :param targets: The target returns for the events, e.g. acquired from a moving volatility estimator. Assumes a log return target.
    :param min_ret: The minimum target return required for running the triple barrier search.
    :param horizontal_barriers: The bottom and top horizontal barrier multipliers for the triple barrier search by which the target is multiplied.
        This setup determines the width of the horizontal barriers. If you want to disable one of the barriers, set it to np.inf.
    :param vertical_barrier: The temporal barrier in seconds.

    :returns: A tuple of 4 elements containing:
        - The label (-1, 0, 1),
        - The first barrier touch index,
        - The return,
        - Maximum return-barrier ratio during the search describing how close the path came to a horizontal barrier.
        This can be used later to calculate weights for 0 label. (If barrier is hit, the ratio is 1.0)
    """

    n_samples = len(close)  # Number of samples in the close price array
    n_events = len(event_idxs)  # Number of events (subset of samples)
    bottom_mult, top_mult = horizontal_barriers
    vertical_barrier_ns = vertical_barrier * 1e9  # Convert to nanoseconds

    labels = np.zeros(n_events, dtype=np.int8)               # The label (-1, 0, 1)
    touch_idxs   = np.empty(n_events, np.int64)               # Index of the first barrier touch
    touch_idxs[:] = -1
    rets = np.full(n_events, np.nan, dtype=np.float64)           # The return corresponding to the given label
    max_rb_ratios = np.full(n_events, np.nan, dtype=np.float64)         # Maximum return/target ratio during the search

    # Loop over the events parallelized
    for i in prange(n_events):
        t0_idx = event_idxs[i]
        tgt = targets[t0_idx]

        # Early skip if `tgt` is below `min_ret`
        if tgt < min_ret:
            continue

        # Upper and lower barriers in log-return space
        upper_barrier = tgt * top_mult
        lower_barrier = -tgt * bottom_mult

        # Find vertical barrier index
        t0 = timestamps[t0_idx]         # Start timestamp
        t1 = t0 + vertical_barrier_ns   # End timestamp
        t1_idx = np.searchsorted(timestamps, t1, side='right') - 1
        if t1_idx <= t0_idx:
            print("Warning: Vertical barrier index is less than or equal to event index. Skipping this event.")
            continue
        if t1_idx >= n_samples:
            # Safety check for vertical barrier index
            # If the vertical barrier index exceeds the number of samples, raise an error
            # This should not happen in practice, but it's a good safeguard.
            raise ValueError(
                f"Vertical barrier index {t1_idx} exceeds the number of samples {n_samples}."
            )

        # Evaluate the path
        label = 0
        touch_idx = t1_idx
        max_rbr = 0.
        base_price = close[t0_idx]  # Base price for calculating returns

        for j in range(t0_idx + 1, t1_idx + 1):
            ret = np.log(close[j] / base_price)

            # # progress towards barrier (skip if barrier is inf/0)
            if ret > 0.0 and np.isfinite(upper_barrier) and upper_barrier != 0.0:
                max_rbr = max(max_rbr, ret / upper_barrier)
            elif ret < 0.0 and np.isfinite(lower_barrier) and lower_barrier != 0.0:
                max_rbr = max(max_rbr, ret / lower_barrier)

            # Check if we touch the barrier
            if ret >= upper_barrier:
                label, touch_idx = 1, j
                break
            if ret <= lower_barrier:
                label, touch_idx = -1, j
                break

        # Assign the label and other values
        final_ret = np.log(close[touch_idx] / base_price)
        labels[i] = label if abs(final_ret) >= min_ret else 0
        touch_idxs[i] = touch_idx
        rets[i] = final_ret
        max_rb_ratios[i] = max_rbr

    return labels, touch_idxs, rets, max_rb_ratios
