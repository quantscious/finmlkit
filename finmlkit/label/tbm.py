"""
Implements the Triple Barrier Method (TBM) for labeling financial data based on
Advances in Financial Machine Learning, Chapter 3.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Tuple, Optional


@njit(nogil=True, parallel=True)
def triple_barrier(
        timestamps: NDArray[np.int64],
        close: NDArray[np.float64],
        event_idxs: NDArray[np.int64],
        targets: NDArray[np.float64],
        horizontal_barriers: Tuple[float, float],
        vertical_barrier: float,
        min_close_time_sec: float,
        side: Optional[NDArray[np.int8]],
        min_ret: float
) -> Tuple[NDArray[np.int8], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Implements the Triple Barrier Method (TBM) for labeling financial data based on
    Advances in Financial Machine Learning, Chapter 3.

    :param timestamps: The timestamps in nanoseconds for the close prices series.
    :param close: The close prices of the asset.
    :param event_idxs: The nanosecond timestamps of the events, e.g. acquired from the cusum filter. (subset of timestamps)
    :param targets: Log-return targets for the events, e.g. acquired from a moving volatility estimator. Length must matchevent_idxs.
    :param horizontal_barriers: The bottom and top horizontal barrier multipliers for the triple barrier search by which the target is multiplied.
        This setup determines the width of the horizontal barriers. If you want to disable the barriers, set it to np.inf or -np.inf.
    :param vertical_barrier: The temporal barrier in seconds. Set it to np.inf to disable the vertical barrier.
    :param min_close_time_sec: The minimum open time in seconds (useful when raw tick data is used). This prevents closing the event prematurely before the minimum open time is reached.
    :param side: Optional array indicating the side of the event (-1 for sell, 1 for buy) for meta labeling. Length must match event_idxs. None for side predication.
    :param min_ret: The minimum target value for meta-labeling. If the return is below this value, the label will be 0, otherwise 1.
    :returns: A tuple of 4 elements containing:

        - The label (-1, 1) for side prediction (barriers should be symmetric); If side is provided, the meta-labels are (0, 1)
        - The first barrier touch index,
        - The return,
        - Maximum return-barrier ratio during the search describing how close the path came to a horizontal barrier.
          This can be used to weight samples. If a barrier is hit, the ratio is 1.0, otherwise it is less than 1.0 – or np.nan if barriers are disabled)
    """
    if vertical_barrier <= 0:
        raise ValueError("The vertical barrier must be greater than zero.")
    if min_ret < 0:
        raise ValueError("The minimum return must be non-negative.")
    if len(timestamps) != len(close):
        raise ValueError("The lengths of timestamps and close must match.")
    if len(event_idxs) != len(targets):
        raise ValueError("The lengths of event_idxs and targets must match.")
    if len(event_idxs) == 0:
        raise ValueError("The event_idxs array must not be empty.")

    is_meta = side is not None
    if is_meta:
        if len(event_idxs) != len(side):
            raise ValueError("The length of event_idxs must match the length of side.")
    else:
        # Create default side array for non-meta labeling
        side = np.ones_like(event_idxs, dtype=np.int8)

    n_events = len(event_idxs)  # Number of events (subset of samples)
    bottom_mult, top_mult = horizontal_barriers
    vertical_barrier_ns = vertical_barrier * 1e9  # Convert to nanoseconds
    min_close_time_ns = min_close_time_sec * 1e9    # Convert to nanoseconds

    log_close = np.log(close)  # Precompute log of close prices for efficiency

    labels = np.zeros(n_events, dtype=np.int8)                # The label (-1, 1) or (0, 1)
    touch_idxs   = np.empty(n_events, dtype=np.int64)               # Index of the first barrier touch
    rets = np.full(n_events, np.nan, dtype=np.float64)             # The return corresponding to the given label
    max_rb_ratios = np.full(n_events, np.nan, dtype=np.float64)    # Maximum return/target ratio during the search

    # Loop over the events parallelized
    for i_event in prange(n_events):
        t0_idx = event_idxs[i_event]
        tgt = targets[i_event]

        # Early skip if `tgt` is below `min_ret`
        # if tgt < min_ret:
        #     continue
        # -> This should be done in a preprocessing step before calling this function.

        # Upper and lower barriers in log-return space
        upper_barrier = tgt * top_mult
        lower_barrier = -tgt * bottom_mult
        # Pre-compute barrier conditions outside the loop
        upper_valid = np.isfinite(upper_barrier) and upper_barrier != 0.0
        lower_valid = np.isfinite(lower_barrier) and lower_barrier != 0.0

        # Find vertical barrier index
        t0 = timestamps[t0_idx]         # Start timestamp
        t1 = t0 + vertical_barrier_ns   # End timestamp
        t1_idx = np.searchsorted(timestamps, t1, side='right') - 1
        if t1_idx <= t0_idx:
            print("Warning: Vertical barrier index is less than or equal to event index. Skipping this event.")
            print(t0, t1)
            continue

        # ---------- Evaluate the path -----------
        side_mult = side[i_event]
        touch_idx = t1_idx
        max_urbr = 0.0
        max_lrbr = 0.0
        base_price = log_close[t0_idx]  # Base price for calculating returns
        ret = 0.
        for j in range(t0_idx + 1, t1_idx + 1):

            dur_ns = timestamps[j] - t0  # Duration in nanoseconds
            if dur_ns < min_close_time_ns:
                # Cannot close until the minimum open time is reached
                continue

            ret = (log_close[j] - base_price) * side_mult

            # progress towards barrier (skip if barrier is inf o r0)
            if ret > 0.0 and upper_valid:
                # Approaching the upper barrier
                max_urbr = max(max_urbr, ret / upper_barrier)
            elif ret < 0.0 and lower_valid:
                # Approaching the lower barrier
                max_lrbr = max(max_lrbr, ret / lower_barrier)

            # Check if we touch the barrier
            if ret >= upper_barrier:
                touch_idx = j
                break
            if ret <= lower_barrier:
                touch_idx = j
                break

        # ---------- Assign the label and other values ------------
        touch_idxs[i_event] = touch_idx
        rets[i_event] = ret

        # Assign the labels
        if is_meta:
            labels[i_event] = 1 if ret >= min_ret else 0
        else:
            sign = np.sign(ret)
            labels[i_event] = sign if sign != 0 else 1


        if touch_idx == t1_idx:
            # Calculate the maximum return-barrier ratio based sample weight
            if ret > 0.:
                max_rbr = max_urbr / (1 + max_lrbr)
                max_rbr = max_rbr if upper_valid else np.nan
            else:
                max_rbr = max_lrbr / (1 + max_urbr)
                max_rbr = max_rbr if lower_valid else np.nan
            max_rb_ratios[i_event] = min(max_rbr, 1.) # Ensure the weight is capped at 1.0
        else:
            max_rb_ratios[i_event] = 1.

    return labels, touch_idxs, rets, max_rb_ratios
