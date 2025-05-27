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
        targets: NDArray[np.float64],
        event_idxs: NDArray[np.int64],
        min_ret: float,
        horizontal_barriers: Tuple[float, float],
        vertical_barrier: float,
        side: Optional[NDArray[np.int8]]
) -> Tuple[NDArray[np.int8], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Implements the Triple Barrier Method (TBM) for labeling financial data based on
    Advances in Financial Machine Learning, Chapter 3.

    :param timestamps: The timestamps in nanoseconds for the close prices series.
    :param close: The close prices of the asset.
    :param targets: Log-return targes for the events, e.g. acquired from a moving volatility estimator. Length must match timestamps and close.
    :param event_idxs: The indices of the events, e.g. acquired from the cusum filter. (subset of timestamps)
    :param min_ret: The minimum target return required for running the triple barrier search.
    :param horizontal_barriers: The bottom and top horizontal barrier multipliers for the triple barrier search by which the target is multiplied.
        This setup determines the width of the horizontal barriers. If you want to disable the barriers, set it to np.inf or -np.inf.
    :param vertical_barrier: The temporal barrier in seconds. Set it to np.inf to disable the vertical barrier.
    :param side: Optional array indicating the side of the event (-1 for sell, 1 for buy) for meta labeling. Length must match event_idxs.

    :returns: A tuple of 4 elements containing:
        - The label (-1, 1), 0 for no label; If side is provided, the meta-labels are (0, 1)
        - The first barrier touch index,
        - The return,
        - Maximum return-barrier ratio during the search describing how close the path came to a horizontal barrier.
          This can be used later to calculate weights for labels. If barrier is hit, the ratio is 1.0, otherwise it is less than 1.0 – or np.nan if barriers are disabled)
    """
    if vertical_barrier <= 0:
        raise ValueError("The vertical barrier must be greater than zero.")
    if min_ret < 0:
        raise ValueError("The minimum return must be non-negative.")
    if len(timestamps) != len(close) or len(timestamps) != len(targets):
        raise ValueError("The lengths of timestamps, close, and targets must match.")
    if len(event_idxs) == 0:
        raise ValueError("The event_idxs array must not be empty.")

    is_meta = side is not None
    if is_meta:
        if len(event_idxs) != len(side):
            raise ValueError("The length of event_idxs must match the length of side.")

    n_samples = len(close)  # Number of samples in the close price array
    n_events = len(event_idxs)  # Number of events (subset of samples)
    bottom_mult, top_mult = horizontal_barriers
    vertical_barrier_ns = vertical_barrier * 1e9  # Convert to nanoseconds
    log_close = np.log(close)  # Precompute log of close prices for efficiency

    labels = np.zeros(n_events, dtype=np.int8)                # The label (-1, 0, 1) or (0, 1)
    touch_idxs   = np.empty(n_events, np.int64)               # Index of the first barrier touch
    touch_idxs[:] = -1
    rets = np.full(n_events, np.nan, dtype=np.float64)             # The return corresponding to the given label
    max_rb_ratios = np.full(n_events, np.nan, dtype=np.float64)    # Maximum return/target ratio during the search


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
        touch_idx = t1_idx
        max_rbr = np.nan
        base_price = log_close[t0_idx]  # Base price for calculating returns
        ret = 0.
        for j in range(t0_idx + 1, t1_idx + 1):
            ret = log_close[j] - base_price
            if is_meta:
                ret *= side[i]  # Adjust return based on the side if meta-labeling

            # progress towards barrier (skip if barrier is inf/0)
            if ret > 0.0 and np.isfinite(upper_barrier) and upper_barrier != 0.0:
                max_rbr = max(max_rbr, ret / upper_barrier)
            elif ret < 0.0 and np.isfinite(lower_barrier) and lower_barrier != 0.0:
                max_rbr = max(max_rbr, ret / lower_barrier)

            # Check if we touch the barrier
            if ret >= upper_barrier:
                touch_idx = j
                break
            if ret <= lower_barrier:
                touch_idx = j
                break

        # Assign the label and other values
        if is_meta:
            labels[i] = 1 if ret > min_ret else 0
        else:
            labels[i] = np.sign(ret)
        touch_idxs[i] = touch_idx
        rets[i] = ret
        max_rb_ratios[i] = max_rbr

    return labels, touch_idxs, rets, max_rb_ratios
