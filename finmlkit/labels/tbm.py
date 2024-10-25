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

    Parameters
    ----------
    timestamps : np.array(np.int64)
        The timestamps in nanoseconds for the close prices series.
    close : np.array(np.float64)
        The close prices of the asset.
    event_idxs : np.array(np.int64)
        The indices of the events, e.g. acquired from the cusum filter. (subset of timestamps)
    targets : np.array(np.float64)
        The target returns for the events, e.g. acquired from a moving volatility estimator.
    min_ret : float
        The minimum target return required for running the triple barrier search.
    horizontal_barriers : tuple(float, float)
        The bottom and top horizontal barrier multipliers for the triple barrier search by which the target is multiplied.
        This setup determines the width of the horizontal barriers. If you want to disable one of the barriers, set it to np.inf.
    vertical_barrier : int
        The temporal barrier in seconds.

    Returns
    -------
    tuple(np.array(np.int8), np.array(np.int32), np.array(np.float64))
        The labels (-1, 0, 1), the first barrier touch index, the return, maximum return to target ratio during the search
        describing how close the path came to a horizontal barrier.
        The latter can be used later to calculate weights for 0 labels.
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


@njit(nogil=True, parallel=True)
def label_concurrency_weights(timestamps: NDArray[np.int64], event_idxs: NDArray[np.int64], lookahead_idxs: NDArray[np.int64]) -> NDArray[np.float64]:
    """
    Calculate the uniqueness weights for the overlapping labels.
    Based on Advances in Financial Machine Learning, Chapter 4. page 61.

    Parameters
    ----------
    timestamps : np.array(np.int64)
        The timestamps in nanoseconds for the close prices series.
    event_idxs : np.array(np.int64)
        The indices of the labeled event_idxs, e.g. acquired from the cusum filter. (subset of timestamps)
    lookahead_idxs : np.array(np.int64)
        The lookahead indices for the given labels.

    Returns
    -------
    np.array(np.float64)
        The uniqueness weights [0, 1] for the labels.
    """
    if len(event_idxs) != len(lookahead_idxs):
        raise ValueError("Timestamps and lookahead indices must have the same length.")

    n = len(timestamps)
    n_events = len(event_idxs)

    concurrency = np.zeros(n, dtype=np.int16)
    weights = np.zeros(n_events, dtype=np.float64)

    # 1.) Calculate the concurrency for each timestamp
    for i in range(n_events):
        start_idx = event_idxs[i]
        end_idx = lookahead_idxs[i]
        concurrency[start_idx:end_idx+2] += 1  # overlaps with [t-1, t]

    # 2.) Calculate the harmonic mean of concurrency for each event
    for i in prange(n_events):
        start_idx = event_idxs[i]
        end_idx = lookahead_idxs[i]
        concurrency_slice = concurrency[start_idx:end_idx+1]

        # The weights are calculated as per the harmonic mean of the inverse concurrency over the label’s duration.
        weights[i] = np.mean(1.0 / concurrency_slice)

    return weights


@njit(nogil=True, parallel=True)
def vertical_barrier_weights(max_return_ratios: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate weights for labels corresponding to vertical barrier hits based on the maximum return ratio.
    The idea is to assign lower weights to labels that approached a horizontal barrier closely but did not touch it.

    Parameters
    ----------
    max_return_ratios : np.array(np.float64)
        The maximum d/D ratio for the given labels.

    Returns
    -------
    np.array(np.float64)
        The vertical barrier weights [0, 1] for the labels.
    """
    if min(max_return_ratios) < 0.0 or max(max_return_ratios) > 1.0:
        raise ValueError("The max_return_ratios must be in the range [0, 1].")

    n_events = len(max_return_ratios)
    weights = np.zeros(n_events, dtype=np.float64)

    for i in prange(n_events):
        if max_return_ratios[i] < 0.5:
            weights[i] = 1.
        else:
            # Linearly decrease the weight as the ratio approaches 1
            weights[i] = 2. - 2. * max_return_ratios[i]

    return weights
