import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Tuple


@njit(nogil=True, parallel=True)
def label_concurrency_weights(
        timestamps: NDArray[np.int64],
        event_idxs: NDArray[np.int64],
        lookahead_idxs: NDArray[np.int64]
) -> NDArray[np.float64]:
    """
    Calculate the uniqueness weights for the overlapping label.
    Based on Advances in Financial Machine Learning, Chapter 4. page 61.

    :param timestamps: The timestamps in nanoseconds for the close prices series.
    :param event_idxs: The indices of the labeled events, e.g. acquired from the cusum filter. (subset of timestamps)
    :param lookahead_idxs: The lookahead indices for the given label.
    :returns: The uniqueness weights [0, 1] for the label.
    :raises ValueError: If timestamps and lookahead indices are of different lengths.
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

    # 2.) Calculate the weight for each label
    for i in prange(n_events):
        start_idx = event_idxs[i]
        end_idx = lookahead_idxs[i]
        concurrency_slice = concurrency[start_idx:end_idx+1]

        # The weights are calculated as per the harmonic mean of the inverse concurrency over the label’s duration.
        weights[i] = np.mean(1.0 / concurrency_slice)

    return weights


@njit(nogil=True, parallel=True)
def vertical_barrier_weights(
        max_return_ratios: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate weights for TBM (triple barrier) label corresponding to vertical barrier hits
    based on the maximum return ratio.

    The idea is to assign lower weights to label that approached a horizontal barrier closely but did not touch it.

    :param max_return_ratios: The maximum d/D ratio for the given label.
    :returns: The vertical barrier weights [0, 1] for the label.
    :raises ValueError: If `max_return_ratios` contains values outside the range [0, 1].
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


@njit(nogil=True)
def class_balance_weights(
        labels: NDArray[np.int8],
        weights: NDArray[np.float64]
) -> Tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the class balance weights for the given label and weights.

    :param labels: The label (e.g., -1, 0, 1) for the given events.
    :param weights: Additional weights for the given label (e.g., uniqueness weights or vertical barrier weights).
        Number of class elements will be calculated as a weighted sum.
    :returns: A tuple containing:
        - The identified classes.
        - Corresponding class weights.
        - Number of class elements per label.
        - Final weights array per sample.
    """

    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    n_class_elements = np.zeros(n_classes, dtype=np.float64)
    class_weights = np.zeros(n_classes, dtype=np.float64)
    final_weights = np.zeros(n_samples, dtype=np.float64)

    # Cumulate weighted sum for each class
    for i in range(n_samples):
        label_idx = np.searchsorted(unique_labels, labels[i])
        n_class_elements[label_idx] += weights[i]

    # Calculate the class balance weights
    for i in range(n_classes):
        class_weights[i] = n_samples / (n_classes * n_class_elements[i])

    # Calculate the final weights
    for i in range(n_samples):
        label_idx = np.searchsorted(unique_labels, labels[i])
        final_weights[i] = class_weights[label_idx] * weights[i]

    return unique_labels, class_weights, n_class_elements, final_weights
