import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Tuple


@njit(nogil=True, parallel=True)
def average_uniqueness(
        timestamps: NDArray[np.int64],
        event_idxs: NDArray[np.int64],
        touch_idxs: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int16]]:
    """
    Calculate the uniqueness weights for the overlapping label.
    Based on Advances in Financial Machine Learning, Chapter 4. page 61.

    :param timestamps: The timestamps in nanoseconds for the close prices series.
    :param event_idxs: The indices of the labeled events, e.g. acquired from the cusum filter. (subset of timestamps)
    :param touch_idxs: The touch indices for the given events.
    :returns: A tuple with two arrays
        - The uniqueness weights [0, 1] for the label.
        - The concurrency array, which indicates how many labels overlap at each timestamp.
    :raises ValueError: If timestamps and touch indices are of different lengths.
    """
    if len(event_idxs) != len(touch_idxs):
        raise ValueError("Timestamps and lookahead indices must have the same length.")

    n = len(timestamps)
    n_events = len(event_idxs)

    concurrency = np.zeros(n, dtype=np.int16)
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

    return weights, concurrency


@njit(nogil=True, parallel=True)
def return_attribution(event_idxs: NDArray[np.int64],
                       touch_idxs: NDArray[np.int64],
                       close: NDArray[np.float64],
                       concurrency: NDArray[np.int16],
                       normalize: bool
                       ) -> NDArray[np.float64]:
    """
    Assign more weights to samples with higher return attribution.
    Advances in Financial Machine Learning, Chapter 4, page 68.

    :param event_idxs: Event indices where the label starts.
    :param touch_idxs: Touch indices where the label ends.
    :param close: Close price array.
    :param concurrency: Concurrency array indicating how many labels overlap at each timestamp. From `label_average_uniqueness` function.
    :param normalize: If True, normalize the returned weights to sum to the number of events.
    :return: NDArray[np.float64]
        An array of return attribution weights for each event.

    """
    n_events = len(event_idxs)
    n = len(close)
    weights = np.zeros(n_events, dtype=np.float64)

    # Compute log returns for the close prices
    log_rets = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if close[i - 1] != 0.0:
            log_rets[i] = np.log(close[i] / close[i - 1])
        else:
            log_rets[i] = np.nan

    # Calculate the return attribution weights for each event
    for i in prange(n_events):
        start_idx = event_idxs[i]
        end_idx = touch_idxs[i]

        weight = 0.0
        for j in range(start_idx, end_idx + 1):
            if concurrency[j] > 0 and not np.isnan(log_rets[j]):
                weight += log_rets[j] / concurrency[j]

        weights[i] = abs(weight)

    if normalize:
        # Normalize the weight to sum up to n_events
        sum_weights = np.sum(weights)
        if sum_weights <= 0.:
            raise ValueError("Sum of weights is zero or negative, cannot normalize.")
        weights *= n_events / sum_weights

    return weights


@njit(nogil=True)
def time_decay(
        avg_uniqueness: NDArray[np.float64],
        last_weight: float
) -> NDArray[np.float64]:
    """
    Apply linear time decay based on the average uniqueness weights.
    Newest observation assigned with 1.0 and oldest with `last_weight`.
    If `last_weight` is negative, the oldest portion (n_events* last_weight) is get erased (assigned with 0.0.)
    Advances in Financial Machine Learning, Chapter 4, page 70.

    :param avg_uniqueness: The average uniqueness weights for the label from `average_uniqueness` function.
    :param last_weight: The weight assigned to the last sample. If 1.0, then there is no decay.
    :return: An array of time-decayed weights [0, 1] for each event.
    :raises ValueError: The sum of all average uniqueness weights must be greater than 0.
    :raises ValueError: If `last_weight` is not in the range [-1, 1].
    """
    if not -1.0 <= last_weight <= 1.0:
        raise ValueError("last_weight must lie in [-1, 1]")

    cum_avg_uniqueness = np.cumsum(avg_uniqueness)
    if cum_avg_uniqueness[-1] == 0.0:
        raise ValueError("The sum of all average uniqueness weights must be grater than 0.")

    if last_weight >= 0.0:
        slope = (1. - last_weight) / cum_avg_uniqueness[-1]
    else:
        slope = 1. / ((last_weight + 1.) * cum_avg_uniqueness[-1])

    const = 1. - slope * cum_avg_uniqueness[-1]
    weights = const + slope * cum_avg_uniqueness

    # clip negative part caused by truncation to exactly zero
    if last_weight < 0.0:
        weights = np.maximum(weights, 0.0)

    return weights



@njit(nogil=True)
def class_balance_weights(
        labels: NDArray[np.int8],
        base_w: NDArray[np.float64]
) -> Tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Run this function after all other sample weights have been calculated and combined into `base_w`.
    Calculate the class balance weights for the given label using the base sample weights.

    :param labels: The label (e.g., -1, 0, 1) for the given events.
    :param base_w: Base weights for the given label (e.g., avg_uniqueness weights, vertical barrier weights, return attribution, time-decay combined).
           Number of class elements will be calculated as a weighted sum.
    :returns: A tuple containing:
        - The identified classes.
        - Corresponding class weights.
        - Number of class elements per label calculated as a sum of sample weights.
        - Final weights array per sample: class weights multiplied by base weights.
    """

    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    sum_w_class = np.zeros(n_classes, dtype=np.float64)
    class_weights = np.zeros(n_classes, dtype=np.float64)
    final_weights = np.zeros(n_samples, dtype=np.float64)

    # Cumulate weighted sum for each class
    for i in range(n_samples):
        label_idx = np.searchsorted(unique_labels, labels[i])
        sum_w_class[label_idx] += base_w[i]

    total_weights = np.sum(sum_w_class)

    # Calculate the class balance weights
    for c in range(n_classes):
        class_weights[c] = total_weights / (n_classes * sum_w_class[c]) if sum_w_class[c] > 0. else 0.0

    # Calculate the final weights
    for i in range(n_samples):
        label_idx = np.searchsorted(unique_labels, labels[i])
        final_weights[i] = base_w[i] * class_weights[label_idx]

    return unique_labels, class_weights, sum_w_class, final_weights