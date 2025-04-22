import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(nogil=True, parallel=True)
def cusum_filter(
    raw_time_series: NDArray[np.float64],
    threshold: NDArray
) -> NDArray[np.int64]:
    """
    Apply the CUSUM filter to detect events based on the cumulative sum of log returns.

    :param raw_time_series: Array of price series.
    :param threshold: Threshold values for event detection.
        - If array has 1 element, a constant threshold is used.
        - If multiple elements, it must be of the same length as `raw_time_series`.
    :returns: Indices where events occurred. These indices correspond to positions in `raw_time_series`.

    .. note::
        This function implements the Symmetric CUSUM Filter, which is designed to detect a shift
        in the mean value of a measured quantity away from a target value. It identifies events
        when the cumulative sum of log returns exceeds a specified threshold.

        This implementation follows the methodology outlined in:

        - Lopez de Prado, Marcos. "Advances in Financial Machine Learning." Wiley, 2018. Snippet 2.4, page 39.
    """
    if len(raw_time_series) <= 1:
        raise ValueError("Input time series must have at least 2 elements.")
    if len(threshold) != 1 and len(threshold) != len(raw_time_series):
        raise ValueError("Threshold array must either contain 1 const. element or len(raw_time_series) elements.")

    n = len(raw_time_series)

    # If one threshold element, repeat it to match the length of the price array
    if len(threshold) == 1:
        tmp = np.empty(n, dtype=np.float64)
        tmp.fill(threshold[0])
        threshold = tmp

    # Calculate log returns
    log_returns = np.zeros(n, dtype=np.float64)
    log_returns[0] = np.nan
    for i in prange(1, n):
        log_returns[i] = np.log(raw_time_series[i] / raw_time_series[i - 1])

    # Container for event indices
    event_indices = np.empty(n, dtype=np.int64)
    num_events = 0

    s_pos = 0.0
    s_neg = 0.0
    for i in range(1, n):
        ret = log_returns[i]
        thrs = threshold[i]

        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        if s_neg < -thrs:
            s_neg = 0.0
            event_indices[num_events] = i
            num_events += 1
        elif s_pos > thrs:
            s_pos = 0.0
            event_indices[num_events] = i
            num_events += 1

    return event_indices[:num_events]


@njit(nogil=True)
def z_score_peak_filter(
    y: NDArray[np.float64],
    window: int,
    threshold: float = 3,
) -> NDArray[np.int64]:
    """
    Implement a z-score peak detection filter.

    :param y: The input time series data of at least length `window + 2`.
    :param window: The window parameter for the moving window (number of observations to use for mean and standard deviation calculations).
    :param threshold: The z-score threshold for detecting peaks.
    :returns: The indices of the events (peaks) in the input time series data `y`.

    .. note::
        This function implements a z-score based peak detection algorithm suitable for real-time data.
        It is optimized for performance using Numba's JIT compilation.

    .. seealso::
        https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data
    """
    raise NotImplementedError("This function is not yet implemented.")
