"""
Implements various volatility estimators.
"""
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from .utils import compute_lagged_returns, logger


@njit(nogil=True)
def ewms(y: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    """
    Calculates the Exponentially Weighted Moving Standard Deviation (EWM_STD) of a one-dimensional numpy array.
    Equivalent to `pandas.Series.ewm(...).std()` with `adjust=True` and `bias=False`.

    :param y: A one-dimensional numpy array of floats.
    :param span: The decay window, or 'span'.
    :returns: The EWM standard deviation vector, same length and shape as `y`.

    .. note::
        This function adjusts for small sample sizes by dividing by the cumulative weight minus the sum of squared weights
        divided by the cumulative weight, matching the behavior of `adjust=True` and `bias=False` in pandas.
    """
    n = y.shape[0]
    ewm_std = np.empty(n, dtype=np.float64)

    if span <= 1:
        ewm_std[:] = np.nan
        print("Span size is less than or equal to 1. Returning NaNs.")
        return ewm_std

    alpha = 2.0 / (span + 1.0)
    one_minus_alpha = 1.0 - alpha

    # Initialize cumulative sums and weights
    S_w = 0.0    # Cumulative sum of weights
    S_w2 = 0.0   # Cumulative sum of squared weights
    S_y = 0.0    # Cumulative sum of weighted y
    S_y2 = 0.0   # Cumulative sum of weighted y^2

    for t in range(n):
        y_t = y[t]

        # Update cumulative weights regardless of NaN
        S_w = one_minus_alpha * S_w + (0.0 if np.isnan(y_t) else 1.0)
        S_w2 = (one_minus_alpha ** 2) * S_w2 + (0.0 if np.isnan(y_t) else 1.0)

        # Update cumulative sums
        if not np.isnan(y_t):
            S_y = one_minus_alpha * S_y + y_t
            S_y2 = one_minus_alpha * S_y2 + y_t ** 2
        else:
            # Decay the cumulative sums without adding new data
            S_y = one_minus_alpha * S_y
            S_y2 = one_minus_alpha * S_y2

        # Calculate mean and variance if cumulative weight is positive
        if S_w > 0.0:
            mean = S_y / S_w
            denominator = S_w - (S_w2 / S_w)
            if denominator > 0.0:
                variance = (S_y2 / S_w - mean ** 2) * S_w / denominator
                variance = max(variance, 0.0)  # Ensure non-negative variance
                ewm_std[t] = np.sqrt(variance)
            else:
                ewm_std[t] = np.nan
        else:
            ewm_std[t] = np.nan

    return ewm_std


@njit(nogil=True, parallel=True)
def standard_volatility_estimator(
        timestamps: NDArray[np.int64],
        close: NDArray[np.float64],
        return_window_sec: float,
        lookback: int = 100
) -> NDArray[np.float64]:
    """
    Implements a simple volatility estimator using an exponentially weighted rolling window
    standard deviation on lagged returns.

    This function works for arbitrary time series data and does not require a fixed frequency.
    It first computes time-lagged returns over a fixed horizon, and then applies
    an exponentially weighted moving standard deviation (EWM Std) over a specified lookback window.

    :param timestamps: Raw trade timestamps in nanoseconds, sorted in ascending order.
    :param close: Raw trade prices corresponding to the timestamps.
    :param return_window_sec: The lag window size in seconds to compute returns.
    :param lookback: The number of points to use for EWM Std lookback. Defaults to 100.
    :returns: The exponentially weighted rolling volatility estimate as a NumPy array.
    :raises ValueError: If `timestamps` and `close` are not the same length.

    .. note::
        This estimator does not assume uniform sampling. It is suitable for event-driven
        data such as trades or tick-level bars. The EWM standard deviation used internally
        has ``adjust=True``, ``bias=False`` behavior, which provides an unbiased estimate.
    """
    if len(timestamps) != len(close):
        raise ValueError("The length of timestamps and close prices must be the same.")

    n = len(close)
    vol_estimates = np.empty(n, dtype=np.float64)
    vol_estimates.fill(np.nan)

    # 1. Compute the lagged returns
    returns = compute_lagged_returns(timestamps, close, return_window_sec)
    # 2. Apply EWM standard deviation over the returns
    vol_estimates = ewms(returns, lookback)

    return vol_estimates


@njit(nogil=True, parallel=True)
def true_range(high: NDArray, low: NDArray, close: NDArray) -> NDArray:
    """
    Calculate True Range using Numba.

    :param high: np.array, high prices
    :param low: np.array, low prices
    :param close: np.array, close prices
    :return: np.array, True Range values
    """
    if not (len(high) == len(low) == len(close)):
        raise ValueError("The length of high, low, and close prices must be the same.")

    tr = np.empty_like(high)
    tr[0] = high[0] - low[0]  # First TR value

    for i in prange(1, len(high)):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1])
                    )  # TR formula

    return tr

