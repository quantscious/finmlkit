"""
General utility functions not exclusively for OHLCV bars data.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(nogil=True)
def ewma(y: NDArray, span: int) -> NDArray[np.float64]:
    """
    Exponentially weighted moving average (EWMA) of a one-dimensional numpy array.
    Calculates https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html with adjust=True (default).
    By using this weighting scheme, the function provides a more accurate and unbiased estimate of the EWMA,
    especially in the early stages of the data series.

    Parameters
    ----------
    y : np.ndarray
        A one-dimensional numpy array of floats.
    span : int
        The decay window, or 'span'. Determines how many past points meaningfully impact the given EWMA value.

    Returns
    -------
    ewma_y : np.ndarray(np.float64)
        The EWMA vector, same length and shape as `y`.

    Notes
    -----
    This function adjusts for small sample sizes by dividing by the cumulative weight. For more information, see:
    https://terbe.dev/blog/posts/exponentially-weighted-moving-average
    """
    n = y.shape[0]
    ewma = np.empty(n, dtype=np.float64)  # Container for the EWMA values

    # If span is less than 1, raise an error
    if span < 1:
        raise ValueError("span size is less than or equal to 1. Please provide a span size greater than 1.")

    alpha = 2.0 / (span + 1.0)
    u_t = y[0]  # Initial value for numerator
    v_t = 1.0   # Initial value for denominator
    ewma[0] = u_t / v_t

    for t in range(1, n):
        # Update the numerator and denominator
        u_t = y[t] + (1.0 - alpha) * u_t
        v_t = 1.0 + (1.0 - alpha) * v_t
        # Calculate the EWMA
        ewma[t] = u_t / v_t

    return ewma


@njit(nogil=True)
def ewms(y: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    """
    Calculates the Exponentially Weighted Moving Standard Deviation (EWM_STD) of a one-dimensional numpy array.
    Similar to pandas' ewm.std with adjust=True (default) and bias=False (default).

    Parameters
    ----------
    y : np.ndarray
        A one-dimensional numpy array of floats.
    span : int
        The decay window, or 'span'.

    Returns
    -------
    ewm_std_y : np.ndarray(np.float64)
        The EWM standard deviation vector, same length and shape as `y`.

    Notes
    -----
    This function adjusts for small sample sizes by dividing by the cumulative weight minus the sum of squared weights
    divided by the cumulative weight, matching the behavior of adjust=True and bias=False in pandas' ewm.std.
    """
    n = y.shape[0]
    ewm_std = np.empty(n, dtype=np.float64)

    if span <= 1:
        ewm_std[:] = np.nan
        print("WARNING! Span size is less than or equal to 1. Returning NaNs.")
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
def compute_lagged_returns(timestamps: NDArray[np.int64], close: NDArray[np.float64], return_window_sec: float):
    """
    Calculate the lagged returns on the given time window.
    This function works for arbitrary time series data and does not require a fixed frequency.

    Parameters
    ----------
    timestamps : np.array(np.int64)
        Timestamps series
    close : np.array(np.float64)
        Close price series
    return_window_sec : float
        Time window in seconds for lagged return calculation. Set it to a small value (e.g. 1e-6) for 1 sample lag.


    Returns
    -------
    np.array(np.float64)
        The lagged returns series

    """
    # return window should be greater than zero
    if return_window_sec <= 0:
        raise ValueError("The return window must be greater than zero.")

    n = len(close)
    returns = np.empty(n, dtype=np.float64)
    returns.fill(np.nan)
    ret_window_ns = return_window_sec * 1e9

    # Find the first index where the lookback window is fully contained
    start_idx = np.searchsorted(timestamps, timestamps[0] + ret_window_ns, side='left')

    for i in prange(start_idx, n):
        target_time = timestamps[i] - ret_window_ns
        lag_idx = np.searchsorted(timestamps, target_time, side='right') - 1
        if 0 <= lag_idx < i:
            if close[lag_idx] != 0.0:
                ret = close[i] / close[lag_idx] - 1.0
                returns[i] = ret
            else:
                print("WARNING! Encountered a zero in price data while calculating lagged return.\n"
                      "Division by zero. Setting return value to inf.")
                returns[i] = np.inf
        else:
            # If we cannot find a valid lagged index, set the return to NaN
            returns[i] = np.nan

    return returns
