"""
General utility functions not exclusively for OHLCV bar data.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from finmlkit.utils.log import get_logger

logger = get_logger(__name__)


@njit(nogil=True, parallel=True)
def comp_lagged_returns(timestamps: NDArray[np.int64], close: NDArray[np.float64], return_window_sec: float, is_log: bool) -> NDArray[np.float64]:
    """
    Calculate the lagged returns on the given time window.

    This function works for arbitrary time series data and does not require a fixed frequency.
    It computes the percentage change in price over a time lag window, aligned to timestamp precision.

    :param timestamps: Timestamps series in nanoseconds.
    :param close: Close price series.
    :param return_window_sec: Time window in seconds for lagged return calculation. Set it to a small value (e.g. 1e-6) for 1 sample lag.
    :param is_log: If True, compute log returns instead of simple returns.
    :returns: The lagged returns series as a float64 array.
    :raises ValueError: If return_window_sec is less than or equal to zero.

    .. note::
        The function searches the closest earlier timestamp matching the lag time difference and computes
        percentage return as `close[i] / close[lag_idx] - 1`. Gaps and irregular timestamps are handled.
        Division by zero results in `inf`. If no valid lagged index is found, the return is `NaN`.
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
                if is_log:
                    # Calculate log return
                    ret = np.log(close[i] / close[lag_idx])
                else:
                    # Calculate simple return
                    ret = close[i] / close[lag_idx] - 1.0
                returns[i] = ret
            else:
                print("Warning: Encountered a zero in price data while calculating lagged return.\n"
                      "Division by zero. Setting return value to inf.")
                returns[i] = np.inf
        else:
            # If we cannot find a valid lagged index, set the return to NaN
            returns[i] = np.nan

    return returns
