"""
Implements various volatility estimators.
"""
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from .utils import ewms

@njit(nogil=True, parallel=True)
def standard_volatility_estimator(
        timestamps: NDArray[np.int64],
        close: NDArray[np.float64],
        return_window_sec: int,
        lookback: int = 100
) -> NDArray[np.float64]:
    """
    Implements a simple volatility estimator using an exponentially weighted rolling window standard deviation on returns.
    TODO: Test this function

    Parameters
    ----------
    timestamps : np.array(np.int64)
        Raw trade timestamps in nanoseconds, sorted in ascending order.
    close : np.array(np.float64)
        Raw trade prices corresponding to the timestamps.
    return_window_sec : int
        The window size in seconds for the lagged return calculation.
    lookback : int, optional
        The number of observations to look back for the volatility estimate (span for the EWM_STD), by default 100

    Returns
    -------
    np.array(np.float64)
        The simple volatility estimate.

    Notes
    -----
    The function works for arbitrary time series data and does not require a fixed frequency.
    """
    if len(timestamps) != len(close):
        raise ValueError("The length of timestamps and close prices must be the same.")

    n = len(close)
    returns = np.empty(n, dtype=np.float64)
    returns.fill(np.nan)
    vol_estimates = np.empty(n, dtype=np.float64)
    vol_estimates.fill(np.nan)

    ret_window_ns = return_window_sec * 1e9

    for i in range(n):
        target_time = timestamps[i] - ret_window_ns
        lag_idx = np.searchsorted(timestamps, target_time, side='right') - 1
        if lag_idx >= 0:
            if close[lag_idx] != 0.0:
                ret = close[i] / close[lag_idx] - 1.0
                returns[i] = ret
            else:
                returns[i] = np.nan
        else:
            returns[i] = np.nan

    # Apply EWM standard deviation over the returns
    vol_estimates = ewms(returns, lookback)

    return vol_estimates
