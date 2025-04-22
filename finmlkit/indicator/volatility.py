"""
Implements various volatility estimators.
"""
import numpy as np
from numpy.typing import NDArray
from numba import njit
from .utils import ewms, compute_lagged_returns

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
