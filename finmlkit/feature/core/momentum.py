from numba import njit, prange
import numpy as np
from numpy.typing import NDArray


@njit(nogil=True, parallel=True)
def roc(price: NDArray, period: int) -> NDArray:
    """
    Calculate the Rate of Change (ROC) feature.

    :param price: np.array, an array of prices.
    :param period: int, the period over which to calculate ROC.
    :return: np.array, ROC values.
    """
    n = len(price)
    roc = np.empty_like(price)
    roc[:period] = np.nan  # First 'period' values are NaN

    for i in range(period, n):
        roc[i] = ((price[i] - price[i - period]) / price[i - period]) * 100

    return roc


@njit(nogil=True)
def rsi_wilder(close: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """
    Calculate the Relative Strength Index (RSI) using Wilder's smoothing method.
    :param close: A one-dimensional numpy array of closing prices.
    :param window: The number of periods to use for the RSI calculation, default is 14.
    :return: A one-dimensional numpy array of RSI values, same length as `close`.
    """
    n = close.size
    out = np.empty(n, np.float64)
    out[:window] = np.nan

    if n <= window:
        return out                      # not enough data

    # ----- first window (indices 1 … window) -----
    sum_gain = 0.0
    sum_loss = 0.0
    for i in range(1, window + 1):
        diff = close[i] - close[i - 1]
        if diff > 0.0:
            sum_gain += diff
        else:
            sum_loss += -diff

    avg_gain = sum_gain / window
    avg_loss = sum_loss / window
    out[window] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) if avg_loss > 0 else np.nan

    # ----- Wilder smoothing -----
    for i in range(window + 1, n):
        diff = close[i] - close[i - 1]
        gain = diff if diff > 0.0 else 0.0
        loss = -diff if diff < 0.0 else 0.0

        avg_gain = ((window - 1) * avg_gain + gain) / window
        avg_loss = ((window - 1) * avg_loss + loss) / window

        out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) if avg_loss > 0 else np.nan

    return out


@njit(nogil=True)
def stoch_k(
        close: NDArray[np.float64],
        low: NDArray[np.float64],
        high: NDArray[np.float64],
        length: int
) -> NDArray[np.float64]:
    """
    Calculate the Stochastic Oscillator %K value.

    :param close: A one-dimensional numpy array of closing prices
    :param low: A one-dimensional numpy array of low prices
    :param high: A one-dimensional numpy array of high prices
    :param length: The lookback period for the stochastic calculation
    :return: A one-dimensional numpy array of %K values, same length as input arrays
    """
    n = close.size
    out = np.empty(n, np.float64)
    out[:length - 1] = np.nan

    if n < length:
        return out

    # initial window
    lo = np.min(low[:length])
    hi = np.max(high[:length])
    out[length - 1] = 100.0 * (close[length - 1] - lo) / (hi - lo) if hi > lo else np.nan

    # rolling update
    for t in range(length, n):
        # update window min/max efficiently
        # if the outgoing bar was the min/max we need full recompute;
        # otherwise only compare the new bar.
        out_idx = t - length

        if low[out_idx] == lo or high[out_idx] == hi:
            lo = np.min(low[t - length + 1:t + 1])
            hi = np.max(high[t - length + 1:t + 1])
        else:
            lo = min(lo, low[t])
            hi = max(hi, high[t])

        out[t] = 100.0 * (close[t] - lo) / (hi - lo) if hi > lo else np.nan

    return out