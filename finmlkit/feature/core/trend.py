"""
Trend indicators for financial time series analysis.
"""
import numpy as np
from numba import njit


@njit
def adx_core(high, low, close, length):
    """
    Calculate the Average Directional Index (ADX) with specified length.

    The ADX is a technical analysis indicator used to determine the strength of a trend.
    Higher values indicate a stronger trend (regardless of direction).

    :param high: Array of high prices
    :param low: Array of low prices
    :param close: Array of close prices
    :param length: Period for ADX calculation
    :return: ADX indicator values
    """
    size = len(high)

    # Arrays to store True Range, +DM, -DM
    tr = np.zeros(size)
    plus_dm = np.zeros(size)
    minus_dm = np.zeros(size)

    # Calculate True Range, +DM, -DM
    for i in range(1, size):
        # True Range = max(high[i] - low[i], |high[i] - close[i-1]|, |low[i] - close[i-1]|)
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

        # Directional Movement
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]

        # +DM
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0

        # -DM
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0

    # Initialize smoothed values using Wilder's smoothing method
    smoothed_tr = np.zeros(size)
    smoothed_plus_dm = np.zeros(size)
    smoothed_minus_dm = np.zeros(size)

    # First calculation is simple average
    if size >= length + 1:
        smoothed_tr[length] = np.sum(tr[1:length+1])
        smoothed_plus_dm[length] = np.sum(plus_dm[1:length+1])
        smoothed_minus_dm[length] = np.sum(minus_dm[1:length+1])

    # Apply Wilder's smoothing for subsequent values
    for i in range(length + 1, size):
        smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / length) + tr[i]
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / length) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / length) + minus_dm[i]

    # Calculate +DI, -DI
    plus_di = np.zeros(size)
    minus_di = np.zeros(size)
    for i in range(length, size):
        if smoothed_tr[i] > 0:
            plus_di[i] = 100 * (smoothed_plus_dm[i] / smoothed_tr[i])
            minus_di[i] = 100 * (smoothed_minus_dm[i] / smoothed_tr[i])

    # Calculate DX
    dx = np.zeros(size)
    for i in range(length, size):
        if (plus_di[i] + minus_di[i]) > 0:
            dx[i] = 100 * (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]))

    # Calculate ADX
    adx = np.zeros(size)

    # First ADX is simple average of DX
    if size >= 2*length:
        adx[2*length-1] = np.mean(dx[length:2*length])

    # Apply Wilder's smoothing for subsequent ADX values
    for i in range(2*length, size):
        adx[i] = ((adx[i-1] * (length-1)) + dx[i]) / length

    return adx
