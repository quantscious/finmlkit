"""
Implements mean reversion features.
"""
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange


@njit(nogil=True)
def vwap_distance(close: NDArray[np.float64],
                  volume: NDArray[np.float64],
                  n_periods: int,
                  is_log: bool
                  ) -> NDArray[np.float64]:
    """
    Calculate the distance of the current price from the VWAP (Volume Weighted Average Price).
    The VWAP is calculated over a specified number of periods, and the distance is expressed as a percentage.
    :param close: close price series
    :param volume: corresponding volume series
    :param n_periods: number of periods to calculate VWAP
    :param is_log: if True, calculate log distance
    :return: array of distances from VWAP
    """
    n = close.size
    out = np.empty(n, np.float64)
    out[:n_periods] = np.nan        # fill first window

    if n < n_periods:
        return out

    wsum = 0.0     # Σ P·V
    vsum = 0.0     # Σ V

    # initial window
    for k in range(n_periods):
        wsum += close[k] * volume[k]
        vsum += volume[k]

    if vsum > 0:
        out[n_periods-1] = close[n_periods-1] / (wsum / vsum) - 1.0

    # rolling update
    for i in range(n_periods, n):
        left_pv = close[i - n_periods] * volume[i - n_periods]
        wsum += close[i] * volume[i] - left_pv
        vsum += volume[i] - volume[i - n_periods]

        if vsum > 0:
            if is_log:
                out[i] = np.log(close[i] / (wsum/vsum))
            else:
                out[i] = close[i] / (wsum / vsum) - 1.0
        else:               # empty 4-h window (extremely rare)
            out[i] = out[i-1]

    return out
