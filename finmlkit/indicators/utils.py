"""
General utility functions not exclusively for OHLCV bars data.
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(nopython=True, nogil=True, fastmath=True)
def ewma(y: NDArray, window: int) -> NDArray[np.float64]:
    """
    Exponentially weighted moving average (EWMA) of a one-dimensional numpy array.
    Calculates https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html with adjust=True.
    By using this weighting scheme, the function provides a more accurate and unbiased estimate of the EWMA,
    especially in the early stages of the data series.

    Parameters
    ----------
    y : np.ndarray
        A one-dimensional numpy array of floats.
    window : int
        The decay window, or 'span'. Determines how many past points meaningfully impact the given EWMA value.

    Returns
    -------
    ewma_y : np.ndarray(np.float64)
        The EWMA vector, same length and shape as `y`.

    Notes
    -----
    This function adjusts for small sample sizes by dividing by the cumulative weight.
    """
    n = y.shape[0]
    ewma = np.empty(n, dtype=np.float64)

    # If window is less than 1, raise an error
    if window < 1:
        raise ValueError("Window size is less than or equal to 1. Please provide a window size greater than 1.")

    alpha = 2.0 / (window + 1.0)
    ewma_prev = y[0]
    ewma[0] = ewma_prev
    weight = 1.0

    for i in range(1, n):
        ewma_prev = ewma_prev * (1.0 - alpha) + y[i]
        weight = weight * (1.0 - alpha) + 1.0
        ewma[i] = ewma_prev / weight

    return ewma


@njit(nopython=True, nogil=True, fastmath=True)
def ewms(y: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """
    Calculates the Exponentially Weighted Moving Standard Deviation (EWM_STD) of a one-dimensional numpy array.
    Similar to pandas' ewm.std with adjust=True.


    Parameters
    ----------
    y : np.ndarray
        A one-dimensional numpy array of floats.
    window : int
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

    # If window is less equal to 1, return NaNs (to mimic pandas behavior)
    if window == 1:
        ewm_std[:] = np.nan
        print("WARNING! Window size is equal to 1. Returning NaNs.")
        return ewm_std
    elif window < 1:
        raise ValueError("Window size is less than 1. Please provide a window size greater than 1.")

    alpha = 2.0 / (window + 1.0)
    beta = 1.0 - alpha

    # Initialize cumulative sums and weights
    S_y = 0.0  # Cumulative sum of weighted y
    S_y2 = 0.0  # Cumulative sum of weighted y^2
    S_w = 0.0  # Cumulative sum of weights
    S_w2 = 0.0  # Cumulative sum of squared weights

    for i in range(n):
        w_i = beta ** (n - i - 1)
        S_w += w_i
        S_w2 += w_i ** 2
        S_y += w_i * y[i]
        S_y2 += w_i * y[i] ** 2

        if S_w > 0.0 and (S_w - S_w2 / S_w) > 0.0:
            mean = S_y / S_w
            variance = (S_y2 / S_w - mean ** 2) * S_w / (S_w - S_w2 / S_w)
            variance = max(variance, 0.0)
            ewm_std[i] = np.sqrt(variance)
        else:
            ewm_std[i] = np.nan

    return ewm_std
