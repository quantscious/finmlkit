import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(nogil=True)
def ewma(y: NDArray, span: int) -> NDArray[np.float64]:
    """
    Exponentially weighted moving average (EWMA) of a one-dimensional numpy array.
    Calculates the equivalent of `pandas.DataFrame.ewm(...).mean()` with `adjust=True`.

    By using this weighting scheme, the function provides a more accurate and unbiased estimate of the EWMA,
    especially in the early stages of the data series.

    :param y: A one-dimensional numpy array of floats.
    :param span: The decay window, or 'span'. Determines how many past points meaningfully impact the EWMA value.
    :returns: The EWMA vector, same length and shape as `y`.
    :raises ValueError: If `span` is less than 1.

    .. note::
        This function adjusts for small sample sizes by dividing by the cumulative weight.
        For more information, see: https://terbe.dev/blog/posts/exponentially-weighted-moving-average
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


@njit(nogil=True, parallel=True)
def sma(array: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """
    Calculate Simple Moving Average (SMA) with Numba for better performance.

    :param array: np.array, input array to calculate SMA
    :param window: int, window for the rolling average
    :return: np.array, SMA values
    """
    sma = np.empty_like(array)
    sma.fill(np.nan)

    N = window
    for i in prange(N-1, len(array)):
        sma[i] = 1./N * np.sum(array[i-N+1 : i+1], dtype=np.float64)

    return sma



