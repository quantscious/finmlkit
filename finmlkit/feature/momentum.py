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