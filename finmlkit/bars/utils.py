from typing import Literal
from numba import njit
import numpy as np


@njit(nopython=True)
def tick_rule(price: float, prev_price: float, prev_tick: int) -> Literal[-1, 1]:
    """
    Implements the tick rule defined on page 29 of "Advances in Financial Machine Learning". It classifies the
    trade as an aggressive buy or sell based on the direction of price change.

    Parameters
    ----------
    price : float
        The current price.
    prev_price : float
        The previous price.
    prev_tick : int
        The previous tick rule.

    Returns
    -------
    int
        The tick rule for the current price change direction (1 for an upward change, -1 for a downward change).
    """
    epsilon = 1e-12  # Small value to avoid floating point error
    dp = price - prev_price

    return np.sign(dp) if abs(dp) > epsilon else prev_tick

