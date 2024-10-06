from typing import Literal
from numba import njit
from numpy.typing import NDArray
import numpy as np

type float64 = float | np.float64  # Float type alias

@njit(nopython=True, nogil=True, fastmath=True)
def comp_trade_side(price: float64, prev_price: float64, prev_tick: int) -> Literal[-1, 1]:
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
        1 for an upward change (buy), -1 for a downward change (sell)
    """
    epsilon = 1e-12  # Small value to avoid floating point error
    dp = price - prev_price

    return np.sign(dp) if abs(dp) > epsilon else prev_tick


@njit(nopython=True, nogil=True)
def comp_trade_side_vector(prices: NDArray[np.float64]) -> NDArray[np.int8]:
    """
    Calculate the trade side for a sequence of raw trades prices.

    Parameters
    ----------
    prices : np.array(np.float64)
        The raw trades data prices sequence.

    Returns
    -------
    np.ndarray(np.int8)
        The sequence of trade sides (1 for buy, -1 for sell).
    """
    n = len(prices)
    trade_sides = np.zeros(n, dtype=np.int8)

    prev_tick_sign = 0
    prev_price = np.float64(prices[0])  # Explicit cast to avoid warnings

    for i in range(1, n):
        price = np.float64(prices[i])  # Explicit cast to avoid warnings
        prev_tick_sign = comp_trade_side(price, prev_price, prev_tick_sign)
        trade_sides[i] = prev_tick_sign
        prev_price = price

    return trade_sides


@njit(nopython=True, nogil=True)
def comp_price_tick_size(prices: NDArray[np.float64]) -> float:
    """
    Compute the price tick size from raw trades prices data.

    Parameters
    ----------
    prices : np.array(np.float64)
        Raw trades prices.

    Returns
    -------
    float
        The price tick size. Returns 0.0 if not determinable from the input prices.

    """
    # Select first 10000 trades to calculate the price tick size
    n_samples = len(prices)
    price_sample = prices[:min(10000, n_samples)]

    # raise value error if prices is empty
    if len(price_sample) == 0:
        raise ValueError("Empty prices array")

    # Round the prices to mitigate floating-point errors
    price_sample = np.round(price_sample, decimals=12)

    # Get the sorted unique prices
    unique_prices = np.unique(price_sample)
    if len(unique_prices) <= 1:
        # No variation in prices; tick size is zero
        return 0.0

    # Calculate the median price tick size
    price_tick_size = np.median(np.diff(unique_prices))

    if price_tick_size == 0.0:
        # Avoid computing log10(0)
        return 0.0

    # Determine the exponent for adaptive rounding
    exponent = np.floor(np.log10(abs(price_tick_size)))

    # Specify the desired number of significant digits
    desired_significant_digits = 2

    # Calculate the number of decimal places to round to
    ndigits = int(desired_significant_digits - 1 - exponent)

    # Round the tick size adaptively based on the exponent
    rounded_tick_size = round(price_tick_size, ndigits)

    return rounded_tick_size
