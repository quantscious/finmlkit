"""
Implements various volatility estimators.
"""
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from .utils import compute_lagged_returns, logger


@njit(nogil=True)
def ewms(y: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    """
    Calculates the Exponentially Weighted Moving Standard Deviation (EWM_STD) of a one-dimensional numpy array.
    Equivalent to `pandas.Series.ewm(...).std()` with `adjust=True` and `bias=False`.

    :param y: A one-dimensional numpy array of floats.
    :param span: The decay window, or 'span'.
    :returns: The EWM standard deviation vector, same length and shape as `y`.

    .. note::
        This function adjusts for small sample sizes by dividing by the cumulative weight minus the sum of squared weights
        divided by the cumulative weight, matching the behavior of `adjust=True` and `bias=False` in pandas.
    """
    n = y.shape[0]
    ewm_std = np.empty(n, dtype=np.float64)

    if span <= 1:
        ewm_std[:] = np.nan
        print("Span size is less than or equal to 1. Returning NaNs.")
        return ewm_std

    alpha = 2.0 / (span + 1.0)
    one_minus_alpha = 1.0 - alpha

    # Initialize cumulative sums and weights
    S_w = 0.0    # Cumulative sum of weights
    S_w2 = 0.0   # Cumulative sum of squared weights
    S_y = 0.0    # Cumulative sum of weighted y
    S_y2 = 0.0   # Cumulative sum of weighted y^2

    for t in range(n):
        y_t = y[t]

        # Update cumulative weights regardless of NaN
        S_w = one_minus_alpha * S_w + (0.0 if np.isnan(y_t) else 1.0)
        S_w2 = (one_minus_alpha ** 2) * S_w2 + (0.0 if np.isnan(y_t) else 1.0)

        # Update cumulative sums
        if not np.isnan(y_t):
            S_y = one_minus_alpha * S_y + y_t
            S_y2 = one_minus_alpha * S_y2 + y_t ** 2
        else:
            # Decay the cumulative sums without adding new data
            S_y = one_minus_alpha * S_y
            S_y2 = one_minus_alpha * S_y2

        # Calculate mean and variance if cumulative weight is positive
        if S_w > 0.0:
            mean = S_y / S_w
            denominator = S_w - (S_w2 / S_w)
            if denominator > 0.0:
                variance = (S_y2 / S_w - mean ** 2) * S_w / denominator
                variance = max(variance, 0.0)  # Ensure non-negative variance
                ewm_std[t] = np.sqrt(variance)
            else:
                ewm_std[t] = np.nan
        else:
            ewm_std[t] = np.nan

    return ewm_std


@njit(nogil=True)
def ewmst_return(
    timestamps: NDArray[np.int64],
    y:          NDArray[np.float64],
    half_life:  float,
    sigma_floor: float = 1e-12
) -> NDArray[np.float64]:
    """
    Unbiased EWMA std-dev with time-decay half-life on returns (zero-mean series).

    σ_t² = U_t / V_t  with
      U_t = α_t * y_t² + (1-α_t) * U_{t-1}
      V_t = α_t       + (1-α_t) * V_{t-1}

    where α_t = 1 - exp(-Δt / half_life), Δt in seconds
    and y_t is your return at timestamp[t].

    :param timestamps: 1D array of event times in nanoseconds.
    :param y:          1D array of floats (e.g. lagged returns).
    :param half_life:  Decay half-life in seconds.
    :param sigma_floor: Minimum σ to enforce stability.
    :returns:          EWMA standard deviation array.
    """
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    # initialize
    U = 0.0
    V = 0.0
    last_ts = timestamps[0]

    # first point: can't compute Δt, mark NaN or zero
    out[0] = np.nan

    for i in range(1, n):
        # time delta in seconds
        dt = (timestamps[i] - last_ts) / 1e9
        last_ts = timestamps[i]

        # decay factor
        alpha = 1.0 - np.exp(-dt / half_life)

        # update U,V with unbiased recursion
        y_t = y[i]
        if np.isnan(y_t):
            # decay only
            U = (1.0 - alpha) * U
            V = (1.0 - alpha) * V
        else:
            U = alpha * (y_t * y_t) + (1.0 - alpha) * U
            V = alpha             + (1.0 - alpha) * V

        # compute variance & floor
        var = U / V if V > 0.0 else np.nan
        if var < 0.0:
            var = 0.0
        sigma = np.sqrt(var)
        if sigma < sigma_floor:
            sigma = sigma_floor

        out[i] = sigma

    return out


@njit(nogil=True)
def ewmst(
    timestamps: NDArray[np.int64],
    y:          NDArray[np.float64],
    half_life:  float,
    sigma_floor: float = 1e-12
) -> NDArray[np.float64]:
    """
    Unbiased time-decay EWMA std-dev (adjust=True, bias=False semantics).

    Maintains:
      V  = sum of weights
      V2 = sum of squared weights
      Sy = EWMA sum of y
      Syy = EWMA sum of y^2

    Then
      mean_t   = Sy / V
      ewma_y2  = Syy / V
      var_raw  = ewma_y2 - mean_t^2
      denom    = V - V2 / V
      var_t    = var_raw * (V / denom)
      σ_t      = sqrt(max(var_t, 0))
    """
    n = y.shape[0]
    out = np.empty(n, np.float64)
    if n == 0:
        return out

    # Initialize
    V   = 0.0    # sum of weights
    V2  = 0.0    # sum of squared weights
    Sy  = 0.0    # EWMA sum of y
    Syy = 0.0    # EWMA sum of y^2
    last_ts = timestamps[0]
    out[0] = np.nan

    for i in range(1, n):
        # time delta in seconds
        dt = (timestamps[i] - last_ts) / 1e9
        last_ts = timestamps[i]

        # decay factor
        alpha = 1.0 - np.exp(-dt / half_life)
        one_minus = 1.0 - alpha

        yi = y[i]

        # update weight sums
        V  = alpha       + one_minus * V
        V2 = alpha*alpha + (one_minus*one_minus) * V2

        # update EWMA sums
        if np.isnan(yi):
            # decay only
            Sy  = one_minus * Sy
            Syy = one_minus * Syy
        else:
            Sy  = alpha * yi        + one_minus * Sy
            Syy = alpha * yi * yi   + one_minus * Syy

        # compute mean & raw variance
        if V > 0.0:
            mean   = Sy  / V
            ewma_y2= Syy / V
            var_raw= ewma_y2 - mean*mean
            # bias correction
            denom = V - (V2 / V) if V > 0.0 else 0.0
            if denom > 0.0 and var_raw > 0.0:
                var = var_raw * (V / denom)
            else:
                var = 0.0

            sigma = np.sqrt(var)
            if sigma < sigma_floor:
                sigma = sigma_floor
            out[i] = sigma
        else:
            out[i] = np.nan

    return out


@njit(nogil=True, parallel=True)
def standard_volatility_estimator(
        timestamps: NDArray[np.int64],
        close: NDArray[np.float64],
        return_window_sec: float,
        half_life_sec: int = 1800
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
    :param half_life_sec: Half life for the exponentially weighted moving standard deviation in seconds.
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
    vol_estimates = ewmst(returns, half_life_sec)

    return vol_estimates


@njit(nogil=True, parallel=True)
def true_range(high: NDArray, low: NDArray, close: NDArray) -> NDArray:
    """
    Calculate True Range using Numba.

    :param high: np.array, high prices
    :param low: np.array, low prices
    :param close: np.array, close prices
    :return: np.array, True Range values
    """
    if not (len(high) == len(low) == len(close)):
        raise ValueError("The length of high, low, and close prices must be the same.")

    tr = np.empty_like(high)
    tr[0] = high[0] - low[0]  # First TR value

    for i in prange(1, len(high)):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1])
                    )  # TR formula

    return tr

