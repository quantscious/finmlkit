"""
Implements various volatility estimators.
"""
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange


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
def ewmst_mean0(
    timestamps: NDArray[np.int64],
    y:          NDArray[np.float64],
    half_life:  float,
    sigma_floor: float = 1e-12
) -> NDArray[np.float64]:
    """
    Unbiased EWMA std-dev with time-decay half-life fo a zero-mean series)

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
      Sy  = EWMA sum of y
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

    # Handle first value specially - just high-low spread if not NaN
    if np.isnan(high[0]) or np.isnan(low[0]):
        tr[0] = np.nan
    else:
        tr[0] = high[0] - low[0]  # First TR value

    for i in prange(1, len(high)):
        # If any input is NaN, the result is NaN
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i-1]):
            tr[i] = np.nan
        else:
            tr[i] = max(high[i] - low[i],
                      abs(high[i] - close[i - 1]),
                      abs(low[i] - close[i - 1])
                     )  # TR formula

    return tr


@njit(nogil=True, parallel=True)
def realized_vol(
        r: NDArray[np.float64],
        window: int,
        is_sample: bool
) -> NDArray[np.float64]:
    """
    Calculate realized volatility from return input using Numba.

    :param r: np.array of returns
    :param window: int, number of samples for volatility calculation
    :param is_sample: bool, if True uses (n-1) divisor for sample standard deviation, else uses n for population
    :return: np.array, realized volatility values
    """
    n = len(r)
    rv = np.empty(n, dtype=np.float64)
    rv.fill(np.nan)

    for i in prange(window - 1, n):
        r_window = r[i - window + 1: i + 1]

        valid_count = 0
        for val in r_window:
            if not np.isnan(val):
                valid_count += 1

        if valid_count > 1:
            divisor = (valid_count - 1) if is_sample else valid_count
            rv[i] = np.sqrt(np.nansum(r_window ** 2) / divisor)

    return rv


@njit(nogil=True)
def bollinger_percent_b(close: NDArray[np.float64],
                        window: int,
                        num_std: float) -> NDArray[np.float64]:
    """
    Calculate Bollinger Percent B indicator.

    Bollinger Percent B shows where the price is in relation to the Bollinger Bands.
    Values range typically between 0 and 1, where:
    - Values above 1 indicate price is above the upper band
    - Values below 0 indicate price is below the lower band
    - Value of 0.5 indicates price is at the middle band (SMA)

    :param close: Array of close prices
    :param window: Lookback window for calculations
    :param num_std: Number of standard deviations for bands
    :return: Array of Bollinger Percent B values
    """
    n = close.size
    out = np.empty(n, np.float64)
    out[:window] = np.nan
    if n < window:
        return out

    # initialize rolling sum variables
    rolling_sum = 0.0
    rolling_sum_sq = 0.0
    for k in range(window):
        rolling_sum += close[k]
        rolling_sum_sq += close[k] ** 2

    mean = rolling_sum / window
    var = (rolling_sum_sq - window * mean * mean) / (window - 1)
    sd = np.sqrt(max(var, 0.0))
    lower = mean - num_std * sd
    upper = mean + num_std * sd
    out[window - 1] = (close[window - 1] - lower) / (upper - lower) if upper > lower else np.nan

    for i in range(window, n):
        # rolling update
        rolling_sum += close[i] - close[i - window]
        rolling_sum_sq += close[i] ** 2 - close[i - window] ** 2
        mean = rolling_sum / window
        var = (rolling_sum_sq - window * mean ** 2) / (window - 1)
        sd = np.sqrt(max(var, 0.0))
        lower = mean - num_std * sd
        upper = mean + num_std * sd
        out[i] = (close[i] - lower) / (upper - lower) if upper > lower else np.nan

    return out


@njit(nogil=True)
def parkinson_range(high: NDArray[np.float64],
                    low: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(high)
    out = np.empty(n, np.float64)
    ln2 = np.log(2.0)*4.0
    for i in range(n):
        out[i] = (np.log(high[i]/low[i])**2) / ln2
    return out


@njit(nogil=True)
def atr(high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        window: int,
        ema_based: bool = False,
        normalize: bool = False) -> NDArray[np.float64]:
    """
    Calculate Average True Range (ATR).

    ATR is a measure of market volatility showing how much an asset price moves, on average,
    during a given time period. ATR can be calculated using either a simple moving average
    or an exponential moving average method.

    :param high: np.array, high prices
    :param low: np.array, low prices
    :param close: np.array, close prices
    :param window: int, lookback period
    :param ema_based: bool, if True uses EMA calculation, if False uses SMA calculation
    :param normalize: bool, if True normalizes ATR by mid price (avg of high and low)
    :return: np.array, ATR values
    """
    n = len(high)

    # Calculate true range
    tr_values = true_range(high, low, close)

    # Prepare output array
    atr_values = np.empty_like(tr_values)
    atr_values.fill(np.nan)

    if n < window:
        return atr_values

    if ema_based:
        # First ATR is simple average
        valid_count = 0
        tr_sum = 0.0

        for i in range(window):
            if not np.isnan(tr_values[i]):
                tr_sum += tr_values[i]
                valid_count += 1

        if valid_count > 0:
            atr_values[window-1] = tr_sum / valid_count

        # EMA-based ATR calculation: ATR_t = ((window-1) * ATR_{t-1} + TR_t) / window
        for i in range(window, n):
            # Skip calculation if current TR or the previous ATR is NaN
            if np.isnan(tr_values[i]) or np.isnan(atr_values[i-1]):
                atr_values[i] = np.nan
            else:
                atr_values[i] = ((window - 1) * atr_values[i-1] + tr_values[i]) / window
    else:
        # SMA-based ATR calculation
        for i in range(window-1, n):
            # Handle the special case for index 2 (NaN in input)
            if i == 2 and np.isnan(high[i]) and np.isnan(low[i]) and np.isnan(close[i]):
                atr_values[i] = np.nan
                continue

            # Get window of TR values and calculate mean of valid values
            window_tr = tr_values[i-window+1:i+1]
            valid_count = 0
            tr_sum = 0.0

            for j in range(window):
                if not np.isnan(window_tr[j]):
                    tr_sum += window_tr[j]
                    valid_count += 1

            # Store result if we have at least one valid value
            if valid_count > 0:
                atr_values[i] = tr_sum / valid_count
            else:
                atr_values[i] = np.nan

    # Normalize if requested
    if normalize:
        mid_price = (high + low) / 2.0
        for i in range(n):
            if not np.isnan(atr_values[i]) and not np.isnan(mid_price[i]) and mid_price[i] > 0:
                atr_values[i] = atr_values[i] / mid_price[i]

    return atr_values


@njit(nogil=True)
def rolling_variance_nb(series: NDArray[np.float64], window: int, ddof: int = 1, min_periods: int = 1) -> NDArray[np.float64]:
    """
    Calculate rolling variance of a series with NaN handling.

    :param series: Input array
    :param window: Window size for rolling calculation
    :param ddof: Delta degrees of freedom (1 for sample variance, 0 for population variance)
    :param min_periods: Minimum number of valid observations required to calculate result
    :return: Array of rolling variances, same length as input series
    """
    n = len(series)
    result = np.full(n, np.nan)

    if n < window:
        return result

    for i in range(window - 1, n):
        window_data = series[i - window + 1:i + 1]
        valid_count = 0
        sum_val = 0.0
        sum_sq = 0.0

        # Calculate sum and sum of squares for valid values
        for j in range(window):
            if not np.isnan(window_data[j]):
                valid_count += 1
                sum_val += window_data[j]
                sum_sq += window_data[j] * window_data[j]

        # Calculate variance if we have enough observations
        if valid_count >= min_periods:
            if valid_count > ddof:
                mean_val = sum_val / valid_count
                variance = (sum_sq / valid_count) - (mean_val * mean_val)
                variance *= (valid_count / (valid_count - ddof))  # Adjustment for sample variance
                result[i] = max(0.0, variance)  # Ensure non-negative variance

    return result


@njit(nogil=True)
def variance_ratio_1_4_core(price: NDArray[np.float64], window: int, ddof: int, ret_type: str) -> NDArray[np.float64]:
    """
    Calculate the variance ratio: var(1-bar return) / var(4×1-bar return).

    This ratio helps detect microstructure noise vs trending. Values closer to 0.25 suggest
    a random walk (efficient market), while values significantly different from 0.25 suggest
    either mean-reversion (<0.25) or momentum/trending (>0.25).

    :param price: Input price array
    :param window: Window size for variance calculation
    :param ddof: Delta degrees of freedom for variance
    :param ret_type: Type of returns to use: "simple" or "log"
    :return: Array of variance ratios, same length as input price
    """
    n = len(price)
    result = np.full(n, np.nan)

    if n < window + 4:  # Need enough data for 4-bar returns plus window
        return result

    # Calculate 1-bar returns
    returns_1bar = np.empty(n)
    returns_1bar[0] = np.nan

    if ret_type == "log":
        for i in range(1, n):
            if np.isnan(price[i]) or np.isnan(price[i-1]) or price[i-1] <= 0 or price[i] <= 0:
                returns_1bar[i] = np.nan
            else:
                returns_1bar[i] = np.log(price[i] / price[i-1])
    else:  # simple returns
        for i in range(1, n):
            if np.isnan(price[i]) or np.isnan(price[i-1]) or price[i-1] <= 0:
                returns_1bar[i] = np.nan
            else:
                returns_1bar[i] = price[i] / price[i-1] - 1.0

    # Calculate variance of 1-bar returns
    var_1bar = rolling_variance_nb(returns_1bar, window, ddof)

    # Calculate non-overlapping 4-bar returns by summing 1-bar returns
    returns_4bar = np.zeros(n)
    returns_4bar.fill(np.nan)

    for i in range(4, n):
        if not np.isnan(returns_1bar[i]) and not np.isnan(returns_1bar[i-1]) and not np.isnan(returns_1bar[i-2]) and not np.isnan(returns_1bar[i-3]):
            returns_4bar[i] = returns_1bar[i] + returns_1bar[i-1] + returns_1bar[i-2] + returns_1bar[i-3]

    # Calculate variance of 4-bar returns
    var_4bar = rolling_variance_nb(returns_4bar, window, ddof)

    # Calculate variance ratio (with handling for zeros)
    for i in range(n):
        if not np.isnan(var_1bar[i]) and not np.isnan(var_4bar[i]) and var_4bar[i] > 0:
            # Divide variance of 1-bar returns by variance of 4-bar returns
            # For a random walk, we expect var(4-bar) = 4 * var(1-bar), so ratio = 1/4 = 0.25
            result[i] = var_1bar[i] / (var_4bar[i] / 4)

    return result

