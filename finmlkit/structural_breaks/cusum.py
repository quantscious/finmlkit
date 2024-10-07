import numpy as np
from numba import njit
from numba import prange
from typing import Tuple
from numpy.typing import NDArray


@njit(nogil=True, parallel=False)
def chu_stinchcombe_white_developing(y: NDArray[np.float64], warmup_period: int = 30) -> (
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """
    Perform the Chu-Stinchcombe-White CUSUM Test on Levels.

    This function detects structural breaks in a financial time series by testing the null hypothesis
    that the series follows a geometric Brownian motion (GBM). It calculates one-sided test statistics
    for upward and downward movements separately, as well as a two-sided test using absolute values.

    Parameters
    ----------
    y : np.ndarray
        Array of price series (log prices).
    warmup_period : int, optional
        Number of initial observations to discard, by default 30.

    Returns
    -------
    s_n_t_values_up : np.ndarray
        Test statistic values for upward movements (one-sided test).
    s_n_t_values_down : np.ndarray
        Test statistic values for downward movements (one-sided test).
    s_n_t_values_abs : np.ndarray
        Test statistic values for absolute movements (two-sided test).
    critical_values : np.ndarray
        Critical values for the test statistics.

    Notes
    -----
    The test detects deviations from the random walk hypothesis in the time series.
    A significant test statistic indicates a structural break in the time series.

    """
    y = np.asarray(y, dtype=np.float64)
    y = np.log(y)

    n = len(y)

    s_n_t_values_up = np.empty(n, dtype=np.float64)
    s_n_t_values_down = np.empty(n, dtype=np.float64)
    s_n_t_values_abs = np.empty(n, dtype=np.float64)
    critical_values = np.empty(n, dtype=np.float64)

    # Initialize arrays with NaNs
    s_n_t_values_up.fill(np.nan)
    s_n_t_values_down.fill(np.nan)
    s_n_t_values_abs.fill(np.nan)
    critical_values.fill(np.nan)

    cum_squared_diff = np.cumsum(np.diff(y) ** 2)

    for t in prange(warmup_period, n):
        sigma_sq_t = np.sqrt(cum_squared_diff[t - 1] / (t - 1))

        max_s_n_value_up = -np.inf
        max_s_n_value_abs = -np.inf
        max_s_n_value_down = np.inf
        max_s_n_critical_value = np.nan

        for k in range(1, t - 1):
            dyk = y[t] - y[k]

            s_n_t_up = max(0, dyk) / (sigma_sq_t * np.sqrt(t - k) + 1e-12)
            s_n_t_down = min(0, dyk) / (sigma_sq_t * np.sqrt(t - k) + 1e-12)
            s_n_t_abs = np.abs(dyk) / (sigma_sq_t * np.sqrt(t - k) + 1e-12)

            if s_n_t_up > max_s_n_value_up:
                max_s_n_value_up = s_n_t_up

            if s_n_t_down < max_s_n_value_down:
                max_s_n_value_down = s_n_t_down

            if s_n_t_abs > max_s_n_value_abs:
                max_s_n_value_abs = s_n_t_abs
                max_s_n_critical_value = np.sqrt(4.6 + np.log(t / k))

        s_n_t_values_up[t] = max_s_n_value_up
        s_n_t_values_down[t] = -max_s_n_value_down  # Flip sign to make it positive
        s_n_t_values_abs[t] = max_s_n_value_abs
        critical_values[t] = max_s_n_critical_value

    return s_n_t_values_up, s_n_t_values_down, s_n_t_values_abs, critical_values


@njit(nogil=True)
def chu_stinchcombe_white_last(y: NDArray[np.float64]) -> Tuple[float, float, float, float]:
    """
    Perform the Chu-Stinchcombe-White CUSUM Test on Levels for the last observation.

    This function detects structural breaks in a financial time series by testing the null hypothesis
    that the series follows a geometric Brownian motion (GBM). It calculates the test statistics
    only for the last observation in the series.

    Parameters
    ----------
    y : np.ndarray
        Array of price series (log prices).

    Returns
    -------
    max_s_n_value_up : float
        Test statistic value for upward movement (one-sided test) at the last observation.
    max_s_n_value_down : float
        Test statistic value for downward movement (one-sided test) at the last observation.
    max_s_n_value_abs : float
        Test statistic value for absolute movement (two-sided test) at the last observation.
    max_s_n_critical_value : float
        Critical value for the test statistic at the last observation.

    Notes
    -----
    A significant test statistic at the last observation indicates a recent structural break
    in the time series.

    """
    y = np.asarray(y, dtype=np.float64)
    y = np.log(y)

    n = len(y)

    cum_squared_diff = np.cumsum(np.diff(y) ** 2)

    # Calculate only for the latest time point
    t = n - 1  # t is the last index

    sigma_sq_t = np.sqrt(cum_squared_diff[t - 1] / (t - 1))

    max_s_n_value_up = -np.inf
    max_s_n_value_abs = -np.inf
    max_s_n_value_down = np.inf
    max_s_n_critical_value = np.nan

    # Calculate statistics for each window from 0 to t-1
    for k in range(1, t - 1):
        dyk = y[t] - y[k]

        s_n_t_up = max(0, dyk) / (sigma_sq_t * np.sqrt(t - k) + 1e-12)
        s_n_t_down = min(0, dyk) / (sigma_sq_t * np.sqrt(t - k) + 1e-12)
        s_n_t_abs = np.abs(dyk) / (sigma_sq_t * np.sqrt(t - k) + 1e-12)

        if s_n_t_up > max_s_n_value_up:
            max_s_n_value_up = s_n_t_up

        if s_n_t_down < max_s_n_value_down:
            max_s_n_value_down = s_n_t_down

        if s_n_t_abs > max_s_n_value_abs:
            max_s_n_value_abs = s_n_t_abs
            max_s_n_critical_value = np.sqrt(4.6 + np.log(t / k))

    # Flip the down value to make it positive
    max_s_n_value_down = -max_s_n_value_down

    return max_s_n_value_up, max_s_n_value_down, max_s_n_value_abs, max_s_n_critical_value


@njit(nogil=True, parallel=True)
def chu_stinchcombe_white_rolling(close_prices: NDArray[np.float64], window_size: int = 1000, warmup_period: int = 30) -> (
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]):
    """
    Perform the Chu-Stinchcombe-White CUSUM Test on Levels over a rolling window.

    This function applies the Chu-Stinchcombe-White CUSUM test to a rolling window over the time series,
    allowing for detection of structural breaks over time.

    Parameters
    ----------
    close_prices : np.ndarray
        Array of close price series.
    window_size : int, optional
        Size of the rolling window, by default 1000.
    warmup_period : int, optional
        Minimum number of observations before the first statistic is calculated, by default 30.

    Returns
    -------
    snt_up : np.ndarray
        Test statistic values for upward movements (one-sided test) over the rolling window.
    snt_down : np.ndarray
        Test statistic values for downward movements (one-sided test) over the rolling window.
    snt_abs : np.ndarray
        Test statistic values for absolute movements (two-sided test) over the rolling window.
    critical_values : np.ndarray
        Critical values for the test statistics over the rolling window.

    Notes
    -----
    This test detects deviations from the random walk hypothesis in the time series over a moving window.
    It helps identify periods of structural breaks in the time series.

    """
    close_prices = np.asarray(close_prices, dtype=np.float64)

    n = len(close_prices)
    snt_up = np.empty(n)
    snt_down = np.empty(n)
    snt_abs = np.empty(n)
    critical_values = np.empty(n)

    # Initialize arrays with NaNs
    snt_up.fill(np.nan)
    snt_down.fill(np.nan)
    snt_abs.fill(np.nan)
    critical_values.fill(np.nan)

    if n > window_size:
        for current_idx in prange(window_size, n):
            start_idx = current_idx - window_size
            current_prices = close_prices[start_idx:current_idx + 1]

            if start_idx == 0:
                # First window after warmup period, compute initial values
                s_n_t_values_up, s_n_t_values_down, s_n_t_values_abs, c_values = chu_stinchcombe_white_developing(current_prices, warmup_period)
                snt_up[start_idx:current_idx + 1] = s_n_t_values_up
                snt_down[start_idx:current_idx + 1] = s_n_t_values_down
                snt_abs[start_idx:current_idx + 1] = s_n_t_values_abs
                critical_values[start_idx:current_idx + 1] = c_values
            else:
                # Subsequent windows, compute and store only the last value
                s_n_t_up, s_n_t_down, s_n_t_abs, c_values = chu_stinchcombe_white_last(current_prices)
                snt_up[current_idx] = s_n_t_up
                snt_down[current_idx] = s_n_t_down
                snt_abs[current_idx] = s_n_t_abs
                critical_values[current_idx] = c_values
    else:
        # Case when the window size is larger than the data
        snt_up, snt_down, snt_abs, critical_values = chu_stinchcombe_white_developing(close_prices, warmup_period)

    return snt_up, snt_down, snt_abs, critical_values
