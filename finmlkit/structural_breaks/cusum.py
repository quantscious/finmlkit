import numpy as np
from numba import njit
from numba import prange
from typing import Tuple
from numpy.typing import NDArray


@njit(nogil=True)
def _comp_max_s_nt(y: NDArray, t: int, sigma_sq_t: float) -> (
        Tuple[float, float, float, float]):
    """
    Compute the maximum S_n values and critical values for upward and downward movements.

    Parameters
    ----------
    y : np.ndarray
        Array of log price series.
    t : int
        Current time index.
    sigma_sq_t : float
        Estimated variance at time t.

    Returns
    -------
    max_s_n_value_up : float
        Maximum test statistic for upward movements.
    max_s_n_value_down : float
        Maximum test statistic for downward movements.
    max_s_n_critical_value_up : float
        Critical value for the upward test statistic.
    max_s_n_critical_value_down : float
        Critical value for the downward test statistic.
    """
    y = np.asarray(y, dtype=np.float64)

    max_s_n_value_up = -np.inf
    max_s_n_value_down = -np.inf
    max_s_n_critical_value_up = np.nan
    max_s_n_critical_value_down = np.nan

    b_alpha = 4.6  # As per Homm and Breitung (2011)

    for n in range(1, t - 1):
        dyn = y[t] - y[n]

        denominator = sigma_sq_t * np.sqrt(t - n) + 1e-12

        # One-sided tests, abs(dyn) split into positive and negative parts
        s_n_t_up = max(0, dyn) / denominator
        s_n_t_down = -min(0, dyn) / denominator  # Mane it positive

        if s_n_t_up > max_s_n_value_up:
            max_s_n_value_up = s_n_t_up
            max_s_n_critical_value_up = np.sqrt(b_alpha + np.log(t / n))

        if s_n_t_down > max_s_n_value_down:
            max_s_n_value_down = s_n_t_down
            max_s_n_critical_value_down = np.sqrt(b_alpha + np.log(t / n))

    return (
        max_s_n_value_up,
        max_s_n_value_down,
        max_s_n_critical_value_up,
        max_s_n_critical_value_down,
    )


@njit(nogil=True, parallel=False)
def cusum_developing(
    y: NDArray, warmup_period: int = 30
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Perform the Chu-Stinchcombe-White CUSUM Test on Levels.

    This implementation follows Homm and Breitung (2011), but the one-sided tests
    are used to detect upward or downward structural breaks, providing directionality.

    Parameters
    ----------
    y : np.ndarray
        Array of price series (log prices).
    warmup_period : int, optional
        Number of initial observations kept for std warmup, by default 30.

    Returns
    -------
    s_n_t_values_up : np.ndarray
        Test statistic values for upward movements (one-sided test).
    s_n_t_values_down : np.ndarray
        Test statistic values for downward movements (one-sided test).
    critical_values_up : np.ndarray
        Critical values for the upward test statistics.
    critical_values_down : np.ndarray
        Critical values for the downward test statistics.

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
    critical_values_up = np.empty(n, dtype=np.float64)
    critical_values_down = np.empty(n, dtype=np.float64)

    # Initialize arrays with NaNs
    s_n_t_values_up.fill(np.nan)
    s_n_t_values_down.fill(np.nan)
    critical_values_up.fill(np.nan)
    critical_values_down.fill(np.nan)

    cum_squared_diff = np.cumsum(np.diff(y) ** 2)

    for t in prange(warmup_period, n):
        sigma_sq_t = np.sqrt(cum_squared_diff[t - 1] / (t - 1))
        (
            max_s_n_value_up,
            max_s_n_value_down,
            max_s_n_critical_value_up,
            max_s_n_critical_value_down,
        ) = _comp_max_s_nt(y, t, sigma_sq_t)
        s_n_t_values_up[t] = max_s_n_value_up
        s_n_t_values_down[t] = max_s_n_value_down
        critical_values_up[t] = max_s_n_critical_value_up
        critical_values_down[t] = max_s_n_critical_value_down

    return (
        s_n_t_values_up,
        s_n_t_values_down,
        critical_values_up,
        critical_values_down,
    )

@njit(nogil=True)
def cusum_last(y: NDArray) -> Tuple[float, float, float, float]:
    """
    Perform the Chu-Stinchcombe-White CUSUM Test on Levels for the last observation.

    This implementation follows Homm and Breitung (2011), but the one-sided tests
    are used to detect upward or downward structural breaks, providing directionality.

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
    max_s_n_critical_value_up : float
        Critical value for the upward test statistic at the last observation.
    max_s_n_critical_value_down : float
        Critical value for the downward test statistic at the last observation.

    Notes
    -----
    A significant test statistic at the last observation indicates a recent structural break
    in the time series.
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.log(y)

    n = len(y)
    cum_squared_diff = np.cumsum(np.diff(y) ** 2)

    t = n - 1  # t is the last index
    sigma_sq_t = np.sqrt(cum_squared_diff[t - 1] / (t - 1))

    (
        max_s_n_value_up,
        max_s_n_value_down,
        max_s_n_critical_value_up,
        max_s_n_critical_value_down,
    ) = _comp_max_s_nt(y, t, sigma_sq_t)

    return (
        max_s_n_value_up,
        max_s_n_value_down,
        max_s_n_critical_value_up,
        max_s_n_critical_value_down,
    )


@njit(nogil=True, parallel=True)
def cusum_rolling(
    close_prices: NDArray, window_size: int = 1000, warmup_period: int = 30
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Perform the Chu-Stinchcombe-White CUSUM Test on Levels over a rolling window.

    This implementation follows Homm and Breitung (2011), where the one-sided tests
    are used to detect upward or downward structural breaks, providing directionality.

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
    critical_values_up : np.ndarray
        Critical values for the upward test statistics over the rolling window.
    critical_values_down : np.ndarray
        Critical values for the downward test statistics over the rolling window.

    Notes
    -----
    This test detects deviations from the random walk hypothesis in the time series over a moving window.
    It helps identify periods of structural breaks in the time series.
    """
    close_prices = np.asarray(close_prices, dtype=np.float64)

    n = len(close_prices)
    snt_up = np.empty(n, dtype=np.float64)
    snt_down = np.empty(n, dtype=np.float64)
    critical_values_up = np.empty(n, dtype=np.float64)
    critical_values_down = np.empty(n, dtype=np.float64)

    # Initialize arrays with NaNs
    snt_up.fill(np.nan)
    snt_down.fill(np.nan)
    critical_values_up.fill(np.nan)
    critical_values_down.fill(np.nan)

    if n > window_size:
        for current_idx in prange(window_size, n):
            start_idx = current_idx - window_size
            current_prices = close_prices[start_idx:current_idx + 1]

            if start_idx == 0:
                # First window after warmup period, compute initial values
                (
                    s_n_t_values_up,
                    s_n_t_values_down,
                    c_values_up,
                    c_values_down,
                ) = cusum_developing(current_prices, warmup_period)
                snt_up[start_idx:current_idx + 1] = s_n_t_values_up
                snt_down[start_idx:current_idx + 1] = s_n_t_values_down
                critical_values_up[start_idx:current_idx + 1] = c_values_up
                critical_values_down[start_idx:current_idx + 1] = c_values_down
            else:
                # Subsequent windows, compute and store only the last value
                (
                    s_n_t_up,
                    s_n_t_down,
                    c_value_up,
                    c_value_down,
                ) = cusum_last(current_prices)
                snt_up[current_idx] = s_n_t_up
                snt_down[current_idx] = s_n_t_down
                critical_values_up[current_idx] = c_value_up
                critical_values_down[current_idx] = c_value_down
    else:
        # Case when the window size is larger than the data
        (
            s_n_t_values_up,
            s_n_t_values_down,
            c_values_up,
            c_values_down,
        ) = cusum_developing(close_prices, warmup_period)
        snt_up = s_n_t_values_up
        snt_down = s_n_t_values_down
        critical_values_up = c_values_up
        critical_values_down = c_values_down

    return snt_up, snt_down, critical_values_up, critical_values_down
