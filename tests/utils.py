import numpy as np
import pandas as pd
import time
from typing import Any, Callable


def measure_execution_time(func: Callable, *args, n_runs=11, **kwargs) -> float:
    """
    Measure the execution time of a function by running it multiple times and averaging the time taken.

    Parameters
    ----------
    func
        The function to measure the execution time of
    args
        Positional arguments to pass to the function
    n_runs
        Number of times to run the function
    kwargs
        Keyword arguments to pass to the function

    Returns
    -------

    """
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    avg_time = sum(times[1:]) / (n_runs-1)  # Exclude the first run where numba compiles the function
    return avg_time


def generate_price_series(n_points=1_000_000, inter_arrival_time=15, seed=42) -> pd.Series:
    """
    Generate a synthetic price series with irregular timestamps.

    Parameters
    ----------
    n_points
        Number of data points
    inter_arrival_time
        Average time between data points in seconds
    seed
        Random seed

    Returns
    -------
    close_series : pd.Series
        Synthetic price series with irregular datetime index

    """
    np.random.seed(seed)

    # Generate irregular timestamps
    inter_arrival_times = np.random.exponential(scale=inter_arrival_time, size=n_points)
    timestamps_sec = np.cumsum(inter_arrival_times)
    start_time = pd.Timestamp('1994-01-01')
    timestamps = start_time + pd.to_timedelta(timestamps_sec, unit='s')

    # Generate synthetic closing prices
    mu = 0.0  # Drift
    sigma = 0.01  # Volatility
    S0 = 100  # Initial price
    dt = 1  # Time step size

    # Simulate log returns
    epsilon = np.random.normal(0, 1, n_points)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * epsilon

    # Generate price series
    log_S = np.log(S0) + np.cumsum(log_returns)
    S = np.exp(log_S)

    # Create pandas Series
    close_series = pd.Series(S, index=timestamps)

    return close_series