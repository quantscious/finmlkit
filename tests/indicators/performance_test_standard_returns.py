import numpy as np
import pandas as pd
import time
from numpy.testing import assert_allclose

# Import the standard_volatility_estimator function
from finmlkit.indicators.volatility import standard_volatility_estimator

# Alternative implementation (as provided)
def get_vol(close: pd.Series, lookback: int = 100,
            days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0) -> pd.Series:
    time_delta = pd.Timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    df0 = close.index.searchsorted(close.index - time_delta, side='right') - 1
    valid_mask = df0 >= 0
    df0 = df0[valid_mask]
    current_indices = np.arange(len(close))[valid_mask]
    lagged_indices = df0
    returns = pd.Series(np.nan, index=close.index)
    returns.iloc[current_indices] = close.iloc[current_indices].values / close.iloc[lagged_indices].values - 1
    returns = returns.ewm(span=lookback, adjust=True).std(bias=False)  # EWM standard deviation
    return returns

def generate_large_dataset(n_points=1_000_000, inter_arrival_time=15, seed=42):
    np.random.seed(seed)
    inter_arrival_times = np.random.exponential(scale=inter_arrival_time, size=n_points)
    timestamps_sec = np.cumsum(inter_arrival_times)
    start_time = pd.Timestamp('2021-01-01')
    timestamps = start_time + pd.to_timedelta(timestamps_sec, unit='s')
    prices = np.cumsum(np.random.randn(n_points)) + 100
    prices = np.abs(prices) + 1
    close_series = pd.Series(prices, index=timestamps)
    return close_series

def measure_execution_time(func, *args, n_runs=10, **kwargs):
    # [As defined above]
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    avg_time = sum(times[1:]) / (n_runs-1)
    return avg_time

def performance_test():
    # Generate data
    close_series = generate_large_dataset(n_points=1000000)

    # Parameters
    lookback = 100
    return_window_seconds = 60  # 1 minute

    # Prepare arguments for the functions
    kwargs_get_vol = {
        'close': close_series,
        'lookback': lookback,
        'seconds': return_window_seconds
    }

    kwargs_standard_volatility_estimator = {
        'timestamps': close_series.index.astype(np.int64).values,
        'close': close_series.values,
        'lookback': lookback,
        'return_window_sec': return_window_seconds
    }

    # Measure execution time for get_vol
    time_get_vol = measure_execution_time(get_vol, **kwargs_get_vol)
    print(f"Average execution time of get_vol: {time_get_vol:.4f} seconds")

    # Measure execution time for standard_volatility_estimator
    time_standard_vol = measure_execution_time(standard_volatility_estimator, **kwargs_standard_volatility_estimator)
    print(f"Average execution time of standard_volatility_estimator: {time_standard_vol:.4f} seconds")

    # Speedup
    speedup = time_get_vol / time_standard_vol
    print(f"Speedup of standard_volatility_estimator over get_vol: {speedup:.2f}x")

    # Compute outputs for comparison
    output_get_vol = get_vol(**kwargs_get_vol)
    output_standard_vol = standard_volatility_estimator(**kwargs_standard_volatility_estimator)

    # pandas series
    output_standard_vol = pd.Series(output_standard_vol, index=close_series.index)

    # Align indices and compare outputs
    common_index = output_get_vol.dropna().index.intersection(output_standard_vol.dropna().index)
    output_get_vol_clean = output_get_vol.loc[common_index]
    output_standard_vol_clean = output_standard_vol.loc[common_index]

    # Verify outputs are close
    try:
        assert_allclose(output_get_vol_clean.values, output_standard_vol_clean.values, rtol=1e-5, atol=1e-8)
        print("Outputs are consistent between get_vol and standard_volatility_estimator.")
    except AssertionError as e:
        print("Outputs differ between get_vol and standard_volatility_estimator.")
        print(str(e))


if __name__ == "__main__":
    performance_test()

    """
    Average execution time of get_vol: 0.1608 seconds
    Average execution time of standard_volatility_estimator: 0.0218 seconds
    Speedup of standard_volatility_estimator over pandas: 7.39x
    """