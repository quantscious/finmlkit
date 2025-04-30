import numpy as np
import pandas as pd
import pytest
import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.feature.utils import compute_lagged_returns
from numpy.testing import assert_allclose


def generate_sample_data():
    # Generate a datetime index
    dates = pd.date_range(start='2021-01-01', periods=100, freq='T')  # 1-minute intervals
    n = len(dates)

    # Generate synthetic closing prices
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(n)) + 100  # Random walk starting from 100

    # Ensure no zeros in prices to avoid division by zero
    prices = np.abs(prices) + 1

    # Create pandas Series
    close_series = pd.Series(prices, index=dates)

    # Convert timestamps to int64 (nanoseconds since epoch)
    timestamps = dates.view(np.int64)

    # Convert prices to numpy array
    close_array = prices.astype(np.float64)

    return timestamps, close_array, close_series

def alternative_return_calculation(close, days=0, hours=0, minutes=0, seconds=0):
    time_delta = pd.Timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    # Use 'side' parameter to match Numba's 'searchsorted' behavior
    df0 = close.index.searchsorted(close.index - time_delta, side='right') - 1

    # Create a mask for valid indices
    valid_mask = df0 >= 0

    # Filter out invalid indices
    df0 = df0[valid_mask]

    # Align indices
    current_indices = np.arange(len(close))[valid_mask]
    lagged_indices = df0

    # Compute returns
    returns = pd.Series(np.nan, index=close.index)
    returns.iloc[current_indices] = close.iloc[current_indices].values / close.iloc[lagged_indices].values - 1

    return returns

def test_returns_equivalence():
    """
    WARNING: The return calculation is hardcoded
    Returns
    -------

    """
    # Generate sample data
    timestamps, close_array, close_series = generate_sample_data()

    # Parameters
    return_window_sec = 1  # For example
    minutes = 1

    # Compute returns in Numba function
    returns_numba = compute_lagged_returns(timestamps, close_array, return_window_sec)

    # Compute returns in pandas function
    returns_pandas = alternative_return_calculation(close_series, days=0, hours=0, minutes=minutes, seconds=0)

    # Align and compare returns
    returns_numba_series = pd.Series(returns_numba, index=close_series.index)
    common_index = returns_pandas.dropna().index.intersection(returns_numba_series.dropna().index)
    returns_pandas_clean = returns_pandas.loc[common_index]
    returns_numba_clean = returns_numba_series.loc[common_index]

    # Assert that the returns are close
    assert_allclose(returns_numba_clean.values, returns_pandas_clean.values, rtol=1e-5, atol=1e-8)


# Test with a large return window
def test_returns_equivalence_large_return_window():
    """
    Test the functions with a large return window.
    """
    # Generate regular data
    dates = pd.date_range(start='2021-01-01', periods=2000, freq='T')  # 1-minute intervals
    n = len(dates)

    # Generate synthetic closing prices
    np.random.seed(78)
    prices = np.cumsum(np.random.randn(n)) + 100

    # Ensure no zeros in prices
    prices = np.abs(prices) + 1

    close_series = pd.Series(prices, index=dates)
    timestamps = dates.view(np.int64)

    # Parameters
    return_window_sec = 3600  # 1 hour

    # Compute returns in both functions
    returns_numba = compute_lagged_returns(timestamps, prices, return_window_sec)
    returns_pandas = alternative_return_calculation(close_series, seconds=return_window_sec)

    # Align and compare returns
    returns_numba_series = pd.Series(returns_numba, index=dates)
    common_index = returns_pandas.dropna().index.intersection(returns_numba_series.dropna().index)
    assert_allclose(returns_numba_series.loc[common_index], returns_pandas.loc[common_index], rtol=1e-5, atol=1e-8)

# Test with zero return window
def test_returns_zero_return_window():
    """
    Test the functions with a zero return window, expecting a ValueError.
    """
    # Generate regular data
    dates = pd.date_range(start='2021-01-01', periods=10, freq='T')
    n = len(dates)
    prices = np.arange(1, n + 1)

    close_series = pd.Series(prices, index=dates)
    timestamps = dates.view(np.int64)

    # Parameters
    return_window_sec = 0

    # Test Numba function raises ValueError
    with pytest.raises(ValueError):
        compute_lagged_returns(timestamps, prices, return_window_sec)


# Test with data containing zeros
def test_returns_data_with_zeros():
    """
    Test the functions with data that includes zeros in prices.
    """
    # Generate regular data
    dates = pd.date_range(start='2021-01-01', periods=100, freq='T')
    n = len(dates)

    # Generate prices with zeros
    np.random.seed(90)
    prices = np.cumsum(np.random.randn(n))
    prices[::10] = 0  # Set every 10th price to zero

    # Ensure prices are non-negative
    prices = np.abs(prices)

    close_series = pd.Series(prices, index=dates)
    timestamps = dates.view(np.int64)

    # Parameters
    return_window_sec = 60

    # Compute returns in both functions
    returns_numba = compute_lagged_returns(timestamps, prices, return_window_sec)
    returns_pandas = alternative_return_calculation(close_series, seconds=return_window_sec)

    # Align and compare returns, allowing NaNs
    returns_numba_series = pd.Series(returns_numba, index=dates)
    common_index = returns_pandas.index.intersection(returns_numba_series.index)
    assert_allclose(returns_numba_series.loc[common_index], returns_pandas.loc[common_index], rtol=1e-5, atol=1e-8, equal_nan=True)


# Test with data containing NaNs
def test_returns_data_with_nans():
    """
    Test the functions with data that includes NaNs in prices.
    """
    # Generate regular data
    dates = pd.date_range(start='2021-01-01', periods=100, freq='T')
    n = len(dates)

    # Generate prices with NaNs
    np.random.seed(123)
    prices = np.cumsum(np.random.randn(n)) + 100
    prices[::10] = np.nan  # Set every 10th price to NaN

    close_series = pd.Series(prices, index=dates)
    timestamps = dates.view(np.int64)

    # Handle NaNs in prices (Numba function may not handle NaNs)
    # For testing purposes, we'll fill NaNs with the last valid price
    prices_filled = pd.Series(prices).fillna(method='ffill').values

    # Parameters
    return_window_sec = 60

    # Compute returns in both functions
    returns_numba = compute_lagged_returns(timestamps, prices_filled, return_window_sec)
    returns_pandas = alternative_return_calculation(close_series.fillna(method='ffill'), seconds=return_window_sec)

    # Align and compare returns, allowing NaNs
    returns_numba_series = pd.Series(returns_numba, index=dates)
    common_index = returns_pandas.index.intersection(returns_numba_series.index)
    assert_allclose(returns_numba_series.loc[common_index], returns_pandas.loc[common_index], rtol=1e-5, atol=1e-8, equal_nan=True)


# Test with truly irregular data
def test_returns_equivalence_irregular_data():
    """
    Test the functions with truly irregular data, simulating volume bar or tick data.
    """
    # Generate truly irregular timestamps
    np.random.seed(42)
    n = 1000  # Number of data points

    # Simulate inter-arrival times (in seconds) from an exponential distribution
    inter_arrival_times = np.random.exponential(scale=30, size=n)  # Average of 30 seconds between events

    # Generate timestamps by cumulatively summing the inter-arrival times
    timestamps_sec = np.cumsum(inter_arrival_times)

    # Convert timestamps to pandas datetime index
    start_time = pd.Timestamp('2021-01-01')
    timestamps = start_time + pd.to_timedelta(timestamps_sec, unit='s')

    # Generate synthetic closing prices
    prices = np.cumsum(np.random.randn(n)) + 100  # Random walk starting from 100

    # Ensure prices are positive to avoid division by zero or negative prices
    prices = np.abs(prices) + 1

    # Create pandas Series
    close_series = pd.Series(prices, index=timestamps)

    # Convert timestamps to int64 (nanoseconds since epoch) for Numba function
    timestamps_ns = timestamps.view(np.int64)

    # Parameters
    return_window_sec = 60  # 1 minutes

    # Compute returns in both functions
    returns_numba = compute_lagged_returns(timestamps_ns, prices, return_window_sec)
    returns_pandas = alternative_return_calculation(close_series, seconds=return_window_sec)

    # Align and compare returns
    returns_numba_series = pd.Series(returns_numba, index=timestamps)
    common_index = returns_pandas.dropna().index.intersection(returns_numba_series.dropna().index)
    returns_numba_clean = returns_numba_series.loc[common_index]
    returns_pandas_clean = returns_pandas.loc[common_index]

    # Assert that the returns are close
    assert_allclose(returns_numba_clean.values, returns_pandas_clean.values, rtol=1e-5, atol=1e-8)