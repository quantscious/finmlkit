import numpy as np
import pandas as pd
import pytest
import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.indicator.volatility import standard_volatility_estimator

from numpy.testing import assert_allclose

# Alternative implementation
def get_vol(close: pd.Series, lookback: int = 100,
            days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0) -> pd.Series:

    time_delta = pd.Timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
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

    returns = returns.ewm(span=lookback, adjust=True).std(bias=False)  # EWM standard deviation

    return returns


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


def test_standard_volatility_estimator_equivalence():
    # Generate sample data
    timestamps, close_array, close_series = generate_sample_data()

    # Parameters
    lookback = 20
    return_window_sec = 60  # 1 minute in seconds
    minutes = 1  # For get_vol

    # Run standard_volatility_estimator
    vol_estimates_numba = standard_volatility_estimator(
        timestamps,
        close_array,
        return_window_sec,
        lookback=lookback
    )

    # Run get_vol
    vol_estimates_pandas = get_vol(
        close_series,
        lookback=lookback,
        minutes=minutes
    )

    # Align indices and compare results
    # Convert vol_estimates_numba to pandas Series with the same index
    vol_estimates_numba_series = pd.Series(vol_estimates_numba, index=close_series.index)

    # Compare the two Series
    # Drop NaNs for comparison
    common_index = vol_estimates_pandas.dropna().index.intersection(vol_estimates_numba_series.dropna().index)

    vol_pandas_clean = vol_estimates_pandas.loc[common_index]
    vol_numba_clean = vol_estimates_numba_series.loc[common_index]

    # Assert that the two volatility estimates are close
    assert_allclose(vol_numba_clean, vol_pandas_clean, rtol=1e-5, atol=1e-8, equal_nan=True)


@pytest.mark.parametrize("return_window_sec,minutes,seconds", [
    (30, 0, 30),  # 30 sec. window
    (60, 1, 0),  # 1-minute window
    (300, 5, 0),  # 5-minute window
    (900, 15, 0),  # 15-minute window
])
def test_standard_volatility_estimator_various_windows(return_window_sec, minutes, seconds):
    # Generate sample data
    timestamps, close_array, close_series = generate_sample_data()

    # Parameters
    lookback = 20

    # Run standard_volatility_estimator
    vol_estimates_numba = standard_volatility_estimator(
        timestamps,
        close_array,
        return_window_sec,
        lookback=lookback
    )

    # Run get_vol
    vol_estimates_pandas = get_vol(
        close_series,
        lookback=lookback,
        minutes=minutes,
        seconds=seconds
    )

    # Align indices and compare results
    vol_estimates_numba_series = pd.Series(vol_estimates_numba, index=close_series.index)
    common_index = vol_estimates_pandas.dropna().index.intersection(vol_estimates_numba_series.dropna().index)
    vol_pandas_clean = vol_estimates_pandas.loc[common_index]
    vol_numba_clean = vol_estimates_numba_series.loc[common_index]

    # Assert that the two volatility estimates are close
    assert_allclose(vol_numba_clean.values, vol_pandas_clean.values, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("lookback", [2, 5, 10, 20, 50])
def test_standard_volatility_estimator_various_lookbacks(lookback):
    # Generate sample data
    timestamps, close_array, close_series = generate_sample_data()

    # Parameters
    return_window_sec = 60  # 1 minute
    minutes = 1

    # Run standard_volatility_estimator
    vol_estimates_numba = standard_volatility_estimator(
        timestamps,
        close_array,
        return_window_sec,
        lookback=lookback
    )

    # Run get_vol
    vol_estimates_pandas = get_vol(
        close_series,
        lookback=lookback,
        minutes=minutes
    )

    # Align indices and compare results
    vol_estimates_numba_series = pd.Series(vol_estimates_numba, index=close_series.index)
    common_index = vol_estimates_pandas.dropna().index.intersection(vol_estimates_numba_series.dropna().index)
    vol_pandas_clean = vol_estimates_pandas.loc[common_index]
    vol_numba_clean = vol_estimates_numba_series.loc[common_index]

    # Assert that the two volatility estimates are close
    assert_allclose(vol_numba_clean.values, vol_pandas_clean.values, rtol=1e-5, atol=1e-8)


def test_standard_volatility_estimator_constant_prices():
    # Generate constant price data
    dates = pd.date_range(start='2021-01-01', periods=100, freq='T')
    n = len(dates)
    prices = np.full(n, 100.0)
    close_series = pd.Series(prices, index=dates)
    timestamps = dates.view(np.int64)
    close_array = prices.astype(np.float64)

    # Parameters
    lookback = 20
    return_window_sec = 60
    minutes = 1

    # Run standard_volatility_estimator
    vol_estimates_numba = standard_volatility_estimator(
        timestamps,
        close_array,
        return_window_sec,
        lookback=lookback
    )

    # Run get_vol
    vol_estimates_pandas = get_vol(
        close_series,
        lookback=lookback,
        minutes=minutes
    )

    # Assert that volatility estimates are zero or NaN where appropriate
    vol_estimates_numba_series = pd.Series(vol_estimates_numba, index=close_series.index)
    vol_estimates_pandas = vol_estimates_pandas.dropna()
    vol_estimates_numba_series = vol_estimates_numba_series.dropna()

    assert_allclose(vol_estimates_numba_series.values, vol_estimates_pandas.values, rtol=1e-5, atol=1e-8)


def test_standard_volatility_estimator_missing_data():
    # Generate sample data with missing prices
    timestamps, close_array, close_series = generate_sample_data()

    # Introduce NaNs into the data
    close_series.iloc[::10] = np.nan
    close_array[::10] = np.nan

    # Parameters
    lookback = 20
    return_window_sec = 60
    minutes = 1

    # Run standard_volatility_estimator
    vol_estimates_numba = standard_volatility_estimator(
        timestamps,
        close_array,
        return_window_sec,
        lookback=lookback
    )

    # Run get_vol
    vol_estimates_pandas = get_vol(
        close_series,
        lookback=lookback,
        minutes=minutes
    )

    # Align indices and compare results
    vol_estimates_numba_series = pd.Series(vol_estimates_numba, index=close_series.index)
    common_index = vol_estimates_pandas.dropna().index.intersection(vol_estimates_numba_series.dropna().index)
    vol_pandas_clean = vol_estimates_pandas.loc[common_index]
    vol_numba_clean = vol_estimates_numba_series.loc[common_index]

    # Assert that the two volatility estimates are close
    assert_allclose(vol_numba_clean.values, vol_pandas_clean.values, rtol=1e-5, atol=1e-8)
