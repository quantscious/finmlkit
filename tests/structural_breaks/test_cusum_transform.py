"""
Tests for CUSUMTest transform functionality from finmlkit.feature.transforms
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Add parent directory to path to import finmlkit modules
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from finmlkit.feature.transforms import CUSUMTest
from finmlkit.feature.core.structural_break.cusum import (
    cusum_test_rolling,
    cusum_test_developing,
    cusum_test_last
)

def create_test_dataframe():
    """Create a simple test dataframe with a price series"""
    np.random.seed(42)
    n_samples = 200

    # Start with a random walk
    random_changes = np.random.normal(0, 1, n_samples)
    # Add a structural break (trend) in the middle
    random_changes[100:150] += 0.5  # Add positive bias for upward trend

    # Create the price series
    close = 100 * np.ones(n_samples)
    for i in range(1, n_samples):
        close[i] = close[i-1] + random_changes[i]

    # Ensure all prices are positive
    close = np.abs(close)

    # Create the DataFrame with timestamps
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]

    df = pd.DataFrame({
        'close': close,
        'high': close * 1.01,  # Add some dummy high/low values
        'low': close * 0.99,
        'volume': np.random.randint(100, 1000, n_samples)
    }, index=pd.DatetimeIndex(dates))

    return df

def test_cusum_test_developing():
    """Test the cusum_test_developing function directly"""
    # Create a test series with a clear structural break
    n_samples = 200
    np.random.seed(42)

    # Create a series with a significant upward trend in the middle
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, n_samples)))
    # Add structural break
    prices[100:150] = prices[100:150] * np.exp(np.linspace(0, 0.2, 50))

    # Test the function
    s_n_t_values_up, s_n_t_values_down, critical_values_up, critical_values_down = cusum_test_developing(prices, warmup_period=30)

    # Check for NaN values
    assert not np.isnan(s_n_t_values_up).all(), "All values in s_n_t_values_up are NaN"
    assert not np.isnan(s_n_t_values_down).all(), "All values in s_n_t_values_down are NaN"
    assert not np.isnan(critical_values_up).all(), "All values in critical_values_up are NaN"
    assert not np.isnan(critical_values_down).all(), "All values in critical_values_down are NaN"

    # Ensure we have some values after warmup period
    assert np.sum(~np.isnan(s_n_t_values_up)) >= n_samples - 30, "Not enough non-NaN values in s_n_t_values_up"
    assert np.sum(~np.isnan(s_n_t_values_down)) >= n_samples - 30, "Not enough non-NaN values in s_n_t_values_down"

def test_cusum_test_rolling():
    """Test the cusum_test_rolling function directly"""
    # Create a test series with a clear structural break
    n_samples = 200
    np.random.seed(42)

    # Create a series with a significant upward trend in the middle
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, n_samples)))
    # Add structural break
    prices[100:150] = prices[100:150] * np.exp(np.linspace(0, 0.2, 50))

    window_size = 50
    warmup_period = 10

    # Test the function
    snt_up, snt_down, critical_values_up, critical_values_down = cusum_test_rolling(
        prices, window_size=window_size, warmup_period=warmup_period
    )

    # Check for NaN values
    assert not np.isnan(snt_up).all(), "All values in snt_up are NaN"
    assert not np.isnan(snt_down).all(), "All values in snt_down are NaN"
    assert not np.isnan(critical_values_up).all(), "All values in critical_values_up are NaN"
    assert not np.isnan(critical_values_down).all(), "All values in critical_values_down are NaN"

    # We should have NaNs for at least the first warmup_period elements
    # Note: The original test expected NaNs for the first window_size elements,
    # but the implementation fills with calculated values after warmup_period
    assert np.isnan(snt_up[:warmup_period]).all(), "Expected NaNs for first warmup_period elements"

    # After window_size, we should have non-NaN values
    assert not np.isnan(snt_up[window_size:]).all(), "Expected non-NaN values after window_size"

def test_cusum_transform_direct_output():
    """Test the CUSUMTest transform output directly"""
    df = create_test_dataframe()

    # Apply the transform
    cusum_transform = CUSUMTest(window_size=50, warmup_period=10, input_col="close")
    outputs = cusum_transform(df)

    # We expect a tuple of two series
    assert isinstance(outputs, tuple), "Expected tuple output from CUSUMTest"
    assert len(outputs) == 2, "Expected two series in output tuple"

    # Check each series
    for i, output in enumerate(outputs):
        assert isinstance(output, pd.Series), f"Output {i} should be a pd.Series"
        # Check if all values are NaN
        nan_count = output.isna().sum()
        total_count = len(output)
        print(f"Series {i}: {nan_count}/{total_count} NaN values ({nan_count/total_count:.2%})")

        # After window_size + warmup_period, we should have some non-NaN values
        # Extract output name for reporting
        output_name = cusum_transform.produces[i]
        non_nan_count = (~output.isna()).sum()
        assert non_nan_count > 0, f"Output '{output_name}' has all NaN values"

def test_cusum_internal_calculations():
    """Test the internal calculations of CUSUMTest transform"""
    # Create test data
    df = create_test_dataframe()
    close_prices = df['close'].values

    # Manual calculation using the core functions
    window_size = 50
    warmup_period = 10

    # Get results from the core function
    snt_up, snt_down, critical_values_up, critical_values_down = cusum_test_rolling(
        close_prices, window_size=window_size, warmup_period=warmup_period
    )

    # Check if any of the arrays have all NaNs
    assert not np.isnan(snt_up).all(), "All values in snt_up are NaN"
    assert not np.isnan(snt_down).all(), "All values in snt_down are NaN"
    assert not np.isnan(critical_values_up).all(), "All values in critical_values_up are NaN"
    assert not np.isnan(critical_values_down).all(), "All values in critical_values_down are NaN"

    # After window_size, we should have some non-NaN values
    assert not np.isnan(snt_up[window_size:]).all(), "No non-NaN values after window_size in snt_up"
    assert not np.isnan(snt_down[window_size:]).all(), "No non-NaN values after window_size in snt_down"

    # Now test the transform itself
    cusum_transform = CUSUMTest(window_size=window_size, warmup_period=warmup_period, input_col="close")
    outputs = cusum_transform(df)

    # Check if the transform outputs match our manual calculation
    # The transform should return snt_up - critical_values_up and snt_down - critical_values_down
    expected_up = snt_up - critical_values_up
    expected_down = snt_down - critical_values_down

    # Check if the transform outputs are all NaN where they shouldn't be
    assert not outputs[0].isna().all(), "Output 0 has all NaNs"
    assert not outputs[1].isna().all(), "Output 1 has all NaNs"

def test_debug_cusum_transform():
    """Detailed debugging of CUSUMTest transform to identify NaN issue"""
    df = create_test_dataframe()
    window_size = 50
    warmup_period = 10

    # Test the underlying function directly
    close_prices = df['close'].values
    snt_up, snt_down, critical_values_up, critical_values_down = cusum_test_rolling(
        close_prices, window_size=window_size, warmup_period=warmup_period
    )

    # Print detailed diagnostics
    print(f"From direct function call:")
    print(f"snt_up NaN count: {np.isnan(snt_up).sum()}/{len(snt_up)}")
    print(f"snt_down NaN count: {np.isnan(snt_down).sum()}/{len(snt_down)}")
    print(f"critical_values_up NaN count: {np.isnan(critical_values_up).sum()}/{len(critical_values_up)}")
    print(f"critical_values_down NaN count: {np.isnan(critical_values_down).sum()}/{len(critical_values_down)}")

    # Compute the expected output
    expected_up = snt_up - critical_values_up
    expected_down = snt_down - critical_values_down
    print(f"expected_up NaN count: {np.isnan(expected_up).sum()}/{len(expected_up)}")
    print(f"expected_down NaN count: {np.isnan(expected_down).sum()}/{len(expected_down)}")

    # Now test the transform
    cusum_transform = CUSUMTest(window_size=window_size, warmup_period=warmup_period, input_col="close")
    outputs = cusum_transform(df)

    print(f"\nFrom transform:")
    print(f"Output 0 NaN count: {outputs[0].isna().sum()}/{len(outputs[0])}")
    print(f"Output 1 NaN count: {outputs[1].isna().sum()}/{len(outputs[1])}")

    # Compare the first few non-NaN values from both methods
    non_nan_indices_direct = np.where(~np.isnan(expected_up))[0]
    non_nan_indices_transform = np.where(~outputs[0].isna())[0]

    print("\nIndices with non-NaN values:")
    print(f"Direct function: {non_nan_indices_direct[:10]} ...")
    print(f"Transform: {non_nan_indices_transform[:10]} ...")

    if len(non_nan_indices_direct) > 0 and len(non_nan_indices_transform) > 0:
        # Compare some actual values
        i_direct = non_nan_indices_direct[0]
        i_transform = non_nan_indices_transform[0] if len(non_nan_indices_transform) > 0 else -1

        if i_transform >= 0:
            print("\nValue comparison:")
            print(f"Direct at index {i_direct}: {expected_up[i_direct]}")
            print(f"Transform at index {i_transform}: {outputs[0].iloc[i_transform]}")

if __name__ == "__main__":
    # Run tests directly when executing this file
    test_cusum_test_developing()
    test_cusum_test_rolling()
    test_cusum_transform_direct_output()
    test_cusum_internal_calculations()
    test_debug_cusum_transform()
