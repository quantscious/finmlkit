"""
Tests for price-volume correlation features
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from finmlkit.feature.core.correlation import rolling_price_volume_correlation
from finmlkit.feature.transforms import PriceVolumeCorrelation


def test_rolling_price_volume_correlation_basic():
    """Test basic functionality of rolling price-volume correlation calculation"""
    # Create sample data with known correlation patterns
    price = np.array([10.0, 11.0, 12.0, 11.5, 11.0, 10.5, 11.0, 12.0, 13.0, 12.5])
    volume = np.array([100.0, 110.0, 120.0, 115.0, 105.0, 95.0, 105.0, 110.0, 120.0, 115.0])

    # Window size of 3
    result = rolling_price_volume_correlation(price, volume, 3)

    # First window-1 results should be NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])

    # Verify the rest have valid correlation values
    for i in range(3, len(result)):
        assert not np.isnan(result[i])
        assert -1.0 <= result[i] <= 1.0  # Correlation values should be in [-1, 1]


def test_rolling_price_volume_correlation_perfect():
    """Test with perfectly correlated data"""
    # Create perfectly correlated data (price up, volume up)
    price = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
    volume = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0])

    # Window size of 4
    result = rolling_price_volume_correlation(price, volume, 4)

    # Check values for perfect positive correlation (ignoring first window-1 NaNs)
    for i in range(4, len(result)):
        assert abs(result[i] - 1.0) < 0.001  # Should be very close to 1.0


def test_rolling_price_volume_correlation_inverse():
    """Test with perfectly anti-correlated data"""
    # Create perfectly anti-correlated data (price up, volume down)
    price = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
    volume = np.array([190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0])

    # Window size of 4
    result = rolling_price_volume_correlation(price, volume, 4)

    # Check values for perfect negative correlation (ignoring first window-1 NaNs)
    for i in range(4, len(result)):
        assert abs(result[i] + 1.0) < 0.001  # Should be very close to -1.0


def test_rolling_price_volume_correlation_nan_handling():
    """Test NaN handling in correlation calculation"""
    # Create data with NaN values
    price = np.array([10.0, 11.0, np.nan, 13.0, 14.0, 15.0, 16.0, np.nan, 18.0, 19.0])
    volume = np.array([100.0, 110.0, 120.0, 130.0, np.nan, 150.0, 160.0, 170.0, 180.0, np.nan])

    # Window size of 4
    result = rolling_price_volume_correlation(price, volume, 4)

    # Check that NaN values are handled correctly
    for i in range(len(result)):
        if i < 4:  # First window-1 values are always NaN
            assert np.isnan(result[i])
        elif np.isnan(price[i]) or np.isnan(volume[i]):
            # Result should be NaN if input at current position is NaN
            assert np.isnan(result[i])


def test_rolling_price_volume_correlation_small_window():
    """Test with a small window size"""
    price = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    volume = np.array([100.0, 110.0, 120.0, 130.0, 140.0])

    # Window size of 2
    result = rolling_price_volume_correlation(price, volume, 2)

    # First window-1 results should be NaN
    assert np.isnan(result[0])
    assert not np.isnan(result[2])  # Should have valid values from index 2 onward


def test_rolling_price_volume_correlation_zero_price():
    """Test handling of zero prices (which would cause division by zero in returns calculation)"""
    price = np.array([10.0, 0.0, 12.0, 13.0, 14.0, 15.0, 16.0, 0.0, 18.0, 19.0])
    volume = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0])

    # Window size of 4
    result = rolling_price_volume_correlation(price, volume, 4)

    # The function should handle zero prices by marking the returns as NaN
    assert np.isnan(result[2])  # Result at index 2 should include a NaN return due to zero price at index 1


def test_transform_basic():
    """Test the PriceVolumeCorrelation transform class"""
    # Create sample data
    dates = [datetime(2025, 6, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({
        'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        'volume': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
    }, index=dates)

    # Test standard transform
    transform = PriceVolumeCorrelation(window=4)
    result = transform(df)

    # Column name should match the pattern
    assert result.name == 'corr_pv_4'

    # Compare with direct calculation
    expected = rolling_price_volume_correlation(df['close'].values, df['volume'].values, 4)
    for i in range(len(expected)):
        if np.isnan(expected[i]):
            assert np.isnan(result.iloc[i])
        else:
            assert result.iloc[i] == pytest.approx(expected[i])


def test_transform_custom_columns():
    """Test the transform with custom column names"""
    # Create sample data with custom column names
    dates = [datetime(2025, 6, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({
        'price': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        'vol': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
    }, index=dates)

    # Test transform with custom column names
    transform = PriceVolumeCorrelation(window=4, input_cols=['price', 'vol'])
    result = transform(df)

    # Column name should match the pattern
    assert result.name == 'corr_pv_4'

    # Compare with direct calculation
    expected = rolling_price_volume_correlation(df['price'].values, df['vol'].values, 4)
    for i in range(len(expected)):
        if np.isnan(expected[i]):
            assert np.isnan(result.iloc[i])
        else:
            assert result.iloc[i] == pytest.approx(expected[i])


def test_transform_pandas_impl():
    """Test the pandas implementation of the transform"""
    # Create sample data
    dates = [datetime(2025, 6, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({
        'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        'volume': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
    }, index=dates)

    # Calculate using pandas implementation
    transform = PriceVolumeCorrelation(window=4)
    result_pd = transform._pd(df)

    # Calculate using numba implementation
    result_nb = transform._nb(df)

    # Both implementations should produce the same results
    for i in range(len(result_pd)):
        if np.isnan(result_pd.iloc[i]):
            assert np.isnan(result_nb.iloc[i])
        else:
            assert result_pd.iloc[i] == pytest.approx(result_nb.iloc[i])


if __name__ == '__main__':
    pytest.main()