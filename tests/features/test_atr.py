import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from finmlkit.feature.core.volatility import atr, true_range
from finmlkit.feature.transforms import ATR


def test_true_range_basic():
    """Test basic functionality of true range calculation"""
    high = np.array([10.0, 11.0, 12.0, 13.0])
    low = np.array([8.0, 9.0, 10.0, 11.0])
    close = np.array([9.0, 10.0, 11.0, 12.0])

    tr = true_range(high, low, close)

    # First TR is just high-low of the first bar
    assert tr[0] == 2.0

    # Expected TR values for other bars
    assert tr[1] == max(11.0-9.0, abs(11.0-9.0), abs(9.0-9.0)) == 2.0
    assert tr[2] == max(12.0-10.0, abs(12.0-10.0), abs(10.0-10.0)) == 2.0
    assert tr[3] == max(13.0-11.0, abs(13.0-11.0), abs(11.0-11.0)) == 2.0


def test_atr_sma_basic():
    """Test basic functionality of ATR calculation with SMA method"""
    high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    low = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    close = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

    # Window size of 3
    result = atr(high, low, close, 3)

    # First window of results should be NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])

    # First ATR value at index 2 should be average of first 3 true range values
    tr_values = true_range(high, low, close)
    expected_atr_2 = np.mean(tr_values[:3])
    assert result[2] == pytest.approx(expected_atr_2)

    # Next values should be rolling window
    expected_atr_3 = np.mean(tr_values[1:4])
    assert result[3] == pytest.approx(expected_atr_3)

    expected_atr_4 = np.mean(tr_values[2:5])
    assert result[4] == pytest.approx(expected_atr_4)

    expected_atr_5 = np.mean(tr_values[3:6])
    assert result[5] == pytest.approx(expected_atr_5)


def test_atr_ema_basic():
    """Test basic functionality of ATR calculation with EMA method"""
    high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    low = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    close = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

    # Window size of 3 with EMA
    result = atr(high, low, close, 3, ema_based=True)

    # First window of results should be NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])

    # First ATR value at index 2 should be average of first 3 true range values
    tr_values = true_range(high, low, close)
    expected_atr_2 = np.mean(tr_values[:3])
    assert result[2] == pytest.approx(expected_atr_2)

    # EMA calculation: ATR_t = ((window-1) * ATR_{t-1} + TR_t) / window
    expected_atr_3 = ((3-1) * expected_atr_2 + tr_values[3]) / 3
    assert result[3] == pytest.approx(expected_atr_3)

    expected_atr_4 = ((3-1) * expected_atr_3 + tr_values[4]) / 3
    assert result[4] == pytest.approx(expected_atr_4)

    expected_atr_5 = ((3-1) * expected_atr_4 + tr_values[5]) / 3
    assert result[5] == pytest.approx(expected_atr_5)


def test_atr_normalize():
    """Test ATR normalization by mid price"""
    high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    low = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    close = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

    # Calculate standard ATR with window size 3
    standard_atr = atr(high, low, close, 3)

    # Calculate normalized ATR
    normalized_atr = atr(high, low, close, 3, normalize=True)

    # Mid prices for each bar
    mid_prices = (high + low) / 2.0

    # Check normalization
    for i in range(len(high)):
        if not np.isnan(standard_atr[i]):
            expected_normalized = standard_atr[i] / mid_prices[i]
            assert normalized_atr[i] == pytest.approx(expected_normalized)
        else:
            assert np.isnan(normalized_atr[i])


def test_atr_transform():
    """Test the ATR transform class"""
    # Create sample data
    dates = [datetime(2025, 6, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({
        'high': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        'low': [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        'close': [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    }, index=dates)

    # Test standard ATR transform
    atr_transform = ATR(window=3)
    result = atr_transform(df)

    # Column name should match the pattern
    assert result.name == 'atr3'

    # Compare with direct calculation
    expected = atr(df['high'].values, df['low'].values, df['close'].values, 3)
    for i in range(len(expected)):
        if np.isnan(expected[i]):
            assert np.isnan(result.iloc[i])
        else:
            assert result.iloc[i] == pytest.approx(expected[i])

    # Test EMA-based ATR transform
    atr_transform_ema = ATR(window=3, ema_based=True)
    result_ema = atr_transform_ema(df)

    # Column name should include 'ema'
    assert result_ema.name == 'atr3_ema'

    # Compare with direct calculation
    expected_ema = atr(df['high'].values, df['low'].values, df['close'].values, 3, ema_based=True)
    for i in range(len(expected_ema)):
        if np.isnan(expected_ema[i]):
            assert np.isnan(result_ema.iloc[i])
        else:
            assert result_ema.iloc[i] == pytest.approx(expected_ema[i])

    # Test normalized ATR transform
    atr_transform_norm = ATR(window=3, normalize=True)
    result_norm = atr_transform_norm(df)

    # Column name should include 'norm'
    assert result_norm.name == 'atr3_norm'

    # Compare with direct calculation
    expected_norm = atr(df['high'].values, df['low'].values, df['close'].values, 3, normalize=True)
    for i in range(len(expected_norm)):
        if np.isnan(expected_norm[i]):
            assert np.isnan(result_norm.iloc[i])
        else:
            assert result_norm.iloc[i] == pytest.approx(expected_norm[i])

    # Test with both EMA and normalization
    atr_transform_both = ATR(window=3, ema_based=True, normalize=True)
    result_both = atr_transform_both(df)

    # Column name should include both 'ema' and 'norm'
    assert result_both.name == 'atr3_ema_norm'

    # Compare with direct calculation
    expected_both = atr(df['high'].values, df['low'].values, df['close'].values, 3,
                      ema_based=True, normalize=True)
    for i in range(len(expected_both)):
        if np.isnan(expected_both[i]):
            assert np.isnan(result_both.iloc[i])
        else:
            assert result_both.iloc[i] == pytest.approx(expected_both[i])


def test_atr_nan_handling():
    """Test ATR calculation with NaN values"""
    high = np.array([10.0, 11.0, np.nan, 13.0, 14.0, 15.0])
    low = np.array([8.0, 9.0, np.nan, 11.0, 12.0, 13.0])
    close = np.array([9.0, 10.0, np.nan, 12.0, 13.0, 14.0])

    # Window size of 3
    result = atr(high, low, close, 3)

    # Check that NaN values are handled correctly
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])  # This is NaN due to the NaN in input

    # The subsequent values should be calculated by skipping NaNs
    tr_values = true_range(high, low, close)

    # From the TR values that are not NaN within the window
    valid_tr_values = [val for val in tr_values[1:4] if not np.isnan(val)]
    if len(valid_tr_values) > 0:
        expected_atr_3 = np.mean(valid_tr_values)
        assert result[3] == pytest.approx(expected_atr_3)


if __name__ == '__main__':
    pytest.main()