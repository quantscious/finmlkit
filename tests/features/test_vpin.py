"""
Tests for VPIN (Volume-synchronized Probability of Informed Trading) feature
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from finmlkit.feature.core.volume import vpin
from finmlkit.feature.transforms import VPIN


def test_vpin_basic():
    """Test basic functionality of VPIN calculation"""
    # Create sample data with known imbalance patterns
    volume_buy = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0])
    volume_sell = np.array([200.0, 150.0, 100.0, 50.0, 0.0, 550.0, 500.0, 450.0, 400.0, 350.0])

    # Window size of 4
    window = 4
    result = vpin(volume_buy, volume_sell, window)

    # First window-1 results should be NaN
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[2])

    # Verify the rest have valid values
    for i in range(window, len(result)):
        assert not np.isnan(result[i])
        assert 0.0 <= result[i] <= 1.0  # VPIN values should be in [0, 1]


def test_vpin_perfect_balance():
    """Test with perfectly balanced buy and sell volumes"""
    # Create perfectly balanced data (volume_buy = volume_sell)
    volume_buy = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    volume_sell = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

    # Window size of 4
    result = vpin(volume_buy, volume_sell, window=4)

    # Check values for perfect balance (ignoring first window-1 NaNs)
    for i in range(4, len(result)):
        assert result[i] == 0.0  # Perfect balance should give VPIN = 0


def test_vpin_perfect_imbalance():
    """Test with perfectly imbalanced data (only buys or only sells)"""
    # Create perfectly imbalanced data (only buys)
    volume_buy = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    volume_sell = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Window size of 4
    result = vpin(volume_buy, volume_sell, window=4)

    # Check values for perfect imbalance (ignoring first window-1 NaNs)
    for i in range(4, len(result)):
        assert result[i] == 1.0  # Perfect imbalance should give VPIN = 1


def test_vpin_nan_handling():
    """Test NaN handling in VPIN calculation"""
    # Create data with NaN values
    volume_buy = np.array([100.0, np.nan, 100.0, 100.0, 100.0, 100.0, 100.0, np.nan, 100.0, 100.0])
    volume_sell = np.array([100.0, 100.0, np.nan, 100.0, 100.0, 100.0, np.nan, 100.0, 100.0, 100.0])

    # Window size of 4
    result = vpin(volume_buy, volume_sell, window=4)

    # Check that NaN values are handled correctly
    for i in range(len(result)):
        if i < 4:  # First window-1 values are always NaN
            assert np.isnan(result[i])


def test_vpin_zero_volume():
    """Test handling of zero volume (which would cause division by zero)"""
    # Create data with zero volume in some bars
    volume_buy = np.array([100.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 100.0, 100.0, 100.0])
    volume_sell = np.array([100.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 100.0, 100.0, 100.0])

    # Window size of 4
    result = vpin(volume_buy, volume_sell, window=4)

    # The function should handle zero volumes correctly
    assert not np.isnan(result[4])  # Should still be a valid result despite zeros in the window


def test_vpin_small_window():
    """Test with a small window size"""
    volume_buy = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    volume_sell = np.array([200.0, 150.0, 100.0, 50.0, 0.0])

    # Window size of 2
    result = vpin(volume_buy, volume_sell, window=2)

    # First window-1 results should be NaN
    assert np.isnan(result[0])
    assert not np.isnan(result[2])  # Should have valid values from index 2 onward


def test_transform_basic():
    """Test the VPIN transform class"""
    # Create sample data
    dates = [datetime(2025, 6, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({
        'volume_buy': [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0],
        'volume_sell': [200.0, 150.0, 100.0, 50.0, 0.0, 550.0, 500.0, 450.0, 400.0, 350.0]
    }, index=dates)

    # Test standard transform
    transform = VPIN(window=4)
    result = transform(df)

    # Column name should match the pattern
    assert result.name == 'vpin_4'

    # Compare with direct calculation
    expected = vpin(df['volume_buy'].values, df['volume_sell'].values, 4)
    for i in range(len(expected)):
        if np.isnan(expected[i]):
            assert np.isnan(result.iloc[i])
        else:
            assert abs(result.iloc[i] - expected[i]) < 1e-10


def test_transform_custom_columns():
    """Test the VPIN transform with custom column names"""
    # Create sample data with custom column names
    dates = [datetime(2025, 6, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({
        'buy_vol': [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0],
        'sell_vol': [200.0, 150.0, 100.0, 50.0, 0.0, 550.0, 500.0, 450.0, 400.0, 350.0]
    }, index=dates)

    # Create transform with custom column names
    transform = VPIN(window=4, input_cols=['buy_vol', 'sell_vol'])
    result = transform(df)

    # Compare with direct calculation
    expected = vpin(df['buy_vol'].values, df['sell_vol'].values, 4)
    for i in range(len(expected)):
        if np.isnan(expected[i]):
            assert np.isnan(result.iloc[i])
        else:
            assert abs(result.iloc[i] - expected[i]) < 1e-10


def test_transform_pandas_impl():
    """Test the pandas implementation of the transform"""
    # Create sample data
    dates = [datetime(2025, 6, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({
        'volume_buy': [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0],
        'volume_sell': [200.0, 150.0, 100.0, 50.0, 0.0, 550.0, 500.0, 450.0, 400.0, 350.0]
    }, index=dates)

    # Calculate using pandas implementation
    transform = VPIN(window=4)
    result_pd = transform._pd(df)

    # Calculate using numba implementation
    result_nb = transform._nb(df)

    # Both implementations should produce the same results
    for i in range(len(result_pd)):
        if np.isnan(result_pd.iloc[i]):
            assert np.isnan(result_nb.iloc[i])
        else:
            assert result_pd.iloc[i] == pytest.approx(result_nb.iloc[i])


def test_vpin_window32():
    """Test the default VPIN-32 configuration"""
    # Create sample data with 40 bars
    volume_buy = np.arange(100, 100 + 40 * 10, 10)
    volume_sell = np.arange(200, 200 + 40 * 5, 5)

    # Window size of 32 (default)
    result = vpin(volume_buy, volume_sell, window=32)

    # First 31 results should be NaN
    for i in range(31):
        assert np.isnan(result[i])

    # The rest should have valid values
    for i in range(32, len(result)):
        assert not np.isnan(result[i])
        assert 0.0 <= result[i] <= 1.0


def test_real_world_scenario():
    """Test a real-world scenario with changing imbalance patterns"""
    # Create a more complex dataset with varying patterns
    n = 100
    volume_buy = np.zeros(n)
    volume_sell = np.zeros(n)

    # First 20 bars: balanced
    volume_buy[:20] = 100
    volume_sell[:20] = 100

    # Next 20 bars: buy heavy
    volume_buy[20:40] = 200
    volume_sell[20:40] = 50

    # Next 20 bars: sell heavy
    volume_buy[40:60] = 50
    volume_sell[40:60] = 200

    # Next 20 bars: volatile
    for i in range(60, 80):
        if i % 2 == 0:
            volume_buy[i] = 300
            volume_sell[i] = 50
        else:
            volume_buy[i] = 50
            volume_sell[i] = 300

    # Last 20 bars: gradually increasing imbalance
    for i in range(80, 100):
        ratio = (i - 80) / 20
        volume_buy[i] = 100 + ratio * 200
        volume_sell[i] = 100

    # Calculate VPIN with window 32
    result = vpin(volume_buy, volume_sell, window=32)

    # Check transitions between different regimes
    # Around bar 52 we should see high VPIN (transition from buy to sell heavy)
    assert 0.5 < result[52]

    # Around bar 80 we should see high VPIN (volatile section)
    assert 0.5 < result[80]


if __name__ == '__main__':
    pytest.main()