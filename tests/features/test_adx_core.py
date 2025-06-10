"""
Test script for the ADX (Average Directional Index) implementation.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path to import finmlkit modules
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from finmlkit.feature.core.trend import adx_core

def test_adx_basic():
    """Basic test to verify ADX calculation gives expected results."""
    # Create a sample price series with a clear trend
    n_samples = 100
    np.random.seed(42)

    # Create an uptrend
    high = np.linspace(100, 200, n_samples) + np.random.normal(0, 2, n_samples)
    low = high - 5 - np.random.uniform(0, 3, n_samples)
    close = (high + low) / 2

    # Calculate ADX with a typical length parameter
    length = 14
    adx = adx_core(high, low, close, length)

    # Check that adx is the right shape
    assert len(adx) == n_samples

    # First values should be NaN/0 due to the calculation requiring historical data
    assert np.all(adx[:2*length-1] == 0)

    # Later values should be positive (ADX is always positive)
    assert np.all(adx[2*length:] > 0)

    # ADX values are bounded between 0 and 100
    assert np.all(adx <= 100)

    # In a strong trend, ADX should be higher (typically > 25 indicates a strong trend)
    # Our data has a clear trend, so later ADX values should be relatively high
    assert np.mean(adx[50:]) > 20

def test_adx_no_trend():
    """Test ADX calculation on a flat, sideways price series."""
    # Create a flat price series with small random movements
    n_samples = 100
    np.random.seed(43)

    # Sideways market
    base = 100
    high = base + np.random.uniform(0, 2, n_samples)
    low = base - np.random.uniform(0, 2, n_samples)
    close = base + np.random.uniform(-1, 1, n_samples)

    # Calculate ADX
    length = 14
    adx = adx_core(high, low, close, length)

    # First values should be NaN/0
    assert np.all(adx[:2*length-1] == 0)

    # In a sideways market, ADX should be low (typically < 20)
    # We'll check the average of the later values
    assert np.mean(adx[50:]) < 25

def test_adx_changing_trend():
    """Test ADX calculation on a price series with changing trend direction."""
    n_samples = 200
    np.random.seed(44)

    # First half: uptrend
    high1 = np.linspace(100, 150, n_samples // 2) + np.random.normal(0, 1, n_samples // 2)
    low1 = high1 - 3 - np.random.uniform(0, 2, n_samples // 2)
    close1 = (high1 + low1) / 2

    # Second half: downtrend
    high2 = np.linspace(150, 100, n_samples // 2) + np.random.normal(0, 1, n_samples // 2)
    low2 = high2 - 3 - np.random.uniform(0, 2, n_samples // 2)
    close2 = (high2 + low2) / 2

    # Combine the data
    high = np.concatenate([high1, high2])
    low = np.concatenate([low1, low2])
    close = np.concatenate([close1, close2])

    # Calculate ADX
    length = 14
    adx = adx_core(high, low, close, length)

    # Check that adx is the right shape
    assert len(adx) == n_samples

    # First values should be NaN/0
    assert np.all(adx[:2*length-1] == 0)

    # ADX should increase at trend change points
    mid_point = n_samples // 2
    # Allow some lag for ADX to adjust to the trend change
    adjustment_period = 15
    assert np.mean(adx[mid_point:mid_point+adjustment_period]) > 0

def test_adx_extreme_values():
    """Test ADX calculation with extreme price movements."""
    n_samples = 100

    # Create a very strong uptrend
    high = np.linspace(100, 300, n_samples)  # Tripling in price
    low = high - 10
    close = (high + low) / 2

    # Calculate ADX
    length = 14
    adx = adx_core(high, low, close, length)

    # In a very strong trend, ADX should be very high (> 45)
    assert np.max(adx[50:]) > 45

def test_adx_parameter_sensitivity():
    """Test how changing the length parameter affects ADX calculation."""
    n_samples = 100
    np.random.seed(45)

    # Create a moderate uptrend
    high = np.linspace(100, 150, n_samples) + np.random.normal(0, 1, n_samples)
    low = high - 4 - np.random.uniform(0, 2, n_samples)
    close = (high + low) / 2

    # Calculate ADX with different lengths
    short_length = 7
    adx_short = adx_core(high, low, close, short_length)

    standard_length = 14
    adx_standard = adx_core(high, low, close, standard_length)

    long_length = 28
    adx_long = adx_core(high, low, close, long_length)

    # Check shapes
    assert len(adx_short) == len(adx_standard) == len(adx_long) == n_samples

    # Shorter length should respond faster to trend changes
    # Test this by comparing later values (where all have valid values)
    valid_index = max(2*short_length, 2*standard_length, 2*long_length)

    # Short length ADX should be more volatile
    assert np.std(adx_short[valid_index:]) > np.std(adx_long[valid_index:])


if __name__ == '__main__':
    pytest.main()