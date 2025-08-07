import pytest
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.feature.core.structural_break.cusum import cusum_test_developing, cusum_test_last, cusum_test_rolling
import numpy as np

def test_chu_stinchcombe_white_developing_basic():
    # Test with a realistic price series that has an upward trend
    # Create prices that increase exponentially to ensure positive values
    y = 100 * np.exp(0.001 * np.arange(100))  # Exponential growth
    s_n_t_values_up, s_n_t_values_down, critical_values_up, critical_values_down = cusum_test_developing(
        y, warmup_period=10
    )
    # Check that outputs are arrays of the correct length
    assert len(s_n_t_values_up) == len(y)
    assert len(s_n_t_values_down) == len(y)
    assert len(critical_values_up) == len(y)
    assert len(critical_values_down) == len(y)
    # Check that the first warmup_period values are NaN
    assert np.all(np.isnan(s_n_t_values_up[:10]))
    assert np.all(np.isnan(s_n_t_values_down[:10]))
    assert np.all(np.isnan(critical_values_up[:10]))
    assert np.all(np.isnan(critical_values_down[:10]))
    # Check that values after warmup are finite
    assert np.all(np.isfinite(s_n_t_values_up[10:]))
    assert np.all(np.isfinite(s_n_t_values_down[10:]))

def test_chu_stinchcombe_white_developing_constant_series():
    # Test with a series that has small random fluctuations around a constant
    # Avoid perfectly constant series to prevent division by zero
    np.random.seed(42)
    y = 100 + 0.01 * np.random.randn(100)  # Small fluctuations around 100
    s_n_t_values_up, s_n_t_values_down, critical_values_up, critical_values_down = cusum_test_developing(y)
    # Check that outputs are finite after warmup period
    assert np.all(np.isfinite(s_n_t_values_up[30:]))
    assert np.all(np.isfinite(s_n_t_values_down[30:]))
    # Test statistics should be relatively small for near-constant series
    assert np.all(np.abs(s_n_t_values_up[30:]) < 10)  # Reasonable bound
    assert np.all(np.abs(s_n_t_values_down[30:]) < 10)

def test_chu_stinchcombe_white_developing_random_walk():
    # Simulate a geometric Brownian motion (GBM)
    np.random.seed(0)
    mu = 0.0      # Drift
    sigma = 0.01  # Volatility
    S0 = 100      # Initial price
    n = 1000      # Number of time steps
    dt = 1        # Time step size

    # Simulate log returns
    epsilon = np.random.normal(0, 1, n)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon

    # Generate price series
    log_S = np.log(S0) + np.cumsum(log_returns)
    S = np.exp(log_S)

    # Apply the Chu-Stinchcombe-White test
    s_n_t_values_up, s_n_t_values_down, critical_values_up, critical_values_down = cusum_test_developing(S)

    # Check that outputs are finite after warmup
    mask_up = ~np.isnan(s_n_t_values_up)
    mask_down = ~np.isnan(s_n_t_values_down)
    assert np.all(np.isfinite(s_n_t_values_up[mask_up]))
    assert np.all(np.isfinite(s_n_t_values_down[mask_down]))
    # For a true random walk, most statistics should be below critical values
    # Allow for some statistical variation
    exceedances_up = np.mean(s_n_t_values_up[mask_up] > critical_values_up[mask_up])
    exceedances_down = np.mean(s_n_t_values_down[mask_down] > critical_values_down[mask_down])
    assert exceedances_up < 0.15  # Less than 15% exceedances
    assert exceedances_down < 0.15

def test_chu_stinchcombe_white_last():
    # Test the last value function with realistic price data
    y = 100 * np.exp(0.001 * np.arange(100))  # Exponential growth
    s_n_t_up, s_n_t_down, c_value_up, c_value_down = cusum_test_last(y)
    assert isinstance(s_n_t_up, float)
    assert isinstance(s_n_t_down, float)
    assert isinstance(c_value_up, float)
    assert isinstance(c_value_down, float)
    # Check that all values are finite
    assert np.isfinite(s_n_t_up)
    assert np.isfinite(s_n_t_down)
    assert np.isfinite(c_value_up)
    assert np.isfinite(c_value_down)

def test_chu_stinchcombe_white_rolling():
    # Test the rolling function with realistic price data
    y = 100 * np.exp(0.0005 * np.arange(2000))  # Slower exponential growth
    snt_up, snt_down, critical_values_up, critical_values_down = cusum_test_rolling(
        y, window_size=1000, warmup_period=30
    )
    assert len(snt_up) == len(y)
    assert len(snt_down) == len(y)
    assert len(critical_values_up) == len(y)
    assert len(critical_values_down) == len(y)
    # Check that initial values are NaN
    assert np.all(np.isnan(snt_up[:30]))
    # Check that computed values are finite
    finite_mask = ~np.isnan(snt_up)
    assert np.all(np.isfinite(snt_up[finite_mask]))
    assert np.all(np.isfinite(snt_down[finite_mask]))

def test_chu_stinchcombe_white_rolling_large_window():
    # Test with a window size larger than data length
    y = 100 * np.exp(0.001 * np.arange(500))  # Exponential growth
    snt_up, snt_down, critical_values_up, critical_values_down = cusum_test_rolling(
        y, window_size=1000, warmup_period=30
    )
    assert len(snt_up) == len(y)
    assert len(snt_down) == len(y)
    # Check that computed values after warmup are finite
    finite_mask = ~np.isnan(snt_up)
    assert np.all(np.isfinite(snt_up[finite_mask]))
    assert np.all(np.isfinite(snt_down[finite_mask]))
    # Check that the function handles window size larger than data length properly
    assert np.sum(finite_mask) > 0  # Should have some computed values

if __name__ == "__main__":
    pytest.main()