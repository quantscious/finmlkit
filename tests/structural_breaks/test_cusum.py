import numpy as np
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.structural_breaks.cusum import chu_stinchcombe_white_developing, chu_stinchcombe_white_last, chu_stinchcombe_white_rolling

def test_chu_stinchcombe_white_developing_basic():
    # Test with a simple increasing series
    y = np.arange(1, 101)
    s_n_t_values_up, s_n_t_values_down, s_n_t_values_abs, critical_values = chu_stinchcombe_white_developing(y, warmup_period=10)
    # Check that outputs are arrays of the correct length
    assert len(s_n_t_values_up) == len(y)
    assert len(s_n_t_values_down) == len(y)
    assert len(s_n_t_values_abs) == len(y)
    assert len(critical_values) == len(y)
    # Check that the first warmup_period values are NaN
    assert np.all(np.isnan(s_n_t_values_up[:10]))
    assert np.all(np.isnan(s_n_t_values_down[:10]))
    assert np.all(np.isnan(s_n_t_values_abs[:10]))
    assert np.all(np.isnan(critical_values[:10]))

def test_chu_stinchcombe_white_developing_constant_series():
    # Test with a constant series
    y = np.ones(100)
    s_n_t_values_up, s_n_t_values_down, s_n_t_values_abs, critical_values = chu_stinchcombe_white_developing(y)
    # Since the series is constant, the test statistics should be zero after warmup
    assert np.allclose(s_n_t_values_up[30:], 0, equal_nan=True)
    assert np.allclose(s_n_t_values_down[30:], 0, equal_nan=True)
    assert np.allclose(s_n_t_values_abs[30:], 0, equal_nan=True)

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
    s_n_t_values_up, s_n_t_values_down, s_n_t_values_abs, critical_values = chu_stinchcombe_white_developing(S)

    # Since the series is a GBM, test statistics should not exceed critical values significantly
    mask = ~np.isnan(s_n_t_values_abs)
    tolerance = 2.
    assert np.all(s_n_t_values_abs[mask] < critical_values[mask] + tolerance)

def test_chu_stinchcombe_white_last():
    # Test the last value function
    y = np.arange(1, 101)
    s_n_t_up, s_n_t_down, s_n_t_abs, c_value = chu_stinchcombe_white_last(y)
    assert isinstance(s_n_t_up, float)
    assert isinstance(s_n_t_down, float)
    assert isinstance(s_n_t_abs, float)
    assert isinstance(c_value, float)

def test_chu_stinchcombe_white_rolling():
    # Test the rolling function
    y = np.arange(1, 2000)
    snt_up, snt_down, snt_abs, critical_values = chu_stinchcombe_white_rolling(y, window_size=1000, warmup_period=30)
    assert len(snt_up) == len(y)
    assert len(snt_down) == len(y)
    assert len(snt_abs) == len(y)
    assert len(critical_values) == len(y)
    # Check that initial values are NaN
    assert np.all(np.isnan(snt_up[:30]))
    # Since the series is increasing, snt_up should be significant
    mask = ~np.isnan(snt_up)
    assert np.all(snt_up[mask] > 0)

def test_chu_stinchcombe_white_rolling_small_window():
    # Test with a window size smaller than data length
    y = np.arange(1, 500)
    snt_up, snt_down, snt_abs, critical_values = chu_stinchcombe_white_rolling(y, window_size=1000, warmup_period=30)
    assert len(snt_up) == len(y)
    # Check that function handles window size larger than data length
    # and produces outputs of correct length