import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.structural_breaks.cusum import chu_stinchcombe_white_developing, chu_stinchcombe_white_last, chu_stinchcombe_white_rolling

import numpy as np

def test_chu_stinchcombe_white_developing_basic():
    # Test with a simple increasing series
    y = np.arange(1, 101)
    s_n_t_values_up, s_n_t_values_down, critical_values_up, critical_values_down = chu_stinchcombe_white_developing(
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
    # Since the series is increasing, s_n_t_values_up should be significant after warmup
    assert np.all(s_n_t_values_up[50:] > critical_values_up[50:])
    # s_n_t_values_down should be close to zero
    assert np.allclose(s_n_t_values_down[10:], 0, atol=1e-8, equal_nan=True)

def test_chu_stinchcombe_white_developing_constant_series():
    # Test with a constant series
    y = np.ones(100)
    s_n_t_values_up, s_n_t_values_down, critical_values_up, critical_values_down = chu_stinchcombe_white_developing(y)
    # Since the series is constant, the test statistics should be zero after warmup
    assert np.allclose(s_n_t_values_up[30:], 0, atol=1e-8, equal_nan=True)
    assert np.allclose(s_n_t_values_down[30:], 0, atol=1e-8, equal_nan=True)

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
    s_n_t_values_up, s_n_t_values_down, critical_values_up, critical_values_down = chu_stinchcombe_white_developing(S)

    # Since the series is a GBM, test statistics should not exceed critical values significantly
    mask_up = ~np.isnan(s_n_t_values_up)
    mask_down = ~np.isnan(s_n_t_values_down)
    tolerance = 2.0
    assert np.all(s_n_t_values_up[mask_up] < critical_values_up[mask_up] + tolerance)
    assert np.all(s_n_t_values_down[mask_down] < critical_values_down[mask_down] + tolerance)

def test_chu_stinchcombe_white_last():
    # Test the last value function
    y = np.arange(1, 101)
    s_n_t_up, s_n_t_down, c_value_up, c_value_down = chu_stinchcombe_white_last(y)
    assert isinstance(s_n_t_up, float)
    assert isinstance(s_n_t_down, float)
    assert isinstance(c_value_up, float)
    assert isinstance(c_value_down, float)
    # Since the series is increasing, s_n_t_up should be significant
    assert s_n_t_up > c_value_up
    # s_n_t_down should be close to zero
    assert np.isclose(s_n_t_down, 0, atol=1e-8)

def test_chu_stinchcombe_white_rolling():
    # Test the rolling function
    y = np.arange(1, 2000)
    snt_up, snt_down, critical_values_up, critical_values_down = chu_stinchcombe_white_rolling(
        y, window_size=1000, warmup_period=30
    )
    assert len(snt_up) == len(y)
    assert len(snt_down) == len(y)
    assert len(critical_values_up) == len(y)
    assert len(critical_values_down) == len(y)
    # Check that initial values are NaN
    assert np.all(np.isnan(snt_up[:30]))
    # Since the series is increasing, snt_up should be significant after warmup
    assert np.all(snt_up[50:] > critical_values_up[50:])
    # snt_down should be close to zero
    assert np.allclose(snt_down[30:], 0, atol=1e-8, equal_nan=True)

def test_chu_stinchcombe_white_rolling_large_window():
    # Test with a window size larger than data length
    y = np.arange(1, 500)
    snt_up, snt_down, critical_values_up, critical_values_down = chu_stinchcombe_white_rolling(
        y, window_size=1000, warmup_period=30
    )
    assert len(snt_up) == len(y)
    assert len(snt_down) == len(y)
    # Check that the function handles window size larger than data length
    # and produces outputs of correct length
    # Since the series is increasing, snt_up should be significant after warmup
    assert np.all(snt_up[50:] > critical_values_up[50:])
    # snt_down should be close to zero
    assert np.allclose(snt_down[30:], 0, atol=1e-8, equal_nan=True)