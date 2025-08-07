import numpy as np
import pytest
# import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)
from finmlkit.bar.base import comp_bar_trade_size_features


def _run(amounts, theta, closes, mult=5.0):
    """Helper that casts to required dtypes."""
    return comp_bar_trade_size_features(
        amounts.astype(np.float64),
        theta.astype(np.float64),
        closes.astype(np.int64),
        mult
    )

# ------------------------------------------------------------------
# 1) Two deterministic 3-trade bars, no blocks
# ------------------------------------------------------------------
def test_two_bars():
    amounts = np.array([10, 20, 30, 40, 50, 60])
    # In the comp_bar_trade_size_features function, bar_close_indices should contain indices where each bar ends
    # The number of bars is len(bar_close_indices) - 1
    closes = np.array([0, 3, 5])  # This means: bar 0 spans [1:3], bar 1 spans [4:5]
    # theta array should match the number of bars exactly
    theta = np.array([10, 10])  # per-bar typical trade sizes (2 bars)
    mult = 5.0  # Define the default multiplier value

    m, p95, pct, g = _run(amounts, theta, closes)

    # Manually calculate expected values for bar 0
    bar0_amounts = amounts[1:4]  # [20, 30, 40]
    bar0_threshold = theta[0] * mult  # 10 * 5 = 50
    bar0_mean = np.mean(bar0_amounts)
    expected_mean0 = np.log1p(bar0_mean / bar0_threshold)  # log1p(30/50)

    # Calculate 95th percentile for bar 0
    bar0_p95 = np.percentile(bar0_amounts, 95)
    expected_p95_0 = np.log1p(bar0_p95 / bar0_threshold)

    # Calculate block percentage for bar 0 (no blocks)
    expected_pct0 = 0.0  # No trades above threshold of 50

    # Manually calculate expected values for bar 1
    bar1_amounts = amounts[4:6]  # [50, 60]
    bar1_threshold = theta[1] * mult  # 10 * 5 = 50
    bar1_mean = np.mean(bar1_amounts)
    expected_mean1 = np.log1p(bar1_mean / bar1_threshold)  # log1p(55/50)

    # Calculate 95th percentile for bar 1
    bar1_p95 = np.percentile(bar1_amounts, 95)
    expected_p95_1 = np.log1p(bar1_p95 / bar1_threshold)

    # Calculate block percentage for bar 1
    bar1_total_volume = np.sum(bar1_amounts)  # 50+60 = 110
    bar1_block_volume = 0.0
    for amount in bar1_amounts:
        if amount > bar1_threshold:  # Check both amounts against 50
            bar1_block_volume += amount
    expected_pct1 = bar1_block_volume / bar1_total_volume if bar1_total_volume > 0 else 0.0

    # bar-0: 20/30/40, thr = 10*5 = 50
    assert np.isclose(m[0], expected_mean0, atol=1e-6)
    assert np.isclose(pct[0], expected_pct0, atol=1e-6)
    assert np.isclose(p95[0], expected_p95_0, atol=1e-6)

    # bar-1: 50/60 , thr = 10*5 = 50
    assert np.isclose(m[1], expected_mean1, atol=1e-6)
    assert np.isclose(p95[1], expected_p95_1, atol=1e-6)
    assert np.isclose(pct[1], expected_pct1, atol=1e-6)

# ------------------------------------------------------------------
# 2) Block volume test  (only trade 60 > thr)
# ------------------------------------------------------------------
def test_block_volume():
    amounts = np.array([1, 1, 50, 60])
    closes = np.array([0, 4])  # One bar spanning [1:4]
    theta = np.array([10])  # theta must match the number of bars (1 bar)
    mult = 5.0

    _, _, pct, _ = _run(amounts, theta, closes)

    # Calculate expected block percentage
    bar_amounts = amounts[1:5]  # [1, 1, 50, 60]
    threshold = theta[0] * mult  # 10 * 5 = 50

    total_volume = np.sum(bar_amounts)  # 1+1+50+60 = 112
    block_volume = 0.0
    for amount in bar_amounts:
        if amount > threshold:  # Only 60 > 50
            block_volume += amount

    expected_pct = block_volume / total_volume  # 60/112 ≈ 0.5357

    # Use direct calculation rather than range
    assert np.isclose(pct[0], expected_pct, atol=1e-6)

# ------------------------------------------------------------------
# 5) θ = 0 guard (warm-up)
# ------------------------------------------------------------------
def test_zero_theta_guard():
    amounts = np.array([1, 1])
    closes = np.array([0, 2])  # One bar with both trades
    theta = np.array([0])   # theta must match the number of bars (1 bar)

    m, p95, pct, g = _run(amounts, theta, closes)

    assert np.isnan(m[0])  # Mean size relative should be NaN when theta is 0
    assert np.isnan(p95[0])  # 95th percentile should be NaN when theta is 0

if __name__ == '__main__':
    pytest.main([__file__])
