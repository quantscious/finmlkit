import numpy as np
from numpy.testing import assert_allclose

from finmlkit.feature.core.utils import (
    comp_lagged_returns,
    comp_zscore,
    comp_burst_ratio,
    pct_change,
)


def test_comp_lagged_returns_log_and_zero_price():
    """Cover log-return branch and division-by-zero handling."""
    ts = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)

    # division by zero path
    close_zero = np.array([0.0, 2.0, 4.0], dtype=np.float64)
    simple = comp_lagged_returns(ts, close_zero, 1, is_log=False)
    assert np.isnan(simple[0])
    assert np.isinf(simple[1])

    # log returns path
    close = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    log_res = comp_lagged_returns(ts, close, 1, is_log=True)
    expected = np.array([np.nan, np.log(2/1), np.log(4/2)], dtype=np.float64)
    assert_allclose(log_res, expected, equal_nan=True)


def test_comp_lagged_returns_no_valid_lag():
    """Unsorted timestamps trigger the missing-lag branch."""
    ts = np.array([0, 1_000_000_000, -1_000_000_000], dtype=np.int64)
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    res = comp_lagged_returns(ts, close, 1, is_log=False)
    assert np.isnan(res[2])


def test_comp_zscore_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = comp_zscore(x, window=3, ddof=0)
    expected = np.array([
        np.nan,
        np.nan,
        1.22474487,
        1.22474487,
        1.22474487,
    ])
    assert_allclose(result, expected, equal_nan=True, rtol=1e-8)


def test_comp_burst_ratio_and_zero_median():
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    res = comp_burst_ratio(series, window=3)
    expected = np.array([np.nan, np.nan, 1.5, 4/3, 1.25])
    assert_allclose(res, expected, equal_nan=True)

    series_zero = np.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    res_zero = comp_burst_ratio(series_zero, window=3)
    assert np.isnan(res_zero[2])


def test_pct_change_basic_and_nonpositive_base():
    x = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    res = pct_change(x, periods=1)
    expected = np.array([np.nan, np.nan, 1.0, 1.0])
    assert_allclose(res, expected, equal_nan=True)

    x_neg = np.array([-1.0, -0.5, 0.0], dtype=np.float64)
    res_neg = pct_change(x_neg, periods=1)
    assert np.isnan(res_neg[1]) and np.isnan(res_neg[2])

