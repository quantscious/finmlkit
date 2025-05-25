import pytest
import numpy as np
import pandas as pd
from finmlkit.feature.transforms import RealizedVolatility


@pytest.fixture
def sample_data():
    # Create sample data with known values including NaN
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    returns = [0.01, 0.02, -0.01, 0.03, np.nan, 0.01, -0.02, 0.02, -0.03, 0.01,
               0.02, 0.01, -0.01, 0.02, np.nan, 0.01, -0.01, 0.03, -0.02, 0.01]
    return pd.DataFrame({'ret': returns}, index=dates)


def test_realised_volatility_implementations_match(sample_data):
    """Test that pandas and numba implementations give the same results"""
    rv = RealizedVolatility(window=5, is_sample=True, input_col='ret')

    # Run both implementations
    pd_result = rv._pd(sample_data)
    nb_result = rv._nb(sample_data)

    # Results should be identical
    pd.testing.assert_series_equal(pd_result, nb_result, rtol=1e-10)


def test_realised_volatility_sample_vs_population(sample_data):
    """Test that sample and population calculations differ correctly"""
    rv_sample = RealizedVolatility(window=5, is_sample=True, input_col='ret')
    rv_pop = RealizedVolatility(window=5, is_sample=False, input_col='ret')

    result_sample = rv_sample._pd(sample_data)
    result_pop = rv_pop._pd(sample_data)

    # Population should be smaller than sample (due to n vs n-1 divisor)
    mask = ~(pd.isna(result_sample) | pd.isna(result_pop))
    assert all(result_pop[mask] < result_sample[mask])


def test_realised_volatility_window_size(sample_data):
    """Test that window size works correctly"""
    rv = RealizedVolatility(window=5, is_sample=True, input_col='ret')
    result = rv._pd(sample_data)

    # First (window-1) values should be NaN
    assert all(pd.isna(result.iloc[:4]))
    # Values after that should be calculated
    assert not pd.isna(result.iloc[4])

    # Check a larger window
    rv_large = RealizedVolatility(window=10, is_sample=True, input_col='ret')
    result_large = rv_large._pd(sample_data)
    assert all(pd.isna(result_large.iloc[:9]))


def test_realised_volatility_handles_nan(sample_data):
    """Test proper handling of NaN values"""
    # Create a window of all NaN values
    all_nan_data = sample_data.copy()
    all_nan_data.iloc[5:10, 0] = np.nan

    rv = RealizedVolatility(window=5, is_sample=True, input_col='ret')
    result = rv._pd(all_nan_data)

    # Windows with less than 2 valid values should be NaN
    assert pd.isna(result.iloc[9])


if __name__ == "__main__":
    pytest.main([__file__])