import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.indicators.utils import ewms

def test_ewm_std_against_pandas():
    """
    Test the ewm_std function against pandas' ewm.std function to ensure correctness.
    """
    arr_in = np.random.rand(1000)
    window = 10
    our_ewm_std = ewms(arr_in, window)
    pandas_ewm_std = pd.Series(arr_in).ewm(span=window, adjust=True).std(bias=False).to_numpy()
    assert_allclose(our_ewm_std, pandas_ewm_std, rtol=1e-5, atol=1e-8, equal_nan=True)

def test_ewm_std_with_known_values():
    """
    Test the ewm_std function with known input and output.
    """
    arr_in = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    window = 2
    our_ewm_std = ewms(arr_in, window)
    pandas_ewm_std = pd.Series(arr_in).ewm(span=window, adjust=True).std(bias=False).to_numpy()
    assert_allclose(our_ewm_std, pandas_ewm_std, rtol=1e-5, atol=1e-8)

def test_ewm_std_with_known_values_and_nans():
    """
    Test the ewm_std function with known input and output.
    """
    arr_in = np.array([1, 2, 3, np.nan, np.nan, 4, 5], dtype=np.float64)
    window = 2
    our_ewm_std = ewms(arr_in, window)
    pandas_ewm_std = pd.Series(arr_in).ewm(span=window, adjust=True).std(bias=False).to_numpy()
    assert_allclose(our_ewm_std, pandas_ewm_std, rtol=1e-5, atol=1e-8)

def test_ewm_std_empty_array():
    """
    Test the ewm_std function with an empty array.
    """
    arr_in = np.array([], dtype=np.float64)
    window = 10
    our_ewm_std = ewms(arr_in, window)
    assert our_ewm_std.size == 0

def test_ewm_std_window_size_one():
    """
    Test the ewm_std function with a window size of one.
    """
    arr_in = np.random.rand(100)
    window = 1
    our_ewm_std = ewms(arr_in, window)
    pandas_ewm_std = pd.Series(arr_in).ewm(span=window, adjust=True).std(bias=False).to_numpy()
    assert_allclose(our_ewm_std, pandas_ewm_std, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])