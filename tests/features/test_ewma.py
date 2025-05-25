import numpy as np
import pandas as pd

import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.feature.ma import ewma


def test_ewma_against_pandas():
    """
    Test the ewma function against pandas' ewm function to ensure correctness.
    """
    arr_in = np.random.rand(1000)
    window = 10
    our_ewma = ewma(arr_in, window)
    pandas_ewma = pd.Series(arr_in).ewm(span=window, adjust=True).mean().to_numpy()
    np.testing.assert_allclose(our_ewma, pandas_ewma, rtol=1e-5, atol=1e-8)

def test_ewma_with_known_values():
    """
    Test the ewma function with known input and output.
    """
    arr_in = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    window = 2
    expected_output = pd.Series(arr_in).ewm(span=window, adjust=True).mean().to_numpy()
    our_ewma = ewma(arr_in, window)
    np.testing.assert_allclose(our_ewma, expected_output, rtol=1e-5, atol=1e-8)

def test_ewma_empty_array():
    """
    Test the ewma function with an empty array.
    """
    arr_in = np.array([], dtype=np.float64)
    window = 10
    our_ewma = ewma(arr_in, window)
    assert our_ewma.size == 0

def test_ewma_window_size_one():
    """
    Test the ewma function with window size of one.
    """
    arr_in = np.random.rand(100)
    window = 1
    our_ewma = ewma(arr_in, window)
    expected_output = pd.Series(arr_in).ewm(span=window, adjust=True).mean().to_numpy()
    np.testing.assert_allclose(our_ewma, expected_output, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])