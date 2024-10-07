import numpy as np
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.bars.logic import time_bar_indexer

def test_time_bar_indexer():
    # Prepare test data
    timestamps = np.array([
        999_999_999,    # 0.999999999 seconds
        1_000_000_000,  # 1 second
        2_000_000_000,  # 2 seconds
        3_000_000_000,  # 3 seconds
        4_000_000_000,  # 4 seconds
        5_000_000_000,  # 5 seconds
        5_999_999_999,  # 5.999999999 seconds
        6_100_000_000,  # 6.1 seconds
        7_000_000_000,  # 7 seconds
    ], dtype=np.int64)
    interval_seconds = 2  # Bar length of 2 seconds

    # Expected output
    expected_bar_open_indices = np.array([2, 4, 7])

    # Call the function
    bar_open_indices = time_bar_indexer(timestamps, interval_seconds)

    # Assert the output is as expected
    np.testing.assert_array_equal(bar_open_indices, expected_bar_open_indices)