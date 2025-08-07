import numpy as np
import pandas as pd
import time

from finmlkit.feature.core.volatility import ewms
from finmlkit.feature.core.ma import ewma


def ewma_perf_test():
    for _ in range(10):
        # Generate a large random array
        arr_in = np.random.rand(1_000_000)
        window = 100

        # Time the function
        start = time.time()
        ewma(arr_in, window)
        end = time.time()
        dur = end - start
        print(f"\nTime taken for ewma: {dur:.5f} seconds")

        # Time the pandas function
        start = time.time()
        pd.Series(arr_in).ewm(span=window, adjust=True).mean()
        end = time.time()
        dur = end - start
        print(f"Time taken for pandas ewma: {dur:.5f} seconds")

def ewm_std_perf_test():
    for _ in range(10):
        # Generate a large random array
        arr_in = np.random.rand(1_000_000)
        window = 100

        # Time the function
        start = time.time()
        ewms(arr_in, window)
        end = time.time()
        dur = end - start
        print(f"\nTime taken for ewm_std: {dur:.5f} seconds")

        # Time the pandas function
        start = time.time()
        pd.Series(arr_in).ewm(span=window, adjust=True).std()
        end = time.time()
        dur = end - start
        print(f"Time taken for pandas ewm_std: {dur:.5f} seconds")


if __name__ == "__main__":
    #ewma_perf_test()
    """
    Time taken for ewma: 0.00227 seconds
    Time taken for pandas ewma: 0.00812 seconds
    
    Time taken for ewma: 0.00170 seconds
    Time taken for pandas ewma: 0.00786 seconds
    
    Time taken for ewma: 0.00194 seconds
    Time taken for pandas ewma: 0.00775 seconds
    
    Time taken for ewma: 0.00206 seconds
    Time taken for pandas ewma: 0.00789 seconds
    
    Time taken for ewma: 0.00220 seconds
    Time taken for pandas ewma: 0.00724 seconds
    
    Time taken for ewma: 0.00164 seconds
    Time taken for pandas ewma: 0.00749 seconds
    
    Time taken for ewma: 0.00186 seconds
    Time taken for pandas ewma: 0.00809 seconds
    
    Time taken for ewma: 0.00180 seconds
    Time taken for pandas ewma: 0.00715 seconds
    
    Time taken for ewma: 0.00195 seconds
    Time taken for pandas ewma: 0.00793 seconds
    
    -> ~4x speedup
    """

    ewm_std_perf_test()
    """
    Time taken for ewm_std: 0.00670 seconds
    Time taken for pandas ewm_std: 0.01016 seconds
    
    Time taken for ewm_std: 0.00658 seconds
    Time taken for pandas ewm_std: 0.01013 seconds
    
    Time taken for ewm_std: 0.00655 seconds
    Time taken for pandas ewm_std: 0.00973 seconds
    
    Time taken for ewm_std: 0.00658 seconds
    Time taken for pandas ewm_std: 0.01040 seconds
    
    Time taken for ewm_std: 0.00657 seconds
    Time taken for pandas ewm_std: 0.00983 seconds
    
    Time taken for ewm_std: 0.00676 seconds
    Time taken for pandas ewm_std: 0.01003 seconds
    
    Time taken for ewm_std: 0.00659 seconds
    Time taken for pandas ewm_std: 0.01141 seconds
    
    Time taken for ewm_std: 0.00684 seconds
    Time taken for pandas ewm_std: 0.01103 seconds
    
    Time taken for ewm_std: 0.00706 seconds
    Time taken for pandas ewm_std: 0.01263 seconds
    
    --> ~1.5x speedup
    """
