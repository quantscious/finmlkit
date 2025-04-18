import numpy as np
import time
from finmlkit.structural_break.cusum import cusum_test_rolling

def csw_rolling_perf_test():
    for _ in range(10):
        # Generate a large random array
        arr_in = np.random.rand(50_000)
        window = 500
        warmup = 30

        # Time the function
        start = time.time()
        cusum_test_rolling(arr_in, window, warmup)
        end = time.time()
        dur = end - start
        print(f"\nTime taken for chu_stinchcombe_white_rolling: {dur:.5f} seconds")


if __name__ == "__main__":
    csw_rolling_perf_test()

    """
    Time taken for chu_stinchcombe_white_rolling: ~2.5 seconds without parallelization
    Time taken for chu_stinchcombe_white_rolling: 0.4 seconds with parallelization
    """