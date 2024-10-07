"""
A multiprocessing engine for task partitioning and parallel processing.

Note that numba function may have internal parallelization. Nested parallelization may cause performance issues:
Avoid Nested Parallelism: If you opt for multiprocessing, consider disabling Numba’s multithreading within the child
processes to prevent oversubscription. You can control this by setting environment variables like NUMBA_NUM_THREADS=1.
"""