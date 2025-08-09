# Tests

Testing is done using the `pytest` framework. Install the package in
editable mode with development dependencies before running tests:

```bash
pip install -e .[dev]
```

> [!CAUTION]
> Note that **testing numba function** are often **tricky and challenging**. I recommend to first disable jit and debug the code in pure python. If that passes, then re-enable jit and run the tests again. Both ways should pass. 

You can disable JIT by setting the environment variable `NUMBA_DISABLE_JIT=1`.
```python test_my_testfile.py
# test_my_testfile.py
import os
os.environ['NUMBA_DISABLE_JIT'] = "1"  # disable jit

# important to do this before importing numba functions
from finmlkit.utils import my_numba_fn
...
```
After you are done, it is important to **re-enable jit** by removing or commenting out the above code (otherwise it can broke the full test pipeline if numba disabling stays there) 

Alternatively, run the entire suite with `NUMBA_DISABLE_JIT=1 pytest`.

For convenience, bash scripts are provided to create a fresh virtual test
environment and run all tests with and without JIT enabled.

Run these from the project root:

```bash
chmod +x local_test.sh
./local_test.sh          # JIT enabled

chmod +x local_test_nojit.sh
./local_test_nojit.sh    # JIT disabled
```

>[!IMPORTANT]
> The github workflow for CI will run the tests with jit disabled, because numba is not compatible with mass testing. Nevertheless, it is important to test with jit enabled locally to ensure that the code works as expected in production.