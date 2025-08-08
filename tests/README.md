# Tests

Testing is done using the `pytest` framework. 

> [!CAUTION]
> Note that **testing numba function** are often **tricky and challenging**. I recommend to first disable jit and debug the code in pure python. If that passes, then re-enable jit and run the tests again. Both ways should pass.

Numba-compiled functions are tested in two modes:
1. **With JIT enabled**: This is the default mode and is used to test the performance and correctness of the code in production.
2. **With JIT disabled**: This is used to debug the code in pure Python without the JIT compilation overhead. This is useful for debugging and ensuring that the logic of the code is correct before enabling JIT.

This is managed automatically via a project-wide `conftest.py` file. Run a test with JIT disabled simply with the `pytest tests/test_something.py` command. Run the test with `pytest --jit tests/test_something.py` to enable.

We have created two scripts for convenience which create a new test environment with the necessary dependencies and run all tests in the `tests` directory. These scripts are located in the root directory of the project.
To run all tests with jit enabled, you can use the following script:
```bash
chmode +x ../local_test.sh
../local_test.sh
```
To run all tests with jit disabled, you can use the following command:
```bash
chmod +x ../local_test_nojit.sh
../local_test_nojit.sh
```

>[!IMPORTANT]
> The github workflow for CI will run the tests with jit enabled. Nevertheless, it is important to test with jit disabled (run `local_test_nojit.sh`) as to ensure there are for example no indexing errors or other issues that might not be caught by the JIT compiler.