# AGENTS Instructions

- Use `rg` for searching instead of `grep -R` or `ls -R`.
- Install the package in editable mode with `pip install -e .[dev]` before running tests.
- Run the test suite from the project root.
- For full test runs, disable Numba JIT: `NUMBA_DISABLE_JIT=1 pytest -q`.
- Helper scripts: `./local_test.sh` (JIT enabled) and `./local_test_nojit.sh` (JIT disabled).
- See `tests/README.md` for detailed testing guidance and troubleshooting.
- Keep commits focused and leave the working tree clean before calling `make_pr`.
