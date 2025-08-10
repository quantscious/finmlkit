# AGENTS Instructions

- Prefer `rg` (ripgrep) for search. If missing, install via `sudo apt-get update && sudo apt-get install -y ripgrep`, or fallback to: `git ls-files | xargs grep -n`.
- Install the package in editable develop mode with `pip install -e .[dev]`.
- Use `pytest` for tests and disable Numba JIT for full test runs: `NUMBA_DISABLE_JIT=1 pytest -q`.
- Helper scripts for testing: `./local_test.sh` (JIT enabled) and `./local_test_nojit.sh` (JIT disabled).
- See `tests/README.md` for detailed testing guidance and troubleshooting.
- Keep commits focused and leave the working tree clean before calling `make_pr`.
