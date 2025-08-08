import os
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--jit", action="store_true", default=False, help="Run tests with Numba JIT enabled"
    )

@pytest.fixture(scope="session", autouse=True)
def numba_mode(request):
    """
    Fixture to control Numba JIT compilation mode.
    If --jit is passed, Numba JIT is enabled; otherwise, it is disabled.
    """
    if request.config.getoption("--jit"):
        os.environ.pop("NUMBA_DISABLE_JIT", None)
        print("Numba JIT is enabled.")
    else:
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        print("Numba JIT is disabled for testing.")
