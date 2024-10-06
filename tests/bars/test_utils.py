import numpy as np
import pytest
import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)

from finmlkit.bars.utils import comp_price_tick_size

@pytest.mark.parametrize(
    "prices, expected_tick_size",
    [
        (np.array([1.0, 1., 1., 1., 1., 1.1, 1.2, 1.3, 1.4]), 0.1),  # Standard case with a 0.1 tick size
        (np.array([1e-5, 1e-5 + 1e-10, 1e-5 + 2e-10, 1e-5 + 3e-10]), 1e-10),  # Very small tick size
        (np.array([1.00000000001, 1.00000000002, 1.00000000003]), 1e-11),  # Precision with tiny differences
        (np.array([1000, 1000.1, 1000.2, 1000.3, 1000.4]), 0.1),  # Tick size of 0.1 with large values
        (np.array([0.5, 0.6, 0.7]), 0.1),  # Small price differences
        (np.array([10, 10, 10]), 0.0),  # No variation in prices, tick size should be 0
        (np.array([1000, 1000.01, 1000.02, 1000.03]), 0.01),  # Smaller tick size of 0.01
    ]
)
def test_comp_price_tick_size(prices, expected_tick_size):
    tick_size = comp_price_tick_size(prices)
    assert tick_size == pytest.approx(expected_tick_size, rel=1e-9)

def test_empty_prices():
    prices = np.array([])
    with pytest.raises(ValueError):
        comp_price_tick_size(prices)


def test_single_price():
    prices = np.array([1.0])
    tick_size = comp_price_tick_size(prices)
    assert tick_size == 0.0  # Only one price, no tick size


def test_identical_prices():
    prices = np.array([5.0, 5.0, 5.0])
    tick_size = comp_price_tick_size(prices)
    assert tick_size == 0.0  # Identical prices, no variation


if __name__ == "__main__":
    pytest.main([__file__])