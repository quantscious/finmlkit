import pytest
import numpy as np
from numpy.typing import NDArray
from finmlkit.bar.base import comp_footprint_features

# ---------- helpers ----------
def _run(price, buy, sell, m=1.5):
    return comp_footprint_features(
        price.astype(np.int32),
        buy.astype(np.float32),
        sell.astype(np.float32),
        m
    )


# ---------- tests ----------
def test_skew_positive_above_vwap():
    price = np.array([100, 101, 102, 103, 104])
    buy   = np.array([  5,  10,  15,  60, 150])
    sell  = np.array([  5,  10,  15,  10,   15])
    _, _, _, _, skew, _ = _run(price, buy, sell)
    print(skew)
    assert skew > 0, f"VP skew should be positive, got {skew}"


def test_skew_negative_below_vwap():
    price = np.array([100, 101, 102, 103, 104])
    buy   = np.array([ 10,  15,  10,   5,   5])
    sell  = np.array([150,  60,  20,  10,   5])
    _, _, _, _, skew, _ = _run(price, buy, sell)
    print(skew)
    assert skew < 0, f"skew should be <0, got {skew}"


@pytest.mark.parametrize("buy,sell,expected", [
    (np.array([5, 60, 70, 80, 40,  5]),
     np.array([10,15, 20, 25, 15, 10]),  +4),   # buy wall
    (np.array([10,15, 10, 15, 20, 10]),
     np.array([ 5,60, 70, 80, 40,  5]),  -3),   # sell wall
    (np.array([10,10, 10]),
     np.array([10,10, 10]),               0)    # no imbalance
])
def test_signed_max_run(buy, sell, expected):
    price = np.arange(len(buy)) + 100
    _, _, run_signed, *_ = _run(price, buy, sell)
    assert run_signed == expected, \
        f"run_signed expected {expected}, got {run_signed}"


def test_gini_bounds():
    price = np.array([100, 101, 102, 103, 104])
    buy   = np.array([20, 20, 20, 20, 20])
    sell  = np.array([20, 20, 20, 20, 20])
    _, _, _, _, _, gini = _run(price, buy, sell)
    assert 0.0 <= gini <= 1.0, f"Gini out of bounds: {gini}"


def test_cot_price():
    price = np.array([100, 101, 102, 103, 104, 105])
    buy   = np.array([10, 40, 70, 30, 15,  5])
    sell  = np.array([ 5, 15, 20, 60, 40, 10])
    _, _, _, cot, *_ = _run(price, buy, sell)
    total = buy + sell
    assert cot == price[np.argmax(total)], "COT level mismatch"


if __name__ == "__main__":
    pytest.main(["-q"])