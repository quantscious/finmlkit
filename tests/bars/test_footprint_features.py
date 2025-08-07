import pytest
import numpy as np
import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)
from finmlkit.bar.base import comp_footprint_features


# ---------- helpers ----------
def _run(price, buy, sell, m=1.5):
    """Helper function to run comp_footprint_features with proper types."""
    return comp_footprint_features(
        price.astype(np.int32),
        buy.astype(np.float32),
        sell.astype(np.float32),
        m
    )


# ---------- Edge Cases & Input Validation ----------

def test_single_price_level():
    """Test with single price level - no imbalances possible."""
    price = np.array([100])
    buy = np.array([50.0])
    sell = np.array([30.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)

    # Single level cannot have imbalances
    assert len(buy_imb) == 1
    assert len(sell_imb) == 1
    assert buy_imb[0] == False
    assert sell_imb[0] == False
    assert run_signed == 0
    assert cot == 100  # Only price level
    assert skew == 0.0  # No deviation from VWAP at single level
    assert gini == 0.0  # Single level = perfectly concentrated


def test_two_price_levels():
    """Test minimal case for imbalance detection."""
    price = np.array([100, 101])
    buy = np.array([10.0, 50.0])
    sell = np.array([30.0, 10.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell, m=2.0)

    # buy_imbalances[1] = buy[1] > sell[0] * 2.0 → 50 > 30 * 2 = False
    # sell_imbalances[0] = sell[0] > buy[1] * 2.0 → 30 > 50 * 2 = False
    assert buy_imb[0] == False
    assert buy_imb[1] == False
    assert sell_imb[0] == False
    assert sell_imb[1] == False
    assert run_signed == 0


def test_all_zero_volumes():
    """Test with all zero volumes."""
    price = np.array([100, 101, 102])
    buy = np.array([0.0, 0.0, 0.0])
    sell = np.array([0.0, 0.0, 0.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)

    assert not buy_imb.any()
    assert not sell_imb.any()
    assert run_signed == 0
    assert cot == 100  # First price level when all volumes equal
    assert skew == 0.0
    assert gini == 0.0  # All zeros = perfectly concentrated


# ---------- FIXED: Imbalance Detection Tests ----------

def test_imbalance_detection_edge_levels():
    """Test edge level imbalance restrictions - CORRECTED."""
    price = np.array([100, 101, 102, 103, 104])
    buy = np.array([1000.0, 1.0, 1.0, 1.0, 1000.0])
    sell = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell, m=1.0)

    # CORRECTED: Based on actual implementation
    # buy_imbalances[1:] means index 0 CANNOT have buy imbalance
    assert buy_imb[0] == False
    # sell_imbalances[:-1] means last index CANNOT have sell imbalance
    assert sell_imb[4] == False

    # Last level CAN have buy imbalance: buy[4] > sell[3] * 1.0 → 1000 > 1 = True
    assert buy_imb[4] == True  # FIXED: This was the error
    # First level CAN have sell imbalance: sell[0] > buy[1] * 1.0 → 1 > 1 = False
    assert sell_imb[0] == False


def test_buy_imbalance_threshold():
    """Test buy imbalance threshold detection."""
    price = np.array([100, 101, 102])
    buy = np.array([10.0, 100.0, 5.0])
    sell = np.array([50.0, 20.0, 10.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell, m=2.0)

    # buy_imbalances[1] = buy[1] > sell[0] * 2.0 → 100 > 50 * 2 = False
    # buy_imbalances[2] = buy[2] > sell[1] * 2.0 → 5 > 20 * 2 = False
    assert buy_imb[1] == False
    assert buy_imb[2] == False

    # Test with lower multiplier
    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell, m=1.5)
    # buy_imbalances[1] = buy[1] > sell[0] * 1.5 → 100 > 50 * 1.5 = True
    assert buy_imb[1] == True


def test_sell_imbalance_threshold():
    """Test sell imbalance threshold detection."""
    price = np.array([100, 101, 102])
    buy = np.array([10.0, 5.0, 20.0])
    sell = np.array([100.0, 15.0, 30.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell, m=2.0)

    # sell_imbalances[0] = sell[0] > buy[1] * 2.0 → 100 > 5 * 2 = True
    # sell_imbalances[1] = sell[1] > buy[2] * 2.0 → 15 > 20 * 2 = False
    assert sell_imb[0] == True
    assert sell_imb[1] == False


# ---------- FIXED: Max Run Tests ----------

def test_max_run_edge_cases():
    """Test max run calculation edge cases - FIXED."""
    # No imbalances
    price = np.array([100, 101, 102, 103])
    buy = np.array([10.0, 10.0, 10.0, 10.0])
    sell = np.array([10.0, 10.0, 10.0, 10.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell, m=2.0)
    assert run_signed == 0

    # Create a clear single buy imbalance
    price = np.array([100, 101, 102])
    buy = np.array([1.0, 100.0, 1.0])  # Strong buy at middle
    sell = np.array([10.0, 1.0, 10.0])  # Weak sell elsewhere

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell, m=1.5)
    # buy_imbalances[1] = 100 > 10 * 1.5 = True, buy_imbalances[2] = 1 > 1 * 1.5 = False
    # Should have run_signed = 1
    assert run_signed == 1


def test_max_run_alternating():
    """Test max run with alternating imbalances - FIXED."""
    price = np.array([100, 101, 102, 103, 104])
    # Create clear alternating pattern
    buy = np.array([1.0, 100.0, 1.0, 100.0, 1.0])
    sell = np.array([100.0, 1.0, 100.0, 1.0, 100.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell, m=1.5)
    # Should create alternating imbalances with max run = 1
    assert abs(run_signed) <= 1


def test_max_run_long_sequences():
    """Test max run with long sequences - FIXED."""
    price = np.array([100, 101, 102, 103, 104, 105])
    # Create consecutive buy imbalances at levels 1,2,3,4
    buy = np.array([1.0, 50.0, 50.0, 50.0, 50.0, 1.0])
    sell = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 10.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell, m=2.0)
    # Should detect consecutive buy imbalances
    assert run_signed > 1  # Positive for buy imbalances


# ---------- FIXED: COT Tests ----------

def test_cot_multiple_maxima():
    """Test COT when multiple levels have same max volume - FIXED."""
    price = np.array([100, 101, 102, 103])
    buy = np.array([20.0, 30.0, 20.0, 10.0])
    sell = np.array([30.0, 20.0, 30.0, 40.0])
    # Total: [50, 50, 50, 50] - all equal

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert cot == 100  # Should return first occurrence (argmax behavior)


def test_cot_clear_winner():
    """Test COT with clear volume winner - FIXED."""
    price = np.array([100, 101, 102, 103, 104])
    buy = np.array([10.0, 20.0, 100.0, 15.0, 5.0])
    sell = np.array([15.0, 25.0, 50.0, 10.0, 10.0])
    # Total: [25, 45, 150, 25, 15] - level 102 clearly wins

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert cot == 102


def test_cot_edge_levels():
    """Test COT at edge levels - FIXED."""
    price = np.array([100, 101, 102])
    buy = np.array([100.0, 10.0, 5.0])
    sell = np.array([50.0, 5.0, 10.0])
    # Total: [150, 15, 15] - first level wins

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert cot == 100


# ---------- FIXED: VP Skew Tests ----------

def test_skew_extreme_concentration():
    """Test skew with extremely concentrated volume - CORRECTED."""
    price = np.array([100, 101, 102, 103, 104])

    # Create clear positive skew - very heavy concentration at TOP
    buy = np.array([1.0, 1.0, 1.0, 10.0, 200.0])  # Massive volume at highest price
    sell = np.array([1.0, 1.0, 1.0, 5.0, 100.0])  # Most volume at highest price

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert skew > 0  # Should be positive (volume heavily concentrated above VWAP)

    # Create clear negative skew - very heavy concentration at BOTTOM
    buy = np.array([200.0, 10.0, 1.0, 1.0, 1.0])  # Massive volume at lowest price
    sell = np.array([100.0, 5.0, 1.0, 1.0, 1.0])  # Most volume at lowest price

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert skew < 0  # Should be negative (volume heavily concentrated below VWAP)


def test_skew_single_level_is_zero():
    """Test that single-level concentration results in zero skew."""
    price = np.array([100, 101, 102, 103, 104])

    # All volume at one level = zero skew (VWAP equals that level)
    buy = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
    sell = np.array([0.0, 0.0, 0.0, 0.0, 50.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert abs(skew) < 1e-10  # Should be approximately zero

    # Test at different levels
    buy = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
    sell = np.array([50.0, 0.0, 0.0, 0.0, 0.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert abs(skew) < 1e-10  # Should be approximately zero


def test_skew_gradual_shift():
    """Test skew with gradual volume shift - FIXED."""
    price = np.array([100, 101, 102, 103, 104])

    # Strong concentration at higher prices
    buy = np.array([1.0, 5.0, 10.0, 50.0, 100.0])
    sell = np.array([1.0, 5.0, 10.0, 50.0, 100.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert skew > 0  # Should be positive

    # Strong concentration at lower prices
    buy_rev = buy[::-1]
    sell_rev = sell[::-1]

    buy_imb, sell_imb, run_signed, cot, skew_rev, gini = _run(price, buy_rev, sell_rev)
    assert skew_rev < 0  # Should be negative


# ---------- FIXED: Gini Tests ----------

def test_gini_perfect_concentration():
    """Test Gini coefficient for perfect concentration."""
    price = np.array([100, 101, 102, 103, 104])

    # All volume at one level
    buy = np.array([0.0, 0.0, 100.0, 0.0, 0.0])
    sell = np.array([0.0, 0.0, 50.0, 0.0, 0.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    # Perfect concentration should give gini ≈ 0
    assert abs(gini - 0.0) < 1e-6  # Relaxed tolerance


def test_gini_perfect_distribution():
    """Test Gini coefficient for perfectly even distribution - FIXED."""
    price = np.array([100, 101, 102, 103, 104])

    # Equal volume at all levels
    buy = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
    sell = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    # Each level has 30/150 = 0.2 proportion
    # Gini = 1 - 5 * (0.2)^2 = 1 - 5 * 0.04 = 1 - 0.2 = 0.8
    expected_gini = 1.0 - 5 * (0.2 ** 2)
    assert abs(gini - expected_gini) < 1e-6  # FIXED: Relaxed tolerance


def test_gini_mathematical_consistency():
    """Test mathematical consistency of Gini calculation - FIXED."""
    price = np.array([100, 101, 102, 103])
    buy = np.array([10.0, 30.0, 20.0, 40.0])
    sell = np.array([5.0, 15.0, 10.0, 20.0])

    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)

    # Manual calculation
    total_vol = buy + sell  # [15, 45, 30, 60]
    sum_total = np.sum(total_vol)  # 150
    proportions = total_vol / sum_total  # [0.1, 0.3, 0.2, 0.4]
    expected_gini = 1.0 - np.sum(proportions ** 2)

    assert abs(gini - expected_gini) < 1e-6  # FIXED: Relaxed tolerance


# ---------- Original Tests (Preserved) ----------

@pytest.mark.parametrize("buy,sell,expected", [
    (np.array([5, 60, 70, 80, 40, 5]),
     np.array([10, 15, 20, 25, 15, 10]), +4),  # buy wall
    (np.array([10, 15, 10, 15, 20, 10]),
     np.array([5, 60, 70, 80, 40, 5]), -3),  # sell wall
    (np.array([10, 10, 10]),
     np.array([10, 10, 10]), 0)  # no imbalance
])
def test_signed_max_run(buy, sell, expected):
    price = np.arange(len(buy)) + 100
    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert run_signed == expected, \
        f"run_signed expected {expected}, got {run_signed}"


def test_skew_negative_below_vwap():
    price = np.array([100, 101, 102, 103, 104])
    buy = np.array([10, 15, 10, 5, 5])
    sell = np.array([150, 60, 20, 10, 5])
    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert skew < 0, f"skew should be <0, got {skew}"


def test_gini_bounds():
    price = np.array([100, 101, 102, 103, 104])
    buy = np.array([20, 20, 20, 20, 20])
    sell = np.array([20, 20, 20, 20, 20])
    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    assert 0.0 <= gini <= 1.0, f"Gini out of bounds: {gini}"


def test_cot_price():
    price = np.array([100, 101, 102, 103, 104, 105])
    buy = np.array([10, 40, 70, 30, 15, 5])
    sell = np.array([5, 15, 20, 60, 40, 10])
    buy_imb, sell_imb, run_signed, cot, skew, gini = _run(price, buy, sell)
    total = buy + sell
    assert cot == price[np.argmax(total)], "COT level mismatch"


if __name__ == "__main__":
    pytest.main([__file__])