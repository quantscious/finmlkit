"""
tests/test_comp_bar_directional_features.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit-tests for finmlkit.bar.base.comp_bar_directional_features
adapted to the current implementation (June 2025).

The last index in *bar_close_indices* must now be the last traded index
(inclusive).  Spreads are added **only** when trade direction changes, so
many of the former “1 tick = 1 spread” expectations are now zero or smaller.
"""
import os
import numpy as np
import pytest

os.environ["NUMBA_DISABLE_JIT"] = "1"          # run the pure-Python path

from finmlkit.bar.base import comp_bar_directional_features


# ---------- helper -------------------------------------------------------------------------------

def _chk(result, exp):
    """
    Compare a 14-tuple returned by the function with the 14 arrays in *exp*.
    """
    for got, want in zip(result, exp):
        if got.dtype.kind in {"f"}:           # floats: use almost-equal
            np.testing.assert_array_almost_equal(got, want)
        else:                                 # ints: exact
            np.testing.assert_array_equal(got, want)


# ---------- 1. simple all-buys --------------------------------------------------------------------

def test_comp_bar_directional_features_simple_case():
    prices   = np.array([100., 101., 102.], dtype=np.float64)
    volumes  = np.array([10.,  15.,  20. ], dtype=np.float64)
    sides    = np.array([0, 1, 1],          dtype=np.int8)     # dummy, buy, buy
    bars     = np.array([0, 2],             dtype=np.int64)    # inclusive last index

    out = comp_bar_directional_features(prices, volumes, bars, sides)

    exp = (
        np.array([2],  dtype=np.int64),                       # ticks_buy
        np.array([0],  dtype=np.int64),                       # ticks_sell
        np.array([35.],  dtype=np.float32),                   # volume_buy
        np.array([0.],   dtype=np.float32),                   # volume_sell
        np.array([3555.], dtype=np.float32),                  # dollars_buy
        np.array([0.],     dtype=np.float32),                 # dollars_sell
        np.array([0.],     dtype=np.float32),                 # mean_spread
        np.array([0.],     dtype=np.float32),                 # max_spread
        np.array([1],  dtype=np.int64),                       # cum_ticks_min
        np.array([2],  dtype=np.int64),                       # cum_ticks_max
        np.array([15.],  dtype=np.float32),                   # cum_volumes_min
        np.array([35.],  dtype=np.float32),                   # cum_volumes_max
        np.array([1515.], dtype=np.float32),                  # cum_dollars_min
        np.array([3555.], dtype=np.float32)                   # cum_dollars_max
    )
    _chk(out, exp)


# ---------- 2. mixed sell → buy ------------------------------------------------------------------

def test_comp_bar_directional_features_mixed_trades():
    prices  = np.array([100.,  99., 100., 101.], dtype=np.float64)
    volumes = np.array([10.,   20., 30., 40. ],  dtype=np.float64)
    sides   = np.array([0, -1, 1, 1],            dtype=np.int8)   # sell, buy, buy
    bars    = np.array([0, 3],                   dtype=np.int64)

    out = comp_bar_directional_features(prices, volumes, bars, sides)

    exp = (
        np.array([2],  dtype=np.int64),                        # ticks_buy
        np.array([1],  dtype=np.int64),                        # ticks_sell
        np.array([70.],  dtype=np.float32),                    # volume_buy
        np.array([20.],  dtype=np.float32),                    # volume_sell
        np.array([7040.], dtype=np.float32),                   # dollars_buy
        np.array([1980.], dtype=np.float32),                   # dollars_sell
        np.array([np.float32(1/3)], dtype=np.float32),         # mean_spread (≈0.3333)
        np.array([1.],      dtype=np.float32),                 # max_spread
        np.array([-1], dtype=np.int64),                        # cum_ticks_min
        np.array([1],  dtype=np.int64),                        # cum_ticks_max
        np.array([-20.], dtype=np.float32),                    # cum_volumes_min
        np.array([50.],  dtype=np.float32),                    # cum_volumes_max
        np.array([-1980.], dtype=np.float32),                  # cum_dollars_min
        np.array([5060.],  dtype=np.float32)                   # cum_dollars_max
    )
    _chk(out, exp)


# ---------- 3. all sells -------------------------------------------------------------------------

def test_comp_bar_directional_features_all_sells():
    prices  = np.array([100.,  99.,  98.], dtype=np.float64)
    volumes = np.array([10.,   20.,  30.], dtype=np.float64)
    sides   = np.array([0, -1, -1],        dtype=np.int8)
    bars    = np.array([0, 2],             dtype=np.int64)

    out = comp_bar_directional_features(prices, volumes, bars, sides)

    exp = (
        np.array([0],  dtype=np.int64),                       # ticks_buy
        np.array([2],  dtype=np.int64),                       # ticks_sell
        np.array([0.],   dtype=np.float32),                   # volume_buy
        np.array([50.],  dtype=np.float32),                   # volume_sell
        np.array([0.],     dtype=np.float32),                 # dollars_buy
        np.array([4920.],  dtype=np.float32),                 # dollars_sell
        np.array([0.],     dtype=np.float32),                 # mean_spread
        np.array([0.],     dtype=np.float32),                 # max_spread
        np.array([-2], dtype=np.int64),                       # cum_ticks_min
        np.array([-1], dtype=np.int64),                       # cum_ticks_max
        np.array([-50.], dtype=np.float32),                   # cum_volumes_min
        np.array([-20.], dtype=np.float32),                   # cum_volumes_max
        np.array([-4920.], dtype=np.float32),                 # cum_dollars_min
        np.array([-1980.], dtype=np.float32)                  # cum_dollars_max
    )
    _chk(out, exp)


# ---------- 4. constant price, two buys ----------------------------------------------------------

def test_comp_bar_directional_features_constant_price():
    prices  = np.array([100., 100., 100.], dtype=np.float64)
    volumes = np.array([10.,  20.,  30. ], dtype=np.float64)
    sides   = np.array([0, 1, 1],          dtype=np.int8)
    bars    = np.array([0, 2],             dtype=np.int64)

    out = comp_bar_directional_features(prices, volumes, bars, sides)

    exp = (
        np.array([2],  dtype=np.int64),                        # ticks_buy
        np.array([0],  dtype=np.int64),                        # ticks_sell
        np.array([50.],  dtype=np.float32),                    # volume_buy
        np.array([0.],   dtype=np.float32),                    # volume_sell
        np.array([5000.], dtype=np.float32),                   # dollars_buy
        np.array([0.],     dtype=np.float32),                  # dollars_sell
        np.array([0.],     dtype=np.float32),                  # mean_spread
        np.array([0.],     dtype=np.float32),                  # max_spread
        np.array([1],  dtype=np.int64),                        # cum_ticks_min
        np.array([2],  dtype=np.int64),                        # cum_ticks_max
        np.array([20.],  dtype=np.float32),                    # cum_volumes_min
        np.array([50.],  dtype=np.float32),                    # cum_volumes_max
        np.array([2000.], dtype=np.float32),                   # cum_dollars_min
        np.array([5000.], dtype=np.float32)                    # cum_dollars_max
    )
    _chk(out, exp)


# ---------- 5. two bars, alternating directions --------------------------------------------------

def test_comp_bar_directional_features_multiple_bars_with_varying_spreads():
    prices  = np.array([100., 101., 102., 101.5, 103., 102.], dtype=np.float64)
    volumes = np.array([10.,  15.,  20.,  25.,   30.,  35. ], dtype=np.float64)
    sides   = np.array([0, 1, 1, -1, 1, -1],   dtype=np.int8)
    bars    = np.array([0, 2, 5],              dtype=np.int64)   # two bars

    out = comp_bar_directional_features(prices, volumes, bars, sides)

    exp = (
        np.array([2, 1],            dtype=np.int64),             # ticks_buy
        np.array([0, 2],            dtype=np.int64),             # ticks_sell
        np.array([35., 30.],        dtype=np.float32),           # volume_buy
        np.array([0.,  60.],        dtype=np.float32),           # volume_sell
        np.array([3555., 3090.],    dtype=np.float32),           # dollars_buy
        np.array([0.,  6107.5],     dtype=np.float32),           # dollars_sell
        np.array([0.,  0.8333333],  dtype=np.float32),           # mean_spread
        np.array([0.,  1.5],        dtype=np.float32),           # max_spread
        np.array([1,  -1],          dtype=np.int64),             # cum_ticks_min
        np.array([2,  0],           dtype=np.int64),             # cum_ticks_max
        np.array([15., -30.],       dtype=np.float32),           # cum_volumes_min
        np.array([35., 5.],         dtype=np.float32),           # cum_volumes_max
        np.array([1515., -3017.5],  dtype=np.float32),           # cum_dollars_min
        np.array([3555.,  552.5],   dtype=np.float32)            # cum_dollars_max
    )
    _chk(out, exp)


# ---------- 6. variable spread, buy-buy-sell -----------------------------------------------------

def test_comp_bar_directional_features_variable_spreads():
    prices  = np.array([100., 101., 105., 103.], dtype=np.float64)
    volumes = np.array([10.,  20.,  30.,  40.], dtype=np.float64)
    sides   = np.array([0, 1, 1, -1],           dtype=np.int8)
    bars    = np.array([0, 3],                  dtype=np.int64)

    out = comp_bar_directional_features(prices, volumes, bars, sides)

    exp = (
        np.array([2],  dtype=np.int64),                         # ticks_buy
        np.array([1],  dtype=np.int64),                         # ticks_sell
        np.array([50.],  dtype=np.float32),                     # volume_buy
        np.array([40.],  dtype=np.float32),                     # volume_sell
        np.array([5170.], dtype=np.float32),                    # dollars_buy
        np.array([4120.], dtype=np.float32),                    # dollars_sell
        np.array([0.6666667], dtype=np.float32),                # mean_spread
        np.array([2.],        dtype=np.float32),                # max_spread
        np.array([1],   dtype=np.int64),                        # cum_ticks_min
        np.array([2],   dtype=np.int64),                        # cum_ticks_max
        np.array([10.], dtype=np.float32),                      # cum_volumes_min
        np.array([50.], dtype=np.float32),                      # cum_volumes_max
        np.array([1050.], dtype=np.float32),                    # cum_dollars_min
        np.array([5170.], dtype=np.float32)                     # cum_dollars_max
    )
    _chk(out, exp)


# ---------- 7. NEW: alternating buy / sell / buy -------------------------------------------------

def test_comp_bar_directional_features_alternating_trades_spread():
    prices  = np.array([100., 101., 100., 101.], dtype=np.float64)
    volumes = np.array([10.,  10.,  10.,  10.], dtype=np.float64)
    sides   = np.array([0, 1, -1, 1],           dtype=np.int8)
    bars    = np.array([0, 3],                  dtype=np.int64)

    out = comp_bar_directional_features(prices, volumes, bars, sides)

    exp = (
        np.array([2],  dtype=np.int64),                         # ticks_buy
        np.array([1],  dtype=np.int64),                         # ticks_sell
        np.array([20.],  dtype=np.float32),                     # volume_buy
        np.array([10.],  dtype=np.float32),                     # volume_sell
        np.array([2020.], dtype=np.float32),                    # dollars_buy
        np.array([1000.], dtype=np.float32),                    # dollars_sell
        np.array([0.6666667], dtype=np.float32),                # mean_spread
        np.array([1.],        dtype=np.float32),                # max_spread
        np.array([0],   dtype=np.int64),                        # cum_ticks_min
        np.array([1],   dtype=np.int64),                        # cum_ticks_max
        np.array([0.],  dtype=np.float32),                      # cum_volumes_min
        np.array([10.], dtype=np.float32),                      # cum_volumes_max
        np.array([10.], dtype=np.float32),                      # cum_dollars_min
        np.array([1020.], dtype=np.float32)                     # cum_dollars_max
    )
    _chk(out, exp)


# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":          # pragma: no-cover
    pytest.main([__file__])