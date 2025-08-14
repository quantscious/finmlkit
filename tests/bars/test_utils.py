import numpy as np
import pandas as pd
import pytest

#os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for testing (we can debug numba functions this way)


from finmlkit.bar.utils import (
    comp_trade_side_vector,
    comp_price_tick_size,
    comp_price_tick_size_old,
    footprint_to_dataframe,
    fast_sort_trades,
    merge_split_trades,
)


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


def test_comp_trade_side_vector_basic():
    # price path: flat -> up -> flat -> down
    prices = np.array([100.0, 100.0, 101.0, 101.0, 100.5], dtype=np.float64)
    sides = comp_trade_side_vector(prices)
    # First element is 0 by design, then: 0, +1, +1 (flat keeps prev), -1
    assert sides.dtype == np.int8
    assert sides.tolist() == [0, 0, 1, 1, -1]


def test_comp_price_tick_size_estimators():
    # Construct grid with 0.5 tick
    prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0], dtype=np.float64)
    est = comp_price_tick_size(prices)
    old = comp_price_tick_size_old(prices)
    assert pytest.approx(est, rel=0, abs=1e-12) == 0.5
    # old estimator may round but should be close to 0.5
    assert abs(old - 0.5) < 1e-6


def test_footprint_to_dataframe_structure_and_sorting():
    # two bars, two levels each
    bar_ts = np.array([
        pd.Timestamp('2021-01-01 00:00:00').value,
        pd.Timestamp('2021-01-01 00:00:01').value,
    ], dtype=np.int64)

    price_levels = [np.array([1, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32)]
    buy_volumes = [np.array([10.0, 20.0], dtype=np.float32), np.array([5.0, 1.0], dtype=np.float32)]
    sell_volumes = [np.array([4.0, 6.0], dtype=np.float32), np.array([2.0, 3.0], dtype=np.float32)]
    buy_ticks = [np.array([1, 2], dtype=np.int32), np.array([3, 1], dtype=np.int32)]
    sell_ticks = [np.array([2, 1], dtype=np.int32), np.array([1, 1], dtype=np.int32)]
    buy_imb = [np.array([False, True]), np.array([True, False])]
    sell_imb = [np.array([True, False]), np.array([False, True])]

    df = footprint_to_dataframe(
        bar_ts, price_levels, buy_volumes, sell_volumes, buy_ticks, sell_ticks, buy_imb, sell_imb, price_tick=0.5
    )

    # MultiIndex names and levels
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ['bar_idx', 'bar_datetime_idx']
    # Price scaling applied (levels 1,2 -> 0.5,1.0 for first bar; 1,3 -> 0.5,1.5 for second)
    first_bar = df.xs(0, level='bar_idx')
    assert first_bar['price_level'].tolist() == sorted([1*0.5, 2*0.5], reverse=True)
    # Sorted descending by price_level within each bar datetime
    for _, grp in df.groupby(level='bar_datetime_idx'):
        pl = grp['price_level'].values
        assert np.all(pl[:-1] >= pl[1:])


def test_fast_sort_trades_and_merge_split_trades():
    # Unsorted timestamps with duplicates (same ts+price) to merge
    ts = np.array([3, 1, 1, 2], dtype=np.int64)
    px = np.array([100.0, 100.0, 100.0, 101.0], dtype=np.float64)
    am = np.array([1.0, 0.4, 0.6, 2.0], dtype=np.float32)
    is_bm = np.array([True, True, True, False])

    s_ts, s_px, s_am, s_isbm = fast_sort_trades(ts, px, am, is_bm)
    assert list(s_ts) == [1, 1, 2, 3]
    assert list(s_px) == [100.0, 100.0, 101.0, 100.0]
    assert list(s_am) == [0.4, 0.6, 2.0, 1.0]
    assert list(s_isbm) == [True, True, False, True]

    # Merge on sorted arrays
    m_ts, m_px, m_am, m_side = merge_split_trades(s_ts, s_px, s_am, s_isbm)
    # First two merged: same ts=1 & price=100.0 & same is_buyer_maker
    assert m_ts.tolist() == [1, 2, 3]
    assert m_px.tolist() == [100.0, 101.0, 100.0]
    assert pytest.approx(m_am.tolist()) == [1.0, 2.0, 1.0]
    # Side mapping: is_buyer_maker True -> side -1; False -> +1
    assert m_side.dtype == np.int8
    assert m_side.tolist() == [-1, 1, -1]

    # Merge without side info
    m2_ts, m2_px, m2_am, m2_side = merge_split_trades(s_ts, s_px, s_am, None)
    assert m2_ts.tolist() == [1, 2, 3]
    assert m2_px.tolist() == [100.0, 101.0, 100.0]
    assert pytest.approx(m2_am.tolist()) == [1.0, 2.0, 1.0]
    assert isinstance(m2_side, np.ndarray) and m2_side.size == 0



if __name__ == "__main__":
    pytest.main([__file__])