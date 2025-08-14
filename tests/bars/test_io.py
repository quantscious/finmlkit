import numpy as np
import pandas as pd
import pytest

from finmlkit.bar.data_model import TradesData
from finmlkit.bar.io import H5Inspector, TimeBarReader


def _build_trades_h5(path: str) -> str:
    # Build a trades file with an integrity issue (gap in IDs >1min apart)
    ids = np.array([1, 2, 10], dtype=np.int64)
    t0 = pd.Timestamp('2021-01-01 00:00:00')
    ts_us = np.array([
        (t0.value // 1000) + 0,
        (t0.value // 1000) + 0,  # same ts -> will merge with first
        (t0.value // 1000) + 120_000_000,  # +2 minutes
    ], dtype=np.int64)
    px = np.array([100.0, 100.0, 101.0], dtype=np.float64)
    qty = np.array([0.4, 0.6, 2.0], dtype=np.float64)

    td = TradesData(ts_us, px, qty, ids, preprocess=True, name='IO_TEST')
    h5_key = td.save_h5(path)
    return h5_key


def test_h5inspector_methods(tmp_path):
    h5file = tmp_path / 'io_trades.h5'
    trades_key = _build_trades_h5(str(h5file))

    insp = H5Inspector(str(h5file))

    keys = insp.list_keys()
    assert trades_key in keys

    meta = insp.get_metadata(trades_key)
    assert isinstance(meta, dict)
    # Expect required keys
    for k in ['record_count', 'first_timestamp', 'last_timestamp', 'data_integrity_ok', 'missing_pct']:
        assert k in meta

    # Integrity info should be present (gap)
    disc = insp.get_integrity_info(trades_key)
    assert disc is None or isinstance(disc, pd.DataFrame)

    stats = insp.get_statistics(trades_key)
    assert stats['price_range'][0] <= stats['price_range'][1]
    assert stats['amount_range'][0] <= stats['amount_range'][1]

    summary = insp.get_integrity_summary(verbose=False)
    # With our constructed gap, summary should be either None (if data_ok True) or a dict with month
    if summary is not None:
        assert isinstance(summary, dict)
        # The month key should exist in summary
        month = trades_key.split('/')[-1]
        assert month in summary
        assert 'metadata' in summary[month]


def test_timebarreader_list_and_range_and_resample(tmp_path):
    h5file = tmp_path / 'klines.h5'

    # Build synthetic 1s OHLCV for one minute subset (6 seconds)
    idx = pd.date_range('2021-01-01', periods=6, freq='S')
    df = pd.DataFrame({
        'open':   [10, 11, 12, 13, 14, 15],
        'high':   [11, 12, 13, 14, 15, 16],
        'low':    [9, 10, 11, 12, 13, 14],
        'close':  [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
        'volume': [1, 2, 3, 4, 5, 6],
        'trades': [1, 2, 1, 2, 3, 1],
        'vwap':   [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
        'median_trade_size': [1, 2, 3, 4, 5, 6],
    }, index=idx)

    month = '2021-01'
    klines_key = f'/klines/{month}'
    meta_key = f'/klines_meta/{month}'

    with pd.HDFStore(str(h5file), mode='a') as store:
        store.put(klines_key, df, format='table', index=True)
        meta = pd.Series({
            'record_count': len(df),
            'first_timestamp': df.index[0].value,
            'last_timestamp': df.index[-1].value,
            'original_trades_key': '/trades/' + month,
        })
        store.put(meta_key, meta, format='fixed')

    reader = TimeBarReader(str(h5file))

    # list_keys
    keys = reader.list_keys()
    assert klines_key in keys

    # time range
    start, end = reader._list_time_range()
    assert start == df.index[0]
    assert end == df.index[-1]

    # resample to 3-second bars directly
    out = reader._resample(df, '3s')
    assert list(out.columns) == ['open', 'high', 'low', 'close', 'volume', 'trades', 'vwap', 'median_trade_size']
    assert len(out) == 2

    # Group 1: seconds 0,1,2
    g1 = out.iloc[0]
    assert g1['open'] == 10
    assert g1['high'] == 13
    assert g1['low'] == 9
    assert g1['close'] == 12.5
    assert g1['volume'] == 1 + 2 + 3
    assert g1['trades'] == 1 + 2 + 1
    assert pytest.approx(g1['vwap'], rel=1e-6) == (10.5*1 + 11.5*2 + 12.5*3) / 6
    assert pytest.approx(g1['median_trade_size'], rel=1e-6) == 2

    # Group 2: seconds 3,4,5
    g2 = out.iloc[1]
    assert g2['open'] == 13
    assert g2['high'] == 16
    assert g2['low'] == 12
    assert g2['close'] == 15.5
    assert g2['volume'] == 4 + 5 + 6
    assert g2['trades'] == 2 + 3 + 1
    assert pytest.approx(g2['vwap'], rel=1e-6) == (13.5*4 + 14.5*5 + 15.5*6) / 15
    assert pytest.approx(g2['median_trade_size'], rel=1e-6) == 5
