import numpy as np
import pandas as pd
import pytest

from finmlkit.bar.data_model import TradesData, _load_single_h5_group, _is_notebook_environment


def make_us_timestamps(base='2021-01-01 00:00:00', offsets_us=(0, 0, 120_000_000)):
    base_ns = pd.Timestamp(base).value
    base_us = base_ns // 1000
    return np.array([base_us + o for o in offsets_us], dtype=np.int64)


def test_tradesdata_type_validations():
    ts = np.array([1, 2, 3], dtype=np.int64)
    px = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    qty = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(TypeError):
        TradesData([1, 2, 3], px, qty)  # ts not ndarray
    with pytest.raises(TypeError):
        TradesData(ts, [100.0], qty)  # px not ndarray
    with pytest.raises(TypeError):
        TradesData(ts, px, [1.0])  # qty not ndarray
    with pytest.raises(TypeError):
        TradesData(ts, px, qty, id=[1, 2, 3])
    with pytest.raises(TypeError):
        TradesData(ts, px, qty, is_buyer_maker=[True, False, True])
    with pytest.raises(TypeError):
        TradesData(ts, px, qty, side=[-1, 1, -1])


def test_preprocess_pipeline_and_rounding_and_side_inference(tmp_path):
    # ids: duplicate 2 and gap 2->10 to trigger integrity and merging
    ids = np.array([1, 2, 2, 10], dtype=np.int64)
    ts_us = np.array([
        *make_us_timestamps(offsets_us=(0,)),             # id 1
        *make_us_timestamps(offsets_us=(0,)),             # id 2 (same ts as id1)
        *make_us_timestamps(offsets_us=(500_000,)),       # id 2 duplicate (later, will be dropped)
        *make_us_timestamps(offsets_us=(120_000_000,)),   # id 10 (2 minutes later)
    ], dtype=np.int64)
    # Flatten created arrays
    ts_us = ts_us.reshape(-1)

    px = np.array([100.0, 100.0, 100.0, 101.0], dtype=np.float64)
    qty = np.array([0.4, 0.6, 0.1, 2.0], dtype=np.float64)

    td = TradesData(
        ts_us, px, qty, ids,
        preprocess=True, proc_res='ms', name='TEST'
    )

    # Timestamp unit inferred as 'us' and converted to ns
    assert td.orig_timestamp_unit == 'us'
    assert (td.data['timestamp'].values % 1_000_000 == 0).all()  # rounded to ms (in ns)

    # After merging same ts+price, there should be 2 rows (id 1&2 merged; id10 remains)
    assert len(td.data) == 2
    # Side inferred (tick rule): first 0, then +1 since price increases
    assert 'side' in td.data.columns
    assert td.data['side'].iloc[0] == 0
    assert td.data['side'].iloc[1] == 1

    # set_view_range slicing
    start = pd.to_datetime(td.data['timestamp'].iloc[0], unit='ns')
    end = start + pd.Timedelta(minutes=1)
    td.set_view_range(start, end)
    sliced = td.data
    assert len(sliced) == 1
    with pytest.raises(ValueError):
        td.set_view_range(end, start)

    # Save to H5 using full dataset (avoid active view range)
    h5_path = tmp_path / 'trades.h5'
    td_full = TradesData(ts_us, px, qty, ids, preprocess=True, proc_res='ms', name='TEST_FULL')
    key = td_full.save_h5(str(h5_path))
    assert key.startswith('/trades/')
    month = key.split('/')[-1]

    # Load sequentially by key
    td2 = TradesData.load_trades_h5(str(h5_path), key=month, enable_multiprocessing=False)
    assert isinstance(td2, TradesData)
    assert len(td2.data) >= 2

    # _load_single_h5_group helper
    df_ok = _load_single_h5_group((str(h5_path), f'/trades/{month}', None))
    assert isinstance(df_ok, pd.DataFrame)
    assert not df_ok.empty
    df_bad = _load_single_h5_group((str(h5_path), '/trades/2099-01', None))
    assert isinstance(df_bad, pd.DataFrame) and df_bad.empty

    # environment detector returns a bool
    assert isinstance(_is_notebook_environment(), bool)
