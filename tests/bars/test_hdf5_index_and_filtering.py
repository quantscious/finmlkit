import os
import pandas as pd
import numpy as np
import pytest

from finmlkit.bar.data_model import TradesData


def _make_sample_trades(n=10, start_ts="2021-01-01 00:00:00", freq="h"):
    # Create evenly spaced timestamps in ns (tz-naive)
    idx = pd.date_range(start=start_ts, periods=n, freq=freq)
    ts = idx.view("int64")  # ns
    px = np.linspace(100.0, 101.0, n).astype(np.float64)
    qty = np.linspace(1.0, 2.0, n).astype(np.float32)
    ids = np.arange(1, n + 1, dtype=np.int64)
    return ts, px, qty, ids, idx


def test_hdf5_writes_and_index_query_succeeds(tmp_path):
    ts, px, qty, ids, idx = _make_sample_trades(n=12, start_ts="2021-01-15 00:00:00", freq="h")

    # Build TradesData (no preprocessing) so timestamps are already ns
    td = TradesData(ts, px, qty, ids, preprocess=False, name="TEST")

    # Save to HDF5
    h5_path = tmp_path / "trades.h5"
    month_key = f"{idx[0].year:04d}-{idx[0].month:02d}"
    h5_key = td.save_h5(str(h5_path), month_key=month_key, mode="w")

    assert h5_key == f"/trades/{month_key}"
    assert os.path.exists(h5_path)

    # Open HDFStore and verify index-based querying WORKS (index is saved and restored)
    with pd.HDFStore(str(h5_path), mode="r") as store:
        # Verify table exists
        assert h5_key in store

        # Query on index (datetime) should work
        df_idx = store.select(h5_key, where=f"index >= Timestamp('{idx[3]}') & index <= Timestamp('{idx[8]}')")
        assert len(df_idx) == 6
        assert isinstance(df_idx.index, pd.DatetimeIndex)

        # Query on data column 'timestamp' (numeric ns) should also work
        start_ns = int(idx[3].value)
        end_ns = int(idx[8].value)
        df_range = store.select(h5_key, where=f"timestamp >= {start_ns} & timestamp <= {end_ns}")
        assert len(df_range) == 6
        # Ensure the stored frame keeps a DatetimeIndex on read via select
        assert isinstance(df_range.index, pd.DatetimeIndex)


def test_load_trades_h5_without_filters_has_datetime_index(tmp_path):
    ts, px, qty, ids, idx = _make_sample_trades(n=6, start_ts="2021-01-03 00:00:00", freq="h")
    td = TradesData(ts, px, qty, ids, preprocess=False, name="TEST2")

    h5_path = tmp_path / "trades2.h5"
    month_key = f"{idx[0].year:04d}-{idx[0].month:02d}"
    td.save_h5(str(h5_path), month_key=month_key, mode="w")

    # Load without any time filtering
    t_loaded = TradesData.load_trades_h5(str(h5_path), key=month_key)

    # The loaded DataFrame should have a DatetimeIndex
    assert isinstance(t_loaded.data.index, pd.DatetimeIndex)
    # Timestamps column should be present and monotonic
    assert "timestamp" in t_loaded.data.columns
    assert t_loaded.data["timestamp"].is_monotonic_increasing


def test_load_trades_h5_with_time_filters_succeeds(tmp_path):
    ts, px, qty, ids, idx = _make_sample_trades(n=24, start_ts="2021-01-10 00:00:00", freq="h")
    td = TradesData(ts, px, qty, ids, preprocess=False, name="TEST3")

    h5_path = tmp_path / "trades3.h5"
    month_key = f"{idx[0].year:04d}-{idx[0].month:02d}"
    td.save_h5(str(h5_path), month_key=month_key, mode="w")

    # Choose a subrange
    start_time = idx[5]
    end_time = idx[10]

    # Time-filtered loading should succeed and return the correct slice
    t_loaded = TradesData.load_trades_h5(str(h5_path), start_time=start_time, end_time=end_time)
    df = t_loaded.data
    assert isinstance(df.index, pd.DatetimeIndex)
    # Expect inclusive endpoints for the HDF5 select where clause (6 rows)
    assert len(df) == (10 - 5 + 1)
    assert df.index[0] == start_time
    assert df.index[-1] == end_time

    # Also verify store-level timestamp filtering returns the same count
    with pd.HDFStore(str(h5_path), mode="r") as store:
        h5_key = f"/trades/{month_key}"
        start_ns = int(start_time.value)
        end_ns = int(end_time.value)
        df_ts = store.select(h5_key, where=f"timestamp >= {start_ns} & timestamp <= {end_ns}")
        assert len(df_ts) == len(df)
