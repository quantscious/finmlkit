import numpy as np
import pandas as pd
from pathlib import Path

from finmlkit.bar.data_model import TradesData, FootprintData
from finmlkit.bar.io import H5Inspector
from finmlkit.bar.kit import TimeBarKit, TickBarKit, VolumeBarKit, DollarBarKit, CUSUMBarKit


def _sample_trades():
    ts = np.array([1000, 1000, 2000, 3000], dtype=np.int64)
    px = np.array([10.0, 10.0, 11.0, 12.0])
    qty = np.array([1.0, 2.0, 1.5, 1.0])
    ids = np.array([2, 1, 3, 4], dtype=np.int64)
    return ts, px, qty, ids


def test_tradesdata_save_load_and_view(tmp_path):
    ts, px, qty, ids = _sample_trades()
    td = TradesData(ts, px, qty, ids, timestamp_unit="ms", preprocess=True)

    # basic properties
    assert td.orig_timestamp_unit == "ms"
    assert len(td.data) == 3  # two first trades are merged

    # set and verify view range
    td.set_view_range(td.data.index[0], td.data.index[-1])
    assert td.start_date <= td.end_date
    assert not td.data.empty

    # save to h5 and reload
    h5file = tmp_path / "trades.h5"
    key = td.save_h5(str(h5file))
    assert key.startswith("/trades/")

    loaded = TradesData.load_trades_h5(str(h5file), start_time=td.start_date, end_time=td.end_date)
    assert isinstance(loaded, TradesData)
    assert not loaded.data.empty

    # H5 inspection utilities
    inspector = H5Inspector(str(h5file))
    keys = inspector.list_keys()
    assert key in keys
    meta = inspector.get_metadata(key)
    assert meta["record_count"] == len(td.data)
    assert inspector.get_integrity_info(key) is None
    stats = inspector.get_statistics(key)
    assert stats["record_count"] == len(td.data)
    gaps = inspector.inspect_gaps(processes=1)
    assert key in gaps
    summary = inspector.get_integrity_summary(verbose=False)
    assert summary is None or key in summary


def test_bar_kits_run(tmp_path):
    ts, px, qty, ids = _sample_trades()
    td = TradesData(ts, px, qty, ids, timestamp_unit="ms", preprocess=True)

    # Time bars
    tkit = TimeBarKit(td, pd.Timedelta(seconds=1))
    tts, tidx = tkit._comp_bar_close()
    assert len(tts) == len(tidx)

    # Tick bars
    tikit = TickBarKit(td, tick_count_thrs=1)
    tts, tidx = tikit._comp_bar_close()
    assert len(tts) == len(tidx)

    # Volume bars
    vkit = VolumeBarKit(td, volume_ths=1.0)
    vts, vidx = vkit._comp_bar_close()
    assert len(vts) == len(vidx)

    # Dollar bars
    dkit = DollarBarKit(td, dollar_thrs=10.0)
    dts, didx = dkit._comp_bar_close()
    assert len(dts) == len(didx)

    # CUSUM bars require sigma vector
    sigma = np.ones(len(td.data))
    ckit = CUSUMBarKit(td, sigma)
    cts, cidx = ckit._comp_bar_close()
    assert len(cts) == len(cidx)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])