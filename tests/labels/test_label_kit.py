import numpy as np
import pandas as pd

from finmlkit.bar.data_model import TradesData
from finmlkit.label.kit import TBMLabel, SampleWeights


def sample_trades():
    ts = np.array([1000, 1000, 2000, 3000], dtype=np.int64)
    px = np.array([10.0, 10.0, 11.0, 12.0])
    qty = np.ones_like(px)
    ids = np.arange(1, 5)
    return TradesData(ts, px, qty, ids, timestamp_unit="ms", preprocess=True)


def test_tbm_label_and_weights():
    trades = sample_trades()
    # features with target returns
    idx = pd.to_datetime([1, 2], unit="s")
    feats = pd.DataFrame({"ret": [0.01, 0.02]}, index=idx)

    tbm = TBMLabel(feats, target_ret_col="ret", min_ret=0.0,
                   horizontal_barriers=(-1.0, 1.0), vertical_barrier=pd.Timedelta(seconds=1))
    f_df, out = tbm.compute_labels(trades)
    assert not out.empty
    assert tbm.event_count == len(out)
    assert not tbm.labels.isna().any()

    weights = tbm.compute_weights(trades)
    assert {"avg_uniqueness", "return_attribution"}.issubset(weights.columns)

    final = SampleWeights.compute_final_weights(weights["avg_uniqueness"],
                                                return_attribution=weights["return_attribution"],
                                                labels=out["labels"])
    assert "weights" in final.columns
    assert len(final) == len(out)
