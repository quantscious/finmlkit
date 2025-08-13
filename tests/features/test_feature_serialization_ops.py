import pandas as pd
import numpy as np

from finmlkit.feature.kit import Feature, FeatureKit
from finmlkit.feature.transforms import SMA, EWMA


def make_df(n=64):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "close": 100 + rng.normal(0, 1, n).cumsum(),
    }, index=idx)


def test_ops_serialization_roundtrip(tmp_path):
    df = make_df()

    f_sma = Feature(SMA(3, input_col="close"))
    f_ewma = Feature(EWMA(5, input_col="close"))

    f_ratio = f_sma / f_ewma
    f_sum_const = f_sma + 2.0
    f_rsub_const = 2.0 - f_sma
    f_abs = f_sma.abs()
    f_min_const = Feature.min(f_sma, 10.0)
    f_max_feat = Feature.max(f_sma, f_ewma)

    kit = FeatureKit([f_sma, f_ewma, f_ratio, f_sum_const, f_rsub_const, f_abs, f_min_const, f_max_feat], retain=["close"])

    out1 = kit.build(df, backend="pd", order="topo")

    # Round trip
    cfg_path = tmp_path / "ops_kit.json"
    kit.save_config(str(cfg_path))
    loaded = FeatureKit.from_config(str(cfg_path))

    out2 = loaded.build(df, backend="pd", order="topo")

    assert set(out1.columns) == set(out2.columns)
    for c in out1.columns:
        assert np.allclose(out1[c].fillna(0).values, out2[c].fillna(0).values, atol=1e-12)
