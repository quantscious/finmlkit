import json
import pandas as pd
import numpy as np

from finmlkit.feature.kit import Feature, FeatureKit
import finmlkit.feature.transforms as tfs


def make_df(n=128):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "close": 100 + rng.normal(0, 1, n).cumsum(),
    }, index=idx)


def test_quickstart_ops_kit_roundtrip(tmp_path):
    df = make_df()

    # Base features
    f_close = Feature(tfs.Identity("close"))
    f_sma3 = Feature(tfs.SMA(3, input_col="close"))
    f_ewma5 = Feature(tfs.EWMA(5, input_col="close"))

    # Arithmetic ops mirroring the notebook
    f_ratio = f_sma3 / (f_ewma5 + 1e-9)
    f_ratio.name = "sma3_over_ewma5"

    f_shifted = f_close - 1000.0  # feature - constant (sub)

    f_abs = (f_close - f_sma3).abs()

    f_clipped = (f_close - f_ewma5).clip(lower=-100.0, upper=100.0)

    f_min_fc = Feature.min(f_close, f_sma3)
    f_min_fc.name = "min_close_sma3"

    f_max_fC = Feature.max(f_close, 100.0)
    f_max_fC.name = "max_close_100"

    kit = FeatureKit([
        f_sma3, f_ewma5, f_ratio, f_shifted, f_abs, f_clipped, f_min_fc, f_max_fC
    ], retain=["close"]) 

    out1 = kit.build(df, backend="pd", order="topo")

    # Round-trip
    cfg_path = tmp_path / "quick_ops_kit.json"
    kit.save_config(str(cfg_path))

    # Optional: read raw JSON for debug if needed
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        assert isinstance(cfg, dict)

    loaded = FeatureKit.from_config(str(cfg_path))
    out2 = loaded.build(df, backend="pd", order="topo")

    assert set(out1.columns) == set(out2.columns)
    for c in out1.columns:
        assert np.allclose(out1[c].fillna(0).values, out2[c].fillna(0).values, atol=1e-12)
