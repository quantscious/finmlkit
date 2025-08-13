import pandas as pd
import numpy as np

from finmlkit.feature.kit import Feature, FeatureKit
from finmlkit.feature.transforms import SMA, EWMA, RSIWilder


def make_sample_df(n=50):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "close": 100 + rng.normal(0, 1, n).cumsum(),
        "volume": rng.integers(1000, 5000, n),
        "high": 100 + rng.normal(0, 1, n).cumsum() + 1.0,
        "low":  100 + rng.normal(0, 1, n).cumsum() - 1.0,
    }, index=idx)
    return df


def test_featurekit_save_load_roundtrip(tmp_path):
    df = make_sample_df(64)

    f_sma = Feature(SMA(5, input_col="close"))
    f_ewma = Feature(EWMA(10, input_col="close"))
    f_rsi_on_sma = Feature(RSIWilder(14, input_col=f_sma.name))  # depends on sma output

    kit = FeatureKit([f_sma, f_ewma, f_rsi_on_sma], retain=["close"]) 

    out1 = kit.build(df, backend="pd", timeit=False, order="topo")

    # Save and load
    cfg_path = tmp_path / "kit.json"
    kit.save_config(str(cfg_path))

    loaded = FeatureKit.load_config(str(cfg_path))
    out2 = loaded.build(df, backend="pd", timeit=False, order="topo")

    assert list(out1.columns) == list(out2.columns)
    for c in out1.columns:
        # allow small numeric differences if any
        assert np.allclose(out1[c].fillna(0).values, out2[c].fillna(0).values, atol=1e-12)
