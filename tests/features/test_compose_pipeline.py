import pandas as pd
import numpy as np

from finmlkit.feature.kit import Feature, FeatureKit
from finmlkit.feature.kit import Compose
from finmlkit.feature.transforms import SMA, EWMA


def make_df(n=64):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(2024)
    return pd.DataFrame({
        "close": 100 + rng.normal(0, 1, n).cumsum(),
    }, index=idx)


def test_compose_chain_execute_and_roundtrip(tmp_path):
    df = make_df()

    t1 = SMA(3, input_col="close")
    t2 = EWMA(5, input_col=t1.output_name)
    pipeline = Compose(t1, t2)

    f_pipeline = Feature(pipeline)

    kit = FeatureKit([f_pipeline], retain=["close"]) 

    out1 = kit.build(df, backend="pd", order="defined")
    assert f_pipeline.name in out1.columns

    # Round-trip config
    cfg_path = tmp_path / "compose_kit.json"
    kit.save_config(str(cfg_path))
    kit2 = FeatureKit.load_config(str(cfg_path))

    out2 = kit2.build(df, backend="pd", order="defined")

    assert list(out1.columns) == list(out2.columns)
    for c in out1.columns:
        assert np.allclose(out1[c].fillna(0).values, out2[c].fillna(0).values, atol=1e-12)
