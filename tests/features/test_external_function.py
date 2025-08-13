import pandas as pd
import numpy as np

from finmlkit.feature.kit import Feature, FeatureKit
from finmlkit.feature.transforms import ExternalFunction


def make_df(n=64):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(2025)
    return pd.DataFrame({
        "close": 100 + rng.normal(0, 1, n).cumsum(),
        "volume": rng.integers(1000, 5000, n)
    }, index=idx)


def test_external_function_single_output_roundtrip(tmp_path):
    df = make_df()

    # Use numpy.log on close (returns ndarray)
    tf_log = ExternalFunction(np.log, input_cols="close", output_cols="ext_log", pass_numpy=True)
    f_log = Feature(tf_log)

    kit = FeatureKit([f_log], retain=["close"]) 

    out1 = kit.build(df, backend="pd")
    assert "ext_log" in out1.columns

    # Round-trip config
    cfg_path = tmp_path / "ext_log_kit.json"
    kit.save_config(str(cfg_path))
    kit2 = FeatureKit.load_config(str(cfg_path))

    out2 = kit2.build(df, backend="pd")

    assert list(out1.columns) == list(out2.columns)
    for c in out1.columns:
        assert np.allclose(out1[c].fillna(0).values, out2[c].fillna(0).values, atol=1e-12)


def test_external_function_multi_output_roundtrip(tmp_path):
    df = make_df()

    # np.modf returns (frac, int) arrays
    tf_modf = ExternalFunction("numpy.modf", input_cols="close", output_cols=["frac", "integ"], pass_numpy=True)
    f_modf = Feature(tf_modf)

    kit = FeatureKit([f_modf], retain=["close"]) 

    out1 = kit.build(df, backend="pd")
    assert set(["frac", "integ"]).issubset(out1.columns)

    # Round-trip config
    cfg_path = tmp_path / "ext_modf_kit.json"
    kit.save_config(str(cfg_path))
    kit2 = FeatureKit.load_config(str(cfg_path))

    out2 = kit2.build(df, backend="pd")

    assert set(out1.columns) == set(out2.columns)
    for c in out1.columns:
        assert np.allclose(out1[c].fillna(0).values, out2[c].fillna(0).values, atol=1e-12)
