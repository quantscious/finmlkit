import numpy as np
import pandas as pd

from finmlkit.feature.kit import Feature, FeatureKit
from finmlkit.feature.transforms import SMA, EWMA
from finmlkit.feature.transforms import ExternalFunction


def make_df(n=64):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "close": 100 + rng.normal(0, 1, n).cumsum(),
    }, index=idx)


def test_external_function_numpy_single_and_multi_roundtrip(tmp_path):
    df = make_df()

    # Single-output external function using numpy.sqrt
    ext_sqrt = ExternalFunction(np.sqrt, input_cols="close", output_cols="sqrt_close", pass_numpy=True)
    f_sqrt = Feature(ext_sqrt)

    # Multi-output external function using numpy.modf -> (frac, int)
    ext_modf = ExternalFunction(np.modf, input_cols="close", output_cols=["frac_close", "int_close"], pass_numpy=True)
    f_modf = Feature(ext_modf)

    kit = FeatureKit([f_sqrt, f_modf], retain=["close"])  # small kit with two external features

    out1 = kit.build(df, backend="pd", order="topo")
    # Sanity: compare against direct numpy
    assert np.allclose(out1["sqrt_close"].fillna(0).values, np.sqrt(df["close"]).fillna(0).values)
    frac, integ = np.modf(df["close"].values)
    assert np.allclose(out1["frac_close"].fillna(0).values, np.nan_to_num(frac))
    assert np.allclose(out1["int_close"].fillna(0).values, np.nan_to_num(integ))

    # Round-trip serialization
    cfg_path = tmp_path / "external_numpy_kit.json"
    kit.save_config(str(cfg_path))
    loaded = FeatureKit.load_config(str(cfg_path))

    out2 = loaded.build(df, backend="pd", order="topo")

    assert set(out1.columns) == set(out2.columns)
    for c in out1.columns:
        assert np.allclose(out1[c].fillna(0).values, out2[c].fillna(0).values, atol=1e-12)


def test_computation_graph_visualize_and_topo():
    df = make_df()

    f_sma = Feature(SMA(3, input_col="close"))
    f_ewma = Feature(EWMA(5, input_col="close"))
    f_ratio = f_sma / (f_ewma + 1e-9)
    f_ratio.name = "sma_over_ewma"

    kit = FeatureKit([f_ratio, f_sma, f_ewma], retain=["close"])  # intentionally misordered

    # Graph visualization contains input and feature node names
    g = kit.build_graph()
    viz = g.visualize()
    assert "input:close" in viz
    assert str(f_sma.name) in viz
    assert str(f_ewma.name) in viz
    assert str(f_ratio.name) in viz

    # Topological order should place dependencies before dependents
    topo = kit.topological_order()
    i_ratio = topo.index(str(f_ratio.name))
    i_sma = topo.index(str(f_sma.name))
    i_ewma = topo.index(str(f_ewma.name))
    assert i_ratio > i_sma
    assert i_ratio > i_ewma

    # Build to ensure no runtime issues
    out = kit.build(df, backend="pd", order="topo")
    assert str(f_ratio.name) in out.columns
