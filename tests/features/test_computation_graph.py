import pandas as pd
import numpy as np

from finmlkit.feature.kit import Feature, FeatureKit
from finmlkit.feature.transforms import SMA, EWMA


def make_df(n=32):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "close": 100 + rng.normal(0, 1, n).cumsum(),
    }, index=idx)


def test_graph_edges_and_topo():
    f_sma = Feature(SMA(5, input_col="close"))
    f_ewma = Feature(EWMA(10, input_col="close"))
    f_ratio = f_sma / f_ewma

    name_sma = str(f_sma.name)
    name_ewma = str(f_ewma.name)
    name_ratio = str(f_ratio.name)

    kit = FeatureKit([f_ratio, f_sma, f_ewma], retain=["close"])  # deliberately unsorted order

    g = kit.build_graph()
    viz = g.visualize()

    # Raw input edges
    assert "input:close -> [" in viz

    # Child dependencies should be present in viz
    assert f"{name_sma} -> [" in viz
    assert f"{name_ewma} -> [" in viz

    # Topological order should place inputs first, then sma/ewma, then ratio
    topo = kit.topological_order()
    assert topo.index(name_sma) < topo.index(name_ratio)
    assert topo.index(name_ewma) < topo.index(name_ratio)

    # Building in topo order should work and include all 3 columns
    df = make_df(64)
    out = kit.build(df, backend="pd", order="topo")
    assert set(["close", name_sma, name_ewma, name_ratio]).issubset(set(out.columns))
