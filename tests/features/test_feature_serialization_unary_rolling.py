import pandas as pd
import numpy as np

from finmlkit.feature.kit import Feature, FeatureKit
import finmlkit.feature.transforms as tfs


def make_df(n=96):
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "close": 100 + rng.normal(0, 1, n).cumsum(),
    }, index=idx)


def test_unary_rolling_ema_lag_and_apply_roundtrip(tmp_path):
    df = make_df()

    # Base
    f_close = Feature(tfs.Identity("close"))
    f_sma3 = Feature(tfs.SMA(3, input_col="close"))

    # Unary via convenience/apply
    f_abs = f_close.abs()
    f_square = f_close.square()
    f_exp = f_close.exp()
    f_log1p = f_close.log1p()
    f_log = (f_close - 90.0).clip(lower=1e-6, upper=None).log()  # ensure positive input for log
    f_sqrt = (f_close - f_sma3).square().sqrt()  # sqrt of square -> non-negative

    # Rolling family
    f_rmean5 = f_close.rolling_mean(5)
    f_rstd7 = f_close.rolling_std(7)
    f_rsum9 = f_close.rolling_sum(9)
    f_ema6 = f_close.ema(6)
    f_lag2 = f_close.lag(2)

    # Alias check via apply: tsum window 4
    f_tsum4 = f_close.apply(lambda x: x.rolling(window=4).sum(), suffix="tsum4")

    # Arithmetic and min/max, constants and features, including right-const ops
    f_add_c = f_close + 2.5
    f_sub_c = f_close - 1.5
    f_mul_c = f_close * 3.0
    f_div_c = f_close / 2.0
    f_rsub_c = 200.0 - f_close
    f_rdiv_c = 1000.0 / (f_close + 1e-9)

    f_min_ff = Feature.min(f_close, f_sma3)
    f_max_fc = Feature.max(f_close, 100.0)

    # Clip via apply helper
    f_clip = (f_close - f_sma3).clip(lower=-0.1, upper=0.1)

    feats = [
        f_close, f_sma3,
        f_abs, f_square, f_exp, f_log1p, f_log, f_sqrt,
        f_rmean5, f_rstd7, f_rsum9, f_ema6, f_lag2, f_tsum4,
        f_add_c, f_sub_c, f_mul_c, f_div_c, f_rsub_c, f_rdiv_c,
        f_min_ff, f_max_fc, f_clip,
    ]

    kit = FeatureKit(feats, retain=["close"]) 

    out1 = kit.build(df, backend="pd", order="topo")

    # Round-trip
    cfg_path = tmp_path / "unary_rolling_kit.json"
    kit.save_config(str(cfg_path))

    loaded = FeatureKit.from_config(str(cfg_path))
    out2 = loaded.build(df, backend="pd", order="topo")

    assert set(out1.columns) == set(out2.columns)
    for c in out1.columns:
        assert np.allclose(out1[c].fillna(0).values, out2[c].fillna(0).values, atol=1e-10)
