import numpy as np
import pandas as pd

from finmlkit.feature.base import SISOTransform
from finmlkit.feature.kit import Feature, Compose, FeatureKit


class AddOneTransform(SISOTransform):
    def __init__(self, input_col):
        super().__init__(input_col, "plus1")

    def _pd(self, x: pd.DataFrame):
        return x[self.requires[0]] + 1

    def _nb(self, x: pd.DataFrame):
        arr = self._prepare_input_nb(x)
        return self._prepare_output_nb(x.index, arr + 1)


def sample_df():
    idx = pd.date_range("2020", periods=5, freq="D")
    return pd.DataFrame({"a": np.arange(5, dtype=float)}, index=idx)


def test_feature_math_and_cache():
    df = sample_df()
    t = AddOneTransform("a")
    feat = Feature(t)

    out1 = feat(df, backend="pd")
    cache = df.copy()
    out2 = feat(df, cache=cache, backend="pd")
    # cached result should be same
    pd.testing.assert_series_equal(out1, out2)

    # arithmetic operations
    f2 = feat + 1
    res = f2(df, backend="pd")
    assert np.allclose(res.values, df["a"].values + 2)
    f3 = 2 + feat
    res = f3(df, backend="pd")
    assert np.allclose(res.values, df["a"].values + 3)
    f4 = feat * 2
    res = f4(df, backend="pd")
    assert np.allclose(res.values, (df["a"] + 1) * 2)
    f5 = 2 / feat
    res = f5(df, backend="pd")
    assert np.allclose(res.values, 2 / (df["a"] + 1))
    f6 = Feature.min(feat, f4)
    res = f6(df, backend="pd")
    assert (res <= f4(df, backend="pd")).all()
    f7 = Feature.max(feat, f4)
    res = f7(df, backend="pd")
    assert (res >= feat(df, backend="pd")).all()

    # function application
    logged = feat.log()(df, backend="pd")
    assert np.allclose(logged.values, np.log(df["a"] + 1))
    clipped = feat.clip(lower=1, upper=3)(df, backend="pd")
    assert clipped.min() >= 1 and clipped.max() <= 3
    rolled = feat.rolling_mean(2)(df, backend="pd")
    assert len(rolled) == len(df)
    lagged = feat.lag(1)(df, backend="pd")
    assert lagged.isna().sum() == 1


def test_compose_and_featurekit():
    df = sample_df()
    t1 = AddOneTransform("a")
    t2 = AddOneTransform("a_plus1")
    comp = Compose(t1, t2)
    comp_res = comp(df, backend="pd")
    assert comp_res.name.endswith("plus1_plus1")

    feat1 = Feature(t1)
    feat2 = Feature(t2)
    kit = FeatureKit([feat1, feat2], retain=["a"])
    built = kit.build(df, backend="pd")
    assert "a" in built.columns
    assert feat1.name in built.columns
