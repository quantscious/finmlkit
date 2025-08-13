import pandas as pd
import numpy as np
import pytest

from finmlkit.feature.base import BinaryOpTransform
from finmlkit.feature.kit import Compose
from finmlkit.feature.transforms import SMA, EWMA


def make_df(n=64):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "close": 100 + rng.normal(0, 1, n).cumsum(),
    }, index=idx)


def test_binaryop_uses_cached_operands(monkeypatch):
    df = make_df()

    # Prepare cached child outputs in the DataFrame
    t_left = SMA(3, input_col="close")
    t_right = EWMA(5, input_col="close")

    s_left = t_left(df, backend="pd")
    s_right = t_right(df, backend="pd")

    df[s_left.name] = s_left
    df[s_right.name] = s_right

    # Ensure children are not called when their outputs are cached
    def boom_left(x, *, backend="pd"):
        raise AssertionError("Left child should not be called when cached")

    def boom_right(x, *, backend="pd"):
        raise AssertionError("Right child should not be called when cached")

    monkeypatch.setattr(t_left, "__call__", boom_left)
    monkeypatch.setattr(t_right, "__call__", boom_right)

    op = BinaryOpTransform(t_left, t_right, "add", lambda a, b: a + b)
    res = op(df, backend="pd")

    expected = df[s_left.name] + df[s_right.name]
    assert np.allclose(res.fillna(0).values, expected.fillna(0).values)
    assert isinstance(res, pd.Series)


def test_compose_skips_cached_steps(monkeypatch):
    df = make_df()

    # Chain: t1 feeds into t2
    t1 = SMA(3, input_col="close")
    t2 = EWMA(5, input_col=t1.output_name)

    s1 = t1(df, backend="pd")
    df[s1.name] = s1

    # For t2 we need input column named as t2.requires[0]
    s2 = t2(pd.DataFrame(s1.values, index=s1.index, columns=[t2.requires[0]]), backend="pd")
    df[s2.name] = s2

    # Compose should reuse cached outputs and not call underlying transforms
    def boom1(x, *, backend="pd"):
        raise AssertionError("First step should not be recomputed")

    def boom2(x, *, backend="pd"):
        raise AssertionError("Second step should not be recomputed")

    monkeypatch.setattr(t1, "__call__", boom1)
    monkeypatch.setattr(t2, "__call__", boom2)

    comp = Compose(t1, t2)
    out = comp(df, backend="pd")

    # Final result should equal the cached second step but be renamed to composed output name
    assert np.allclose(out.fillna(0).values, df[s2.name].fillna(0).values)
    assert out.name == comp.output_name


def test_compose_short_circuit_final(monkeypatch):
    df = make_df()

    t1 = SMA(3, input_col="close")
    t2 = EWMA(5, input_col=t1.output_name)
    comp = Compose(t1, t2)

    # Precompute final output and place directly into df with the composed final name
    s1 = t1(df, backend="pd")
    s2 = t2(pd.DataFrame(s1.values, index=s1.index, columns=[t2.requires[0]]), backend="pd")
    final_name = comp.output_name
    df[final_name] = s2.values  # same values, final column name

    # Any calls to underlying transforms should not happen when final is cached
    def boom1(x, *, backend="pd"):
        raise AssertionError("First step should not be called when final is cached")

    def boom2(x, *, backend="pd"):
        raise AssertionError("Second step should not be called when final is cached")

    monkeypatch.setattr(t1, "__call__", boom1)
    monkeypatch.setattr(t2, "__call__", boom2)

    out = comp(df, backend="pd")
    assert isinstance(out, pd.Series)
    assert out.name == final_name
    assert np.allclose(out.fillna(0).values, s2.fillna(0).values)
