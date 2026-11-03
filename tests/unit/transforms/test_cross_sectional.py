from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.transforms.base import TRANSFORM_REGISTRY, apply_transforms, build_transform

T1, T2 = pd.Timestamp("2026-10-01 14:00", tz="UTC"), pd.Timestamp("2026-10-01 14:01", tz="UTC")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": list("ABCDEABCDE"),
            "decision_ts": [T1] * 5 + [T2] * 5,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 10.0, 30.0, 0.0, 20.0],
        }
    )


def test_registry_keys():
    assert {"rank", "zscore", "winsorize", "demean", "neutralize"} <= set(TRANSFORM_REGISTRY)
    with pytest.raises(ValueError, match="unknown transform"):
        build_transform({"type": "nope"})


def test_rank_zscore_hand():
    f = _frame()
    rank = build_transform({"type": "rank", "column": "x"}).compute(f)
    assert rank.iloc[:5].tolist() == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    # T2: values 10,10,30,0,20 -> ranks 2.5,2.5,5,1,4 -> pct = rank / 5
    assert rank.iloc[5:].tolist() == pytest.approx([0.5, 0.5, 1.0, 0.2, 0.8])
    raw = build_transform({"type": "rank", "column": "x", "pct": False}).compute(f)
    assert raw.iloc[5:].tolist() == [2.5, 2.5, 5.0, 1.0, 4.0]
    z = build_transform({"type": "zscore", "column": "x"}).compute(f)
    s = np.sqrt(2.5)  # sample std of 1..5
    assert z.iloc[:5].tolist() == pytest.approx([-2 / s, -1 / s, 0, 1 / s, 2 / s])
    v = np.array([10, 10, 30, 0, 20.0])
    assert z.iloc[5:].tolist() == pytest.approx(((v - 14) / v.std(ddof=1)).tolist())
    assert z.index.equals(f.index)


def test_no_cross_time_mixing():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2026-10-01 14:00", periods=6, freq="1min", tz="UTC")
    f = pd.DataFrame(
        {"symbol": list("ABCDEFGH") * 6, "decision_ts": np.repeat(ts, 8), "x": rng.normal(size=48),
         "b": rng.normal(size=48)}
    )
    specs = [{"type": "rank"}, {"type": "zscore"}, {"type": "winsorize"}, {"type": "demean"},
             {"type": "neutralize", "exposures": ["b"], "min_count": 5}]
    target = ts[2]
    other = f["decision_ts"] != target
    g = f.copy()
    g.loc[other, ["x", "b"]] = g.loc[other, ["x", "b"]].sample(frac=1, random_state=1).to_numpy()
    for spec in specs:
        t = build_transform({**spec, "column": "x"})
        a, b = t.compute(f), t.compute(g)
        m = f["decision_ts"] == target
        np.testing.assert_allclose(a[m], b[m], err_msg=spec["type"])


def test_min_count():
    f = _frame().iloc[:8]  # T1 has 5 rows, T2 has 3
    for spec in ({"type": "rank"}, {"type": "zscore"}, {"type": "winsorize"}):
        out = build_transform({**spec, "column": "x", "min_count": 4}).compute(f)
        assert out.iloc[:5].notna().all() and out.iloc[5:].isna().all(), spec
    # NaN inputs do not count toward min_count
    g = _frame()
    g.loc[[0, 1], "x"] = np.nan
    out = build_transform({"type": "rank", "column": "x", "min_count": 4}).compute(g)
    assert out.iloc[:5].isna().all() and out.iloc[5:].notna().all()


def test_winsorize_and_demean():
    f = _frame().iloc[:5]
    w = build_transform({"type": "winsorize", "column": "x", "lower": 0.25, "upper": 0.75}).compute(f)
    assert w.tolist() == [2.0, 2.0, 3.0, 4.0, 4.0]
    f = f.assign(g=["u", "u", "v", "v", "v"])
    assert build_transform({"type": "demean", "column": "x"}).compute(f).tolist() == [-2, -1, 0, 1, 2]
    assert build_transform({"type": "demean", "column": "x", "by": "g"}).compute(f).tolist() == [-0.5, 0.5, -1, 0, 1]


def test_apply_transforms_adds_prefixed_float_columns():
    f = _frame()
    out = apply_transforms(
        f, [build_transform({"type": "rank", "column": "x"}), build_transform({"type": "zscore", "column": "x"})]
    )
    assert {"rank_x", "zscore_x"} <= set(out.columns)
    assert out["rank_x"].dtype == np.float64
    assert list(f.columns) == ["symbol", "decision_ts", "x"]  # input untouched
