from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.transforms.base import build_transform


def _planted_frame() -> pd.DataFrame:
    syms = [f"X{i:02d}" for i in range(30)]
    bars = synthetic_bars(
        symbols=syms,
        start="2026-03-02",
        end="2026-08-28",
        factors={"market": 0.0005, "sectors": 3},
        xs_alpha={"ic": 0.1, "contaminate": {"beta": 0.5}},
    )
    planted = bars.attrs["planted"]
    score = planted["xs_alpha"]["score"]
    beta = planted["beta"].reindex(score.columns)
    day = bars["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    close = bars.assign(day=day).groupby(["day", "symbol"])["close"].last().unstack("symbol")
    nxt = np.log(close).diff().shift(-1)  # the session d + 1 return, known only after d + 1
    long = score.stack().rename("score").reset_index()
    long.columns = ["day", "symbol", "score"]
    long["day"] = pd.to_datetime(long["day"])
    long["decision_ts"] = long["day"].dt.tz_localize("UTC")
    long["beta"] = long["symbol"].map(beta)
    long["fwd"] = [nxt.at[d, s] if d in nxt.index else np.nan for d, s in zip(long["day"], long["symbol"], strict=True)]
    return long


def _mean_corr(frame: pd.DataFrame, col: str) -> float:
    return float(np.mean([np.corrcoef(g[col], g["beta"])[0, 1] for _, g in frame.groupby("decision_ts")]))


def _mean_ic(frame: pd.DataFrame, col: str) -> float:
    groups = frame.dropna(subset=["fwd"]).groupby("decision_ts")
    return float(np.mean([spearmanr(g[col], g["fwd"])[0] for _, g in groups]))


def test_planted_beta_removed():
    f = _planted_frame()
    assert _mean_corr(f, "score") > 0.2  # contaminated going in
    t = build_transform({"type": "neutralize", "column": "score", "exposures": ["beta"]})
    f["neut"] = t.compute(f)
    assert abs(_mean_corr(f, "neut")) < 0.05
    assert _mean_ic(f, "neut") > 0


def test_proportion():
    rng = np.random.default_rng(3)
    ts = pd.date_range("2026-10-01 14:00", periods=4, freq="1min", tz="UTC")
    n = 20
    beta = rng.normal(size=(4, n))
    x = 2.0 * beta + rng.normal(size=(4, n))
    f = pd.DataFrame(
        {"symbol": [f"S{i}" for i in range(n)] * 4, "decision_ts": np.repeat(ts, n), "x": x.ravel(),
         "beta": beta.ravel()}
    )

    def out(p: float) -> pd.Series:
        return build_transform({"type": "neutralize", "column": "x", "exposures": ["beta"], "proportion": p}).compute(f)

    np.testing.assert_allclose(out(0.0), f["x"])

    def slope(y: pd.Series, t: pd.Timestamp) -> float:
        m = f["decision_ts"] == t
        return float(np.polyfit(f.loc[m, "beta"], y[m], 1)[0])

    for t in ts:
        s0 = slope(f["x"], t)
        assert abs(slope(out(0.5), t) - 0.5 * s0) < 1e-9
        assert abs(slope(out(1.0), t)) < 1e-9


def test_collinear_sector_dummies_and_min_count():
    rng = np.random.default_rng(0)
    ts = pd.Timestamp("2026-10-01 14:00", tz="UTC")
    n = 12
    sec = rng.integers(0, 3, n)
    f = pd.DataFrame({"symbol": [f"S{i}" for i in range(n)], "decision_ts": ts, "x": rng.normal(size=n)})
    for k in range(3):
        f[f"s{k}"] = (sec == k).astype(float)
    out = build_transform({"type": "neutralize", "column": "x", "exposures": ["s0", "s1", "s2"]}).compute(f)
    assert out.notna().all()
    gm = out.groupby(sec).mean()
    assert np.allclose(gm, gm.iloc[0])  # every sector has the same mean after neutralizing
    small = build_transform({"type": "neutralize", "column": "x", "exposures": ["s0"], "min_count": 13})
    assert small.compute(f).isna().all()
