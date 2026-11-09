"""C1o: planted effects in `synthetic_bars` are opt-in, recoverable, and leave the defaults untouched."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import coint

from shpe_research.contracts.bars import validate_bars
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.experiments.sources import load_bars

# sha256 of pd.util.hash_pandas_object(synthetic_bars(), index=True), taken before C1o changed fixtures.py.
DEFAULT_HASH = "b08d2f43fa2468b9297a1d37dd5fd9bdcaad12974d5ec5ee0881b07c7ec54a0e"


def _session(bars: pd.DataFrame) -> pd.Series:
    return bars["ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)


def _within_session_returns(g: pd.DataFrame, lags: int = 0) -> pd.DataFrame:
    """Log returns of one symbol, dropping a session's first `lags + 1` bars so no overnight gap enters."""
    g = g.reset_index(drop=True)
    sess = _session(g)
    r = np.log(g["close"]).diff()
    out = pd.DataFrame({"r": r, "session": sess})
    for k in range(1, lags + 1):
        out[f"l{k}"] = r.shift(k)
    return out[sess.eq(sess.shift(lags + 1))].dropna()


def _ou_half_life(spread: np.ndarray) -> float:
    y, x = spread[1:], spread[:-1]
    x = x - x.mean()
    y = y - y.mean()
    phi = float(x @ y / (x @ x))
    return float(-np.log(2) / np.log(phi))


def test_defaults_unchanged():
    bars = synthetic_bars()
    digest = hashlib.sha256(pd.util.hash_pandas_object(bars, index=True).values.tobytes()).hexdigest()
    assert digest == DEFAULT_HASH
    assert bars.attrs["planted"] == {}


def test_deterministic_and_composable():
    kwargs = dict(
        symbols=["AAA", "BBB", "CCC", "DDD"],
        start="2026-03-02",
        end="2026-08-28",
        factors={"market": 0.0005, "sectors": 2},
        xs_alpha={"ic": 0.05},
        null_symbols=2,
    )
    a, b = synthetic_bars(**kwargs), synthetic_bars(**kwargs)
    pd.testing.assert_frame_equal(a, b)
    validate_bars(a)
    assert {"N01", "N02"} <= set(a["symbol"])
    assert set(a.attrs["planted"]) >= {"beta", "sector", "xs_alpha", "null_symbols"}
    # Effects are additive: base symbols keep their draws whether or not the null symbols exist.
    plain = synthetic_bars(symbols=kwargs["symbols"], start="2026-03-02", end="2026-08-28", null_symbols=2)
    pd.testing.assert_frame_equal(plain[plain["symbol"] == "N01"], a[a["symbol"] == "N01"])


def test_null_symbols_named_and_listed():
    bars = synthetic_bars(symbols=["AAA"], start="2026-03-02", end="2026-03-06", null_symbols=["N01", "ZZZ"])
    assert set(bars["symbol"]) == {"AAA", "N01", "ZZZ"}
    with pytest.raises(ValueError, match="already"):
        synthetic_bars(symbols=["AAA"], start="2026-03-02", end="2026-03-06", null_symbols=["AAA"])


def test_reversal_recovered():
    ar = [-0.2, -0.05]
    bars = synthetic_bars(symbols=["AAA", "BBB"], start="2026-03-02", end="2026-03-27", reversal={"ar": ar})
    assert bars.attrs["planted"]["reversal"]["ar"] == ar
    for _, g in bars.groupby("symbol"):
        d = _within_session_returns(g, lags=2)
        X = np.column_stack([d["l1"], d["l2"]])
        coef = np.linalg.lstsq(X, d["r"].to_numpy(), rcond=None)[0]
        assert abs(coef[0] - ar[0]) < 0.05
        assert abs(coef[1] - ar[1]) < 0.05


def test_pair_half_life():
    pairs = [{"beta": 1.5, "half_life_bars": 60}, {"beta": 0.8, "half_life_bars": 200}]
    bars = synthetic_bars(symbols=["AAA"], start="2026-03-02", end="2026-04-03", cointegrated_pairs=pairs)
    planted = bars.attrs["planted"]["cointegrated_pairs"]["pairs"]
    assert len(planted) == 2
    close = bars.pivot(index="ts", columns="symbol", values="close")
    for spec in planted:
        s1, s2 = spec["symbols"]
        spread = np.log(close[s1]) - spec["beta"] * np.log(close[s2])
        hl = _ou_half_life(spread.to_numpy())
        assert abs(hl - spec["half_life_bars"]) <= 0.3 * spec["half_life_bars"]
        assert coint(np.log(close[s1]), np.log(close[s2]), trend="c")[1] < 0.01


def test_trap_not_cointegrated():
    bars = synthetic_bars(
        symbols=["AAA"],
        start="2026-03-02",
        end="2026-03-27",
        cointegrated_pairs={"pairs": [{"beta": 1.0, "half_life_bars": 60}], "trap": True},
    )
    s1, s2 = bars.attrs["planted"]["cointegrated_pairs"]["trap"]["symbols"]
    log = np.log(bars.pivot(index="ts", columns="symbol", values="close"))
    assert log[s1].diff().corr(log[s2].diff()) > 0.4  # correlated ...
    assert coint(log[s1], log[s2], trend="c")[1] > 0.1  # ... but not cointegrated


def test_trap_flag_inside_list():
    bars = synthetic_bars(
        symbols=["AAA"],
        start="2026-03-02",
        end="2026-03-13",
        cointegrated_pairs=[{"beta": 1.0, "half_life_bars": 60}, {"trap": True}],
    )
    assert "trap" in bars.attrs["planted"]["cointegrated_pairs"]


def _mean_rv_acf1(bars: pd.DataFrame) -> float:
    out = []
    for _, g in bars.groupby("symbol"):
        d = _within_session_returns(g)
        rv = (d["r"] ** 2).groupby(d["session"]).sum()
        out.append(rv.autocorr(1))
    return float(np.mean(out))


def test_vol_persistence():
    syms = ["AAA", "BBB", "CCC", "DDD"]
    bars = synthetic_bars(symbols=syms, start="2026-03-02", end="2026-04-03", vol_clustering={"persistence": 0.95})
    assert _session(bars).nunique() >= 126  # the option extends `end` to its minimum span
    plain = synthetic_bars(symbols=syms, start="2026-03-02", end="2026-08-28")
    assert _mean_rv_acf1(bars) > 0.5
    assert _mean_rv_acf1(plain) < 0.3
    assert bars.attrs["planted"]["vol_clustering"]["persistence"] == 0.95


def test_factors_beta():
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    bars = synthetic_bars(symbols=syms, start="2026-03-02", end="2026-04-03", factors={"market": 0.0005, "sectors": 2})
    planted = bars.attrs["planted"]
    assert set(planted["beta"].index) == set(syms)
    assert set(planted["sector"]) <= {"SEC1", "SEC2"}
    market = planted["factors"]["market_return"]
    for sym in syms:
        g = bars[bars["symbol"] == sym].reset_index(drop=True)
        d = _within_session_returns(g)
        m = market.reindex(g["ts"]).to_numpy()[d.index]
        m = m - m.mean()
        beta_hat = float(m @ (d["r"].to_numpy() - d["r"].mean()) / (m @ m))
        assert abs(beta_hat - planted["beta"][sym]) < 0.1


def _session_returns(bars: pd.DataFrame) -> pd.DataFrame:
    last = bars.assign(session=_session(bars)).groupby(["symbol", "session"])["close"].last()
    return np.log(last).groupby(level="symbol").diff().unstack("symbol")


def test_xs_alpha_ic():
    syms = [f"X{i:02d}" for i in range(30)]
    ic = 0.1
    bars = synthetic_bars(symbols=syms, start="2026-03-02", end="2026-08-28", xs_alpha={"ic": ic})
    score = bars.attrs["planted"]["xs_alpha"]["score"]
    nxt = _session_returns(bars).shift(-1)
    days = score.index.intersection(nxt.index)[:-1]
    ics = [spearmanr(score.loc[d], nxt.loc[d].reindex(score.columns))[0] for d in days]
    assert abs(float(np.mean(ics)) - ic) < 0.02


def test_xs_alpha_contaminated_score_loads_on_beta():
    syms = [f"X{i:02d}" for i in range(24)]
    bars = synthetic_bars(
        symbols=syms,
        start="2026-03-02",
        end="2026-08-28",
        factors={"market": 0.0005, "sectors": 3},
        xs_alpha={"ic": 0.1, "contaminate": {"beta": 1.0, "sector": 1.0}},
    )
    planted = bars.attrs["planted"]
    raw, clean = planted["xs_alpha"]["score"], planted["xs_alpha"]["score_clean"]
    beta = planted["beta"].reindex(raw.columns).to_numpy()
    corr_raw = np.mean([np.corrcoef(raw.loc[d], beta)[0, 1] for d in raw.index])
    corr_clean = np.mean([np.corrcoef(clean.loc[d], beta)[0, 1] for d in clean.index])
    assert corr_raw > 0.3 > abs(corr_clean)


def test_xs_momentum():
    bars = synthetic_bars(daily={"years": 3, "n_symbols": 50, "xs_momentum": True})
    validate_bars(bars)
    assert bars["symbol"].nunique() >= 50
    close = np.log(bars.pivot(index="ts", columns="symbol", values="close"))
    assert len(close) >= 3 * 250
    spreads = []
    for t in range(252, len(close) - 21, 21):
        score = close.iloc[t - 21] - close.iloc[t - 252]  # 12-1: skip the latest month
        fwd = close.iloc[t + 21] - close.iloc[t]
        order = score.sort_values().index
        spreads.append(fwd[order[-5:]].mean() - fwd[order[:5]].mean())
    spreads = np.array(spreads)
    assert spreads.mean() > 0
    assert spreads.mean() / (spreads.std(ddof=1) / np.sqrt(len(spreads))) > 2


def test_daily_bars_contract():
    bars = synthetic_bars(daily={"years": 3, "n_symbols": 50})
    assert (bars["ts_available"] - bars["ts"] > pd.Timedelta(hours=3)).all()
    assert bars.groupby("symbol").size().nunique() == 1
    with pytest.raises(ValueError, match="years"):
        synthetic_bars(daily={"years": 1, "n_symbols": 50})
    with pytest.raises(ValueError, match="intraday"):
        synthetic_bars(daily={"years": 3, "n_symbols": 50}, momentum={"breakout_prob": 0.5})


def _breakout_checks(bars: pd.DataFrame, events: pd.DataFrame, range_minutes: int) -> pd.Series:
    """Assert each event bar is the first close above the opening-range high; return 30-minute forward returns."""
    out = []
    by_symbol = {s: g.reset_index(drop=True) for s, g in bars.groupby("symbol")}
    for ev in events.itertuples():
        g = by_symbol[ev.symbol]
        sess = _session(g)
        pos = int(np.flatnonzero((g["ts"] == ev.ts).to_numpy())[0])
        day = g[sess == sess.iloc[pos]].reset_index(drop=True)
        k = int(np.flatnonzero((day["ts"] == ev.ts).to_numpy())[0])
        high = day["high"].iloc[:range_minutes].max()
        assert day["close"].iloc[k] > high
        assert (day["close"].iloc[range_minutes:k] <= high).all()
        out.append(np.log(day["close"].iloc[k + 30] / day["close"].iloc[k]) if k + 30 < len(day) else np.nan)
    return pd.Series(out, index=events.index)


def test_orb_rvol_conditional():
    thr = 1.5
    bars = synthetic_bars(
        symbols=["AAA", "BBB", "CCC", "DDD"],
        start="2026-03-02",
        end="2026-05-29",
        orb_rvol={"rvol_threshold": thr, "drift_bps": 40, "drift_minutes": 30, "context": "SPY"},
    )
    assert "SPY" in set(bars["symbol"])
    planted = bars.attrs["planted"]["orb_rvol"]
    events = planted["events"]
    assert len(events) > 40
    fwd = _breakout_checks(bars, events, planted["range_minutes"])
    above = events["rvol"] >= thr
    assert above.any() and (~above).any()
    assert fwd[above].mean() > fwd[~above].mean()
    assert fwd[above].mean() > 0.001


def test_momentum_breakout_drifts_up():
    bars = synthetic_bars(
        symbols=["AAA", "BBB", "CCC", "DDD"],
        start="2026-03-02",
        end="2026-05-29",
        momentum={"breakout_prob": 0.5, "drift_bps": 40, "drift_minutes": 30},
    )
    planted = bars.attrs["planted"]["momentum"]
    events = planted["events"]
    assert len(events) > 40
    fwd = _breakout_checks(bars, events, planted["range_minutes"])
    assert fwd.mean() > 0.001


def test_sources_passthrough():
    cfg = {
        "source": "fixture",
        "symbols": ["AAA"],
        "start": "2026-03-02",
        "end": "2026-03-13",
        "reversal": {"ar": [-0.3]},
    }
    bars = load_bars(cfg)
    assert bars.attrs["planted"]["reversal"]["ar"] == [-0.3]
    pd.testing.assert_frame_equal(bars, synthetic_bars(**{k: v for k, v in cfg.items() if k != "source"}))
    with pytest.raises(TypeError):
        load_bars({"source": "fixture", "no_such_effect": {}})
