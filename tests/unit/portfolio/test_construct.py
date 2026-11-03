from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.portfolio import construct
from shpe_research.portfolio.construct import (
    neutralize_weights,
    portfolio_returns,
    rebalance,
    scores_to_weights,
)
from shpe_research.trade.types import CostModel

SCHEMES = ["top_k", "rank", "zscore", "vol_scaled"]


def _scores(n_days: int = 6, n_sym: int = 12, seed: int = 3) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-03-02", periods=n_days, freq="D", tz="UTC")
    syms = [f"S{i:02d}" for i in range(n_sym)]
    idx = pd.MultiIndex.from_product([days, syms], names=["decision_ts", "symbol"])
    return (
        pd.Series(rng.normal(size=len(idx)), index=idx, name="score"),
        pd.Series(rng.uniform(0.5, 2.0, size=len(idx)), index=idx, name="vol"),
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_dollar_neutral_and_gross(scheme):
    scores, vol = _scores()
    w = scores_to_weights(scores, scheme, gross=1.5, top_k=3, vol=vol)
    by = w.groupby(level="decision_ts")
    assert np.allclose(by.sum(), 0.0, atol=1e-12)
    assert np.allclose(w.abs().groupby(level="decision_ts").sum(), 1.5)
    if scheme == "top_k":
        assert (w != 0).groupby(level="decision_ts").sum().eq(6).all()
        assert np.isclose(w.max(), 1.5 / 6)


def test_missing_score_gets_zero_weight_and_is_reported():
    scores, _ = _scores()
    scores.iloc[[2, 15]] = np.nan
    w = scores_to_weights(scores, "rank", gross=1.0)
    assert (w.iloc[[2, 15]] == 0).all()
    assert list(w.attrs["missing_score"]) == list(scores.index[[2, 15]])
    assert np.allclose(w.abs().groupby(level="decision_ts").sum(), 1.0)


def test_top_k_skips_dates_with_too_few_names():
    scores, _ = _scores(n_sym=12)
    first = scores.index.get_level_values("decision_ts").unique()[0]
    scores = scores[~((scores.index.get_level_values("decision_ts") == first) & (np.arange(len(scores)) % 12 > 4))]
    w = scores_to_weights(scores, "top_k", gross=1.0, top_k=3)
    assert (w.xs(first, level="decision_ts") == 0).all()
    assert list(w.attrs["skipped_periods"]) == [first]


def test_vol_scaled_needs_vol_and_downweights_volatile_names():
    scores, vol = _scores()
    with pytest.raises(ValueError):
        scores_to_weights(scores, "vol_scaled", gross=1.0)
    base = scores_to_weights(scores, "zscore", gross=1.0)
    scaled = scores_to_weights(scores, "vol_scaled", gross=1.0, vol=vol * 0 + 1.0)
    assert np.allclose(base.abs().groupby(level=0).sum(), scaled.abs().groupby(level=0).sum())
    assert np.corrcoef(base, scaled)[0, 1] > 0.99


def _beta_frame():
    syms = [f"X{i:02d}" for i in range(30)]
    bars = synthetic_bars(
        symbols=syms, start="2026-03-02", end="2026-08-28",
        factors={"market": 0.0005, "sectors": 3}, xs_alpha={"ic": 0.1, "contaminate": {"beta": 0.5}},
    )
    planted = bars.attrs["planted"]
    score = planted["xs_alpha"]["score"]
    long = score.stack()
    long.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(long.index.get_level_values(0)).tz_localize("UTC"), long.index.get_level_values(1)],
        names=["decision_ts", "symbol"],
    )
    beta = planted["beta"].reindex(score.columns)
    exposures = pd.DataFrame({"beta": long.index.get_level_values("symbol").map(beta).to_numpy()}, index=long.index)
    clean = planted["xs_alpha"]["score_clean"].stack()
    clean.index = long.index
    return long, exposures, clean


def test_neutralized_beta():
    scores, exposures, clean = _beta_frame()
    raw = scores_to_weights(scores, "top_k", gross=1.0, top_k=5)
    neutral = neutralize_weights(raw, exposures, 1.0)
    book_beta = lambda w: (w * exposures["beta"]).groupby(level="decision_ts").sum()  # noqa: E731
    assert book_beta(raw).abs().mean() > 0.05  # the contaminated score loads on beta
    assert book_beta(neutral).abs().max() < 0.05
    assert np.allclose(neutral.abs().groupby(level="decision_ts").sum(), 1.0)
    assert np.allclose(neutral.groupby(level="decision_ts").sum(), 0.0, atol=1e-12)
    assert np.corrcoef(neutral, clean)[0, 1] > 0.1  # the planted alpha survives


def test_neutralize_proportion_zero_is_identity():
    scores, exposures, _ = _beta_frame()
    raw = scores_to_weights(scores, "rank", gross=1.0)
    same = neutralize_weights(raw, exposures, 0.0)
    assert np.allclose(same, raw)


def _drifting_targets(n_periods: int = 40, n_sym: int = 10, seed: int = 5) -> list[pd.Series]:
    rng = np.random.default_rng(seed)
    syms = [f"S{i}" for i in range(n_sym)]
    w = pd.Series(rng.normal(size=n_sym), index=syms)
    out = []
    for _ in range(n_periods):
        w = 0.97 * w + 0.03 * pd.Series(rng.normal(size=n_sym), index=syms)
        out.append(w / w.abs().sum())
    return out


def _turnover(targets, band):
    current = pd.Series(0.0, index=targets[0].index)
    total = 0.0
    for t in targets:
        current, trades = rebalance(t, current, no_trade_band=band, partial_adjustment=1.0)
        total += trades.abs().sum() / 2
    return total


def test_no_trade_band_reduces_turnover():
    targets = _drifting_targets()
    assert _turnover(targets, 0.01) < _turnover(targets, 0.0)


def test_partial_adjustment():
    target = pd.Series({"A": 0.4, "B": -0.4, "C": 0.0})
    current = pd.Series({"A": 0.0, "B": -0.2, "D": 0.1})
    new, trades = rebalance(target, current, no_trade_band=0.0, partial_adjustment=0.5)
    assert np.isclose(new["A"], 0.2) and np.isclose(new["B"], -0.3) and np.isclose(new["D"], 0.05)
    assert np.allclose(trades, new - current.reindex(new.index).fillna(0.0))
    skipped, tr = rebalance(target, current, no_trade_band=0.25, partial_adjustment=1.0)
    assert skipped["A"] == pytest.approx(0.4) and tr["A"] == pytest.approx(0.4)  # gap 0.4 >= band: traded
    assert skipped["B"] == -0.2 and tr["B"] == 0.0  # gap 0.2 < band: skipped
    with pytest.raises(ValueError):
        rebalance(target, current, 0.0, 1.5)


def _bars(n: int = 12, symbols=("AAA", "BBB", "CCC")) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    ts = pd.date_range("2026-03-02 14:30", periods=n, freq="min", tz="UTC")
    frames = []
    for s in symbols:
        op = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        frames.append(pd.DataFrame({
            "symbol": s, "ts": ts, "ts_available": ts + pd.Timedelta(minutes=1),
            "open": op, "high": op * 1.001, "low": op * 0.999, "close": op * 1.0005, "volume": 1000.0,
        }))
    return pd.concat(frames, ignore_index=True)


def _weights(bars: pd.DataFrame, decisions) -> pd.Series:
    syms = sorted(bars["symbol"].unique())
    idx = pd.MultiIndex.from_product([decisions, syms], names=["decision_ts", "symbol"])
    return pd.Series(np.tile([0.5, -0.5, 0.0], len(decisions)), index=idx)


def test_portfolio_returns_hand_case_fills_at_next_open():
    bars = _bars()
    decisions = [bars["ts"].iloc[2] + pd.Timedelta(minutes=1), bars["ts"].iloc[6] + pd.Timedelta(minutes=1)]
    w = _weights(bars, decisions)  # decision at ts_available of bar 2 -> fills at the open of bar 3
    free = CostModel(half_spread_bps=0, slippage_bps=0, commission_per_share=0)
    out = portfolio_returns(w, bars, free)
    px = bars.pivot(index="ts", columns="symbol", values="open")
    expect = 0.5 * (px["AAA"].iloc[7] / px["AAA"].iloc[3] - 1) - 0.5 * (px["BBB"].iloc[7] / px["BBB"].iloc[3] - 1)
    assert len(out) == 1  # the last decision has no following open
    assert out["gross_return"].iloc[0] == pytest.approx(expect)
    assert out["cost"].iloc[0] == 0.0 and out["net_return"].iloc[0] == pytest.approx(expect)
    assert out["turnover"].iloc[0] == pytest.approx(0.5)  # building a gross-1 book from cash


def test_costs_charged_once_and_turnover_uses_drifted_weights():
    bars = _bars()
    decisions = [bars["ts"].iloc[i] + pd.Timedelta(minutes=1) for i in (1, 4, 8)]
    w = _weights(bars, decisions)  # same target every period
    model = CostModel(half_spread_bps=2, slippage_bps=5, commission_per_share=0.0)
    out = portfolio_returns(w, bars, model, equity=1_000_000.0)
    assert out["turnover"].iloc[0] == pytest.approx(0.5)
    assert 0 < out["turnover"].iloc[1] < 0.05  # only the drift is re-traded, not the whole book
    assert out["cost"].iloc[0] == pytest.approx(7e-4 * 1.0, rel=1e-6)  # 7 bps on traded gross 1.0
    assert np.allclose(out["net_return"], out["gross_return"] - out["cost"])


def test_next_open_timing():
    bars = _bars()
    decisions = [bars["ts"].iloc[i] + pd.Timedelta(minutes=1) for i in (1, 4, 7, 10)]
    w = _weights(bars, decisions)
    model = CostModel()
    base = portfolio_returns(w, bars, model)
    later = bars.copy()
    later.loc[later["ts"] > decisions[2], ["open", "high", "low", "close"]] *= 1.3  # after the third entry open
    shifted = portfolio_returns(w, later, model)
    pd.testing.assert_frame_equal(base.iloc[:2], shifted.iloc[:2])  # periods ending at or before it unchanged
    assert not np.isclose(base["gross_return"].iloc[2], shifted["gross_return"].iloc[2])
    # the decision bar's own close is not a fill price
    closed = bars.copy()
    closed.loc[closed["ts"] < decisions[1], "close"] *= 2
    pd.testing.assert_frame_equal(base, portfolio_returns(w, closed, model))


def test_missing_bar_is_reported_and_earns_nothing():
    bars = _bars()
    decisions = [bars["ts"].iloc[i] + pd.Timedelta(minutes=1) for i in (1, 5, 9)]
    w = _weights(bars, decisions)
    end = decisions[1] + pd.Timedelta(minutes=2)
    hole = (bars["symbol"] == "BBB") & (bars["ts"] >= decisions[1]) & (bars["ts"] < end)
    holes = bars[~hole]
    out = portfolio_returns(w, holes, CostModel())
    assert (out["n_missing_price"] == 0).all()  # the hole delays BBB's fill to its next bar; it is still priced
    gone = bars[~((bars["symbol"] == "BBB") & (bars["ts"] >= decisions[1]))]
    out2 = portfolio_returns(w, gone, CostModel())
    assert out2["n_missing_price"].iloc[0] == 1  # BBB has no bar at the second fill
    assert np.isfinite(out[["gross_return", "cost", "net_return", "turnover"]].to_numpy()).all()


def test_module_exports():
    assert set(construct.__all__) >= {"scores_to_weights", "neutralize_weights", "rebalance", "portfolio_returns"}
