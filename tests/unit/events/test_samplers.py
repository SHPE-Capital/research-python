"""Decision samplers (C1b): timing, hand cases, the planted breakout fixture and the candidates sampler."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from shpe_research.contracts.base import ContractError
from shpe_research.contracts.candidates import validate_candidates
from shpe_research.data.calendar import get_calendar, session_date
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.events import breakouts, samplers, thresholds  # noqa: F401  (importing registers the samplers)
from shpe_research.pipeline.registry import SAMPLER_REGISTRY, build_sampler
from shpe_research.strategies.registry import STRATEGY_REGISTRY
from tests.conftest import make_bars

SYMBOLS = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def small_bars() -> pd.DataFrame:
    bars = synthetic_bars(SYMBOLS, start="2026-03-02", end="2026-03-13")
    by_symbol = bars.groupby("symbol")["close"]
    return bars.assign(ret=by_symbol.pct_change())


@pytest.fixture(scope="module")
def breakout_bars() -> pd.DataFrame:
    return synthetic_bars(
        SYMBOLS, momentum={"breakout_prob": 0.5, "range_minutes": 15, "max_delay_minutes": 60, "drift_bps": 30}
    )


KEYS = {
    "every_n_minutes": {"n": 15},
    "session_open": {"offset_minutes": 5},
    "session_close": {"offset_minutes": 1},
    "threshold": {"column": "ret", "upper": 0.001, "lower": -0.001},
    "cusum": {"column": "ret", "h": 0.004},
    "range_breakout": {"range_minutes": 15},
}


def sample(key: str, bars: pd.DataFrame, **params) -> pd.DataFrame:
    return build_sampler({"type": key, **params}).sample(bars)


def test_all_keys_registered():
    assert {*KEYS, "candidates"} <= set(SAMPLER_REGISTRY)


@pytest.mark.parametrize("key", KEYS)
def test_decision_is_bar_ts_available(key, small_bars):
    out = sample(key, small_bars, **KEYS[key])
    assert len(out) > 0
    validate_candidates(out)
    known = small_bars.groupby("symbol")["ts_available"].apply(set)
    assert all(ts in known[sym] for sym, ts in zip(out["symbol"], out["decision_ts"], strict=True))
    assert set(out["side"]) == {"long"}
    assert out["strategy_id"].str.startswith(f"{key}:").all()
    assert out.index.equals(pd.RangeIndex(len(out)))
    order = out.sort_values(["decision_ts", "symbol"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(out, order)


def test_every_n_minutes_legacy_alias():
    assert build_sampler({"type": "every_n_minutes", "minutes": 15}).n == 15
    assert build_sampler({"type": "every_n_minutes", "n": 15}).minutes == 15


def test_every_n_minutes_and_session_open(small_bars):
    cal = get_calendar()
    every = sample("every_n_minutes", small_bars, n=30)
    assert (every["decision_ts"].dt.minute.isin([0, 30])).all()
    opened = sample("session_open", small_bars, offset_minutes=0)
    first = small_bars.groupby(["symbol", session_date(small_bars["ts"])])["ts_available"].min()
    assert len(opened) == len(first)
    days = session_date(opened["decision_ts"])
    opens = cal.schedule(days.min(), days.max())["open"].reindex(days.to_numpy())
    assert ((opened["decision_ts"].to_numpy() - opens.to_numpy()) == pd.Timedelta(minutes=1)).all()


def test_session_skips_missing_bars_never_fills(small_bars):
    dropped = small_bars[small_bars["ts"].dt.minute != 0]  # no bar at any :00 start
    out = sample("session_open", dropped, offset_minutes=0)
    assert set(out["decision_ts"]).issubset(set(dropped["ts_available"]))
    known = dropped.groupby("symbol")["ts_available"].apply(set)
    assert all(ts in known[s] for s, ts in zip(out["symbol"], out["decision_ts"], strict=True))


def test_session_close_daily_and_minute(small_bars):
    cal = get_calendar()
    sessions = small_bars.groupby("symbol")["ts"].apply(lambda s: session_date(s).nunique())
    out = sample("session_close", small_bars, offset_minutes=0)
    assert out.groupby("symbol").size().to_dict() == sessions.to_dict()
    days = session_date(out["decision_ts"] - pd.Timedelta(minutes=1))
    closes = cal.schedule(days.min(), days.max())["close"].reindex(days.to_numpy())
    assert (out["decision_ts"].to_numpy() == closes.to_numpy()).all()
    default = sample("session_close", small_bars)
    assert (default["decision_ts"].to_numpy() == out["decision_ts"].to_numpy() - pd.Timedelta(minutes=1)).all()

    daily = synthetic_bars(daily={"years": 3, "n_symbols": 50})
    rows = sample("session_close", daily)
    assert len(rows) == len(daily)
    merged = daily.merge(rows, left_on=["symbol", "ts_available"], right_on=["symbol", "decision_ts"], how="outer")
    assert len(merged) == len(daily)
    assert merged["decision_ts"].notna().all()
    assert (merged["decision_ts"] == merged["ts_available"]).all()


def test_range_breakout_planted(breakout_bars):
    planted = breakout_bars.attrs["planted"]["momentum"]["events"]
    assert len(planted) > 10
    out = sample("range_breakout", breakout_bars, range_minutes=15)
    validate_candidates(out)
    ny = out["decision_ts"].dt.tz_convert("America/New_York")
    assert (ny.dt.hour * 60 + ny.dt.minute >= 9 * 60 + 45).all()
    assert not out.assign(day=session_date(out["decision_ts"])).duplicated(["symbol", "day"]).any()
    want = planted.assign(decision_ts=planted["ts"] + pd.Timedelta(minutes=1))[["symbol", "decision_ts"]]
    found = want.merge(out[["symbol", "decision_ts"]], how="left", indicator=True)
    assert (found["_merge"] == "both").all()


def test_range_breakout_options(breakout_bars):
    base = sample("range_breakout", breakout_bars, range_minutes=15)
    early = sample("range_breakout", breakout_bars, range_minutes=15, entry_cutoff=25)
    ny = early["decision_ts"].dt.tz_convert("America/New_York")
    assert len(early) < len(base)
    assert (ny.dt.hour * 60 + ny.dt.minute <= 9 * 60 + 30 + 25).all()
    touch = sample("range_breakout", breakout_bars, range_minutes=15, confirm="touch")
    assert len(touch) >= len(base)
    short = sample("range_breakout", breakout_bars, range_minutes=15, side="short")
    assert set(short["side"]) == {"short"}
    with pytest.raises(ValueError, match="confirm"):
        sample("range_breakout", breakout_bars, confirm="never")


@pytest.mark.parametrize("key", ["range_breakout", "cusum", "threshold", "session_close", "every_n_minutes"])
def test_range_breakout_no_lookahead(key, breakout_bars):
    bars = breakout_bars.assign(ret=breakout_bars.groupby("symbol")["close"].pct_change())
    params = KEYS[key]
    full = sample(key, bars, **params)
    for cut in full["decision_ts"].quantile([0.2, 0.5, 0.8], interpolation="nearest"):
        truncated = sample(key, bars[bars["ts_available"] <= cut], **params)
        expected = full[full["decision_ts"] <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(truncated, expected)


def test_cusum_hand_case():
    bars = make_bars(n=9)
    bars["x"] = [0.0, 0.4, 0.4, 0.4, -0.5, -0.6, 0.1, 0.9, 0.2]
    # up-sum 0.4, 0.8, 1.2 > 1 at index 3 -> reset; then -0.5, -1.1 < -1 at index 5 -> reset; then 0.1, 1.0 (not > 1),
    # 1.2 > 1 at index 8
    out = sample("cusum", bars, column="x", h=1.0)
    assert out["decision_ts"].tolist() == bars.loc[[3, 5, 8], "ts_available"].tolist()
    assert out["trigger_value"].round(6).tolist() == [1.2, -1.1, 1.2]


def test_threshold_hand_case():
    bars = make_bars(n=7)
    bars["z"] = [0.0, 1.0, 2.0, 1.0, 3.0, -2.0, -3.0]
    out = sample("threshold", bars, column="z", upper=1.5, lower=-1.5)
    assert out["decision_ts"].tolist() == bars.loc[[2, 4, 5], "ts_available"].tolist()
    only_upper = sample("threshold", bars, column="z", upper=1.5)
    assert only_upper["decision_ts"].tolist() == bars.loc[[2, 4], "ts_available"].tolist()


def test_threshold_ignores_missing_values():
    bars = make_bars(n=5)
    bars["z"] = [0.0, float("nan"), 2.0, 3.0, 0.0]
    out = sample("threshold", bars, column="z", upper=1.5)
    assert out["decision_ts"].tolist() == bars.loc[[2], "ts_available"].tolist()


def test_ignores_extended_hours():
    bars = make_bars(n=10)
    bars.loc[:4, "session"] = "pre"
    out = sample("every_n_minutes", bars, n=1)
    assert out["decision_ts"].isin(bars.loc[5:, "ts_available"]).all()
    assert not out["decision_ts"].isin(bars.loc[:4, "ts_available"]).any()


class _Dummy:
    strategy_type = "dummy_c1b"

    def __init__(self, **params):
        self.params = params

    def generate(self, bars, context=None):
        first = bars.groupby("symbol", as_index=False).first()
        return pd.DataFrame(
            {
                "symbol": first["symbol"],
                "decision_ts": first["ts_available"],
                "side": "short",
                "strategy_id": f"dummy_c1b:{self.params.get('tag', 'x')}",
            }
        )


class _Broken(_Dummy):
    strategy_type = "broken_c1b"

    def generate(self, bars, context=None):
        return super().generate(bars, context).assign(side="sideways")


def test_candidates_sampler_dispatch(small_bars, tmp_path, monkeypatch):
    monkeypatch.setitem(STRATEGY_REGISTRY, "dummy_c1b", _Dummy)
    monkeypatch.setitem(STRATEGY_REGISTRY, "broken_c1b", _Broken)
    path = tmp_path / "dummy.yaml"
    path.write_text(yaml.safe_dump({"type": "dummy_c1b", "tag": "t1"}))
    out = sample("candidates", small_bars, strategy=str(path))
    assert sorted(out["symbol"]) == SYMBOLS
    assert set(out["side"]) == {"short"}
    assert set(out["strategy_id"]) == {"dummy_c1b:t1"}
    assert out.index.equals(pd.RangeIndex(len(out)))
    inline = sample("candidates", small_bars, strategy={"type": "dummy_c1b", "tag": "t2"})
    assert set(inline["strategy_id"]) == {"dummy_c1b:t2"}
    with pytest.raises(ContractError):
        sample("candidates", small_bars, strategy={"type": "broken_c1b"})
    with pytest.raises(ValueError, match="unknown strategy"):
        sample("candidates", small_bars, strategy={"type": "nope_c1b"})
