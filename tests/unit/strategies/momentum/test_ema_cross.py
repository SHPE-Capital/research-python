"""SM2 acceptance tests: EMA crossover baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from shpe_research.config.strategy import StrategySpec
from shpe_research.contracts.base import to_utc_ns
from shpe_research.events.samplers import Candidates
from shpe_research.strategies.momentum.ema_cross import EmaCross, ema
from shpe_research.strategies.registry import STRATEGY_REGISTRY, build_strategy
from shpe_research.testing.checks import check_strategy
from shpe_research.testing.fixtures import check_bars

CONFIG = Path(__file__).parents[4] / "configs" / "strategies" / "momentum" / "ema_cross.yaml"


def _bars(closes, start="2026-03-02T14:30:00Z", symbol="AAA") -> pd.DataFrame:
    """Minute bars whose closes are `closes` (regular session; 09:30 New York is 14:30 UTC in early March)."""
    n = len(closes)
    ts = to_utc_ns(pd.date_range(start, periods=n, freq="1min"))
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": ts,
            "ts_available": ts + pd.Timedelta(minutes=1),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1000,
            "session": "regular",
        }
    )


def _zigzag() -> list[float]:
    return [10, 10, 10, 10, 11, 12, 13, 12, 10, 8, 7, 8, 10, 12, 13, 12, 10, 8]


def test_passes_check_strategy():
    check_strategy(EmaCross(fast=3, slow=6, direction="both"), check_bars())
    check_strategy(EmaCross(), check_bars())
    check_strategy(EmaCross(fast=3, slow=6, carry=True), check_bars())


def test_ema_seed_matches_platform():
    closes = np.array([3.0, 5.0, 4.0, 8.0, 9.0, 7.0])
    out = ema(closes, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(closes[:3].mean())  # SMA of the first n closes
    k = 2 / (3 + 1)
    assert out[3] == pytest.approx(closes[3] * k + out[2] * (1 - k))
    assert np.isnan(ema(closes, 10)).all()  # fewer closes than the period: never seeded


def test_cross_on_completed_bars():
    bars = _bars(_zigzag())
    gen = EmaCross(fast=2, slow=4, direction="both")
    full = gen.generate(bars)
    assert len(full) > 2
    for k in range(5, len(bars)):
        cut = bars.iloc[:k]
        part = gen.generate(cut)
        cutoff = cut["ts_available"].max()
        want = full[full["decision_ts"] <= cutoff].reset_index(drop=True)
        got = part[part["decision_ts"] <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(want, got)
    # decision_ts is the ts_available of the bar whose close completed the cross
    assert set(full["decision_ts"]) <= set(bars["ts_available"])


def test_entry_exit_alternate():
    bars = pd.concat([_bars(_zigzag(), "2026-03-02T14:30:00Z"), _bars(_zigzag(), "2026-03-03T14:30:00Z")])
    for carry in (False, True):
        for direction in ("long", "both"):
            out = EmaCross(fast=2, slow=4, direction=direction, carry=carry).generate(bars)
            assert not out.empty
            out = out.assign(_exit=(out["action"] != "exit").astype(int), _day=out["decision_ts"].dt.date)
            keys = ["symbol"] if carry else ["symbol", "_day"]
            for _, g in out.groupby(keys):
                g = g.sort_values(["decision_ts", "_exit"], kind="stable")  # an exit precedes a flip entry
                seq = list(g["action"])
                assert seq[0] == "enter"
                assert all(a != b for a, b in zip(seq, seq[1:], strict=False)), seq
            exits = out[out["action"] == "exit"]
            assert (exits["exit_reason"] == "exit").all()


def test_rule_branches_by_hand():
    # Flat, then up: fast crosses above slow once (golden), then back below (death).
    bars = _bars([10, 10, 10, 10, 12, 14, 14, 9, 6, 6])
    long_only = EmaCross(fast=2, slow=4, direction="long").generate(bars)
    assert list(long_only["action"]) == ["enter", "exit"] and list(long_only["side"]) == ["long", "long"]
    both = EmaCross(fast=2, slow=4, direction="both").generate(bars)
    assert list(zip(both["action"], both["side"], strict=True)) == [
        ("enter", "long"), ("exit", "long"), ("enter", "short"),
    ]
    assert both["decision_ts"].iloc[1] == both["decision_ts"].iloc[2]  # the flip shares one decision time
    # Entry fires on the bar of the cross: closes 10,10,10,10,12 -> index 4, known at that bar's ts_available.
    assert long_only["decision_ts"].iloc[0] == bars["ts_available"].iloc[4]


def test_no_candidates_before_slow_bars_and_session_boundary():
    # Day 1 ends high (fast above slow); day 2 starts low. With carry=False the EMA restarts each session.
    day1 = _bars([10, 10, 10, 10, 12, 14, 16, 18], "2026-03-02T14:30:00Z")
    day2 = _bars([20, 20, 20, 20, 20, 20, 20, 20], "2026-03-03T14:30:00Z")
    bars = pd.concat([day1, day2])
    out = EmaCross(fast=2, slow=4).generate(bars)
    assert (out["decision_ts"] <= day1["ts_available"].max()).all()  # no spurious cross at the day boundary
    first_ok = day1["ts_available"].iloc[4]  # the slow EMA seeds at bar index 3; first possible cross is index 4
    assert out["decision_ts"].min() >= first_ok
    carried = EmaCross(fast=2, slow=4, carry=True).generate(bars)
    assert carried["decision_ts"].min() >= first_ok


def test_missing_bar_is_skipped_not_filled():
    bars = _bars(_zigzag())
    gapped = bars.drop(index=[6]).reset_index(drop=True)
    out = EmaCross(fast=2, slow=4, direction="both").generate(gapped)
    assert set(out["decision_ts"]) <= set(gapped["ts_available"])
    check_strategy(EmaCross(fast=2, slow=4, direction="both"), gapped)


def test_symbols_are_independent_and_extended_hours_ignored():
    a, b = _bars(_zigzag(), symbol="AAA"), _bars(_zigzag(), symbol="BBB")
    pre = _bars([1, 2, 3, 4, 5, 6], "2026-03-02T12:00:00Z", "AAA").assign(session="pre")
    both = EmaCross(fast=2, slow=4).generate(pd.concat([a, b, pre]))
    only_a = EmaCross(fast=2, slow=4).generate(a)
    pd.testing.assert_frame_equal(
        both[both["symbol"] == "AAA"].reset_index(drop=True), only_a.reset_index(drop=True)
    )
    assert set(both["symbol"]) == {"AAA", "BBB"}


def test_parameters_defaults_and_validation():
    gen = EmaCross()
    assert (gen.fast, gen.slow, gen.direction, gen.carry) == (9, 21, "long", False)
    with pytest.raises(ValueError):
        EmaCross(fast=21, slow=9)
    with pytest.raises(ValueError):
        EmaCross(direction="short")


def test_registered_and_deterministic():
    assert STRATEGY_REGISTRY["ema_cross"] is EmaCross
    built = build_strategy({"type": "ema_cross", "fast": 3, "slow": 6})
    bars = check_bars()
    a, b = built.generate(bars), built.generate(bars)
    pd.testing.assert_frame_equal(a, b)
    assert a["strategy_id"].str.startswith("ema_cross:").all()
    other = EmaCross(fast=4, slow=8).generate(bars)
    assert set(other["strategy_id"]).isdisjoint(a["strategy_id"])


def test_config_runs_on_fixture():
    spec = StrategySpec(**yaml.safe_load(CONFIG.read_text()))
    assert spec.strategy_type == "ema_cross"
    out = Candidates(strategy=str(CONFIG)).sample(check_bars(symbols=("AAA", "BBB")))
    assert not out.empty and set(out["symbol"]) <= {"AAA", "BBB"}
