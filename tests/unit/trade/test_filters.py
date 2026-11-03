"""T3 acceptance tests: entry filters."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from shpe_research.trade import filters as filters_mod
from shpe_research.trade.filters import FILTER_REGISTRY, apply_filters, build_filter


def _cands(times, **cols):
    ts = pd.to_datetime(times).tz_localize("America/New_York").tz_convert("UTC")
    df = pd.DataFrame({"symbol": "AAA", "decision_ts": ts, "side": "long", "strategy_id": "t:1"})
    for k, v in cols.items():
        df[k] = v
    return df


def test_time_window():
    c = _cands(["2026-01-05 09:35", "2026-01-05 15:55"])
    keep, reason = apply_filters(c, c, [{"filter_type": "time_window", "start": "09:30", "end": "15:30"}])
    assert keep.tolist() == [True, False]
    assert reason.tolist() == ["", "time_window"]


def test_missing_column_raises():
    c = _cands(["2026-01-05 09:35"])
    with pytest.raises(KeyError, match="rvol_30"):
        apply_filters(c, c, [{"filter_type": "min_rvol", "column": "rvol_30", "min": 1.5}])
    with pytest.raises(KeyError, match="min_rvol"):
        apply_filters(c, c, [{"filter_type": "min_rvol", "column": "rvol_30", "min": 1.5}])


def test_first_reason_wins():
    c = _cands(["2026-01-05 09:35", "2026-01-05 10:00"], rvol=[0.5, 2.0], adv=[1.0, 1e9])
    specs = [
        {"filter_type": "min_rvol", "column": "rvol", "min": 1.0},
        {"filter_type": "min_liquidity", "column": "adv", "min": 1e6},
    ]
    keep, reason = apply_filters(c, c, specs)
    assert keep.tolist() == [False, True]
    assert reason.iloc[0] == "min_rvol"
    assert reason.iloc[1] == ""
    assert c.shape[1] == 6  # inputs untouched
    keep, reason, all_reasons = apply_filters(c, c, specs, return_all=True)
    assert all_reasons.iloc[0] == ["min_rvol", "min_liquidity"]
    assert all_reasons.iloc[1] == []


def test_blackout_known_dates_only():
    events = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-07", "2026-01-08"]),
            "known_ts": pd.to_datetime(["2025-12-01 00:00", "2026-01-06 12:00"]).tz_localize("UTC"),
        }
    )
    c = _cands(["2026-01-05 10:00", "2026-01-07 10:00", "2026-01-08 10:00", "2026-01-12 10:00"])
    spec = {"filter_type": "blackout", "calendar": events, "sessions_before": 0, "sessions_after": 0}
    keep, reason = apply_filters(c, c, [spec])
    # 01-07 known in advance -> blocked; 01-08 was announced after its own decision time of 01-05 only
    # but is known by 01-08 10:00 itself -> blocked there.
    assert keep.tolist() == [True, False, False, True]
    # A row decided on 01-05 must not be blocked by the 01-08 event even with a wide window.
    wide = {"filter_type": "blackout", "calendar": events, "sessions_before": 3, "sessions_after": 0}
    keep, _ = apply_filters(c.iloc[[0]], c.iloc[[0]], [wide])
    assert keep.tolist() == [False]  # blocked only by the 01-07 event, which was known
    only_late = events.iloc[[1]]
    keep, _ = apply_filters(c.iloc[[0]], c.iloc[[0]], [{**wide, "calendar": only_late}])
    assert keep.tolist() == [True]


def test_blackout_sessions_before_after_and_symbol():
    events = pd.DataFrame({"event_date": pd.to_datetime(["2026-01-07"]), "symbol": ["BBB"]})
    c = _cands(["2026-01-06 10:00", "2026-01-08 10:00", "2026-01-09 10:00"])
    c2 = pd.concat([c, c.assign(symbol="BBB")], ignore_index=True)
    spec = {"filter_type": "blackout", "calendar": events, "sessions_before": 1, "sessions_after": 1}
    keep, _ = apply_filters(c2, c2, [spec])
    assert keep.tolist() == [True, True, True, False, False, True]


def test_blackout_calendar_from_path(tmp_path):
    p = tmp_path / "fomc.csv"
    pd.DataFrame({"event_date": ["2026-01-07"]}).to_csv(p, index=False)
    c = _cands(["2026-01-07 10:00"])
    keep, _ = apply_filters(c, c, [{"filter_type": "blackout", "calendar": str(p)}])
    assert keep.tolist() == [False]


def test_unknown_named_calendar_raises():
    c = _cands(["2026-01-07 10:00"])
    with pytest.raises(ValueError, match="earnings"):
        apply_filters(c, c, [{"filter_type": "blackout", "calendar": "earnings"}])


def test_band_filters_and_nan():
    c = _cands(["2026-01-05 10:00"] * 4, spread=[1.0, 9.0, float("nan"), 5.0], reg=[0.1, 0.5, 0.9, float("nan")],
               p=[0.9, 0.4, 0.5, float("nan")])
    keep, reason = apply_filters(c, c, [{"filter_type": "max_spread", "column": "spread", "max_bps": 5.0}])
    assert keep.tolist() == [True, False, False, True]
    keep, _ = apply_filters(c, c, [{"filter_type": "regime", "column": "reg", "min": 0.2, "max": 0.8}])
    assert keep.tolist() == [False, True, False, False]
    keep, _ = apply_filters(c, c, [{"filter_type": "meta_score", "column": "p", "min_probability": 0.5}])
    assert keep.tolist() == [True, False, True, False]


def test_columns_come_from_frame_and_index_aligned():
    c = _cands(["2026-01-05 10:00", "2026-01-05 10:01"])
    c.index = [10, 20]
    frame = c.assign(rvol=[3.0, 0.1])
    keep, reason = apply_filters(c, frame, [{"filter_type": "min_rvol", "column": "rvol", "min": 1.0}])
    assert list(keep.index) == [10, 20]
    assert keep.tolist() == [True, False]


def test_empty_filters_and_unknown_type():
    c = _cands(["2026-01-05 10:00"])
    keep, reason = apply_filters(c, c, [])
    assert keep.tolist() == [True] and reason.tolist() == [""]
    with pytest.raises(ValueError, match="nope"):
        build_filter({"filter_type": "nope"})


def test_registry_has_all_keys():
    assert {"time_window", "min_liquidity", "max_spread", "regime", "blackout", "min_rvol", "meta_score"} <= set(
        FILTER_REGISTRY
    )


def test_no_feature_computation():
    tree = ast.parse(Path(filters_mod.__file__).read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    assert not [m for m in mods if m.startswith("shpe_research.features")]


def test_strategy_spec_filters_build_through_registry():
    from shpe_research.config.strategy import StrategySpec

    spec = StrategySpec(
        strategy_type="orb",
        filters=[{"filter_type": "time_window", "start": "09:35", "end": "11:00"},
                 {"filter_type": "min_rvol", "column": "rvol", "min": 1.5}],
    )
    built = [build_filter(f) for f in spec.filters]
    assert [type(f).__name__ for f in built] == ["TimeWindow", "MinRvol"]
