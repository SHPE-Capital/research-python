"""Tests for the D7 0-A artifact builder and the atomic run-directory writer."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from shpe_research.calibration.acceptance import Finalist, StageMetrics, select_finalist, validation_acceptance
from shpe_research.calibration.artifact import (
    account_settings,
    build_artifact,
    decision_cutoff_ts,
    peak_rss_mb,
    write_run_dir,
)
from shpe_research.calibration.universe import StaticUniverseProvider
from shpe_research.calibration.window import Archive, iso_monday, plan_window
from shpe_research.config.momentum import ResolvedProtocol, load_protocol
from shpe_research.contracts.weekly_config import from_json, to_json
from shpe_research.strategies.momentum.breakout.grid import Config, config_id
from shpe_research.trade.account_types import AccountConfig

AS_OF = date(2024, 5, 31)
RUN_FILES = {
    "weekly_config.json",
    "audit.jsonl",
    "finalists.json",
    "protocol.resolved.yaml",
    "overrides.json",
    "run.json",
}
CFG_A = Config.make("v1", ["breakout", "rvol"], {"breakout.n": 10, "breakout.b": 0, "rvol.w": 3, "rvol.v": 1})
CFG_B = Config.make("v1", ["breakout", "vwap"], {"breakout.n": 20, "breakout.b": 0, "vwap.e": 3})


def _metrics(trades: int, j: float, dd: float = 0.01, invalid: bool = False) -> StageMetrics:
    return StageMetrics(
        trades=trades,
        active_sessions=trades,
        j=math.nan if invalid else j,
        max_drawdown=math.nan if invalid else dd,
        max_single_stock_share=0.2,
        invalid_accounting=invalid,
        unresolved=0,
    )


def _finalist(cfg: Config, rp: ResolvedProtocol, val: StageMetrics, rep: StageMetrics) -> Finalist:
    return Finalist(cfg, config_id(cfg), 1, 0.001, val, rep, validation_acceptance(val, rp.protocol.acceptance))


@pytest.fixture(scope="module")
def setup(synth_archive_root: Path, small_protocol: ResolvedProtocol):
    archive = Archive.open(synth_archive_root)
    p = small_protocol.protocol
    plan = plan_window(
        archive,
        AS_OF,
        validation_sessions=p.validation.sessions,
        min_prior_sessions=max(p.cache.warmup_sessions, p.weights.descriptor_lookback + 1),
        granularity=p.cache.granularity,
    )
    universe = StaticUniverseProvider.from_archive(archive).week(iso_monday(plan.activation_sessions[0]))
    return plan, universe


def _build(rp, plan, universe, selection, search_reasons=()):
    return build_artifact(
        run_id="breakout-uniform-test",
        rp=rp,
        data_version="synth-7",
        policy="uniform",
        plan=plan,
        universe=universe,
        selection=selection,
        search_reasons=search_reasons,
        search_ess=12.5,
        logical=48,
        actual=30,
        seeds=[1, 2],
        live_account=AccountConfig(cost_bps=rp.protocol.acceptance.selection_bps),
        runtime_s=1.5,
        peak_mb=200.0,
    )


def test_account_settings_mirror_dataclass() -> None:
    assert account_settings(AccountConfig()).model_dump() == dataclasses.asdict(AccountConfig())
    odd = AccountConfig(cost_bps=20.0, search_replay_reference_equity=5.0, entry_score_threshold=0.3)
    assert account_settings(odd).model_dump() == dataclasses.asdict(odd)


def test_decision_cutoff_is_session_close_utc() -> None:
    ts = decision_cutoff_ts(date(2024, 5, 31))
    assert ts == datetime(2024, 5, 31, 20, 0, tzinfo=UTC)  # 16:00 New York (EDT)
    assert ts.utcoffset() is not None


def test_build_selected_and_no_deployment_round_trip(setup, small_protocol) -> None:
    plan, universe = setup
    rp = small_protocol
    good = _finalist(CFG_A, rp, _metrics(5, 0.002), _metrics(5, 0.001))
    bad = _finalist(CFG_B, rp, _metrics(1, -0.001), _metrics(1, -0.002))
    selected = _build(rp, plan, universe, select_finalist([good, bad], rp.protocol.acceptance))
    assert from_json(to_json(selected)) == selected
    assert selected.selected is not None and selected.selected.config_id == config_id(CFG_A)
    assert selected.selected.params == CFG_A.as_dict()
    assert selected.acceptance.passed
    names = [c.name for c in selected.acceptance.checks]
    report = f"report_j_at_{int(rp.protocol.acceptance.report_bps)}bps"
    assert names[-1] == report and names[:-1] == [c.name for c in good.verdict.checks]
    assert selected.acceptance.checks[-1].value == pytest.approx(0.001)
    assert selected.acceptance.checks[-1].threshold is None and selected.acceptance.checks[-1].passed
    assert selected.protocol_hash == rp.hash
    assert selected.decision_cutoff == decision_cutoff_ts(plan.cutoff)
    assert (selected.search_range.start, selected.search_range.end) == (
        plan.search_sessions[0],
        plan.search_sessions[-1],
    )
    assert [(u.asset_id, u.symbol) for u in selected.universe] == list(universe.members)
    assert selected.account.cost_bps == rp.protocol.acceptance.selection_bps
    assert len(selected.code_commit) >= 7

    broken = _finalist(CFG_A, rp, _metrics(5, 0.0, invalid=True), _metrics(5, 0.0, invalid=True))
    nodep = _build(
        rp,
        plan,
        universe,
        select_finalist([bad, broken], rp.protocol.acceptance),
        search_reasons=("search:search_min_trades x3",),
    )
    assert from_json(to_json(nodep)) == nodep
    assert nodep.selected is None and not nodep.acceptance.passed
    assert nodep.no_deployment.reason.endswith("; search:search_min_trades x3")
    assert "validation_min_trades x1" in nodep.no_deployment.reason
    assert "valid_accounting x1" in nodep.no_deployment.reason
    assert [c.name for c in nodep.acceptance.checks] == [c.name for c in bad.verdict.checks]  # highest validation J

    empty = _build(rp, plan, universe, select_finalist([], rp.protocol.acceptance), ("search:search_min_trades x48",))
    assert from_json(to_json(empty)) == empty
    assert empty.no_deployment.reason == "no_feasible_finalists; search:search_min_trades x48"
    [check] = empty.acceptance.checks
    assert (check.name, check.passed, check.value, check.threshold) == ("no_feasible_finalists", False, 0.0, 1.0)


def test_run_dir_atomic_files(setup, small_protocol, tmp_path) -> None:
    plan, universe = setup
    rp = small_protocol
    good = _finalist(CFG_A, rp, _metrics(5, 0.002), _metrics(5, 0.001))
    selection = select_finalist([good], rp.protocol.acceptance)
    art = _build(rp, plan, universe, selection)
    out = tmp_path / "runs" / art.run_id
    for _ in range(2):  # the second write replaces the first
        path = write_run_dir(out, art, audit=['{"kind": "status"}'], finalists=selection, rp=rp, extra={"x": 1})
    assert path == out
    assert {p.name for p in out.iterdir()} == RUN_FILES
    assert [p.name for p in out.parent.iterdir()] == [out.name]  # no .tmp sibling left
    assert from_json((out / "weekly_config.json").read_text(encoding="utf-8")) == art
    assert json.loads((out / "run.json").read_text(encoding="utf-8")) == {"x": 1}
    assert (out / "audit.jsonl").read_text(encoding="utf-8") == '{"kind": "status"}\n'
    assert load_protocol(out / "protocol.resolved.yaml").hash == rp.hash
    assert json.loads((out / "overrides.json").read_text(encoding="utf-8"))["overrides"] == list(rp.overrides)
    fin = json.loads((out / "finalists.json").read_text(encoding="utf-8"))
    assert fin["status"] == "selected" and fin["selected"] == config_id(CFG_A)
    assert [c["name"] for c in fin["finalists"][0]["checks"]] == [c.name for c in good.verdict.checks]

    write_run_dir(out, None, audit=[], finalists=selection, rp=rp, extra={"status": "incomplete_time_limit"})
    assert {p.name for p in out.iterdir()} == RUN_FILES - {"weekly_config.json"}


def test_peak_rss_positive() -> None:
    assert peak_rss_mb() > 0
