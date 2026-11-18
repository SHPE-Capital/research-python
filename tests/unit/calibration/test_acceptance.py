"""Tests for D5 acceptance floors and finalist selection."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from shpe_research.calibration.acceptance import (
    Finalist,
    StageMetrics,
    dedupe_finalists,
    search_eligibility,
    select_finalist,
    stage_metrics,
    validation_acceptance,
)
from shpe_research.config.momentum import AcceptanceSection, load_protocol
from shpe_research.strategies.momentum.breakout.grid import Config, config_id

CFG = AcceptanceSection()


def _m(**kw) -> StageMetrics:
    base = dict(
        trades=200,
        active_sessions=100,
        j=0.01,
        max_drawdown=0.05,
        max_single_stock_share=0.2,
        invalid_accounting=False,
        unresolved=0,
    )
    base.update(kw)
    return StageMetrics(**base)


def _cfg(n: int, subset=("breakout",)) -> Config:
    return Config.make("v", subset, {"breakout.n": n})


def _fin(n, *, vj=0.01, rj=0.0, sj=0.0, subset=("breakout",), **mkw) -> Finalist:
    c = _cfg(n, subset)
    v = _m(j=vj, **mkw)
    return Finalist(c, config_id(c), 0, sj, v, _m(j=rj), validation_acceptance(v, CFG))


def test_stage_metrics_hand_worked():
    m = stage_metrics(
        np.array([0.05, -0.08, 0.02]),
        [(0, "A", False), (0, "B", False), (2, "A", False)],
        np.full(3, 1 / 3),
        downside_coeff=1.0,
    )
    assert m.trades == 3
    assert m.active_sessions == 2
    assert m.max_drawdown == pytest.approx(0.08)
    assert m.max_single_stock_share == pytest.approx(2 / 3)
    assert m.invalid_accounting is False
    assert m.unresolved == 0


def test_stage_metrics_no_trades_and_invalid():
    m = stage_metrics(np.array([0.0, np.nan]), [], np.full(2, 0.5), downside_coeff=1.0)
    assert m.trades == 0 and m.max_single_stock_share == 0.0
    assert m.invalid_accounting is True
    assert stage_metrics(np.array([0.0, -1.0]), [], None, downside_coeff=1.0).invalid_accounting


def test_drawdown_fail_and_pass():
    e = [(0, "A", False)]
    m = stage_metrics(np.array([0.05, -0.12, 0.02]), e, None, downside_coeff=1.0)
    assert m.max_drawdown == pytest.approx(0.12)
    assert "max_drawdown" in validation_acceptance(m, CFG).reasons
    ok = validation_acceptance(_m(max_drawdown=0.10), CFG)
    assert ok.passed


def test_search_floor_pass_fail():
    assert search_eligibility(_m(trades=100, active_sessions=50), CFG).passed
    assert search_eligibility(_m(trades=99, active_sessions=50), CFG).reasons == ("search_min_trades",)
    assert search_eligibility(_m(trades=100, active_sessions=49), CFG).reasons == ("search_min_sessions",)


def test_validation_floor_pass_fail():
    assert validation_acceptance(_m(trades=30, active_sessions=20, j=0.001), CFG).passed
    assert validation_acceptance(_m(trades=30, active_sessions=20, j=0.0), CFG).reasons == ("validation_min_j",)
    assert validation_acceptance(_m(trades=29, active_sessions=20), CFG).reasons == ("validation_min_trades",)
    assert validation_acceptance(_m(trades=30, active_sessions=19), CFG).reasons == ("validation_min_sessions",)


@pytest.mark.parametrize("stage", [search_eligibility, validation_acceptance])
def test_single_stock_share(stage):
    assert stage(_m(max_single_stock_share=0.30), CFG).passed
    assert stage(_m(max_single_stock_share=0.31), CFG).reasons == ("max_single_stock_share",)


@pytest.mark.parametrize("stage", [search_eligibility, validation_acceptance])
def test_invalid_accounting_and_unresolved_always_reject(stage):
    assert stage(_m(invalid_accounting=True), CFG).reasons == ("valid_accounting",)
    v = stage(_m(unresolved=1), CFG)
    assert v.reasons == ("no_unresolved_data_errors",)
    chk = next(c for c in v.checks if c.name == "no_unresolved_data_errors")
    assert (chk.value, chk.threshold) == (1, 0)
    assert stage(_m(), CFG).passed


def test_dedupe_across_seeds_and_cap():
    cfgs = [_cfg(i) for i in range(30)]
    s0 = [(cfgs[i], 0, float(i)) for i in range(12)]
    s1 = [(cfgs[i], 1, 99.0) for i in range(6)] + [(cfgs[i], 1, float(i)) for i in range(12, 18)]
    s2 = [(cfgs[i], 2, float(i)) for i in range(18, 30)]
    out = dedupe_finalists([s0, s1, s2], 100)
    assert len(out) == 30
    assert out[0] == (cfgs[0], 0, 0.0)  # first occurrence kept
    capped = dedupe_finalists([s0, s1, s2], 10)
    assert [c for c, _, _ in capped] == cfgs[20:30]  # top 10 J, evaluation order


def test_selection_ties():
    a = _fin(1, subset=("breakout", "adx"))
    b = _fin(2)
    assert a.n_filters == 1 and b.n_filters == 0
    assert select_finalist([a, b], CFG).selected is b
    c, d = _fin(3), _fin(4)
    want = min((c, d), key=lambda f: f.config_id)
    sel = select_finalist([c, d], CFG)
    assert sel.status == "selected" and sel.selected is want and sel.reasons == ()


def test_no_deployment_reasons():
    fs = [_fin(1, max_drawdown=0.5), _fin(2, max_drawdown=0.5, trades=5), _fin(3, trades=5), _fin(4, trades=5)]
    sel = select_finalist(fs, CFG)
    assert sel.status == "no_deployment" and sel.selected is None
    assert sel.reasons == ("max_drawdown x2", "validation_min_trades x3")
    assert sel.finalists == tuple(fs)
    empty = select_finalist([], CFG)
    assert empty.status == "no_deployment" and empty.reasons == ("no_feasible_finalists",)


def test_floor_change_flips_outcome_without_code_edit():
    m = _m(trades=35)
    assert validation_acceptance(m, CFG).passed
    cfg40 = load_protocol(None, ["acceptance.validation_min_trades=40"]).protocol.acceptance
    assert validation_acceptance(m, cfg40).reasons == ("validation_min_trades",)


def test_report_metrics_carried():
    good_val = _fin(1, vj=0.05, rj=-0.5)
    good_report = _fin(2, vj=0.02, rj=0.5)
    sel = select_finalist([good_report, good_val], CFG)
    assert sel.selected is good_val
    assert sel.selected.report.j == -0.5
    assert dataclasses.is_dataclass(sel.selected.report)
