"""Tests for the D9 strategy registry and the chunk-major multi-strategy scheduler."""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from shpe_research.calibration.cache import EvalCache
from shpe_research.calibration.descriptors import PanelProxyDescriptors
from shpe_research.calibration.keys import ChunkSpan, chunk_spans
from shpe_research.calibration.replay import ChunkReplayer, ReplayRequest
from shpe_research.calibration.scheduler import SchedulerReport, run_strategies
from shpe_research.calibration.strategies import BREAKOUT, STRATEGY_BASES, StrategySpec, load_strategies
from shpe_research.calibration.universe import StaticUniverseProvider
from shpe_research.calibration.weekly import (
    CalibrationInputs,
    calibrate_week,
    default_replayer,
    locked_cutoff,
    replay_account,
    run_id_for,
)
from shpe_research.calibration.window import Archive
from shpe_research.config.momentum import ResolvedProtocol, load_protocol
from shpe_research.strategies.momentum.breakout.grid import DEFAULT_DOMAIN, Config
from shpe_research.trade.account_types import AccountConfig
from tests.unit.calibration.conftest import SMALL_PROTOCOL_OVERRIDES
from tests.unit.calibration.helpers import strip_cache_state

AS_OF = date(2024, 5, 31)
POLICY = "uniform"
DATA_VERSION = "synth-7"
TIGHT_STOP_ATR = 1.0
TIGHT_BUDGET = 12
TIGHT_SEEDS = [9]
KILL_AFTER_SPANS = 3
STRATEGIES_OVERRIDE = (
    "search.strategies=["
    "{id: breakout},"
    "{id: breakout_b},"
    f"{{id: breakout_tight, account_overrides: {{exit.initial_stop_atr: {TIGHT_STOP_ATR}}}, "
    f"budget: {TIGHT_BUDGET}, seeds: {TIGHT_SEEDS}}}]"
)
IDS = ("breakout", "breakout_b", "breakout_tight")
CONFIG = Config.make("v1", ["breakout", "rvol"], {"breakout.n": 10, "breakout.b": 0, "rvol.w": 3, "rvol.v": 1})


class _Kill(Exception):
    """Simulated kill of the weekend job between two committed spans."""


def _protocol(*extra: str) -> ResolvedProtocol:
    return load_protocol(None, [*SMALL_PROTOCOL_OVERRIDES, *extra])


@pytest.fixture(scope="module")
def archive(synth_archive_root: Path) -> Archive:
    return Archive.open(synth_archive_root)


@pytest.fixture(scope="module")
def rp() -> ResolvedProtocol:
    return _protocol(STRATEGIES_OVERRIDE)


@pytest.fixture(scope="module")
def specs(rp: ResolvedProtocol) -> tuple[StrategySpec, ...]:
    return load_strategies(rp)


def _inputs(archive: Archive, rp: ResolvedProtocol, cache_path: Path, out_root: Path) -> CalibrationInputs:
    return CalibrationInputs(
        rp=rp,
        archive=archive,
        universe=StaticUniverseProvider.from_archive(archive),
        descriptors=PanelProxyDescriptors(
            archive, lookback=rp.protocol.weights.descriptor_lookback, through=locked_cutoff(archive, AS_OF)
        ),
        cache=EvalCache(cache_path, busy_timeout_s=rp.protocol.cache.busy_timeout_s),
        data_version=DATA_VERSION,
        out_root=out_root,
    )


def _artifact(run_dir: Path) -> dict:
    return strip_cache_state(json.loads((run_dir / "weekly_config.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def joint(archive, rp, specs, tmp_path_factory):
    """All three strategies through one ``run_strategies`` call on a fresh cache."""
    root = tmp_path_factory.mktemp("joint")
    outcomes, report = run_strategies(_inputs(archive, rp, root / "c.sqlite", root / "runs"), AS_OF, POLICY, specs)
    return outcomes, report


@pytest.fixture(scope="module")
def alone(archive, rp, specs, tmp_path_factory):
    """Each strategy through ``calibrate_week`` alone on its own fresh cache: ``{id: (outcome, replayer)}``."""
    out = {}
    for spec in specs:
        root = tmp_path_factory.mktemp(f"alone_{spec.strategy_id}")
        inputs = _inputs(archive, rp, root / "c.sqlite", root / "runs")
        rep = default_replayer(inputs)
        out[spec.strategy_id] = (calibrate_week(inputs, AS_OF, POLICY, spec, replayer=rep), rep)
    return out


# ------------------------------------------------------------------------------ registry


def test_load_strategies_from_protocol(rp, specs) -> None:
    assert set(STRATEGY_BASES) == {"breakout"} and STRATEGY_BASES["breakout"] is BREAKOUT
    assert [s.strategy_id for s in specs] == list(IDS)
    base, tight = specs[0], specs[2]
    assert base.account_config(AccountConfig()) == AccountConfig()
    assert base.budget is None and base.seeds is None
    assert tight.account_config(AccountConfig()).exit.initial_stop_atr == TIGHT_STOP_ATR
    assert AccountConfig().exit.initial_stop_atr != TIGHT_STOP_ATR
    assert tight.budget == TIGHT_BUDGET and tight.seeds == tuple(TIGHT_SEEDS)
    assert tight.domain == BREAKOUT.domain and tight.signal_mask is BREAKOUT.signal_mask
    two = load_strategies(
        _protocol(
            "search.strategies=[{id: breakout}, "
            f"{{id: breakout_tight, account_overrides: {{exit.initial_stop_atr: 1}}, budget: {TIGHT_BUDGET}}}]"
        )
    )
    assert [s.strategy_id for s in two] == ["breakout", "breakout_tight"]
    assert two[1].account_config(AccountConfig()).exit.initial_stop_atr == TIGHT_STOP_ATR
    assert two[1].budget == TIGHT_BUDGET and two[1].seeds is None
    with pytest.raises(KeyError, match="breakout"):  # the message lists the known bases
        load_strategies(_protocol("search.strategies=[{id: x, base: nope}]"))
    with pytest.raises(ValueError, match="nope"):
        load_strategies(_protocol("search.strategies=[{id: x, account_overrides: {exit.nope: 1.0}}]"))


def test_run_strategies_refuses_duplicate_ids(archive, rp, tmp_path) -> None:
    inputs = _inputs(archive, rp, tmp_path / "c.sqlite", tmp_path / "runs")
    with pytest.raises(ValueError, match="breakout"):
        run_strategies(inputs, AS_OF, POLICY, [BREAKOUT, BREAKOUT])


# ------------------------------------------------------------------------------ signal code in the context


def _write_module(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


SIGNAL_SRC = (
    "from shpe_research.strategies.momentum.breakout import signals\n"
    "import d9_own_helper\n\n\n"
    "def signal_mask(panel, config, cache):\n"
    "    return d9_own_helper.wrap(signals.signal_mask(panel, config, cache))\n"
)
HELPER_SRC = "def wrap(mask):\n    return mask\n"


def _own_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StrategySpec:
    _write_module(tmp_path / "d9_own_signal.py", SIGNAL_SRC)
    _write_module(tmp_path / "d9_own_helper.py", HELPER_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("d9_own_signal", "d9_own_helper"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    mod = importlib.import_module("d9_own_signal")
    return StrategySpec("own", DEFAULT_DOMAIN, mod.signal_mask, code_modules=("d9_own_helper",))


def _replayer(archive: Archive, rp: ResolvedProtocol, cache_path: Path) -> ChunkReplayer:
    return default_replayer(_inputs(archive, rp, cache_path, cache_path.parent / "runs"))


def test_own_signal_module_in_context_code_version(archive, rp, tmp_path, monkeypatch) -> None:
    spec = _own_spec(tmp_path, monkeypatch)
    assert spec.signal_modules == ("d9_own_signal", "d9_own_helper")
    assert BREAKOUT.signal_modules == ("shpe_research.strategies.momentum.breakout.signals",)
    acct = replay_account(rp, spec, rp.protocol.acceptance.selection_bps)
    rep = _replayer(archive, rp, tmp_path / "c.sqlite")
    base = rep.context(acct)
    assert rep.context(acct, BREAKOUT) == base  # the breakout signals module is already replay code
    own = rep.context(acct, spec)
    assert own.code_version != base.code_version and own.account_hash == base.account_hash
    # an edit of the signal module, or of a listed helper module, gives a new context (new replayer)
    _write_module(tmp_path / "d9_own_signal.py", SIGNAL_SRC + "# edited\n")
    edited = _replayer(archive, rp, tmp_path / "c.sqlite").context(acct, spec)
    assert edited.code_version != own.code_version
    _write_module(tmp_path / "d9_own_helper.py", HELPER_SRC + "# edited\n")
    assert _replayer(archive, rp, tmp_path / "c.sqlite").context(acct, spec).code_version != edited.code_version


def _first_span(archive: Archive, rp: ResolvedProtocol) -> ChunkSpan:
    warmup = rp.protocol.cache.warmup_sessions
    return next(sp for sp in chunk_spans(archive.sessions, "chunk") if archive.sessions.index(sp.sessions[0]) >= warmup)


def test_run_keys_own_signal_strategy_by_its_code(archive, rp, tmp_path, monkeypatch) -> None:
    """Same mask and account: a breakout-based id shares the path; an own-module strategy does not."""
    spec = _own_spec(tmp_path, monkeypatch)
    span = _first_span(archive, rp)
    rep = _replayer(archive, rp, tmp_path / "c.sqlite")
    acct = replay_account(rp, BREAKOUT, rp.protocol.acceptance.selection_bps)
    same = dataclasses.replace(BREAKOUT, strategy_id="breakout_b")
    reqs = [ReplayRequest(BREAKOUT, CONFIG, acct), ReplayRequest(same, CONFIG, acct), ReplayRequest(spec, CONFIG, acct)]
    out = rep.run(reqs, [span], span.sessions[-1])
    assert [(r.replays, r.mask_hits) for r in out] == [(1, 0), (0, 1), (1, 0)]


# ------------------------------------------------------------------------------ scheduler


def test_shared_indicators_built_once(joint, alone) -> None:
    _, report = joint
    shared = report.shared
    assert shared.spans_opened > 0
    assert shared.atr_builds == shared.panels_built == shared.spans_opened
    _, rep_alone = alone["breakout"]
    assert shared.atr_builds == rep_alone.counters.atr_builds
    assert shared.spans_opened == rep_alone.counters.spans_opened


def test_identical_configs_across_strategies_hit(joint) -> None:
    _, report = joint
    b = report.by_strategy["breakout_b"]
    assert b.replays == 0
    assert b.mask_hits > 0
    assert report.by_strategy["breakout"].replays > 0


def test_different_exits_miss(joint, alone) -> None:
    _, report = joint
    _, rep_alone = alone["breakout_tight"]
    together = report.by_strategy["breakout_tight"]
    by_itself = rep_alone.by_strategy["breakout_tight"]
    assert together.replays > 0
    assert (together.replays, together.mask_hits) == (by_itself.replays, by_itself.mask_hits)


def test_artifact_identical_to_running_alone(joint, alone, rp) -> None:
    outcomes, _ = joint
    assert list(outcomes) == list(IDS)
    for sid in IDS:
        out_alone, _ = alone[sid]
        together = outcomes[sid]
        assert together.run_dir.name == out_alone.run_dir.name
        assert together.run_dir.name == run_id_for(
            dataclasses.replace(BREAKOUT, strategy_id=sid), POLICY, together.plan.cutoff, rp.hash
        )
        assert _artifact(together.run_dir) == _artifact(out_alone.run_dir), sid


def test_resume_after_kill_no_duplicate_chunks(archive, rp, specs, joint, tmp_path) -> None:
    outcomes_clean, report_clean = joint
    cache_path = tmp_path / "c.sqlite"
    first_inputs = _inputs(archive, rp, cache_path, tmp_path / "first")
    killer = default_replayer(first_inputs)
    done: list[ChunkSpan] = []

    def kill_after(span: ChunkSpan) -> None:
        done.append(span)
        if len(done) == KILL_AFTER_SPANS:
            raise _Kill

    killer.on_span_done = kill_after
    with pytest.raises(_Kill):
        run_strategies(first_inputs, AS_OF, POLICY, specs, replayer=killer)
    replays_first = first_inputs.cache.stats.replays  # serial replay: the parent cache saw every lookup
    assert replays_first > 0
    assert not any((tmp_path / "first").glob("*/weekly_config.json"))
    second_inputs = _inputs(archive, rp, cache_path, tmp_path / "second")
    outcomes, report = run_strategies(second_inputs, AS_OF, POLICY, specs, replayer=default_replayer(second_inputs))
    assert replays_first + report.shared.replays == report_clean.shared.replays
    assert report.shared.alias_hits > 0
    for sid in IDS:
        assert _artifact(outcomes[sid].run_dir) == _artifact(outcomes_clean[sid].run_dir), sid


def test_per_strategy_budget_and_seeds_from_config(joint) -> None:
    outcomes, _ = joint
    art = outcomes["breakout_tight"].artifact
    assert art.evaluations.logical == TIGHT_BUDGET
    assert art.seeds == TIGHT_SEEDS
    base = outcomes["breakout"].artifact
    assert base.seeds == [1, 2]


def test_report_counters(joint) -> None:
    _, report = joint
    assert isinstance(report, SchedulerReport)
    assert set(report.by_strategy) == set(IDS)
    for field in ("replays", "mask_hits", "alias_hits", "masks_computed", "atr_builds"):
        assert getattr(report.shared, field) == sum(getattr(c, field) for c in report.by_strategy.values()), field
    assert report.shared.replays > 0
    assert report.cache.replays == report.shared.replays
    assert report.cache.mask_hits == report.shared.mask_hits
    assert report.cache.writes > 0
