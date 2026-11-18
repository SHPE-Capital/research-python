"""Tests for the D7 weekend calibrator, the reserved-period guard and `shpe momentum calibrate`."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from shpe_research.calibration.cache import EvalCache
from shpe_research.calibration.descriptors import PanelProxyDescriptors
from shpe_research.calibration.reserved import ReservedGuard, ReservedPeriodError
from shpe_research.calibration.strategies import BREAKOUT
from shpe_research.calibration.universe import StaticUniverseProvider
from shpe_research.calibration.weekly import (
    CalibrationInputs,
    IncompleteRunError,
    StaleArchiveError,
    WeeklyCalibration,
    calibrate_week,
    default_replayer,
    drive,
    live_account,
    locked_cutoff,
    replay_account,
    run_id_for,
)
from shpe_research.calibration.window import Archive, iso_monday, plan_window
from shpe_research.cli import main
from shpe_research.config.momentum import ResolvedProtocol, dump_protocol_yaml, load_protocol
from shpe_research.contracts.weekly_config import DateRange, from_json
from tests.unit.calibration.conftest import SMALL_PROTOCOL_OVERRIDES
from tests.unit.calibration.helpers import PlantedEdgeReplayer, every_config, short_breakout, strip_cache_state

AS_OF = date(2024, 5, 31)
PLANTED_GAIN = 0.002  # added to every daily return of the short-breakout configs: they select by design
PLANTED_LOSS = -0.6  # added on the first validation session: a drawdown above the small protocol's 0.5
DATA_VERSION = "synth-7"
LOGICAL = 24 * 2  # small protocol: budget 24 x seeds [1, 2]
RUN_JSON_KEYS = {
    "run_id",
    "status",
    "cutoff",
    "policy",
    "strategy_id",
    "search_ess",
    "validation_ess",
    "search_weight_parameter",
    "validation_weight_parameter",
    "weight_fallbacks",
    "scaler_medians",
    "scaler_iqrs",
    "chunk_replays",
    "cache_stats",
    "overrides",
}


@pytest.fixture(scope="module")
def archive(synth_archive_root: Path) -> Archive:
    return Archive.open(synth_archive_root)


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One cache file reused by tests that do not count replays (speeds the module up)."""
    return tmp_path_factory.mktemp("weekly_cache") / "c.sqlite"


def _protocol(*extra: str) -> ResolvedProtocol:
    return load_protocol(None, [*SMALL_PROTOCOL_OVERRIDES, *extra])


def _inputs(
    archive: Archive,
    rp: ResolvedProtocol,
    cache_path: Path,
    out_root: Path,
    *,
    descriptors=None,
    expected_hash: str | None = None,
    as_of: date = AS_OF,
) -> CalibrationInputs:
    if descriptors is None:
        descriptors = _descriptors(archive, rp, as_of)
    return CalibrationInputs(
        rp=rp,
        archive=archive,
        universe=StaticUniverseProvider.from_archive(archive),
        descriptors=descriptors,
        cache=EvalCache(cache_path, busy_timeout_s=rp.protocol.cache.busy_timeout_s),
        data_version=DATA_VERSION,
        out_root=out_root,
        expected_hash=expected_hash,
    )


def _descriptors(archive: Archive, rp: ResolvedProtocol, as_of: date = AS_OF) -> PanelProxyDescriptors:
    """The proxy descriptors bounded at the cutoff for ``as_of`` (nothing later is read)."""
    return PanelProxyDescriptors(
        archive, lookback=rp.protocol.weights.descriptor_lookback, through=locked_cutoff(archive, as_of)
    )


def _planted(inputs: CalibrationInputs, *, loss_on: date | None = None) -> PlantedEdgeReplayer:
    """The real replayer with a planted gain on short-breakout configs (and optionally a loss on one session)."""
    gain = PlantedEdgeReplayer(default_replayer(inputs), short_breakout, PLANTED_GAIN)
    return gain if loss_on is None else PlantedEdgeReplayer(gain, every_config, PLANTED_LOSS, [loss_on])


def _plan(archive: Archive, rp: ResolvedProtocol, as_of: date = AS_OF):
    p = rp.protocol
    return plan_window(
        archive,
        as_of,
        validation_sessions=p.validation.sessions,
        min_prior_sessions=max(p.cache.warmup_sessions, p.weights.descriptor_lookback + 1),
        granularity=p.cache.granularity,
    )


def _audit(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def _run_json(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _finalists(run_dir: Path) -> dict:
    return json.loads((run_dir / "finalists.json").read_text(encoding="utf-8"))


def _artifact_dict(run_dir: Path) -> dict:
    return json.loads((run_dir / "weekly_config.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ helpers and guard


def test_run_id_and_accounts(small_protocol) -> None:
    rp = small_protocol
    assert run_id_for(BREAKOUT, "recency", date(2024, 5, 31), rp.hash) == f"breakout-recency-20240531-{rp.hash[:8]}"
    acct = replay_account(rp, BREAKOUT, 10.0)
    assert acct.search_replay_reference_equity == rp.protocol.search.reference_equity and acct.cost_bps == 10.0
    live = live_account(rp, BREAKOUT)
    assert live.search_replay_reference_equity is None
    assert live.cost_bps == rp.protocol.acceptance.selection_bps


def test_reserved_guard_rules(small_protocol, tmp_path) -> None:
    rp = _protocol("walkforward.reserved_start=2024-05-01")
    guard = ReservedGuard.from_protocol(rp, None)
    assert guard.reserved_start == date(2024, 5, 1) and not guard.unlocked()
    guard.check_calibration(date(2024, 4, 30))
    guard.check_replay([date(2024, 4, 29), date(2024, 4, 30)])
    with pytest.raises(ReservedPeriodError):
        guard.check_calibration(date(2024, 5, 1))
    with pytest.raises(ReservedPeriodError):
        guard.check_replay([date(2024, 4, 30), date(2024, 5, 1)])
    assert not ReservedGuard.from_protocol(rp, rp.hash).unlocked()  # matching hash but not frozen
    unset = ReservedGuard.from_protocol(_protocol("walkforward.reserved_start=null"), None)
    with pytest.raises(ReservedPeriodError):
        unset.require_set()
    unset.check_calibration(date(2024, 5, 31))  # nothing reserved
    path = tmp_path / "frozen.yaml"
    path.write_text(dump_protocol_yaml(rp.protocol.model_copy(update={"frozen": True})), encoding="utf-8")
    frozen = load_protocol(path)
    assert ReservedGuard.from_protocol(frozen, frozen.hash).unlocked()
    assert not ReservedGuard.from_protocol(frozen, "0" * 64).unlocked()
    ReservedGuard.from_protocol(frozen, frozen.hash).check_calibration(date(2024, 5, 31))


# ------------------------------------------------------------------------------ end to end


@pytest.mark.parametrize("policy", ["uniform", "recency", "similarity"])
def test_end_to_end_each_policy(policy, archive, small_protocol, shared_cache, tmp_path) -> None:
    rp = small_protocol
    out = calibrate_week(_inputs(archive, rp, shared_cache, tmp_path), AS_OF, policy)
    plan = _plan(archive, rp)
    assert out.status in {"selected", "no_deployment"}
    assert out.plan == plan
    assert out.run_dir == tmp_path / run_id_for(BREAKOUT, policy, plan.cutoff, rp.hash)
    art = from_json((out.run_dir / "weekly_config.json").read_text(encoding="utf-8"))
    assert art == out.artifact
    assert art.policy == policy and art.protocol_hash == rp.hash and art.data_version == DATA_VERSION
    assert art.search_range == DateRange(start=plan.search_sessions[0], end=plan.search_sessions[-1])
    assert art.validation_range == DateRange(start=plan.validation_sessions[0], end=plan.validation_sessions[-1])
    monday = iso_monday(plan.cutoff) + timedelta(days=7)
    assert art.activation == DateRange(start=monday, end=monday + timedelta(days=4))
    assert art.evaluations.logical == LOGICAL
    assert art.seeds == [1, 2]
    assert art.ess == pytest.approx(out.search_weights.ess)
    week = StaticUniverseProvider.from_archive(archive).week(monday)
    assert [(u.asset_id, u.symbol) for u in art.universe] == list(week.members)
    assert (art.selected is not None) == (out.status == "selected") == art.acceptance.passed
    audit = _audit(out.run_dir)
    kinds = {line["kind"] for line in audit}
    assert {"proposal", "selection", "evaluation", "status", "replay_counters"} <= kinds
    assert audit[-1]["kind"] == "replay_counters"
    assert all(line["policy"] == policy and line["strategy_id"] == "breakout" for line in audit[:-1])
    assert all(line["cutoff"] == plan.cutoff.isoformat() for line in audit[:-1])
    run = _run_json(out.run_dir)
    assert set(run) == RUN_JSON_KEYS
    assert run["status"] == out.status and run["policy"] == policy
    assert run["validation_ess"] == pytest.approx(out.validation_weights.ess)
    assert load_protocol(out.run_dir / "protocol.resolved.yaml").hash == rp.hash
    assert (policy == "similarity") == (run["scaler_medians"] is not None)


def test_rerun_is_identical_and_replays_nothing(archive, small_protocol, tmp_path) -> None:
    cache = tmp_path / "c.sqlite"
    first = calibrate_week(_inputs(archive, small_protocol, cache, tmp_path / "a"), AS_OF, "uniform")
    second = calibrate_week(_inputs(archive, small_protocol, cache, tmp_path / "b"), AS_OF, "uniform")
    assert first.artifact.evaluations.actual > 0
    assert strip_cache_state(_artifact_dict(first.run_dir)) == strip_cache_state(_artifact_dict(second.run_dir))
    assert strip_cache_state({"a": _audit(first.run_dir)}) == strip_cache_state({"a": _audit(second.run_dir)})
    assert _audit(second.run_dir)[-1]["shared"]["replays"] == 0
    assert second.artifact.evaluations.actual == 0
    assert _run_json(second.run_dir)["chunk_replays"] == 0


def test_policies_share_cache(archive, small_protocol, tmp_path) -> None:
    cache = tmp_path / "c.sqlite"
    uni = calibrate_week(_inputs(archive, small_protocol, cache, tmp_path), AS_OF, "uniform")
    rec = calibrate_week(_inputs(archive, small_protocol, cache, tmp_path), AS_OF, "recency")

    def asked(run_dir: Path) -> set[str]:
        return {line["config_id"] for line in _audit(run_dir) if line["kind"] == "evaluation"}

    new = asked(rec.run_dir) - asked(uni.run_dir)
    n_final = len(_finalists(rec.run_dir)["finalists"])
    bound = len(new) * len(rec.plan.search_spans) + n_final * 2 * len(rec.plan.validation_spans)
    assert uni.artifact.evaluations.actual > 0
    assert rec.artifact.evaluations.actual <= bound
    assert _audit(rec.run_dir)[-1]["shared"]["replays"] == rec.artifact.evaluations.actual


def test_all_infeasible_is_no_deployment(archive, shared_cache, tmp_path) -> None:
    rp = _protocol("acceptance.search_min_trades=1000000")
    out = calibrate_week(_inputs(archive, rp, shared_cache, tmp_path), AS_OF, "uniform")
    assert out.status == "no_deployment"
    reason = out.artifact.no_deployment.reason
    assert reason.startswith("no_feasible_finalists")
    assert "search:search_min_trades x" in reason
    assert out.artifact.evaluations.logical == LOGICAL
    assert [c.name for c in out.artifact.acceptance.checks] == ["no_feasible_finalists"]


def test_floor_change_flips_outcome(archive, small_protocol, shared_cache, tmp_path) -> None:
    inputs = _inputs(archive, small_protocol, shared_cache, tmp_path / "a")
    base = calibrate_week(inputs, AS_OF, "uniform", replayer=_planted(inputs))
    assert base.status == "selected"
    assert short_breakout(base.selection.selected.config)
    assert base.selection.selected.validation.trades >= small_protocol.protocol.acceptance.validation_min_trades
    rp = _protocol("acceptance.validation_min_trades=100000")  # an override only; same data, same plant
    inputs = _inputs(archive, rp, shared_cache, tmp_path / "b")
    flipped = calibrate_week(inputs, AS_OF, "uniform", replayer=_planted(inputs))
    assert flipped.status == "no_deployment"
    assert "validation_min_trades" in flipped.artifact.no_deployment.reason


def test_artifact_records_rejections(archive, shared_cache, tmp_path) -> None:
    rp = _protocol("acceptance.validation_min_trades=100000")
    inputs = _inputs(archive, rp, shared_cache, tmp_path)
    loss_day = _plan(archive, rp).validation_sessions[0]
    out = calibrate_week(inputs, AS_OF, "uniform", replayer=_planted(inputs, loss_on=loss_day))
    assert out.status == "no_deployment"
    fin = _finalists(out.run_dir)
    assert fin["finalists"], "the search should leave feasible finalists"
    failed = {c["name"] for f in fin["finalists"] for c in f["checks"] if not c["passed"]}
    assert {"validation_min_trades", "max_drawdown"} <= failed
    limit = rp.protocol.acceptance.max_drawdown
    for f in fin["finalists"]:  # the planted loss fails every finalist's drawdown, traded or not
        [dd] = [c for c in f["checks"] if c["name"] == "max_drawdown"]
        assert not dd["passed"] and dd["value"] > limit
    # Not a vacuous pass: the finalists traded, and the reported (best) finalist did.
    assert all(f["validation"]["trades"] > 0 for f in fin["finalists"])
    [trades] = [c for c in out.artifact.acceptance.checks if c.name == "validation_min_trades"]
    assert trades.value > 0
    for name in failed:
        assert name in out.artifact.no_deployment.reason
    for f in fin["finalists"]:
        assert [c["name"] for c in f["checks"]] == [c.name for c in out.selection.finalists[0].verdict.checks]
    assert fin["reasons"] == list(out.selection.reasons)


class _ScaledValidation:
    """Descriptor provider that multiplies the rows of ``sessions`` by ``factor``."""

    def __init__(self, base: PanelProxyDescriptors, sessions, factor: float) -> None:
        self.base, self.sessions, self.factor = base, frozenset(sessions), factor
        self.version = "scaled-validation"

    def rows(self, sessions):
        out = self.base.rows(sessions).copy()
        out[[s in self.sessions for s in sessions]] *= self.factor
        return out

    def current(self, cutoff):
        return self.base.current(cutoff)


class _ConstantDescriptors:
    version = "constant"

    def rows(self, sessions):
        return np.ones((len(sessions), 3))

    def current(self, cutoff):
        return np.ones(3)


def test_validation_descriptors_never_fit_scaler(archive, small_protocol, shared_cache, tmp_path) -> None:
    rp = small_protocol
    base = _descriptors(archive, rp)
    plain = calibrate_week(_inputs(archive, rp, shared_cache, tmp_path / "a", descriptors=base), AS_OF, "similarity")
    scaled_desc = _ScaledValidation(base, _plan(archive, rp).validation_sessions, 100.0)
    scaled = calibrate_week(
        _inputs(archive, rp, shared_cache, tmp_path / "b", descriptors=scaled_desc), AS_OF, "similarity"
    )
    a, b = _run_json(plain.run_dir), _run_json(scaled.run_dir)
    assert a["scaler_medians"] is not None
    assert (a["scaler_medians"], a["scaler_iqrs"]) == (b["scaler_medians"], b["scaler_iqrs"])
    assert plain.search_weights.weights.tobytes() == scaled.search_weights.weights.tobytes()


def test_similarity_fallback_recorded(archive, small_protocol, shared_cache, tmp_path) -> None:
    inputs = _inputs(archive, small_protocol, shared_cache, tmp_path, descriptors=_ConstantDescriptors())
    out = calibrate_week(inputs, AS_OF, "similarity")
    fallbacks = _run_json(out.run_dir)["weight_fallbacks"]
    assert fallbacks["search"] == "all_descriptors_zero_scale"
    assert fallbacks["validation"] == "all_descriptors_zero_scale"
    assert out.search_weights.policy_used == "uniform"


def test_reserved_guard_in_calibrate(archive, shared_cache, tmp_path) -> None:
    rp = _protocol("walkforward.reserved_start=2024-05-01")
    blind = _ConstantDescriptors()  # refused runs read no descriptors (the CLI builds them after the guard)
    with pytest.raises(ReservedPeriodError):
        calibrate_week(_inputs(archive, rp, shared_cache, tmp_path, descriptors=blind), AS_OF, "uniform")
    assert not any(tmp_path.iterdir())  # refused before anything was written
    path = tmp_path / "frozen.yaml"
    path.write_text(dump_protocol_yaml(rp.protocol.model_copy(update={"frozen": True})), encoding="utf-8")
    frozen = load_protocol(path)
    with pytest.raises(ReservedPeriodError):  # frozen but no expected hash
        calibrate_week(_inputs(archive, frozen, shared_cache, tmp_path / "x", descriptors=blind), AS_OF, "uniform")
    out = calibrate_week(
        _inputs(archive, frozen, shared_cache, tmp_path / "ok", expected_hash=frozen.hash), AS_OF, "uniform"
    )
    assert out.status in {"selected", "no_deployment"}


def test_stale_archive_is_refused(archive, small_protocol, shared_cache, tmp_path) -> None:
    # The archive ends 2024-06-28; an as_of weeks later would silently lock that old session.
    with pytest.raises(StaleArchiveError):
        calibrate_week(_inputs(archive, small_protocol, shared_cache, tmp_path), date(2024, 7, 19), "uniform")
    # A weekend as_of right after the last session is fine (the last session is the latest calendar session).
    late = date(2024, 6, 30)
    WeeklyCalibration(_inputs(archive, small_protocol, shared_cache, tmp_path, as_of=late), late, "uniform", BREAKOUT)
    # Good Friday 2024-03-29 is a holiday: the latest session is Thursday, which the archive holds.
    assert locked_cutoff(archive, date(2024, 3, 29)) == date(2024, 3, 28)


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def test_time_limit_raises_incomplete(archive, shared_cache, tmp_path) -> None:
    rp = _protocol("compute.max_wall_hours=0.0001")
    with pytest.raises(IncompleteRunError):
        calibrate_week(_inputs(archive, rp, shared_cache, tmp_path), AS_OF, "uniform", clock=_FakeClock())
    [run_dir] = list(tmp_path.iterdir())
    assert _run_json(run_dir)["status"] == "incomplete_time_limit"
    assert not (run_dir / "weekly_config.json").exists()
    assert (run_dir / "audit.jsonl").exists()


def test_drive_requires_shared_plan(archive, small_protocol, shared_cache, tmp_path) -> None:
    inputs = _inputs(archive, small_protocol, shared_cache, tmp_path)
    a = WeeklyCalibration(inputs, AS_OF, "uniform", BREAKOUT)
    b = WeeklyCalibration(inputs, date(2024, 6, 7), "uniform", BREAKOUT)
    with pytest.raises(ValueError, match="cutoff"):
        drive([a, b], replayer=None)  # type: ignore[arg-type]


def _load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    return (
        strip_cache_state(_artifact_dict(run_dir)),
        strip_cache_state({"audit": _audit(run_dir)}),
        _finalists(run_dir),
    )


def test_joint_drive_equals_each_policy_alone(archive, small_protocol, tmp_path) -> None:
    policies = ("uniform", "recency", "similarity")
    joint_inputs = _inputs(archive, small_protocol, tmp_path / "joint.sqlite", tmp_path / "joint")
    joint = drive(
        [WeeklyCalibration(joint_inputs, AS_OF, pol, BREAKOUT) for pol in policies], default_replayer(joint_inputs)
    )
    assert [o.artifact.policy for o in joint] == list(policies)
    for pol, together in zip(policies, joint, strict=True):
        alone_inputs = _inputs(archive, small_protocol, tmp_path / f"{pol}.sqlite", tmp_path / pol)
        alone = calibrate_week(alone_inputs, AS_OF, pol)
        assert _load_run(together.run_dir) == _load_run(alone.run_dir), pol


def test_drive_refuses_conflicting_accounts(archive, small_protocol, shared_cache, tmp_path) -> None:
    inputs = _inputs(archive, small_protocol, shared_cache, tmp_path)
    other = dataclasses.replace(BREAKOUT, account_overrides=(("max_positions", 5),))  # same strategy_id
    a = WeeklyCalibration(inputs, AS_OF, "uniform", BREAKOUT)
    b = WeeklyCalibration(inputs, AS_OF, "recency", other)
    with pytest.raises(ValueError, match="replay account"):
        drive([a, b], default_replayer(inputs))


# ------------------------------------------------------------------------------ CLI


def _cli_args(root: Path, out: Path, *extra: str, cutoff: str = "2024-05-31", policy: str = "uniform") -> list[str]:
    args = [
        "momentum",
        "calibrate",
        "--panel-root",
        str(root),
        "--cutoff",
        cutoff,
        "--policy",
        policy,
        "--data-version",
        "synth-7",
        "--out",
        str(out),
        "--set",
        f"cache.path={(out / 'c.sqlite').as_posix()}",
    ]
    for item in (*SMALL_PROTOCOL_OVERRIDES, *extra):
        args += ["--set", item]
    return args


def test_cli_calibrate(synth_archive_root, tmp_path, capsys) -> None:
    assert main(_cli_args(synth_archive_root, tmp_path)) == 0
    [artifact] = list(tmp_path.glob("*/weekly_config.json"))
    art = from_json(artifact.read_text(encoding="utf-8"))
    assert art.data_version == "synth-7" and art.policy == "uniform"
    printed = capsys.readouterr().out
    assert str(artifact) in printed
    assert ("selected" if art.selected else "no_deployment") in printed
    assert (tmp_path / "c.sqlite").exists()


def test_cli_reserved_exit_code(synth_archive_root, tmp_path, capsys) -> None:
    assert main(_cli_args(synth_archive_root, tmp_path, "walkforward.reserved_start=2024-05-01")) == 2
    assert "reserved" in capsys.readouterr().err.lower()


def test_cli_insufficient_history_exit_code(synth_archive_root, tmp_path, capsys) -> None:
    assert main(_cli_args(synth_archive_root, tmp_path, cutoff="2024-01-12")) == 2
    assert "InsufficientHistoryError" in capsys.readouterr().err


def _record_chunk_reads(monkeypatch) -> list[Path]:
    """Record every chunk opened by the descriptors and by panel assembly."""
    import shpe_research.calibration.descriptors as descriptors_mod
    import shpe_research.calibration.window as window_mod

    opened: list[Path] = []
    real = window_mod.open_panel_chunk

    def recording(path):
        opened.append(Path(path))
        return real(path)

    monkeypatch.setattr(descriptors_mod, "open_panel_chunk", recording)
    monkeypatch.setattr(window_mod, "open_panel_chunk", recording)
    return opened


def test_cli_reads_nothing_reserved(synth_archive_root, tmp_path, monkeypatch) -> None:
    from shpe_research.data.panel import read_meta

    opened = _record_chunk_reads(monkeypatch)
    # Cutoff inside the reserved period, no unlock: refused before any chunk is opened.
    assert main(_cli_args(synth_archive_root, tmp_path / "a", "walkforward.reserved_start=2024-05-01")) == 2
    assert opened == []
    # Cutoff before the reserved period: the run (similarity reads the descriptors) opens no chunk
    # holding a session after the cutoff, hence none on or after reserved_start.
    extra = "walkforward.reserved_start=2024-05-06"
    assert main(_cli_args(synth_archive_root, tmp_path / "b", extra, cutoff="2024-04-26", policy="similarity")) == 0
    assert opened
    last = max(date.fromisoformat(d) for path in set(opened) for d in read_meta(path)["sessions"])
    assert last <= date(2024, 4, 26) < date(2024, 5, 6)
