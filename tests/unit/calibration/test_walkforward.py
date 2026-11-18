"""Tests for the D8 walk-forward driver, its forward accounts, the reserved guard and `shpe momentum walkforward`."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import shutil
from datetime import date
from pathlib import Path

import pytest

import shpe_research.calibration.walkforward as wf
from shpe_research.calibration.cache import EvalCache
from shpe_research.calibration.descriptors import PanelProxyDescriptors
from shpe_research.calibration.reserved import ReservedGuard, ReservedPeriodError
from shpe_research.calibration.universe import StaticUniverseProvider
from shpe_research.calibration.walkforward import (
    CHECKPOINT_SCHEMA,
    RECORD_SCHEMA,
    StepRecord,
    WalkforwardResult,
    run_walkforward,
    weekly_cutoffs,
)
from shpe_research.calibration.weekly import CalibrationInputs, default_replayer
from shpe_research.calibration.window import Archive, activation_sessions
from shpe_research.cli import main
from shpe_research.config.momentum import ResolvedProtocol, dump_protocol_yaml, load_protocol
from tests.unit.calibration.conftest import SMALL_PROTOCOL_OVERRIDES
from tests.unit.calibration.helpers import PlantedEdgeReplayer, short_breakout, strip_cache_state

FIRST = date(2024, 5, 10)
N_WEEKS = 3
DATA_VERSION = "synth-7"
POLICIES = ("uniform", "recency", "similarity")
PLANTED_GAIN = 0.002  # added to the short-breakout configs' calibration returns: they select by design
CSV_HEADER = ["session", "cutoff", "policy", "active", "config_id", "daily_return", "equity"]


@pytest.fixture(scope="module")
def archive(synth_archive_root: Path) -> Archive:
    return Archive.open(synth_archive_root)


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One cache file for the module; outputs are compared after ``strip_cache_state``."""
    return tmp_path_factory.mktemp("walkforward_cache") / "c.sqlite"


def _protocol(*extra: str) -> ResolvedProtocol:
    return load_protocol(None, [*SMALL_PROTOCOL_OVERRIDES, *extra])


def _inputs(
    archive: Archive, rp: ResolvedProtocol, cache_path: Path, *, first: date = FIRST, n_weeks: int = N_WEEKS
) -> CalibrationInputs:
    through = weekly_cutoffs(archive, first, n_weeks)[-1]
    return CalibrationInputs(
        rp=rp,
        archive=archive,
        universe=StaticUniverseProvider.from_archive(archive),
        descriptors=PanelProxyDescriptors(archive, lookback=rp.protocol.weights.descriptor_lookback, through=through),
        cache=EvalCache(cache_path, busy_timeout_s=rp.protocol.cache.busy_timeout_s),
        data_version=DATA_VERSION,
        out_root=cache_path.parent / "unused",
    )


def _planted(inputs: CalibrationInputs) -> PlantedEdgeReplayer:
    return PlantedEdgeReplayer(default_replayer(inputs), short_breakout, PLANTED_GAIN)


def _run(inputs: CalibrationInputs, out_dir: Path, **kwargs) -> WalkforwardResult:
    first = kwargs.pop("first", FIRST)
    n_weeks = kwargs.pop("n_weeks", N_WEEKS)
    return run_walkforward(inputs, first, n_weeks, out_dir, replayer=_planted(inputs), **kwargs)


@pytest.fixture(scope="module")
def clean(archive, small_protocol, shared_cache, tmp_path_factory) -> WalkforwardResult:
    """One uninterrupted 3-week x 3-policy walk-forward (the reference every resume test compares to)."""
    inputs = _inputs(archive, small_protocol, shared_cache)
    return _run(inputs, tmp_path_factory.mktemp("clean") / "wf")


def _records(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_dir / "records").glob("*.json"))]


def _runs(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (out_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()]


def _csv_rows(out_dir: Path, policy: str) -> list[dict]:
    with (out_dir / "daily" / f"{policy}.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_HEADER
        return list(reader)


def _assert_same_outputs(out_dir: Path, ref_dir: Path) -> None:
    assert [strip_cache_state(r) for r in _records(out_dir)] == [strip_cache_state(r) for r in _records(ref_dir)]
    assert [strip_cache_state(r) for r in _runs(out_dir)] == [strip_cache_state(r) for r in _runs(ref_dir)]
    for policy in POLICIES:
        name = f"daily/{policy}.csv"
        assert (out_dir / name).read_bytes() == (ref_dir / name).read_bytes()


# ------------------------------------------------------------------------------ calendar


def test_weekly_cutoffs_are_last_session_of_each_week(archive) -> None:
    assert weekly_cutoffs(archive, FIRST, N_WEEKS) == [date(2024, 5, 10), date(2024, 5, 17), date(2024, 5, 24)]
    # a mid-week first cutoff still takes its week's last archive session
    assert weekly_cutoffs(archive, date(2024, 5, 8), 1) == [date(2024, 5, 10)]
    # weeks past the archive end are not listed (the archive ends 2024-06-28)
    assert weekly_cutoffs(archive, date(2024, 6, 14), 4) == [date(2024, 6, 14), date(2024, 6, 21), date(2024, 6, 28)]


# ------------------------------------------------------------------------------ the loop


def test_runs_consecutive_weeks_and_writes_daily_series(clean) -> None:
    assert clean.stopped == "complete"
    assert len(clean.records) == N_WEEKS * len(POLICIES)
    assert all(isinstance(r, StepRecord) and r.schema == RECORD_SCHEMA for r in clean.records)
    assert len(list((clean.out_dir / "records").glob("*.json"))) == 9
    assert len(_runs(clean.out_dir)) == 9
    expected_sessions = [s for c in sorted({r.cutoff for r in clean.records}) for s in activation_sessions(c)]
    assert len(expected_sessions) == 14  # 5 + 5 + 4 (Memorial Day 2024-05-27)
    for policy in POLICIES:
        rows = _csv_rows(clean.out_dir, policy)
        assert [date.fromisoformat(r["session"]) for r in rows] == expected_sessions
        assert {r["policy"] for r in rows} == {policy}
    checkpoint = json.loads((clean.out_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["schema"] == CHECKPOINT_SCHEMA
    assert checkpoint["in_progress"] is None
    assert checkpoint["policies"] == list(POLICIES)
    for rec in clean.records:
        assert (clean.out_dir / rec.artifact_path).is_file()
        assert rec.activation == activation_sessions(rec.cutoff)
        assert len(rec.daily_returns) == len(rec.activation)


def test_equity_carries_without_reset(clean) -> None:
    assert any(r.active and any(x != 0.0 for x in r.daily_returns) for r in clean.records)
    for policy in POLICIES:
        recs = sorted((r for r in clean.records if r.policy == policy), key=lambda r: r.cutoff)
        assert recs[0].equity_start == 100000.0
        for prev, cur in zip(recs, recs[1:], strict=False):
            assert cur.equity_start == prev.equity_end
        for rec in recs:
            assert rec.equity_end == pytest.approx(rec.equity_start * math.prod(1 + r for r in rec.daily_returns))
            assert rec.active == (rec.status == "selected") == (rec.config_id is not None)
        rows = _csv_rows(clean.out_dir, policy)
        assert float(rows[-1]["equity"]) == pytest.approx(recs[-1].equity_end)


def test_no_deployment_week_is_inactive(archive, shared_cache, tmp_path) -> None:
    rp = _protocol("acceptance.search_min_trades=1000000")
    result = _run(_inputs(archive, rp, shared_cache), tmp_path / "wf")
    assert len(result.records) == 9
    for rec in result.records:
        assert rec.status == "no_deployment" and rec.active is False and rec.config_id is None
        assert all(x == 0.0 for x in rec.daily_returns)
        assert rec.equity_start == rec.equity_end == 100000.0
    for policy in POLICIES:
        assert {r["active"] for r in _csv_rows(result.out_dir, policy)} == {"false"}


# ------------------------------------------------------------------------------ resume


class _Kill(RuntimeError):
    pass


def test_resume_after_kill_matches_clean_run(archive, small_protocol, shared_cache, clean, tmp_path) -> None:
    inputs = _inputs(archive, small_protocol, shared_cache)
    done: list[StepRecord] = []

    def kill_after_four(rec: StepRecord) -> None:
        done.append(rec)
        if len(done) == 4:
            raise _Kill

    with pytest.raises(_Kill):
        _run(inputs, tmp_path / "wf", on_step_done=kill_after_four)
    assert len(list((tmp_path / "wf" / "records").glob("*.json"))) == 4
    resumed: list[StepRecord] = []
    result = _run(inputs, tmp_path / "wf", on_step_done=resumed.append)
    assert len(resumed) == 5  # the four finished steps are not run again
    keys = [(r.cutoff, r.policy) for r in done + resumed]
    assert len(keys) == len(set(keys)) == 9
    assert result.stopped == "complete" and len(result.records) == 9
    assert not (tmp_path / "wf" / "events.jsonl").exists() or not (tmp_path / "wf" / "events.jsonl").read_text()
    _assert_same_outputs(tmp_path / "wf", clean.out_dir)


def test_kill_after_artifact_before_record(archive, small_protocol, shared_cache, clean, tmp_path, monkeypatch) -> None:
    inputs = _inputs(archive, small_protocol, shared_cache)
    real = wf._write_record
    calls = {"n": 0}

    def flaky(out_dir: Path, rec: StepRecord) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise _Kill
        real(out_dir, rec)

    monkeypatch.setattr(wf, "_write_record", flaky)
    with pytest.raises(_Kill):
        _run(inputs, tmp_path / "wf")
    assert len(list((tmp_path / "wf" / "records").glob("*.json"))) == 1
    assert list((tmp_path / "wf" / "artifacts").glob(f"{FIRST:%Y%m%d}/breakout-recency-*/weekly_config.json"))
    _run(inputs, tmp_path / "wf")
    events = [json.loads(x) for x in (tmp_path / "wf" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == [
        {"kind": "operational_failure", "cutoff": FIRST.isoformat(), "policy": "recency", "detected_at_step": 1}
    ]
    _assert_same_outputs(tmp_path / "wf", clean.out_dir)


def test_kill_after_last_record_before_outputs(
    archive, small_protocol, shared_cache, clean, tmp_path, monkeypatch
) -> None:
    inputs = _inputs(archive, small_protocol, shared_cache)
    real = wf._write_outputs
    killed = {"done": False}

    def flaky(out_dir: Path, records, policies) -> None:
        if len(records) == 9 and not killed["done"]:  # the last step's regeneration
            killed["done"] = True
            raise _Kill
        real(out_dir, records, policies)

    monkeypatch.setattr(wf, "_write_outputs", flaky)
    with pytest.raises(_Kill):
        _run(inputs, tmp_path / "wf")
    assert len(list((tmp_path / "wf" / "records").glob("*.json"))) == 9
    assert len(_runs(tmp_path / "wf")) == 8  # stale: the last record is missing from the outputs
    resumed: list[StepRecord] = []
    result = _run(inputs, tmp_path / "wf", on_step_done=resumed.append)
    assert resumed == [] and result.stopped == "complete"
    _assert_same_outputs(tmp_path / "wf", clean.out_dir)


def test_header_mismatch_refuses_resume(archive, small_protocol, shared_cache, clean, tmp_path) -> None:
    out = tmp_path / "wf"
    out.mkdir()
    shutil.copy(clean.out_dir / "checkpoint.json", out / "checkpoint.json")
    inputs = _inputs(archive, small_protocol, shared_cache)
    other = dataclasses.replace(inputs, data_version="synth-8")
    with pytest.raises(ValueError, match="data_version"):
        _run(other, out)
    assert not (out / "records").exists()


def test_archive_end_stops_cleanly(archive, shared_cache, tmp_path) -> None:
    rp = _protocol("walkforward.policies=[uniform]")
    first = date(2024, 6, 14)
    result = _run(_inputs(archive, rp, shared_cache, first=first, n_weeks=4), tmp_path / "wf", first=first, n_weeks=4)
    assert result.stopped == "archive_end"
    assert [(r.cutoff, r.policy) for r in result.records] == [
        (date(2024, 6, 14), "uniform"),
        (date(2024, 6, 21), "uniform"),
    ]
    checkpoint = json.loads((tmp_path / "wf" / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["in_progress"] is None


# ------------------------------------------------------------------------------ reserved guard


def _no_calibration(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("calibrate_week must not run")

    monkeypatch.setattr(wf, "calibrate_week", refuse)
    monkeypatch.setattr(wf, "forward_replay", refuse)


def test_refuses_when_reserved_start_unset(archive, shared_cache, tmp_path, monkeypatch) -> None:
    _no_calibration(monkeypatch)
    rp = _protocol("walkforward.reserved_start=null")
    with pytest.raises(ReservedPeriodError, match="reserved_start"):
        _run(_inputs(archive, rp, shared_cache), tmp_path / "wf")
    assert not (tmp_path / "wf" / "checkpoint.json").exists()


def test_refuses_calibration_on_or_after_reserved(archive, shared_cache, tmp_path, monkeypatch) -> None:
    _no_calibration(monkeypatch)
    rp = _protocol("walkforward.reserved_start=2024-05-17")
    first = date(2024, 5, 17)
    with pytest.raises(ReservedPeriodError, match="cutoff 2024-05-17"):
        _run(_inputs(archive, rp, shared_cache, first=first), tmp_path / "wf", first=first)
    assert not list((tmp_path / "wf" / "records").glob("*.json"))


def test_refuses_replay_into_reserved(archive, shared_cache, tmp_path, monkeypatch) -> None:
    _no_calibration(monkeypatch)
    rp = _protocol("walkforward.reserved_start=2024-05-14")
    with pytest.raises(ReservedPeriodError, match="session 2024-05-14"):
        _run(_inputs(archive, rp, shared_cache, n_weeks=1), tmp_path / "wf", n_weeks=1)
    assert not list((tmp_path / "wf" / "records").glob("*.json"))
    checkpoint = json.loads((tmp_path / "wf" / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["in_progress"] is None


def _frozen(small_protocol: ResolvedProtocol, tmp_path: Path) -> ResolvedProtocol:
    p = small_protocol.protocol
    walk = p.walkforward.model_copy(update={"reserved_start": date(2024, 5, 14)})
    path = tmp_path / "frozen.yaml"
    path.write_text(dump_protocol_yaml(p.model_copy(update={"frozen": True, "walkforward": walk})), encoding="utf-8")
    rp = load_protocol(path)
    assert rp.protocol.frozen and rp.protocol.walkforward.reserved_start == date(2024, 5, 14)
    return rp


def test_frozen_with_matching_hash_allowed(archive, small_protocol, shared_cache, tmp_path) -> None:
    rp = _frozen(small_protocol, tmp_path)
    inputs = dataclasses.replace(_inputs(archive, rp, shared_cache, n_weeks=1), expected_hash=rp.hash)
    result = _run(inputs, tmp_path / "wf", n_weeks=1)
    assert result.stopped == "complete"
    assert len(result.records) == 3
    assert all(r.protocol_hash == rp.hash and r.activation[-1] >= date(2024, 5, 14) for r in result.records)


def test_frozen_with_wrong_hash_refused(archive, small_protocol, shared_cache, tmp_path, monkeypatch) -> None:
    _no_calibration(monkeypatch)
    rp = _frozen(small_protocol, tmp_path)
    with pytest.raises(ReservedPeriodError):
        inputs = dataclasses.replace(_inputs(archive, rp, shared_cache, n_weeks=1), expected_hash="0" * 64)
        _run(inputs, tmp_path / "wf", n_weeks=1)
    assert not list((tmp_path / "wf" / "records").glob("*.json"))


def test_run_walkforward_has_one_expected_hash_path() -> None:
    import inspect

    assert "expected_hash" not in inspect.signature(run_walkforward).parameters


def test_guard_allows_calibration(small_protocol, tmp_path) -> None:
    rp = _protocol("walkforward.reserved_start=2024-05-17")
    guard = ReservedGuard.from_protocol(rp, None)
    assert guard.allows_calibration(date(2024, 5, 16))
    assert not guard.allows_calibration(date(2024, 5, 17))
    frozen = _frozen(small_protocol, tmp_path)
    assert ReservedGuard.from_protocol(frozen, frozen.hash).allows_calibration(date(2024, 6, 28))
    assert not ReservedGuard.from_protocol(frozen, None).allows_calibration(date(2024, 6, 28))
    unset = _protocol("walkforward.reserved_start=null")
    assert ReservedGuard.from_protocol(unset, None).allows_calibration(date(2024, 6, 28))


def test_refused_resume_does_not_duplicate_operational_failure(archive, shared_cache, tmp_path, monkeypatch) -> None:
    _no_calibration(monkeypatch)
    rp = _protocol("walkforward.reserved_start=2024-05-17")
    first = date(2024, 5, 17)
    out = tmp_path / "wf"
    inputs = _inputs(archive, rp, shared_cache, first=first)
    with pytest.raises(ReservedPeriodError):
        _run(inputs, out, first=first)
    checkpoint = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint["in_progress"] = {"cutoff": "2024-05-17", "policy": "uniform"}
    (out / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    for _ in range(2):
        with pytest.raises(ReservedPeriodError):
            _run(inputs, out, first=first)
    assert not (out / "events.jsonl").exists()


# ------------------------------------------------------------------------------ resume header


def test_header_pins_universe_and_descriptor_versions(archive, small_protocol, shared_cache, clean, tmp_path) -> None:
    checkpoint = json.loads((clean.out_dir / "checkpoint.json").read_text(encoding="utf-8"))
    inputs = _inputs(archive, small_protocol, shared_cache)
    assert checkpoint["universe_version"] == inputs.universe.version
    assert checkpoint["descriptors_version"] == inputs.descriptors.version
    out = tmp_path / "wf"
    out.mkdir()
    shutil.copy(clean.out_dir / "checkpoint.json", out / "checkpoint.json")
    other = dataclasses.replace(inputs, universe=StaticUniverseProvider.from_archive(archive, version="other"))
    with pytest.raises(ValueError, match="universe_version"):
        _run(other, out)
    other = dataclasses.replace(
        inputs,
        descriptors=PanelProxyDescriptors(
            archive, lookback=small_protocol.protocol.weights.descriptor_lookback + 1, through=FIRST
        ),
    )
    with pytest.raises(ValueError, match="descriptors_version"):
        _run(other, out)


def _copy_run(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("artifacts"))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("protocol_hash", "f" * 64, "protocol_hash"),
        ("data_version", "synth-8", "data_version"),
        ("strategy_id", "other", "strategy_id"),
        ("cutoff", "2024-04-26", "not a step"),
        ("policy", "bogus", "not a step"),
    ],
)
def test_resume_refuses_foreign_record(
    archive, small_protocol, shared_cache, clean, tmp_path, monkeypatch, field, value, match
) -> None:
    _no_calibration(monkeypatch)
    out = tmp_path / "wf"
    _copy_run(clean.out_dir, out)
    path = sorted((out / "records").glob("*.json"))[0]
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec[field] = value
    path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _run(_inputs(archive, small_protocol, shared_cache), out)


# ------------------------------------------------------------------------------ CLI


def _cli_args(root: Path, out: Path, *extra: str) -> list[str]:
    args = [
        "momentum",
        "walkforward",
        "--panel-root",
        str(root),
        "--first-cutoff",
        "2024-05-10",
        "--weeks",
        "2",
        "--data-version",
        DATA_VERSION,
        "--policies",
        "uniform,recency",
        "--out",
        str(out),
        "--set",
        f"cache.path={(out.parent / 'c.sqlite').as_posix()}",
    ]
    for item in (*SMALL_PROTOCOL_OVERRIDES, *extra):
        args += ["--set", item]
    return args


def test_cli_walkforward(synth_archive_root, tmp_path, capsys, monkeypatch) -> None:
    out = tmp_path / "wf"
    assert main(_cli_args(synth_archive_root, out)) == 0
    assert len(_records(out)) == 4
    assert {r["policy"] for r in _records(out)} == {"uniform", "recency"}
    assert (out / "daily" / "uniform.csv").is_file() and (out / "daily" / "recency.csv").is_file()
    assert "complete" in capsys.readouterr().out
    _no_calibration(monkeypatch)  # the rerun resumes: every step has its record, nothing is replayed
    assert main(_cli_args(synth_archive_root, out)) == 0
    assert len(_records(out)) == 4


def test_cli_walkforward_reserved_exit_code(synth_archive_root, tmp_path, capsys, monkeypatch) -> None:
    _no_calibration(monkeypatch)
    assert main(_cli_args(synth_archive_root, tmp_path / "wf", "walkforward.reserved_start=2024-05-01")) == 2
    assert "reserved" in capsys.readouterr().err.lower()
    assert main(_cli_args(synth_archive_root, tmp_path / "wf2", "walkforward.reserved_start=null")) == 2
    assert "reserved_start" in capsys.readouterr().err


def test_cli_walkforward_reserved_boundary_stop(synth_archive_root, tmp_path, capsys) -> None:
    # Week 1 (activation 2024-05-13..17) runs; week 2's activation reaches reserved_start: a boundary stop.
    out = tmp_path / "wf"
    args = [*_cli_args(synth_archive_root, out, "walkforward.reserved_start=2024-05-20"), "--policies", "uniform"]
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "boundary" in err.lower() and "2024-05-20" in err
    assert "rerun the same command to resume" not in err
    assert len(_records(out)) == 1


def test_cli_walkforward_insufficient_history_exit_code(synth_archive_root, tmp_path, capsys) -> None:
    args = _cli_args(synth_archive_root, tmp_path / "wf")
    args[args.index("--first-cutoff") + 1] = "2024-01-12"
    assert main(args) == 2
    assert "InsufficientHistoryError" in capsys.readouterr().err


def test_cli_walkforward_needs_one_strategy(synth_archive_root, tmp_path) -> None:
    two = "search.strategies=[{id: breakout, base: breakout}, {id: breakout_b, base: breakout}]"
    with pytest.raises(ValueError, match="exactly one"):
        main(_cli_args(synth_archive_root, tmp_path / "wf", two))
