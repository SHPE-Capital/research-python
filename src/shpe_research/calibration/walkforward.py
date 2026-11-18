"""D8 walk-forward: every historical weekend x every policy, each replayed forward on its own account.

Each step calibrates one (cutoff, policy) with D7 (``calibrate_week``) and replays the activation
week with the chosen config on that policy's forward account. Every account starts at
``walkforward.starting_cash``, flat, at the first cutoff and carries its equity across weeks (no
weekly reset). A ``no_deployment`` week is recorded as inactive: zero returns, equity unchanged.

Out-dir layout:

- ``checkpoint.json``: the run header (``CHECKPOINT_SCHEMA``, protocol hash, data version, strategy,
  first cutoff, weeks, policies) plus ``in_progress`` (the step being run, or null), rewritten
  atomically;
- ``records/<cutoff:%Y%m%d>_<policy>.json``: one ``StepRecord`` per finished step, written
  atomically. The existence of a record is the checkpoint;
- ``artifacts/<cutoff:%Y%m%d>/<run_id>/``: the D7 run directories;
- ``events.jsonl``: ``operational_failure`` events (a step that was started and never recorded);
- ``runs.jsonl`` and ``daily/<policy>.csv``: regenerated from ``records/`` after every step.

A crashed step is an operational failure: no record is written, the exception propagates and the
next run with the same arguments reruns that step. Nothing is ever filled with an invented result.
Nothing on or after ``walkforward.reserved_start`` is calibrated or replayed unless the protocol is
frozen and the caller passes its hash; the walk-forward refuses to start while ``reserved_start``
is unset.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np

from shpe_research.calibration.keys import ChunkSpan
from shpe_research.calibration.replay import ChunkReplayer
from shpe_research.calibration.reserved import ReservedGuard
from shpe_research.calibration.strategies import BREAKOUT, StrategySpec
from shpe_research.calibration.weekly import (
    CalibrationInputs,
    CalibrationOutcome,
    calibrate_week,
    default_replayer,
    live_account,
)
from shpe_research.calibration.window import Archive, activation_sessions, iso_monday
from shpe_research.config.momentum import Policy
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.trade.account import simulate_account

CHECKPOINT_SCHEMA = "walkforward_checkpoint.v1"
RECORD_SCHEMA = "walkforward_run_record.v1"

_DAYS_PER_WEEK = 7
_HEADER_FIELDS = ("schema", "protocol_hash", "data_version", "strategy_id", "first_cutoff", "n_weeks", "policies")
_CSV_COLUMNS = ("session", "cutoff", "policy", "active", "config_id", "daily_return", "equity")
# Cache-state artifact fields: excluded from ``artifact_sha256`` so a resumed run hashes like a clean one.
_ARTIFACT_CACHE_STATE = (("runtime",), ("evaluations", "actual"))


@dataclass(frozen=True)
class StepRecord:
    """One finished (cutoff, policy) step of the walk-forward (``walkforward_run_record.v1``).

    Attributes:
        schema: ``RECORD_SCHEMA``.
        run_id: The D7 run id of the calibration.
        cutoff: The decision cutoff session.
        policy: The weighting policy.
        strategy_id: The calibrated strategy.
        activation: The activation week's sessions (replayed forward).
        status: The calibration's decision.
        config_id: The selected 0-B config id, or None for ``no_deployment``.
        artifact_path: The artifact's path relative to the walk-forward out dir (POSIX separators).
        artifact_sha256: SHA-256 of the canonical artifact JSON (sorted keys, compact) with ``runtime``
            and ``evaluations.actual`` removed.
        protocol_hash: Hash of the resolved protocol.
        code_commit: The artifact's code commit.
        data_version: Identity of the data snapshot.
        seeds: The search seeds.
        logical_evaluations: Logical search evaluations over every seed.
        actual_evaluations: Chunks actually replayed (cache state; excluded from identity comparisons).
        search_ess: Realized ESS of the search weights.
        equity_start: The policy's forward equity before the activation week.
        equity_end: ``equity_start`` compounded over ``daily_returns``.
        daily_returns: One forward return per activation session; zeros when inactive.
        active: False for ``no_deployment``.
    """

    schema: str
    run_id: str
    cutoff: date
    policy: Policy
    strategy_id: str
    activation: tuple[date, ...]
    status: Literal["selected", "no_deployment"]
    config_id: str | None
    artifact_path: str
    artifact_sha256: str
    protocol_hash: str
    code_commit: str
    data_version: str
    seeds: tuple[int, ...]
    logical_evaluations: int
    actual_evaluations: int
    search_ess: float
    equity_start: float
    equity_end: float
    daily_returns: tuple[float, ...]
    active: bool

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, ISO dates, lists for tuples; NaN refused). Pure."""
        d = dataclasses.asdict(self)
        d["cutoff"] = self.cutoff.isoformat()
        d["activation"] = [s.isoformat() for s in self.activation]
        d["seeds"] = list(self.seeds)
        d["daily_returns"] = list(self.daily_returns)
        return json.dumps(d, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> StepRecord:
        """Parse ``to_json`` output. Raises ValueError for another schema or missing/unknown fields. Pure."""
        d = json.loads(text)
        if d.get("schema") != RECORD_SCHEMA:
            raise ValueError(f"not a {RECORD_SCHEMA} record: schema {d.get('schema')!r}")
        names = {f.name for f in dataclasses.fields(cls)}
        if set(d) != names:
            raise ValueError(f"record fields differ from {RECORD_SCHEMA}: {sorted(set(d) ^ names)}")
        d["cutoff"] = date.fromisoformat(d["cutoff"])
        d["activation"] = tuple(date.fromisoformat(s) for s in d["activation"])
        d["seeds"] = tuple(int(s) for s in d["seeds"])
        d["daily_returns"] = tuple(float(x) for x in d["daily_returns"])
        return cls(**d)


@dataclass(frozen=True)
class WalkforwardResult:
    """Outcome of one ``run_walkforward`` call.

    Attributes:
        records: Every record in the out dir (this call's and earlier calls'), ordered by cutoff and
            then by ``walkforward.policies`` order.
        stopped: ``"complete"`` when every requested week ran, ``"archive_end"`` when the archive ran
            out of full activation weeks first.
        out_dir: The walk-forward out dir.
    """

    records: tuple[StepRecord, ...]
    stopped: Literal["complete", "archive_end"]
    out_dir: Path


def weekly_cutoffs(archive: Archive, first_cutoff: date, n_weeks: int) -> list[date]:
    """The walk-forward's decision cutoffs: the last archive session of each consecutive ISO week.

    Inputs: the archive (session dates only), the first cutoff (any date in the first week) and the
    number of weeks (>= 1). Output: up to ``n_weeks`` cutoffs, starting with ``first_cutoff``'s ISO
    week; the list stops at the first week with no archive session (past the archive end). Timing:
    each cutoff is the last session of its week known to the archive; no bar is read.
    """
    if n_weeks < 1:
        raise ValueError("n_weeks must be >= 1")
    last_of_week: dict[date, date] = {}
    for s in archive.sessions:
        last_of_week[iso_monday(s)] = s
    out: list[date] = []
    monday = iso_monday(first_cutoff)
    for k in range(n_weeks):
        cutoff = last_of_week.get(monday + timedelta(days=_DAYS_PER_WEEK * k))
        if cutoff is None:
            break
        out.append(cutoff)
    return out


def _equity_path(equity_start: float, returns: Sequence[float]) -> list[float]:
    """Equity after each session, compounding from ``equity_start`` in session order."""
    path, equity = [], float(equity_start)
    for r in returns:
        equity *= 1.0 + float(r)
        path.append(equity)
    return path


def forward_replay(
    inputs: CalibrationInputs, strategy: StrategySpec, outcome: CalibrationOutcome, equity_start: float
) -> tuple[np.ndarray, float]:
    """Replay the selected config over its activation week on the policy's forward account.

    Inputs: the calibration inputs (archive, universe, protocol), the strategy, a ``selected``
    outcome (else ValueError) and the account's equity before the week. The panel is the activation
    week with ``cache.warmup_sessions`` earlier sessions in front and the activation week's universe
    membership; the account is ``live_account(rp, strategy)`` (``cost_bps = selection_bps``) with
    ``starting_cash = equity_start`` and ``search_replay_reference_equity = None`` (equity
    compounds); the kernel is ``compute.kernel``. Output: ``(daily returns, one per activation
    session; equity_end = equity_start * prod(1 + r))``. Never cached. Timing: reads the activation
    week, which is after the cutoff; the config was fixed at the cutoff. The caller applies the
    reserved guard to the activation sessions first.
    """
    selected = outcome.selection.selected
    if outcome.status != "selected" or selected is None:
        raise ValueError(f"forward_replay needs a selected outcome, got {outcome.status!r}")
    p = inputs.rp.protocol
    sessions = outcome.plan.activation_sessions
    week = inputs.universe.week(iso_monday(sessions[0]))
    panel, n_prefix = inputs.archive.build_panel(
        ChunkSpan(sessions), p.cache.warmup_sessions, frozenset(a for a, _ in week.members)
    )
    mask = strategy.signal_mask(panel, selected.config, IndicatorCache(panel))
    account = dataclasses.replace(
        live_account(inputs.rp, strategy), starting_cash=float(equity_start), search_replay_reference_equity=None
    )
    if p.compute.kernel == "fast":
        from shpe_research.trade.account_fast import prepare_panel, simulate_account_fast  # lazy: imports numba

        prepared = prepare_panel(panel, account.ranking, account.liquidity_lookback)
        res = simulate_account_fast(prepared, mask, account, warmup_sessions=n_prefix)
    else:
        res = simulate_account(panel, mask, account, warmup_sessions=n_prefix)
    returns = np.asarray(res.daily_returns, dtype=np.float64)
    if returns.shape != (len(sessions),):
        raise ValueError(f"forward replay returned {returns.shape} daily returns for {len(sessions)} sessions")
    return returns, _equity_path(equity_start, returns.tolist())[-1]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _record_path(out_dir: Path, cutoff: date, policy: str) -> Path:
    return out_dir / "records" / f"{cutoff:%Y%m%d}_{policy}.json"


def _write_record(out_dir: Path, rec: StepRecord) -> None:
    """Write one step's record atomically; its existence marks the step done."""
    _atomic_write(_record_path(out_dir, rec.cutoff, rec.policy), rec.to_json() + "\n")


def _load_records(out_dir: Path) -> dict[tuple[date, str], StepRecord]:
    records: dict[tuple[date, str], StepRecord] = {}
    for path in sorted((out_dir / "records").glob("*.json")):
        rec = StepRecord.from_json(path.read_text(encoding="utf-8"))
        records[(rec.cutoff, rec.policy)] = rec
    return records


def _ordered(records: Mapping[tuple[date, str], StepRecord], policies: Sequence[str]) -> list[StepRecord]:
    rank = {p: i for i, p in enumerate(policies)}
    return sorted(records.values(), key=lambda r: (r.cutoff, rank.get(r.policy, len(rank)), r.policy))


def _write_outputs(out_dir: Path, records: Mapping[tuple[date, str], StepRecord], policies: Sequence[str]) -> None:
    """Regenerate ``runs.jsonl`` and ``daily/<policy>.csv`` from the records."""
    ordered = _ordered(records, policies)
    _atomic_write(out_dir / "runs.jsonl", "".join(f"{r.to_json()}\n" for r in ordered))
    for policy in policies:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(_CSV_COLUMNS)
        for rec in (r for r in ordered if r.policy == policy):
            equity = _equity_path(rec.equity_start, rec.daily_returns)
            for session, ret, eq in zip(rec.activation, rec.daily_returns, equity, strict=True):
                active = "true" if rec.active else "false"
                writer.writerow(
                    [
                        session.isoformat(),
                        rec.cutoff.isoformat(),
                        policy,
                        active,
                        rec.config_id or "",
                        repr(ret),
                        repr(eq),
                    ]
                )
        _atomic_write(out_dir / "daily" / f"{policy}.csv", buf.getvalue())


def _open_checkpoint(out_dir: Path, header: dict[str, Any]) -> dict[str, Any]:
    """Write a new checkpoint, or verify an existing header field by field and return it."""
    path = out_dir / "checkpoint.json"
    if not path.exists():
        checkpoint = {**header, "in_progress": None}
        _write_checkpoint(out_dir, checkpoint)
        return checkpoint
    existing = json.loads(path.read_text(encoding="utf-8"))
    for name in _HEADER_FIELDS:
        if existing.get(name) != header[name]:
            raise ValueError(
                f"checkpoint.json {name} mismatch: existing {existing.get(name)!r} != {header[name]!r}; "
                "resume with the same arguments or use a new out dir"
            )
    return {**header, "in_progress": existing.get("in_progress")}


def _write_checkpoint(out_dir: Path, checkpoint: Mapping[str, Any]) -> None:
    _atomic_write(out_dir / "checkpoint.json", json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")


def _append_event(out_dir: Path, event: Mapping[str, Any]) -> None:
    with (out_dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def _artifact_sha256(outcome: CalibrationOutcome) -> str:
    doc = outcome.artifact.model_dump(mode="json")
    for path in _ARTIFACT_CACHE_STATE:
        node = doc
        for key in path[:-1]:
            node = node[key]
        node.pop(path[-1], None)
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_walkforward(
    inputs: CalibrationInputs,
    first_cutoff: date,
    n_weeks: int,
    out_dir: Path,
    *,
    strategy: StrategySpec = BREAKOUT,
    expected_hash: str | None = None,
    replayer: ChunkReplayer | None = None,
    on_step_done: Callable[[StepRecord], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> WalkforwardResult:
    """Run (or resume) the walk-forward over ``weekly_cutoffs(archive, first_cutoff, n_weeks)``.

    Inputs: the calibration inputs (their ``out_root`` and ``expected_hash`` are replaced per step;
    ``inputs.descriptors`` must not read past the last cutoff), the first cutoff, the number of
    weeks, the out dir, the strategy, the protocol hash that unlocks the reserved period, an
    optional replayer for the calibrations (default ``default_replayer(inputs)``), a callback run
    after each finished step and the wall clock for ``compute.max_wall_hours``.

    For each cutoff, for each policy in ``walkforward.policies``: a step with a record is skipped; a
    step named by the checkpoint's ``in_progress`` gets an ``operational_failure`` event and is rerun;
    the reserved guard is applied to the cutoff and the activation sessions; the run stops with
    ``archive_end`` if the activation week is not fully in the archive; otherwise ``in_progress`` is
    set, the week is calibrated (``calibrate_week``), a ``selected`` week is replayed forward
    (``forward_replay``) from the policy's last ``equity_end`` (``walkforward.starting_cash`` for its
    first week) and a ``no_deployment`` week is recorded inactive with zero returns. Each record is
    written atomically, then ``runs.jsonl`` and the CSVs are regenerated, ``in_progress`` is cleared
    and ``on_step_done`` is called.

    Output: a ``WalkforwardResult``. Raises ReservedPeriodError before anything is written while
    ``reserved_start`` is unset, and before any calibration or replay that would read a session on
    or after it unless the protocol is frozen and ``expected_hash`` equals its hash; ValueError when
    an existing checkpoint header differs (naming the field). IncompleteRunError and every other
    exception propagate with no record for the failed step (operational failure; rerun to resume).
    Timing: each step reads data through its cutoff to decide and only then the activation week.
    """
    rp = inputs.rp
    p = rp.protocol
    guard = ReservedGuard.from_protocol(rp, expected_hash)
    guard.require_set()
    out_dir = Path(out_dir)
    policies: list[Policy] = list(p.walkforward.policies)
    header = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol_hash": rp.hash,
        "data_version": inputs.data_version,
        "strategy_id": strategy.strategy_id,
        "first_cutoff": first_cutoff.isoformat(),
        "n_weeks": int(n_weeks),
        "policies": list(policies),
    }
    cutoffs = weekly_cutoffs(inputs.archive, first_cutoff, n_weeks)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _open_checkpoint(out_dir, header)
    records = _load_records(out_dir)
    rep = replayer if replayer is not None else default_replayer(inputs)
    archived = set(inputs.archive.sessions)
    past_end = False
    step = -1
    for cutoff in cutoffs:
        activation = activation_sessions(cutoff)
        for policy in policies:
            step += 1
            if (cutoff, policy) in records:
                continue
            marker = {"cutoff": cutoff.isoformat(), "policy": policy}
            if checkpoint["in_progress"] == marker:
                _append_event(out_dir, {"kind": "operational_failure", **marker, "detected_at_step": step})
            guard.check_calibration(cutoff)
            guard.check_replay(activation)
            if not archived.issuperset(activation):
                past_end = True
                break
            checkpoint["in_progress"] = marker
            _write_checkpoint(out_dir, checkpoint)
            step_inputs = dataclasses.replace(
                inputs, expected_hash=expected_hash, out_root=out_dir / "artifacts" / f"{cutoff:%Y%m%d}"
            )
            outcome = calibrate_week(step_inputs, cutoff, policy, strategy, replayer=rep, clock=clock)
            prior = [r for r in records.values() if r.policy == policy and r.cutoff < cutoff]
            equity_start = max(prior, key=lambda r: r.cutoff).equity_end if prior else p.walkforward.starting_cash
            if outcome.status == "selected":
                returns, equity_end = forward_replay(step_inputs, strategy, outcome, equity_start)
                daily = tuple(float(x) for x in returns)
            else:
                daily, equity_end = tuple(0.0 for _ in activation), equity_start
            art = outcome.artifact
            rec = StepRecord(
                schema=RECORD_SCHEMA,
                run_id=art.run_id,
                cutoff=cutoff,
                policy=policy,
                strategy_id=strategy.strategy_id,
                activation=activation,
                status=outcome.status,
                config_id=art.selected.config_id if art.selected is not None else None,
                artifact_path=(outcome.run_dir / "weekly_config.json").relative_to(out_dir).as_posix(),
                artifact_sha256=_artifact_sha256(outcome),
                protocol_hash=rp.hash,
                code_commit=art.code_commit,
                data_version=art.data_version,
                seeds=tuple(int(s) for s in art.seeds),
                logical_evaluations=art.evaluations.logical,
                actual_evaluations=art.evaluations.actual,
                search_ess=float(art.ess),
                equity_start=float(equity_start),
                equity_end=float(equity_end),
                daily_returns=daily,
                active=outcome.status == "selected",
            )
            _write_record(out_dir, rec)
            records[(cutoff, policy)] = rec
            _write_outputs(out_dir, records, policies)
            checkpoint["in_progress"] = None
            _write_checkpoint(out_dir, checkpoint)
            if on_step_done is not None:
                on_step_done(rec)
        if past_end:
            break
    if not (out_dir / "runs.jsonl").exists():
        _write_outputs(out_dir, records, policies)
    # archive_end: an activation week past the archive, or fewer archive weeks than requested
    stopped: Literal["complete", "archive_end"] = "archive_end" if past_end or len(cutoffs) < n_weeks else "complete"
    return WalkforwardResult(records=tuple(_ordered(records, policies)), stopped=stopped, out_dir=out_dir)
