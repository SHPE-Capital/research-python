"""D7 weekend calibrator: one cutoff -> one 0-A artifact (packet §11 steps 1-6).

``WeeklyCalibration.__init__`` runs steps 1-4 (lock the cutoff and apply the reserved guard, pick the
activation week's universe, plan the windows, compute the policy weights) and builds one
``SearchRun`` per seed, so no calibration can skip the guard. ``drive`` runs step 5: it steps every
calibration's seeds in lockstep, issuing one ``ChunkReplayer.run`` per round over the shared search
spans, then one combined validation replay over the validation spans. ``finish`` runs step 6:
validate the deduplicated finalists at ``selection_bps`` (and again at ``report_bps`` for the report),
select one, build the artifact and write the run directory.

Search and validation are replayed as separate ``ChunkReplayer.run`` calls (each over its own span
list), so a stage's ``ambiguous_bars`` never mixes the other stage's spans.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np

from shpe_research.calibration.acceptance import (
    Finalist,
    Selection,
    StageMetrics,
    dedupe_finalists,
    search_eligibility,
    select_finalist,
    stage_metrics,
    validation_acceptance,
)
from shpe_research.calibration.artifact import build_artifact, peak_rss_mb, write_run_dir
from shpe_research.calibration.cache import EvalCache
from shpe_research.calibration.descriptor_scaling import RobustScaler, fit_robust_scaler
from shpe_research.calibration.descriptors import DescriptorProvider
from shpe_research.calibration.replay import ChunkReplayer, ReplayCounters, ReplayRequest, ReplayResponse
from shpe_research.calibration.reserved import ReservedGuard
from shpe_research.calibration.search import AUDIT_SCHEMA, EvalOutcome, SearchRun, SeedResult
from shpe_research.calibration.strategies import BREAKOUT, StrategySpec
from shpe_research.calibration.universe import UniverseProvider
from shpe_research.calibration.weights import WeightResult, stage_weights, uniform_weights
from shpe_research.calibration.window import (
    Archive,
    InsufficientHistoryError,
    WindowPlan,
    activation_sessions,
    iso_monday,
    lock_cutoff,
    plan_window,
)
from shpe_research.config.momentum import Policy, ResolvedProtocol
from shpe_research.contracts.weekly_config import WeeklyConfig
from shpe_research.data.calendar import get_calendar
from shpe_research.strategies.momentum.breakout.grid import Config, config_id
from shpe_research.trade.account_types import AccountConfig

logger = logging.getLogger(__name__)

INCOMPLETE = "incomplete_time_limit"
_SECONDS_PER_HOUR = 3600.0


class IncompleteRunError(RuntimeError):
    """A seed hit ``compute.max_wall_hours``; the run directory was written without ``weekly_config.json``."""


class StaleArchiveError(InsufficientHistoryError):
    """The archive does not reach the latest exchange session on or before ``as_of``."""


@dataclass(frozen=True)
class CalibrationInputs:
    """Everything a weekly calibration reads, apart from the run date, policy and strategy.

    Attributes:
        rp: The resolved protocol.
        archive: The A6 chunk archive.
        universe: The weekly universe schedule (picklable).
        descriptors: Market descriptors for the similarity policy.
        cache: The evaluation cache.
        data_version: Identity of the data snapshot (part of every cache key and the artifact).
        out_root: Run directories are written to ``out_root / run_id``.
        expected_hash: Protocol hash that unlocks the reserved period (with ``frozen: true``).
    """

    rp: ResolvedProtocol
    archive: Archive
    universe: UniverseProvider
    descriptors: DescriptorProvider
    cache: EvalCache
    data_version: str
    out_root: Path
    expected_hash: str | None = None


@dataclass(frozen=True)
class CalibrationOutcome:
    """Result of one finished calibration (``status`` mirrors the artifact)."""

    status: Literal["selected", "no_deployment"]
    artifact: WeeklyConfig
    run_dir: Path
    plan: WindowPlan
    selection: Selection
    seed_results: tuple[SeedResult, ...]
    search_weights: WeightResult
    validation_weights: WeightResult


def run_id_for(strategy: StrategySpec, policy: Policy, cutoff: date, protocol_hash: str) -> str:
    """Return ``f"{strategy_id}-{policy}-{cutoff:%Y%m%d}-{protocol_hash[:8]}"``. Pure; no timing dependence."""
    return f"{strategy.strategy_id}-{policy}-{cutoff:%Y%m%d}-{protocol_hash[:8]}"


def replay_account(rp: ResolvedProtocol, strategy: StrategySpec, cost_bps: float) -> AccountConfig:
    """The search/validation replay account at ``cost_bps``.

    Output: ``strategy.account_config(AccountConfig(search_replay_reference_equity=search.reference_equity,
    cost_bps=cost_bps))``; the fixed reference equity makes each session's return independent of the
    others, which is what makes week-chunk caching valid. Pure.
    """
    base = AccountConfig(search_replay_reference_equity=rp.protocol.search.reference_equity, cost_bps=cost_bps)
    return strategy.account_config(base)


def live_account(rp: ResolvedProtocol, strategy: StrategySpec) -> AccountConfig:
    """The account the artifact hands the platform: defaults + strategy overrides, ``cost_bps = selection_bps``.

    ``search_replay_reference_equity`` is None (live equity compounds). TODO(MO-0G-rest): build from the
    protocol ``account`` section once it exists. Pure.
    """
    return strategy.account_config(AccountConfig(cost_bps=rp.protocol.acceptance.selection_bps))


def locked_cutoff(archive: Archive, as_of: date) -> date:
    """``lock_cutoff`` plus the stale-archive check.

    Inputs: the archive and the run date. Output: the cutoff session. Raises StaleArchiveError when the
    archive's last session on or before ``as_of`` is not the exchange calendar's latest session on or
    before ``as_of`` (the archive is missing recent sessions, so the cutoff and its activation week
    would silently be in the past).
    """
    cutoff = lock_cutoff(archive, as_of)
    latest = get_calendar().previous_sessions(as_of + timedelta(days=1), 1)[0].date()
    if cutoff != latest:
        raise StaleArchiveError(
            f"archive's last session on or before {as_of} is {cutoff}, but the calendar's is {latest}; "
            "update the archive before calibrating"
        )
    return cutoff


def default_replayer(inputs: CalibrationInputs) -> ChunkReplayer:
    """The replayer ``calibrate_week`` uses when none is given (kernel, warm-up and workers from the protocol)."""
    p = inputs.rp.protocol
    return ChunkReplayer(
        inputs.archive,
        inputs.universe,
        inputs.cache,
        data_version=inputs.data_version,
        kernel=p.compute.kernel,
        warmup_sessions=p.cache.warmup_sessions,
        workers=p.compute.workers,
        threads_per_worker=p.compute.threads_per_worker,
    )


def _metrics(resp: ReplayResponse, sessions: Sequence[date], weights: np.ndarray, coeff: float) -> StageMetrics:
    window = resp.window.slice_sessions(sessions)
    return stage_metrics(window.daily_returns, window.entries, weights, downside_coeff=coeff)


class WeeklyCalibration:
    """One (cutoff, policy, strategy) calibration, stepped by ``drive``.

    Attributes:
        plan: The window plan at the cutoff.
        search_account: The replay account of every search (and selection-cost validation) replay.
        run_id: ``run_id_for(strategy, policy, plan.cutoff, rp.hash)``.
    """

    def __init__(
        self,
        inputs: CalibrationInputs,
        as_of: date,
        policy: Policy,
        strategy: StrategySpec,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Run steps 1-4 and build the seed searches.

        Inputs: the calibration inputs, the run date, the weighting policy, the strategy and the clock
        used for ``compute.max_wall_hours``. Timing: the cutoff is the last archive session on or before
        ``as_of``; every planned session is on or before it, the activation week after it. Raises
        StaleArchiveError, ReservedPeriodError (reserved period without the frozen unlock) or
        InsufficientHistoryError before any replay.
        """
        self._t0 = time.perf_counter()
        self.inputs, self.policy, self.strategy = inputs, policy, strategy
        rp = inputs.rp
        p = rp.protocol
        # (1) cutoff and reserved guard
        cutoff = locked_cutoff(inputs.archive, as_of)
        self.guard = ReservedGuard.from_protocol(rp, inputs.expected_hash)
        if self.guard.reserved_start is None:
            logger.warning("walkforward.reserved_start is unset; the reserved-period guard cannot refuse anything")
        self.guard.check_calibration(cutoff)
        # (2) the activation week's universe
        self.universe_week = inputs.universe.week(iso_monday(activation_sessions(cutoff)[0]))
        # (3) windows
        self.plan = plan_window(
            inputs.archive,
            as_of,
            validation_sessions=p.validation.sessions,
            min_prior_sessions=max(p.cache.warmup_sessions, p.weights.descriptor_lookback + 1),
            granularity=p.cache.granularity,
        )
        self.guard.check_replay(self.plan.search_sessions + self.plan.validation_sessions)
        # (4) weights (never look at returns)
        self.search_weights, self.validation_weights, self.scaler = self._weights()
        # (5) seed searches
        self.search_account = replay_account(rp, strategy, p.acceptance.selection_bps)
        self.seeds: tuple[int, ...] = tuple(strategy.seeds or p.search.seeds)
        hours = p.compute.max_wall_hours
        self._runs = [
            SearchRun(
                strategy.domain,
                p.search,
                p.surrogate,
                seed,
                budget=strategy.budget or p.search.budget,
                max_wall_seconds=None if hours is None else hours * _SECONDS_PER_HOUR,
                clock=clock,
            )
            for seed in self.seeds
        ]
        self.run_id = run_id_for(strategy, policy, self.plan.cutoff, rp.hash)
        self._pending: dict[int, list[Config]] = {}
        self._search_reasons: dict[str, tuple[str, ...]] = {}  # config_id -> failed search checks
        self._actual = 0
        self._finalists: list[tuple[Config, int, float]] | None = None

    def _weights(self) -> tuple[WeightResult, WeightResult, RobustScaler | None]:
        """Search and validation weights; similarity scaling is fitted on search rows only."""
        cfg = self.inputs.rp.protocol.weights
        search, validation = self.plan.search_sessions, self.plan.validation_sessions
        if self.policy != "similarity":
            return stage_weights(self.policy, len(search), cfg), stage_weights(self.policy, len(validation), cfg), None
        desc = self.inputs.descriptors
        rows_s = np.asarray(desc.rows(search), dtype=np.float64)
        if not np.all(np.isfinite(rows_s)):
            logger.warning("non-finite search descriptors; similarity weights fall back to uniform")

            def fallback(n: int) -> WeightResult:
                return dataclasses.replace(
                    uniform_weights(n, cfg),
                    policy_requested="similarity",
                    target_reached=False,
                    fallback_reason="non_finite_descriptors",
                )

            return fallback(len(search)), fallback(len(validation)), None
        scaler = fit_robust_scaler(rows_s, iqr_floor=cfg.iqr_floor)
        current = desc.current(self.plan.cutoff)
        rows_v = desc.rows(validation)
        w_s = stage_weights("similarity", len(search), cfg, stage_rows=rows_s, current=current, scaler=scaler)
        w_v = stage_weights("similarity", len(validation), cfg, stage_rows=rows_v, current=current, scaler=scaler)
        return w_s, w_v, scaler

    @property
    def done(self) -> bool:
        """True once every seed search has stopped."""
        return all(r.done for r in self._runs)

    @property
    def incomplete(self) -> bool:
        """True if any seed stopped on ``compute.max_wall_hours``."""
        return any(r.status == INCOMPLETE for r in self._runs)

    def ask(self) -> list[Config]:
        """The union of the unfinished seeds' next batches, deduplicated and sorted by config_id.

        Output: ``[]`` once every seed has stopped. Raises RuntimeError if the previous batch has not
        been told. Timing: uses only search outcomes already told.
        """
        if self._pending:
            raise RuntimeError("ask called before the previous batch was told")
        union: dict[str, Config] = {}
        for i, run in enumerate(self._runs):
            if run.done:
                continue
            batch = run.ask()
            if batch:
                self._pending[i] = batch
                union.update((config_id(c), c) for c in batch)
        return [union[k] for k in sorted(union)]

    def tell(self, responses: Mapping[str, ReplayResponse]) -> None:
        """Score the last ``ask`` and tell every seed its outcomes.

        Input: one search-span response per asked config_id. Each window is sliced to the search
        sessions and scored with the search weights (``stage_metrics``); ``search_eligibility`` gives
        feasibility. A non-finite J (invalid accounting) scores ``-inf``. ``cache_hit`` = no chunk was
        replayed. Timing: search-window data only (all on or before the cutoff).
        """
        if not self._pending:
            raise RuntimeError("tell called without a pending ask")
        asked = sorted({config_id(c) for batch in self._pending.values() for c in batch})
        missing = [cid for cid in asked if cid not in responses]
        if missing:
            raise ValueError(f"missing responses for {len(missing)} asked configs, e.g. {missing[0]}")
        p = self.inputs.rp.protocol
        outcomes: dict[str, EvalOutcome] = {}
        for cid in asked:
            resp = responses[cid]
            m = _metrics(resp, self.plan.search_sessions, self.search_weights.weights, p.score.downside_coeff)
            verdict = search_eligibility(m, p.acceptance)
            score = m.j if math.isfinite(m.j) else -math.inf
            outcomes[cid] = EvalOutcome(cid, score, verdict.passed, verdict.reasons, cache_hit=resp.replays == 0)
            self._search_reasons[cid] = verdict.reasons
            self._actual += resp.replays
        pending, self._pending = self._pending, {}
        for i, batch in pending.items():
            self._runs[i].tell([outcomes[config_id(c)] for c in batch])

    def _costs(self) -> list[float]:
        acc = self.inputs.rp.protocol.acceptance
        return sorted({acc.selection_bps, acc.report_bps})

    def validation_requests(self) -> list[ReplayRequest]:
        """Deduplicated finalists x {selection_bps, report_bps} replay requests (``[]`` if incomplete).

        Finalists: ``dedupe_finalists`` over each seed's finalists (seeds in order), capped at
        ``acceptance.max_finalists``. Raises RuntimeError before the search is done. Timing: the
        finalists depend on search-window scores only.
        """
        if not self.done:
            raise RuntimeError("validation_requests called before the search is done")
        if self.incomplete:
            return []
        if self._finalists is None:
            per_seed = [[(s.config, r.seed, s.score) for s in r.finalists] for r in (x.result() for x in self._runs)]
            self._finalists = dedupe_finalists(per_seed, self.inputs.rp.protocol.acceptance.max_finalists)
        rp = self.inputs.rp
        return [
            ReplayRequest(self.strategy, cfg, replay_account(rp, self.strategy, bps))
            for cfg, _, _ in self._finalists
            for bps in self._costs()
        ]

    def _audit_lines(self, results: Sequence[SeedResult], counters: Mapping[str, Any] | None) -> list[str]:
        tag = {"policy": self.policy, "strategy_id": self.strategy.strategy_id, "cutoff": self.plan.cutoff.isoformat()}
        lines = [json.dumps({**json.loads(rec.to_json()), **tag}, sort_keys=True) for r in results for rec in r.audit]
        counters = counters or {}
        lines.append(
            json.dumps(
                {
                    "schema": AUDIT_SCHEMA,
                    "kind": "replay_counters",
                    **tag,
                    "shared": counters.get("shared", dataclasses.asdict(ReplayCounters())),
                    "by_strategy": counters.get("by_strategy", {}),
                    "cache_stats": dataclasses.asdict(self.inputs.cache.stats),
                },
                sort_keys=True,
            )
        )
        return lines

    def _run_json(self, status: str) -> dict[str, Any]:
        fallbacks = {
            stage: w.fallback_reason
            for stage, w in (("search", self.search_weights), ("validation", self.validation_weights))
            if w.fallback_reason is not None
        }
        return {
            "run_id": self.run_id,
            "status": status,
            "cutoff": self.plan.cutoff.isoformat(),
            "policy": self.policy,
            "strategy_id": self.strategy.strategy_id,
            "search_ess": self.search_weights.ess,
            "validation_ess": self.validation_weights.ess,
            "search_weight_parameter": self.search_weights.parameter,
            "validation_weight_parameter": self.validation_weights.parameter,
            "weight_fallbacks": fallbacks,
            "scaler_medians": None if self.scaler is None else self.scaler.medians.tolist(),
            "scaler_iqrs": None if self.scaler is None else self.scaler.iqrs.tolist(),
            "chunk_replays": self._actual,
            "cache_stats": dataclasses.asdict(self.inputs.cache.stats),
            "overrides": list(self.inputs.rp.overrides),
        }

    def finish(
        self,
        responses: Mapping[tuple[str, float], ReplayResponse],
        *,
        counters: Mapping[str, Any] | None = None,
    ) -> CalibrationOutcome:
        """Run step 6 and write the run directory.

        Inputs: validation-span responses keyed ``(config_id, cost_bps)`` for every
        ``validation_requests()`` item, and optionally the replay counters for the audit's
        ``replay_counters`` line (``{"shared": ..., "by_strategy": ...}``). Output: the outcome.
        If any seed ended ``incomplete_time_limit`` the run directory is written without
        ``weekly_config.json`` and IncompleteRunError is raised. Timing: validation-window data only.
        """
        if not self.done:
            raise RuntimeError("finish called before the search is done")
        rp = self.inputs.rp
        p = rp.protocol
        results = tuple(r.result() for r in self._runs)
        run_dir = self.inputs.out_root / self.run_id
        if self.incomplete:
            selection = Selection("no_deployment", None, (), (INCOMPLETE,))
            write_run_dir(
                run_dir,
                None,
                audit=self._audit_lines(results, counters),
                finalists=selection,
                rp=rp,
                extra=self._run_json(INCOMPLETE),
            )
            raise IncompleteRunError(f"{self.run_id}: a seed hit compute.max_wall_hours; see {run_dir}")
        if self._finalists is None:
            self.validation_requests()
        assert self._finalists is not None
        sel_bps, rep_bps = p.acceptance.selection_bps, p.acceptance.report_bps
        coeff, weights = p.score.downside_coeff, self.validation_weights.weights
        sessions = self.plan.validation_sessions
        finalists = []
        for cfg, seed, search_j in self._finalists:
            cid = config_id(cfg)
            m_sel = _metrics(responses[(cid, sel_bps)], sessions, weights, coeff)
            m_rep = _metrics(responses[(cid, rep_bps)], sessions, weights, coeff)
            verdict = validation_acceptance(m_sel, p.acceptance)
            finalists.append(Finalist(cfg, cid, seed, search_j, m_sel, m_rep, verdict))
        self._actual += sum(
            responses[(config_id(cfg), bps)].replays for cfg, _, _ in self._finalists for bps in self._costs()
        )
        selection = select_finalist(finalists, p.acceptance)
        counts = Counter(name for reasons in self._search_reasons.values() for name in reasons)
        artifact = build_artifact(
            run_id=self.run_id,
            rp=rp,
            data_version=self.inputs.data_version,
            policy=self.policy,
            plan=self.plan,
            universe=self.universe_week,
            selection=selection,
            search_reasons=sorted(f"search:{name} x{n}" for name, n in counts.items()),
            search_ess=self.search_weights.ess,
            logical=sum(r.logical_evaluations for r in results),
            actual=self._actual,
            seeds=self.seeds,
            live_account=live_account(rp, self.strategy),
            runtime_s=time.perf_counter() - self._t0,
            peak_mb=peak_rss_mb(),
        )
        write_run_dir(
            run_dir,
            artifact,
            audit=self._audit_lines(results, counters),
            finalists=selection,
            rp=rp,
            extra=self._run_json(selection.status),
        )
        return CalibrationOutcome(
            status=selection.status,
            artifact=artifact,
            run_dir=run_dir,
            plan=self.plan,
            selection=selection,
            seed_results=results,
            search_weights=self.search_weights,
            validation_weights=self.validation_weights,
        )


def _snapshot(replayer: ChunkReplayer) -> tuple[ReplayCounters, dict[str, ReplayCounters]]:
    return dataclasses.replace(replayer.counters), {k: dataclasses.replace(v) for k, v in replayer.by_strategy.items()}


def _delta(after: ReplayCounters, before: ReplayCounters | None) -> dict[str, int]:
    a = dataclasses.asdict(after)
    b = dataclasses.asdict(before) if before is not None else {}
    return {k: v - b.get(k, 0) for k, v in a.items()}


def _batched_run(
    replayer: ChunkReplayer, requests: Mapping[tuple, ReplayRequest], spans, cutoff: date
) -> dict[tuple, ReplayResponse]:
    if not requests:
        return {}
    keys = list(requests)
    return dict(zip(keys, replayer.run([requests[k] for k in keys], spans, cutoff), strict=True))


def _add_request(requests: dict[tuple, ReplayRequest], key: tuple, req: ReplayRequest) -> None:
    prev = requests.setdefault(key, req)
    if prev.account != req.account:
        raise ValueError(f"calibrations disagree on the replay account for {key}")


def drive(calibrations: Sequence[WeeklyCalibration], replayer: ChunkReplayer) -> list[CalibrationOutcome]:
    """Step every calibration to completion in lockstep (step 5) and finish each (step 6).

    Inputs: calibrations sharing one cutoff and window plan (else ValueError) and the replayer. Each
    round collects every unfinished calibration's ``ask()``, issues one ``replayer.run`` over the
    shared ``plan.search_spans`` with every ``(strategy_id, config_id)`` request (deduplicated; the
    same request asked by two calibrations is replayed once and its response routed to both) and
    calls each ``tell``. Then one combined ``replayer.run`` over ``plan.validation_spans`` serves every
    ``validation_requests()``, and each calibration's ``finish`` writes its run directory. If any
    calibration is incomplete, every other one is still finished and the first IncompleteRunError
    is raised at the end. Output: outcomes in input order. Timing: no span ends after the cutoff.
    """
    if not calibrations:
        return []
    plan = calibrations[0].plan
    for cal in calibrations[1:]:
        if cal.plan.cutoff != plan.cutoff:
            raise ValueError(f"calibrations in one drive must share the cutoff: {cal.plan.cutoff} != {plan.cutoff}")
        if cal.plan != plan:
            raise ValueError("calibrations in one drive must share the window plan")
    before, before_by = _snapshot(replayer)
    while not all(cal.done for cal in calibrations):
        asks = [(cal, cal.ask()) for cal in calibrations if not cal.done]
        search: dict[tuple, ReplayRequest] = {}
        for cal, configs in asks:
            for cfg in configs:
                req = ReplayRequest(cal.strategy, cfg, cal.search_account)
                _add_request(search, (cal.strategy.strategy_id, config_id(cfg)), req)
        responses = _batched_run(replayer, search, plan.search_spans, plan.cutoff)
        for cal, configs in asks:
            if configs:
                sid = cal.strategy.strategy_id
                cal.tell({config_id(c): responses[(sid, config_id(c))] for c in configs})
    validation: dict[tuple, ReplayRequest] = {}
    owned: list[list[tuple]] = []
    for cal in calibrations:
        keys = []
        for req in cal.validation_requests():
            key = (cal.strategy.strategy_id, config_id(req.config), req.account.cost_bps)
            _add_request(validation, key, req)
            keys.append(key)
        owned.append(keys)
    v_responses = _batched_run(replayer, validation, plan.validation_spans, plan.cutoff)
    after, after_by = _snapshot(replayer)
    counters = {
        "shared": _delta(after, before),
        "by_strategy": {sid: _delta(c, before_by.get(sid)) for sid, c in sorted(after_by.items())},
    }
    outcomes: list[CalibrationOutcome] = []
    errors: list[IncompleteRunError] = []
    for cal, keys in zip(calibrations, owned, strict=True):
        try:
            outcomes.append(cal.finish({(k[1], k[2]): v_responses[k] for k in keys}, counters=counters))
        except IncompleteRunError as exc:
            errors.append(exc)
    if errors:
        raise errors[0]
    return outcomes


def calibrate_week(
    inputs: CalibrationInputs,
    as_of: date,
    policy: Policy,
    strategy: StrategySpec = BREAKOUT,
    *,
    replayer: ChunkReplayer | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CalibrationOutcome:
    """Calibrate one strategy under one policy at the cutoff for ``as_of``: ``drive([WeeklyCalibration(...)])[0]``.

    Inputs: the calibration inputs, the run date, the policy, the strategy, an optional replayer
    (default ``default_replayer(inputs)``) and the wall clock for ``compute.max_wall_hours``. Output:
    the outcome; the run directory is ``inputs.out_root / run_id``. The reserved guard is enforced
    whenever ``walkforward.reserved_start`` is set (a WARNING is logged while it is unset). Raises
    StaleArchiveError, ReservedPeriodError, InsufficientHistoryError or IncompleteRunError. Timing:
    reads nothing after the cutoff session.
    """
    cal = WeeklyCalibration(inputs, as_of, policy, strategy, clock=clock)
    return drive([cal], replayer if replayer is not None else default_replayer(inputs))[0]
