"""D9 multi-strategy scheduler: several strategies, one panel load and indicator build per span.

``run_strategies`` builds one ``WeeklyCalibration`` per strategy (same cutoff, policy and window
plan) and hands them all to ``weekly.drive``, the single lockstep driver. Every ``ChunkReplayer.run``
call then carries every strategy's requests, so each span's panel and ``IndicatorCache`` are built
once per call for all strategies. Requests with the same mask and the same replay account resolve to
one cached return path through the D6b mask hash; a strategy with different account settings has a
different context and reuses nothing of the others'. Each strategy's artifact is the one it gets
when run alone (``calibrate_week``), apart from the cache-state fields.

Resume after a kill: the D6 cache is the journal. Every span's paths and aliases are committed
before the next span starts, so a rerun of the same weekend resolves the committed spans as alias
hits and replays only the rest; there is no separate scheduler checkpoint.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from shpe_research.calibration.cache import CacheStats
from shpe_research.calibration.replay import ChunkReplayer, ReplayCounters
from shpe_research.calibration.strategies import StrategySpec
from shpe_research.calibration.weekly import (
    CalibrationInputs,
    CalibrationOutcome,
    WeeklyCalibration,
    default_replayer,
    drive,
)
from shpe_research.config.momentum import Policy


@dataclass(frozen=True)
class SchedulerReport:
    """Work done by one ``run_strategies`` call (cache state; never part of a deterministic output).

    Attributes:
        shared: Replay counters over every strategy (spans opened, panels and ATR built once each).
        by_strategy: The same counters split by ``strategy_id`` (``spans_opened``/``panels_built``
            count the spans each strategy took part in).
        cache: The parent cache's lookups during the call (with ``compute.workers > 1`` the workers'
            lookups are not included; use ``shared``).
    """

    shared: ReplayCounters
    by_strategy: Mapping[str, ReplayCounters]
    cache: CacheStats


def _delta(after: object, before: object) -> dict[str, int]:
    return {f.name: getattr(after, f.name) - getattr(before, f.name) for f in dataclasses.fields(after)}


def run_strategies(
    inputs: CalibrationInputs,
    as_of: date,
    policy: Policy,
    specs: Sequence[StrategySpec],
    *,
    replayer: ChunkReplayer | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, CalibrationOutcome], SchedulerReport]:
    """Calibrate every strategy in ``specs`` at the cutoff for ``as_of`` under one policy, chunk-major.

    Inputs: the calibration inputs, the run date, the weighting policy, the strategies (unique
    ``strategy_id``, else ValueError), an optional replayer (default ``default_replayer(inputs)``)
    and the wall clock for ``compute.max_wall_hours``. Each spec's budget, seeds and account overrides
    apply to its own calibration. Output: outcomes keyed by ``strategy_id`` in ``specs`` order (each
    run directory is ``inputs.out_root / run_id_for(spec, policy, cutoff, rp.hash)``) and the work
    report for this call. Every ``WeeklyCalibration`` is built (cutoff lock, reserved guard, window
    plan, weights) before any replay, so a refused cutoff replays nothing. Raises what
    ``WeeklyCalibration`` and ``drive`` raise (StaleArchiveError, ReservedPeriodError,
    InsufficientHistoryError, IncompleteRunError after every other strategy is finished). Timing:
    reads nothing after the cutoff session.
    """
    ids = [spec.strategy_id for spec in specs]
    dupes = sorted({sid for sid in ids if ids.count(sid) > 1})
    if dupes:
        raise ValueError(f"strategy ids must be unique in one run: {dupes}")
    calibrations = [WeeklyCalibration(inputs, as_of, policy, spec, clock=clock) for spec in specs]
    rep = replayer if replayer is not None else default_replayer(inputs)
    before = dataclasses.replace(rep.counters)
    before_by = {sid: dataclasses.replace(c) for sid, c in rep.by_strategy.items()}
    before_cache = dataclasses.replace(inputs.cache.stats)
    outcomes = drive(calibrations, rep)
    report = SchedulerReport(
        shared=ReplayCounters(**_delta(rep.counters, before)),
        by_strategy={
            sid: ReplayCounters(**_delta(c, before_by.get(sid, ReplayCounters())))
            for sid, c in sorted(rep.by_strategy.items())
            if sid in ids
        },
        cache=CacheStats(**_delta(inputs.cache.stats, before_cache)),
    )
    return {spec.strategy_id: out for spec, out in zip(specs, outcomes, strict=True)}, report
