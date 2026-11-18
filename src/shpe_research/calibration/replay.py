"""D7 chunk-major replay: every requested config over every planned chunk, through the D6/D6b cache.

For each chunk span the panel (warm-up prefix plus chunk) is built once, its ``Chunk`` key is the
``panel_chunk_id`` of that exact panel, one ``IndicatorCache`` and (fast kernel) one ``prepare_panel``
per distinct ``(ranking, liquidity_lookback)`` are shared by every request, and each request goes
through ``EvalCache.get_window_dedup``: alias hit, stored mask path, or a replay of the mask through
the account engine. The alias -> mask -> replay logic lives only in the cache, so each chunk is
counted once in ``EvalCache.stats``.

The cache context is ``(data_version, universe.version, execution_model, account_hash(account),
code_version(REPLAY_CODE_MODULES), warmup_sessions)``; ``execution_model`` names both the account
kernel and the indicator kernel mode (Numba or the NumPy reference), so the modes never share entries.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from shpe_research.calibration.cache import (
    CacheContext,
    Chunk,
    ChunkResult,
    EvalCache,
    LookaheadError,
    MaskedReplay,
    WindowResult,
)
from shpe_research.calibration.keys import (
    CODE_MODULES,
    ChunkSpan,
    account_hash,
    calendar_version,
    code_version,
    content_digest,
    indicator_key,
    mask_hash,
    panel_chunk_id,
)
from shpe_research.calibration.strategies import StrategySpec
from shpe_research.calibration.window import Archive, UniverseWeek, iso_monday
from shpe_research.features.momentum.wilder import ATR_LENGTH, use_numba
from shpe_research.strategies.momentum.breakout.grid import Config
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.trade.account import simulate_account
from shpe_research.trade.account_types import AccountConfig, AccountResult, MarketPanel, SignalMask

Kernel = Literal["fast", "oracle"]
Outcome = Literal["alias", "mask", "replay"]

# Calibration-side replay glue on top of the engine modules: this module (kernel call, ChunkResult
# construction, entry mapping), window (prefix assembly and truncation), strategies (alias ids and
# account overrides) and cache (ChunkResult normalisation and serialisation).
REPLAY_CODE_MODULES: tuple[str, ...] = CODE_MODULES + (
    "shpe_research.calibration.replay",
    "shpe_research.calibration.window",
    "shpe_research.calibration.strategies",
    "shpe_research.calibration.cache",
)
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
_UNRESOLVED = "unresolved"


class UniverseLike(Protocol):
    """Weekly universe schedule (Task 10's ``UniverseProvider`` satisfies it); must be picklable."""

    version: str

    def week(self, week_start: date) -> UniverseWeek: ...


@dataclass(frozen=True)
class ReplayRequest:
    """One config of one strategy to replay.

    Attributes:
        strategy: The strategy (signal rule and alias namespace).
        config: The candidate config.
        account: The replay account, already strategy-adjusted (``strategy.account_config(base)``).
    """

    strategy: StrategySpec
    config: Config
    account: AccountConfig


@dataclass(frozen=True)
class ReplayResponse:
    """A request's result over the requested spans.

    Attributes:
        window: Return path over the union of the spans' sessions, in order. ``ambiguous_bars`` is the
            total over those spans (not per session; ``slice_sessions`` keeps it unchanged).
        replays: Chunks actually replayed for this request.
        alias_hits: Chunks resolved by the alias table.
        mask_hits: Chunks resolved by an already stored path of the same mask.
    """

    window: WindowResult
    replays: int
    alias_hits: int
    mask_hits: int


@dataclass
class ReplayCounters:
    """Work counters (process-local; cache state, never part of a deterministic output)."""

    spans_opened: int = 0
    panels_built: int = 0
    indicator_hits: int = 0
    indicator_misses: int = 0
    atr_builds: int = 0  # IndicatorCache.misses["atr"] summed per span
    masks_computed: int = 0
    replays: int = 0
    alias_hits: int = 0
    mask_hits: int = 0

    def add(self, other: ReplayCounters) -> None:
        """Add ``other`` into this instance, field by field (bookkeeping only; no timing dependence)."""
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


def chunk_result(res: AccountResult, panel: MarketPanel, n_prefix: int, sessions: tuple[date, ...]) -> ChunkResult:
    """Convert one account replay over a prefix-plus-chunk panel into the chunk's cached result.

    Inputs: the engine result (run with ``warmup_sessions=n_prefix`` and a full ledger), the panel it
    ran on, the prefix length and the chunk's sessions. Output: a ``ChunkResult`` whose entries are
    ``(local session ordinal of entry_minute - n_prefix, asset_id, exit_reason == "unresolved")``,
    sorted. Raises ValueError if the daily returns do not match ``sessions``, an entry falls in the
    prefix or outside the panel, or trades were counted but the ledger is missing (fast kernel without
    ``return_ledger=True``). Pure; the prefix sessions are history only and carry no return.
    """
    sessions = tuple(sessions)
    daily = np.asarray(res.daily_returns, dtype=np.float64)
    if daily.shape != (len(sessions),):
        raise ValueError(f"daily_returns has shape {daily.shape}, expected ({len(sessions)},)")
    if int(res.summary.get("trades", len(res.ledger))) != len(res.ledger):
        raise ValueError("the result counts trades but has no matching ledger; replay with return_ledger=True")
    sidx = np.asarray(panel.session_index)
    local = np.cumsum(np.r_[0, sidx[1:] != sidx[:-1]])
    if int(local[-1]) + 1 != n_prefix + len(sessions):
        raise ValueError(f"panel has {int(local[-1]) + 1} sessions, expected {n_prefix} + {len(sessions)}")
    entries = []
    for row in res.ledger:
        if not 0 <= row.entry_minute < len(local):
            raise ValueError(f"ledger entry_minute {row.entry_minute} outside the panel")
        offset = int(local[row.entry_minute]) - n_prefix
        if offset < 0:
            raise ValueError(f"ledger entry at minute {row.entry_minute} is inside the warm-up prefix")
        entries.append((offset, row.asset_id, row.exit_reason == _UNRESOLVED))
    return ChunkResult(
        sessions=sessions, daily_returns=daily, entries=tuple(sorted(entries)), ambiguous_bars=res.ambiguous_bars
    )


def _join_windows(parts: Sequence[WindowResult]) -> WindowResult:
    """Join per-span windows in order, re-indexing entries and summing ambiguous bars."""
    sessions: list[date] = []
    entries = []
    for part in parts:
        offset = len(sessions)
        sessions.extend(part.sessions)
        entries.extend((offset + pos, asset, unresolved) for pos, asset, unresolved in part.entries)
    return WindowResult(
        sessions=tuple(sessions),
        daily_returns=np.concatenate([p.daily_returns for p in parts]),
        entries=tuple(entries),
        ambiguous_bars=sum(p.ambiguous_bars for p in parts),
    )


def _execution_model(kernel: Kernel) -> str:
    """Account kernel plus indicator kernel mode, e.g. ``"fast+indicators:numba"``."""
    return f"{kernel}+indicators:{'numba' if use_numba() else 'numpy'}"


def _replay_mask(
    kernel: Kernel,
    panel: MarketPanel,
    prepared: Callable[[AccountConfig], Any],
    mask: SignalMask,
    account: AccountConfig,
    n_prefix: int,
    sessions: tuple[date, ...],
) -> ChunkResult:
    """Replay one mask over one panel with one account (all in the cache key); no config input."""
    if kernel == "fast":
        from shpe_research.trade.account_fast import simulate_account_fast  # lazy: imports numba

        res = simulate_account_fast(prepared(account), mask, account, warmup_sessions=n_prefix, return_ledger=True)
    else:
        res = simulate_account(panel, mask, account, warmup_sessions=n_prefix)
    return chunk_result(res, panel, n_prefix, sessions)


@dataclass
class _SpanOutput:
    windows: list[WindowResult]
    outcomes: list[Outcome]
    totals: ReplayCounters
    by_strategy: dict[str, ReplayCounters] = field(default_factory=dict)


@dataclass(frozen=True)
class _Env:
    archive: Archive
    universe: UniverseLike
    cache: EvalCache
    data_version: str
    kernel: Kernel
    warmup_sessions: int
    calendar_version: str
    code_version: str


def _replay_span(
    env: _Env, span: ChunkSpan, requests: Sequence[ReplayRequest], contexts: Sequence[CacheContext], cutoff: date
) -> _SpanOutput:
    """Build ``span``'s panel once and resolve every request on it through ``get_window_dedup``."""
    per: dict[str, ReplayCounters] = {}
    for r in requests:  # every strategy in the request list shares this span's panel
        per.setdefault(r.strategy.strategy_id, ReplayCounters(spans_opened=1, panels_built=1))
    week = env.universe.week(iso_monday(span.sessions[0]))
    panel, n_prefix = env.archive.build_panel(span, env.warmup_sessions, frozenset(a for a, _ in week.members))
    chunk = Chunk(
        sessions=span.sessions,
        panel_digest=panel_chunk_id(
            data_version=env.data_version,
            universe_version=env.universe.version,
            week_start=span.sessions[0],
            asset_ids=panel.asset_ids,
            calendar_version=env.calendar_version,
            dtype=panel.close.dtype.name,
            content=content_digest(panel),
        ),
    )
    atr_key = indicator_key(chunk.panel_digest, env.warmup_sessions, "atr", {"length": ATR_LENGTH}, env.code_version)
    ind = IndicatorCache(panel)
    preps: dict[tuple[str, int], Any] = {}

    def prepared(account: AccountConfig) -> Any:
        key = (account.ranking, account.liquidity_lookback)
        if key not in preps:
            from shpe_research.trade.account_fast import prepare_panel  # lazy: imports numba

            preps[key] = prepare_panel(panel, account.ranking, account.liquidity_lookback)
        return preps[key]

    out = _SpanOutput(windows=[], outcomes=[], totals=ReplayCounters(spans_opened=1, panels_built=1), by_strategy=per)
    stats = env.cache.stats
    for req, ctx in zip(requests, contexts, strict=True):
        counter = per[req.strategy.strategy_id]

        def mask_for(ch: Chunk, req: ReplayRequest = req, counter: ReplayCounters = counter) -> MaskedReplay:
            mask = req.strategy.signal_mask(panel, req.config, ind)
            if mask.atr is not ind.atr14():
                raise ValueError(f"strategy {req.strategy.strategy_id!r} returned an ATR array other than ATR14")
            counter.masks_computed += 1
            account = req.account
            return MaskedReplay(
                mask_hash=mask_hash(mask.signal, atr_key),
                replay=lambda: _replay_mask(env.kernel, panel, prepared, mask, account, n_prefix, ch.sessions),
            )

        before = (stats.alias_hits, stats.mask_hits, stats.replays)
        ind_before = (sum(ind.hits.values()), sum(ind.misses.values()), ind.misses["atr"])
        window = env.cache.get_window_dedup(ctx, req.strategy.namespaced_id(req.config), [chunk], cutoff, mask_for)
        delta = (stats.alias_hits - before[0], stats.mask_hits - before[1], stats.replays - before[2])
        outcome: Outcome = {(1, 0, 0): "alias", (0, 1, 0): "mask", (0, 0, 1): "replay"}[delta]
        counter.alias_hits += delta[0]
        counter.mask_hits += delta[1]
        counter.replays += delta[2]
        counter.indicator_hits += sum(ind.hits.values()) - ind_before[0]
        counter.indicator_misses += sum(ind.misses.values()) - ind_before[1]
        counter.atr_builds += ind.misses["atr"] - ind_before[2]
        out.windows.append(window)
        out.outcomes.append(outcome)
    for c in per.values():
        out.totals.add(dataclasses.replace(c, spans_opened=0, panels_built=0))
    return out


# ------------------------------------------------------------------------------ spawn workers

_WORKER: dict[str, Any] = {}


def _worker_env(
    archive_root: str,
    cache_path: str,
    busy_timeout_s: float,
    universe: UniverseLike,
    data_version: str,
    kernel: Kernel,
    warmup_sessions: int,
    cal_version: str,
    code_ver: str,
) -> _Env:
    """This worker process's own Archive and EvalCache (opened once per process from paths)."""
    key = (archive_root, cache_path)
    if _WORKER.get("key") != key:
        if "cache" in _WORKER:
            _WORKER["cache"].close()
        _WORKER.update(
            key=key, archive=Archive.open(archive_root), cache=EvalCache(cache_path, busy_timeout_s=busy_timeout_s)
        )
    return _Env(
        _WORKER["archive"], universe, _WORKER["cache"], data_version, kernel, warmup_sessions, cal_version, code_ver
    )


def _worker_span(
    env_args: tuple,
    expected_execution_model: str,
    span: ChunkSpan,
    requests: Sequence[ReplayRequest],
    contexts: Sequence[CacheContext],
    cutoff: date,
) -> _SpanOutput:
    """Spawn-worker task: replay one span into the shared cache file and return its results."""
    env = _worker_env(*env_args)
    if _execution_model(env.kernel) != expected_execution_model:
        raise RuntimeError(
            f"worker execution model {_execution_model(env.kernel)!r} != parent {expected_execution_model!r}"
        )
    return _replay_span(env, span, requests, contexts, cutoff)


class ChunkReplayer:
    """Chunk-major replay of many requests over many spans into one ``EvalCache``.

    Attributes:
        cache: The parent's cache (workers open their own on the same file).
        counters: Totals over every ``run`` call of this instance.
        by_strategy: The same counters split by ``strategy_id``.
        on_span_done: Called with each span after its cache writes are committed.
    """

    def __init__(
        self,
        archive: Archive,
        universe: UniverseLike,
        cache: EvalCache,
        *,
        data_version: str,
        kernel: Kernel,
        warmup_sessions: int,
        workers: int,
        threads_per_worker: int,
        on_span_done: Callable[[ChunkSpan], None] | None = None,
    ) -> None:
        """Bind the replayer.

        Inputs: the archive, the weekly universe schedule, the cache, the data snapshot version,
        ``compute.kernel``, ``cache.warmup_sessions``, ``compute.workers``,
        ``compute.threads_per_worker`` and an optional per-span callback. Computes the replay code
        version once (a later source edit needs a new replayer).
        """
        if kernel not in ("fast", "oracle"):
            raise ValueError(f"unknown kernel {kernel!r}; expected 'fast' or 'oracle'")
        if warmup_sessions < 0 or workers < 1 or threads_per_worker < 1:
            raise ValueError("warmup_sessions must be >= 0, workers and threads_per_worker >= 1")
        self.archive = archive
        self.universe = universe
        self.cache = cache
        self.data_version = data_version
        self.kernel: Kernel = kernel
        self.warmup_sessions = warmup_sessions
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self.on_span_done = on_span_done
        self.counters = ReplayCounters()
        self.by_strategy: dict[str, ReplayCounters] = {}
        self._code_version = code_version(REPLAY_CODE_MODULES)
        self._calendar_version = calendar_version()

    def context(self, account: AccountConfig) -> CacheContext:
        """Return the cache context for replays with ``account``.

        Input: the exact (strategy-adjusted) account the kernel is called with. Output:
        ``CacheContext(data_version, universe.version, execution_model, account_hash(account),
        code_version(REPLAY_CODE_MODULES), warmup_sessions)`` where ``execution_model`` is the account
        kernel plus the indicator kernel mode read now from ``SHPE_MOMENTUM_NUMBA``. Raises ValueError
        if ``account.search_replay_reference_equity`` is None: without a fixed reference equity a
        session's return depends on earlier sessions and week chunks cannot be cached.
        """
        if account.search_replay_reference_equity is None:
            raise ValueError(
                "chunk replays need AccountConfig.search_replay_reference_equity (search.reference_equity)"
            )
        return CacheContext(
            data_version=self.data_version,
            universe_version=self.universe.version,
            execution_model=_execution_model(self.kernel),
            account_hash=account_hash(account),
            code_version=self._code_version,
            warmup_sessions=self.warmup_sessions,
        )

    def _env(self) -> _Env:
        return _Env(
            self.archive,
            self.universe,
            self.cache,
            self.data_version,
            self.kernel,
            self.warmup_sessions,
            self._calendar_version,
            self._code_version,
        )

    def run(self, requests: Sequence[ReplayRequest], spans: Sequence[ChunkSpan], cutoff: date) -> list[ReplayResponse]:
        """Replay every request over every span (chunk-major) and return one response per request.

        Inputs: the requests, the spans in window order (strictly increasing, non-overlapping) and the
        decision cutoff session. Output: responses in request order; each window covers the union of
        the spans' sessions. Every chunk path is written to the cache before ``on_span_done(span)``.

        Timing: before any work, LookaheadError is raised if any span ends after ``cutoff``. With
        ``workers > 1`` spans run in spawn-context worker processes, each with its own Archive and
        EvalCache on the same files and ``OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS = threads_per_worker``
        (``self.cache.stats`` then reflects only the parent's lookups; use ``counters``).
        """
        spans = list(spans)
        late = [sp for sp in spans if sp.sessions[-1] > cutoff]
        if late:
            raise LookaheadError(f"span ending {late[0].sessions[-1]} is after the cutoff session {cutoff}")
        for prev, cur in zip(spans, spans[1:], strict=False):
            if not cur.sessions[0] > prev.sessions[-1]:
                raise ValueError(
                    f"spans must be strictly increasing and non-overlapping: {prev.sessions[-1]} then {cur.sessions[0]}"
                )
        requests = list(requests)
        if not requests or not spans:
            raise ValueError("run needs at least one request and one span")
        contexts = [self.context(r.account) for r in requests]
        outputs = (
            self._run_parallel(requests, spans, contexts, cutoff)
            if self.workers > 1
            else (self._run_serial(requests, spans, contexts, cutoff))
        )
        return self._collect(requests, outputs)

    def _run_serial(self, requests, spans, contexts, cutoff) -> list[_SpanOutput]:
        env = self._env()
        outputs = []
        for span in spans:
            outputs.append(_replay_span(env, span, requests, contexts, cutoff))
            if self.on_span_done is not None:
                self.on_span_done(span)
        return outputs

    def _run_parallel(self, requests, spans, contexts, cutoff) -> list[_SpanOutput]:
        env_args = (
            str(self.archive.root),
            str(Path(self.cache.path)),
            self.cache.busy_timeout_s,
            self.universe,
            self.data_version,
            self.kernel,
            self.warmup_sessions,
            self._calendar_version,
            self._code_version,
        )
        expected = _execution_model(self.kernel)
        saved = {k: os.environ.get(k) for k in _THREAD_ENV}
        os.environ.update({k: str(self.threads_per_worker) for k in _THREAD_ENV})
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(self.workers, len(spans)), mp_context=ctx) as pool:
                futures = [
                    pool.submit(_worker_span, env_args, expected, sp, requests, contexts, cutoff) for sp in spans
                ]
                outputs = []
                for span, fut in zip(spans, futures, strict=True):
                    outputs.append(fut.result())
                    if self.on_span_done is not None:
                        self.on_span_done(span)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return outputs

    def _collect(self, requests: list[ReplayRequest], outputs: list[_SpanOutput]) -> list[ReplayResponse]:
        for out in outputs:
            self.counters.add(out.totals)
            for sid, c in out.by_strategy.items():
                self.by_strategy.setdefault(sid, ReplayCounters()).add(c)
        responses = []
        for i in range(len(requests)):
            codes = [out.outcomes[i] for out in outputs]
            responses.append(
                ReplayResponse(
                    window=_join_windows([out.windows[i] for out in outputs]),
                    replays=codes.count("replay"),
                    alias_hits=codes.count("alias"),
                    mask_hits=codes.count("mask"),
                )
            )
        return responses
