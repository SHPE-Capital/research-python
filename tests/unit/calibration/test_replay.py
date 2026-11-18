"""Tests for D7 chunk-major replay into the D6/D6b cache and the StrategySpec."""

from __future__ import annotations

import dataclasses
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from shpe_research.calibration.cache import Chunk, ChunkResult, EvalCache, LookaheadError, PathKey, WindowResult
from shpe_research.calibration.keys import (
    CODE_MODULES,
    ChunkSpan,
    account_hash,
    calendar_version,
    chunk_spans,
    code_version,
    content_digest,
    indicator_key,
    mask_hash,
    panel_chunk_id,
)
from shpe_research.calibration.replay import (
    REPLAY_CODE_MODULES,
    ChunkReplayer,
    ReplayRequest,
    ReplayResponse,
    chunk_result,
)
from shpe_research.calibration.strategies import BREAKOUT, StrategySpec
from shpe_research.calibration.window import Archive
from shpe_research.config.momentum import CacheSection, MomentumProtocol
from shpe_research.features.momentum.wilder import ATR_LENGTH, NUMBA_ENV
from shpe_research.strategies.momentum.breakout.grid import DEFAULT_DOMAIN, Config, config_id
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.strategies.momentum.breakout.signals import signal_mask
from shpe_research.trade.account_types import AccountConfig, AccountResult, LedgerEntry, MarketPanel, SignalMask
from tests.unit.calibration.helpers import AllMembersUniverse, strip_cache_state

BUSY_TIMEOUT_S = CacheSection().busy_timeout_s
WARMUP = CacheSection().warmup_sessions
REF_EQUITY = MomentumProtocol().search.reference_equity
DATA_VERSION = "synth-seed7"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")

CONFIGS = (
    Config.make("v1", ["breakout", "rvol"], {"breakout.n": 10, "breakout.b": 0, "rvol.w": 3, "rvol.v": 1}),
    Config.make("v1", ["breakout", "vwap"], {"breakout.n": 20, "breakout.b": 0, "vwap.e": 3}),
    Config.make("v1", ["adx", "breakout"], {"breakout.n": 10, "breakout.b": 0.1, "adx.period": 7, "adx.q": 15}),
    Config.make(
        "v1", ["breakout", "ema"], {"breakout.n": 40, "breakout.b": 0, "ema.fast": 5, "ema.slow": 20, "ema.d": 0}
    ),
    Config.make("v1", ["breakout", "vwap"], {"breakout.n": 10, "breakout.b": 0, "vwap.e": 1}),
)
FIXED_CONFIG = CONFIGS[1]


def _fixed_mask(panel: MarketPanel, config: Config, cache: IndicatorCache) -> SignalMask:
    """Module-level (picklable) signal rule that ignores the config: every config gives one mask."""
    return signal_mask(panel, FIXED_CONFIG, cache)


FIXED = StrategySpec("fixed", DEFAULT_DOMAIN, _fixed_mask)


def _account(**kw) -> AccountConfig:
    return AccountConfig(search_replay_reference_equity=REF_EQUITY, **kw)


@pytest.fixture(scope="module")
def archive(synth_archive_root: Path) -> Archive:
    return Archive.open(synth_archive_root)


def _week_spans(archive: Archive, n: int, first: int = 0) -> list[ChunkSpan]:
    spans = chunk_spans(archive.sessions, "chunk")
    usable = [sp for sp in spans if archive.sessions.index(sp.sessions[0]) >= WARMUP]
    return usable[first : first + n]


def _replayer(
    archive: Archive, universe: AllMembersUniverse, cache_path: Path, *, kernel="fast", workers=1, on_span_done=None
) -> ChunkReplayer:
    return ChunkReplayer(
        archive,
        universe,
        EvalCache(cache_path, busy_timeout_s=BUSY_TIMEOUT_S),
        data_version=DATA_VERSION,
        kernel=kernel,
        warmup_sessions=WARMUP,
        workers=workers,
        threads_per_worker=1,
        on_span_done=on_span_done,
    )


def _requests(configs, strategy: StrategySpec = BREAKOUT, account: AccountConfig | None = None) -> list[ReplayRequest]:
    acct = strategy.account_config(account or _account())
    return [ReplayRequest(strategy=strategy, config=c, account=acct) for c in configs]


def _bytes(w: WindowResult) -> bytes:
    return ChunkResult(w.sessions, w.daily_returns, w.entries, w.ambiguous_bars).to_bytes()


def _all_bytes(responses: list[ReplayResponse]) -> list[bytes]:
    return [_bytes(r.window) for r in responses]


def test_deterministic_and_cached(archive, all_members_universe, tmp_path) -> None:
    spans = _week_spans(archive, 4)
    cutoff = spans[-1].sessions[-1]
    first = _replayer(archive, all_members_universe, tmp_path / "c.sqlite")
    out1 = first.run(_requests(CONFIGS[:3]), spans, cutoff)
    assert first.counters.replays + first.counters.mask_hits == 12
    second = _replayer(archive, all_members_universe, tmp_path / "c.sqlite")
    out2 = second.run(_requests(CONFIGS[:3]), spans, cutoff)
    assert _all_bytes(out1) == _all_bytes(out2)
    assert second.counters.replays == 0
    assert second.counters.alias_hits == 12
    assert second.counters.masks_computed == 0
    for resp in out1:
        assert resp.window.sessions == tuple(s for sp in spans for s in sp.sessions)
        assert resp.replays + resp.mask_hits + resp.alias_hits == 4
    for resp in out2:
        assert (resp.replays, resp.mask_hits, resp.alias_hits) == (0, 0, 4)


def test_chunk_major_shares_panel_and_indicators(archive, all_members_universe, tmp_path) -> None:
    spans = _week_spans(archive, 3)
    rep = _replayer(archive, all_members_universe, tmp_path / "c.sqlite")
    rep.run(_requests(CONFIGS), spans, spans[-1].sessions[-1])
    assert rep.counters.spans_opened == 3
    assert rep.counters.panels_built == 3
    assert rep.counters.atr_builds == 3
    assert rep.counters.masks_computed == 15
    assert rep.counters.replays + rep.counters.mask_hits == 15
    assert rep.counters.indicator_hits > 0
    assert rep.by_strategy["breakout"] == rep.counters


def test_oracle_fast_parity(archive, all_members_universe, tmp_path) -> None:
    spans = _week_spans(archive, 1, first=2)
    cutoff = spans[-1].sessions[-1]
    cfg = CONFIGS[0]
    fast = _replayer(archive, all_members_universe, tmp_path / "f.sqlite", kernel="fast")
    oracle = _replayer(archive, all_members_universe, tmp_path / "o.sqlite", kernel="oracle")
    wf = fast.run(_requests([cfg]), spans, cutoff)[0].window
    wo = oracle.run(_requests([cfg]), spans, cutoff)[0].window
    assert len(wo.entries) > 0
    assert wf.entries == wo.entries
    assert wf.ambiguous_bars == wo.ambiguous_bars
    assert np.allclose(wf.daily_returns, wo.daily_returns, rtol=1e-12, atol=0)


def test_workers_two_equals_serial(archive, all_members_universe, tmp_path) -> None:
    spans = _week_spans(archive, 3)
    cutoff = spans[-1].sessions[-1]
    done: list[ChunkSpan] = []
    serial = _replayer(archive, all_members_universe, tmp_path / "s.sqlite")
    parallel = _replayer(archive, all_members_universe, tmp_path / "p.sqlite", workers=2, on_span_done=done.append)
    out_s = serial.run(_requests(CONFIGS[:2]), spans, cutoff)
    env_before = {k: os.environ.get(k) for k in THREAD_ENV}
    out_p = parallel.run(_requests(CONFIGS[:2]), spans, cutoff)
    assert {k: os.environ.get(k) for k in THREAD_ENV} == env_before  # restored after the pool
    assert _all_bytes(out_s) == _all_bytes(out_p)
    assert parallel.counters == serial.counters
    assert sorted(done, key=lambda sp: sp.sessions) == spans
    # the workers wrote the cache: a serial rerun on it replays nothing
    again = _replayer(archive, all_members_universe, tmp_path / "p.sqlite")
    assert _all_bytes(again.run(_requests(CONFIGS[:2]), spans, cutoff)) == _all_bytes(out_s)
    assert again.counters.replays == 0


def test_span_after_cutoff_raises(archive, all_members_universe, tmp_path) -> None:
    spans = _week_spans(archive, 2)
    cutoff = spans[-1].sessions[-2]  # last span ends after the cutoff
    rep = _replayer(archive, all_members_universe, tmp_path / "c.sqlite")
    with pytest.raises(LookaheadError):
        rep.run(_requests(CONFIGS[:1]), spans, cutoff)
    assert rep.counters.spans_opened == 0
    assert rep.counters.replays == 0
    assert rep.cache.stats.writes == 0


def test_strategy_spec_namespacing_and_overrides() -> None:
    cfg = CONFIGS[0]
    assert BREAKOUT.strategy_id == "breakout"
    assert BREAKOUT.grid_version == DEFAULT_DOMAIN.version
    assert BREAKOUT.namespaced_id(cfg) == f"breakout/v1/{config_id(cfg)}"
    spec = StrategySpec(
        "tight", DEFAULT_DOMAIN, signal_mask, account_overrides=(("exit.initial_stop_atr", 1.0), ("max_positions", 5))
    )
    base = _account()
    acct = spec.account_config(base)
    assert acct.exit.initial_stop_atr == 1.0
    assert acct.max_positions == 5
    assert acct.exit.trail_distance_atr == base.exit.trail_distance_atr
    assert dataclasses.replace(acct, max_positions=base.max_positions, exit=base.exit) == base
    assert BREAKOUT.account_config(base) == base
    # an int override of a float field is stored as float, so the account hash is canonical
    as_int = StrategySpec("x", DEFAULT_DOMAIN, signal_mask, account_overrides=(("exit.initial_stop_atr", 1),))
    as_float = StrategySpec("y", DEFAULT_DOMAIN, signal_mask, account_overrides=(("exit.initial_stop_atr", 1.0),))
    assert account_hash(as_int.account_config(base)) == account_hash(as_float.account_config(base))
    with pytest.raises(ValueError):
        StrategySpec("x", DEFAULT_DOMAIN, signal_mask, account_overrides=(("exit.nope", 1.0),)).account_config(base)
    with pytest.raises(ValueError):
        StrategySpec("x", DEFAULT_DOMAIN, signal_mask, account_overrides=(("nope", 1.0),)).account_config(base)


def test_session_granularity_byte_identical_on_rerun(archive, all_members_universe, tmp_path) -> None:
    week = _week_spans(archive, 1, first=1)[0]
    spans = chunk_spans(week.sessions, "session")
    cutoff = spans[-1].sessions[-1]
    a = _replayer(archive, all_members_universe, tmp_path / "a.sqlite").run(_requests(CONFIGS[:2]), spans, cutoff)
    b = _replayer(archive, all_members_universe, tmp_path / "b.sqlite").run(_requests(CONFIGS[:2]), spans, cutoff)
    assert _all_bytes(a) == _all_bytes(b)
    rerun = _replayer(archive, all_members_universe, tmp_path / "a.sqlite")
    c = rerun.run(_requests(CONFIGS[:2]), spans, cutoff)
    assert _all_bytes(c) == _all_bytes(a)
    assert rerun.counters.replays == 0
    assert all(len(r.window.sessions) == len(week.sessions) for r in a)


def _tiny_panel(n_sessions: int, per_session: int) -> MarketPanel:
    n = n_sessions * per_session
    px = np.ones((1, n), np.float32)
    return MarketPanel(
        asset_ids=("A",),
        open=px,
        high=px,
        low=px,
        close=px,
        volume=px,
        session_index=np.repeat(np.arange(10, 10 + n_sessions, dtype=np.int32), per_session),
        eligible=np.ones((1, n), bool),
    )


def test_chunk_result_maps_entries() -> None:
    panel = _tiny_panel(3, 4)  # one prefix session + two chunk sessions
    sessions = (date(2024, 3, 4), date(2024, 3, 5))
    ledger = (
        LedgerEntry("A", 9, 10, 1, 1.0, 1.0, "stop", 0.0),  # column 9: local session 2 -> offset 1
        LedgerEntry("A", 5, 7, 1, 1.0, 1.0, "unresolved", 0.0),  # column 5: local session 1 -> offset 0
    )
    res = AccountResult(daily_returns=np.array([0.01, -0.02]), ledger=ledger, ambiguous_bars=3, unresolved_positions=1)
    out = chunk_result(res, panel, 1, sessions)
    assert out.sessions == sessions
    assert out.entries == ((0, "A", True), (1, "A", False))
    assert out.ambiguous_bars == 3
    np.testing.assert_array_equal(out.daily_returns, [0.01, -0.02])
    with pytest.raises(ValueError):  # an entry inside the warm-up prefix
        early = (LedgerEntry("A", 1, 2, 1, 1.0, 1.0, "stop", 0.0),)
        chunk_result(AccountResult(np.zeros(2), early), panel, 1, sessions)
    with pytest.raises(ValueError):  # daily returns do not match the sessions
        chunk_result(AccountResult(np.zeros(3)), panel, 1, sessions)
    with pytest.raises(ValueError):  # trades counted but no ledger (fast kernel without return_ledger)
        chunk_result(AccountResult(np.zeros(2), summary={"trades": 2.0}), panel, 1, sessions)


def test_cache_keys_match_assembled_panel(archive, all_members_universe, tmp_path) -> None:
    """(a) Chunk.panel_digest is panel_chunk_id of the assembled panel; (b) account_hash of the replayed account."""
    span = _week_spans(archive, 1, first=1)[0]
    rep = _replayer(archive, all_members_universe, tmp_path / "c.sqlite")
    acct = _account()
    rep.run(_requests([CONFIGS[0]], account=acct), [span], span.sessions[-1])
    ctx = rep.context(acct)
    assert ctx.account_hash == account_hash(acct)
    assert ctx.code_version == code_version(REPLAY_CODE_MODULES)
    assert ctx.warmup_sessions == WARMUP
    assert ctx.universe_version == all_members_universe.version
    assert ctx.data_version == DATA_VERSION
    members = frozenset(a for a, _ in all_members_universe.week(span.sessions[0]).members)
    panel, n_prefix = archive.build_panel(span, WARMUP, members)
    digest = panel_chunk_id(
        data_version=DATA_VERSION,
        universe_version=all_members_universe.version,
        week_start=span.sessions[0],
        asset_ids=panel.asset_ids,
        calendar_version=calendar_version(),
        dtype=panel.close.dtype.name,
        content=content_digest(panel),
    )
    chunk = Chunk(span.sessions, digest)
    mask = signal_mask(panel, CONFIGS[0], IndicatorCache(panel))
    atr_key = indicator_key(digest, WARMUP, "atr", {"length": ATR_LENGTH}, ctx.code_version)
    key = PathKey(ctx, chunk, mask_hash(mask.signal, atr_key))
    stored = rep.cache.get_path(key)
    assert stored is not None
    assert rep.cache.get_alias(ctx, chunk, BREAKOUT.namespaced_id(CONFIGS[0])) == key.mask_hash
    assert n_prefix == WARMUP


def test_context_covers_code_kernel_and_indicator_mode(archive, all_members_universe, tmp_path, monkeypatch) -> None:
    """(c) replay glue in the code version; (e) account kernel and indicator kernel mode in execution_model."""
    for mod in (
        *CODE_MODULES,
        "shpe_research.calibration.replay",
        "shpe_research.calibration.window",
        "shpe_research.calibration.strategies",
        "shpe_research.calibration.cache",
    ):
        assert mod in REPLAY_CODE_MODULES
    assert REPLAY_CODE_MODULES[: len(CODE_MODULES)] == CODE_MODULES
    acct = _account()
    fast = _replayer(archive, all_members_universe, tmp_path / "c.sqlite", kernel="fast").context(acct)
    oracle = _replayer(archive, all_members_universe, tmp_path / "c.sqlite", kernel="oracle").context(acct)
    assert fast.execution_model != oracle.execution_model
    monkeypatch.setenv(NUMBA_ENV, "0")
    numpy_ref = _replayer(archive, all_members_universe, tmp_path / "c.sqlite", kernel="fast").context(acct)
    assert numpy_ref.execution_model != fast.execution_model
    assert numpy_ref.digest() != fast.digest()
    assert _account(cost_bps=20.0) != acct
    assert (
        fast.account_hash
        != _replayer(archive, all_members_universe, tmp_path / "c.sqlite").context(_account(cost_bps=20.0)).account_hash
    )


def test_context_requires_reference_equity(archive, all_members_universe, tmp_path) -> None:
    rep = _replayer(archive, all_members_universe, tmp_path / "c.sqlite")
    with pytest.raises(ValueError):
        rep.context(AccountConfig())


def test_identical_masks_replay_once(archive, all_members_universe, tmp_path) -> None:
    """(d) the replay depends only on the mask (and panel and context): equal masks share one replay."""
    spans = _week_spans(archive, 2)
    rep = _replayer(archive, all_members_universe, tmp_path / "c.sqlite")
    out = rep.run(_requests(CONFIGS[:3], strategy=FIXED), spans, spans[-1].sessions[-1])
    assert rep.counters.replays == 2
    assert rep.counters.mask_hits == 4
    assert [r.replays for r in out] == [2, 0, 0]
    assert [r.mask_hits for r in out] == [0, 2, 2]
    assert _bytes(out[0].window) == _bytes(out[1].window) == _bytes(out[2].window)
    assert set(rep.by_strategy) == {"fixed"}


def test_two_strategies_split_counters(archive, all_members_universe, tmp_path) -> None:
    spans = _week_spans(archive, 2)
    rep = _replayer(archive, all_members_universe, tmp_path / "c.sqlite")
    reqs = _requests(CONFIGS[:2]) + _requests(CONFIGS[:1], strategy=FIXED)
    out = rep.run(reqs, spans, spans[-1].sessions[-1])
    assert len(out) == 3
    b, f = rep.by_strategy["breakout"], rep.by_strategy["fixed"]
    assert b.masks_computed == 4 and f.masks_computed == 2
    assert b.replays + f.replays == rep.counters.replays
    assert b.atr_builds + f.atr_builds == rep.counters.atr_builds == 2
    assert rep.counters.panels_built == 2


def test_on_span_done_called_after_each_span(archive, all_members_universe, tmp_path) -> None:
    spans = _week_spans(archive, 3)
    seen: list[tuple[ChunkSpan, int]] = []
    rep = _replayer(archive, all_members_universe, tmp_path / "c.sqlite", on_span_done=None)
    rep.on_span_done = lambda sp: seen.append((sp, rep.cache.stats.writes))
    rep.run(_requests(CONFIGS[:1]), spans, spans[-1].sessions[-1])
    assert [sp for sp, _ in seen] == spans
    writes = [w for _, w in seen]
    assert writes == sorted(writes) and writes[0] > 0


def test_strip_cache_state() -> None:
    artifact = {
        "selected": {"config_id": "abc"},
        "runtime": {"wall_s": 3.2},
        "evaluations": {"logical": 10, "actual": 4},
        "steps": [{"week": 1, "actual_evaluations": 2, "logical_evaluations": 5}],
        "audit": [{"kind": "evaluation", "cache_hit": True, "score": 1.0}, {"kind": "replay_counters", "replays": 3}],
        "run": {"chunk_replays": 7, "cache_stats": {"hits": 1}, "seed": 1},
    }
    out = strip_cache_state(artifact)
    assert out == {
        "selected": {"config_id": "abc"},
        "evaluations": {"logical": 10},
        "steps": [{"week": 1, "logical_evaluations": 5}],
        "audit": [{"kind": "evaluation", "score": 1.0}],
        "run": {"seed": 1},
    }
    assert artifact["runtime"] == {"wall_s": 3.2}  # input untouched
