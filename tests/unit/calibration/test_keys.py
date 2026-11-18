"""Tests for D6b: mask-hash dedup, the alias table and the cache-key spec."""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import zlib
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shpe_research.calibration.cache import (
    CacheContext,
    CacheStats,
    Chunk,
    ChunkResult,
    EvalCache,
    LookaheadError,
    MaskedReplay,
    PathKey,
)
from shpe_research.calibration.keys import (
    CODE_MODULES,
    ChunkSpan,
    DedupeReport,
    account_hash,
    calendar_version,
    chunk_spans,
    code_version,
    content_digest,
    dedupe_report,
    indicator_key,
    mask_hash,
    panel_chunk_id,
)
from shpe_research.config.momentum import CacheSection
from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import generate_synthetic_panel, open_panel_chunk
from shpe_research.features.momentum.wilder import ATR_LENGTH
from shpe_research.strategies.momentum.breakout.grid import config_id, enumerate_configs
from shpe_research.strategies.momentum.breakout.indicator_cache import IndicatorCache
from shpe_research.strategies.momentum.breakout.signals import signal_mask
from shpe_research.trade.account_types import AccountConfig, MarketPanel

BUSY_TIMEOUT_S = CacheSection().busy_timeout_s
WARMUP = CacheSection().warmup_sessions
FIRST_MONDAY = date(2024, 1, 1)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")


# ---------------------------------------------------------------------------
# helpers


def _context(**overrides) -> CacheContext:
    base = dict(
        data_version="data-v1",
        universe_version="univ-v1",
        execution_model="fast",
        account_hash=account_hash(AccountConfig()),
        code_version="code-1",
        warmup_sessions=WARMUP,
    )
    base.update(overrides)
    return CacheContext(**base)


def _week(i: int) -> Chunk:
    monday = FIRST_MONDAY + timedelta(weeks=i)
    return Chunk(sessions=tuple(monday + timedelta(days=d) for d in range(5)), panel_digest=f"panel-{i}")


def _weeks(n: int) -> list[Chunk]:
    return [_week(i) for i in range(n)]


def _fake_result(chunk: Chunk, salt: str) -> ChunkResult:
    seed = zlib.crc32(f"{salt}|{chunk.sessions}|{chunk.panel_digest}".encode())
    rng = np.random.default_rng(seed)
    n = len(chunk.sessions)
    entries = tuple(sorted((int(rng.integers(0, n)), f"A{int(rng.integers(0, 9))}", False) for _ in range(2)))
    return ChunkResult(
        sessions=chunk.sessions, daily_returns=rng.normal(0.0, 0.01, size=n), entries=entries, ambiguous_bars=0
    )


def _atr_key(chunk: Chunk, context: CacheContext) -> str:
    return indicator_key(
        chunk.panel_digest, context.warmup_sessions, "atr", {"length": ATR_LENGTH}, context.code_version
    )


def _signal(chunk: Chunk) -> np.ndarray:
    """A deterministic fake signal mask for ``chunk`` (the same for every alias)."""
    rng = np.random.default_rng(zlib.crc32(chunk.panel_digest.encode()))
    return rng.random((3, 7)) < 0.2


class MaskFor:
    """A ``mask_for`` callback: fixed mask per chunk, hash keyed on an atr key; counts replays."""

    def __init__(self, context: CacheContext, atr_salt: str = "") -> None:
        self.context = context
        self.atr_salt = atr_salt
        self.mask_calls: list[Chunk] = []
        self.replays: list[Chunk] = []

    def __call__(self, chunk: Chunk) -> MaskedReplay:
        self.mask_calls.append(chunk)
        h = mask_hash(_signal(chunk), _atr_key(chunk, self.context) + self.atr_salt)

        def replay() -> ChunkResult:
            self.replays.append(chunk)
            return _fake_result(chunk, h)  # the path depends only on the mask (and the chunk)

        return MaskedReplay(mask_hash=h, replay=replay)


@pytest.fixture
def cache(tmp_path: Path):
    c = EvalCache(tmp_path / "eval.sqlite", busy_timeout_s=BUSY_TIMEOUT_S)
    yield c
    c.close()


def _sessions(a: str, b: str) -> pd.DatetimeIndex:
    return get_calendar().sessions(a, b)


def _copy_panel(panel: MarketPanel, **arrays: np.ndarray) -> MarketPanel:
    fields = {n: np.array(getattr(panel, n)) for n in ("open", "high", "low", "close", "volume", "eligible")}
    fields.update(arrays)
    return MarketPanel(asset_ids=panel.asset_ids, session_index=np.array(panel.session_index), **fields)


def _panel_id(panel: MarketPanel, **overrides) -> str:
    base = dict(
        data_version="data-v1",
        universe_version="univ-v1",
        week_start=date(2024, 3, 4),
        asset_ids=panel.asset_ids,
        calendar_version=calendar_version(),
        dtype="float32",
        content=content_digest(panel),
    )
    base.update(overrides)
    return panel_chunk_id(**base)


# ---------------------------------------------------------------------------
# mask hash and dedup


def test_mask_hash_inputs():
    rng = np.random.default_rng(3)
    sig = rng.random((2, 5)) < 0.5
    h = mask_hash(sig, "atr-a")
    assert HEX64.match(h)
    assert mask_hash(sig.copy(), "atr-a") == h
    flipped = sig.copy()
    flipped[1, 3] = ~flipped[1, 3]
    assert mask_hash(flipped, "atr-a") != h
    assert mask_hash(sig, "atr-b") != h
    assert mask_hash(sig.reshape(5, 2), "atr-a") != h
    assert mask_hash(np.asfortranarray(sig), "atr-a") == h  # C-order content, not memory layout
    with pytest.raises(ValueError):
        mask_hash(sig.astype(np.int8), "atr-a")


def test_identical_masks_replay_once_byte_identical(cache):
    ctx = _context()
    chunks = _weeks(4)
    mask_for = MaskFor(ctx)
    win_a = cache.get_window_dedup(ctx, "momentum.breakout:cfg-a", chunks, chunks[-1].end, mask_for)
    win_b = cache.get_window_dedup(ctx, "momentum.breakout:cfg-b", chunks, chunks[-1].end, mask_for)

    assert len(mask_for.replays) == 4
    assert win_a.daily_returns.tobytes() == win_b.daily_returns.tobytes()
    assert win_a.entries == win_b.entries
    assert win_a.sessions == win_b.sessions == tuple(s for c in chunks for s in c.sessions)
    assert cache.stats.mask_hits == 4
    assert cache.stats.replays == 4
    assert cache.stats.alias_hits == 0
    assert cache.stats.dedupe_rate == 0.5

    rows = cache.alias_rows(ctx)
    assert len(rows) == 8
    assert rows == sorted(rows)
    by_chunk: dict[tuple[date, date], dict[str, str]] = {}
    for start, end, alias_id, h in rows:
        by_chunk.setdefault((start, end), {})[alias_id] = h
    assert sorted(by_chunk) == [(c.start, c.end) for c in chunks]
    for mapping in by_chunk.values():
        assert set(mapping) == {"momentum.breakout:cfg-a", "momentum.breakout:cfg-b"}
        assert len(set(mapping.values())) == 1

    # Third pass: aliases resolve directly; no mask is built.
    calls_before = len(mask_for.mask_calls)
    win_a2 = cache.get_window_dedup(ctx, "momentum.breakout:cfg-a", chunks, chunks[-1].end, mask_for)
    assert len(mask_for.mask_calls) == calls_before
    assert cache.stats.alias_hits == 4
    assert win_a2.daily_returns.tobytes() == win_a.daily_returns.tobytes()


def test_alias_lookup_and_path_round_trip(cache):
    ctx = _context()
    chunk = _week(0)
    assert cache.get_alias(ctx, chunk, "ns:cfg") is None
    cache.put_alias(ctx, chunk, "ns:cfg", "h1")
    cache.put_alias(ctx, chunk, "ns:cfg", "h1")  # idempotent
    assert cache.get_alias(ctx, chunk, "ns:cfg") == "h1"
    assert cache.get_alias(_context(code_version="code-2"), chunk, "ns:cfg") is None
    with pytest.raises(ValueError, match="alias"):
        cache.put_alias(ctx, chunk, "ns:cfg", "h2")  # one alias cannot map to two masks

    key = PathKey(context=ctx, chunk=chunk, mask_hash="h1")
    assert key.digest().startswith("p:") and HEX64.match(key.digest()[2:])
    assert cache.get_path(key) is None
    result = _fake_result(chunk, "x")
    cache.put_path(key, result)
    got = cache.get_path(key)
    assert got is not None and got.to_bytes() == result.to_bytes()


@pytest.mark.parametrize("change", ["account", "universe", "code", "atr"])
def test_miss_on_account_universe_atr_code_change(cache, change):
    ctx = _context()
    chunks = _weeks(2)
    first = MaskFor(ctx)
    cache.get_window_dedup(ctx, "ns:cfg", chunks, chunks[-1].end, first)
    assert len(first.replays) == 2

    salt, alias = "", "ns:cfg"
    if change == "account":
        ctx2 = replace(ctx, account_hash=account_hash(replace(AccountConfig(), cost_bps=20.0)))
    elif change == "universe":
        ctx2 = replace(ctx, universe_version="univ-v2")
    elif change == "code":
        ctx2 = replace(ctx, code_version="code-2")
    else:
        # Same signal bits under a different ATR key. Every real ATR-key input (panel digest,
        # warm-up, code version) is also in the context or chunk, so a changed ATR key in the same
        # context and chunk only arises for another config: model it with another alias id.
        ctx2, salt, alias = ctx, "|other-atr", "ns:cfg-other-atr"
    second = MaskFor(ctx2, atr_salt=salt)
    cache.get_window_dedup(ctx2, alias, chunks, chunks[-1].end, second)
    assert len(second.replays) == 2

    # Unchanged: no replay.
    third = MaskFor(ctx)
    cache.get_window_dedup(ctx, "ns:other", chunks, chunks[-1].end, third)
    assert third.replays == []


def test_warmup_in_key(cache):
    ctx = _context()
    ctx2 = replace(ctx, warmup_sessions=ctx.warmup_sessions + 1)
    chunk = _week(0)
    assert indicator_key("pid", 20, "atr", {"length": ATR_LENGTH}, "c") != indicator_key(
        "pid", 21, "atr", {"length": ATR_LENGTH}, "c"
    )
    # Same mask hash, different warm-up length in the context: a different path key.
    assert PathKey(ctx, chunk, "h").digest() != PathKey(ctx2, chunk, "h").digest()

    calls: list[int] = []

    def fixed_mask(c: Chunk) -> MaskedReplay:
        def replay() -> ChunkResult:
            calls.append(1)
            return _fake_result(c, "h")

        return MaskedReplay(mask_hash="h", replay=replay)

    cache.get_window_dedup(ctx, "ns:cfg", [chunk], chunk.end, fixed_mask)
    cache.get_window_dedup(ctx2, "ns:cfg", [chunk], chunk.end, fixed_mask)
    assert len(calls) == 2


def test_tamper_one_bar_misses_everywhere(tmp_path):
    (path,) = generate_synthetic_panel(tmp_path / "panel", 6, _sessions("2024-03-04", "2024-03-08"), seed=1)
    panel = _copy_panel(open_panel_chunk(path))
    close = np.array(panel.close)
    col = int(np.flatnonzero(np.isfinite(close[2]))[100])
    close[2, col] *= np.float32(1.01)
    tampered = _copy_panel(panel, close=close)

    assert content_digest(panel) == content_digest(_copy_panel(panel))
    assert content_digest(tampered) != content_digest(panel)
    pid, pid_t = _panel_id(panel), _panel_id(tampered)
    assert pid != pid_t
    atr, atr_t = (indicator_key(p, WARMUP, "atr", {"length": ATR_LENGTH}, "code-1") for p in (pid, pid_t))
    assert atr != atr_t

    cfg = next(iter(enumerate_configs()))
    sig = signal_mask(panel, cfg, IndicatorCache(panel)).signal
    sig_t = signal_mask(tampered, cfg, IndicatorCache(tampered)).signal
    assert mask_hash(sig, atr) != mask_hash(sig_t, atr_t)


def test_content_digest_covers_every_array(tmp_path):
    (path,) = generate_synthetic_panel(tmp_path / "panel", 4, _sessions("2024-03-04", "2024-03-05"), seed=2)
    panel = _copy_panel(open_panel_chunk(path))
    base = content_digest(panel)
    assert HEX64.match(base)
    for name in ("open", "high", "low", "volume"):
        arr = np.array(getattr(panel, name))
        arr[0, 0] = arr[0, 0] + 1
        assert content_digest(_copy_panel(panel, **{name: arr})) != base, name
    elig = np.array(panel.eligible)
    elig[0, 0] = ~elig[0, 0]
    assert content_digest(_copy_panel(panel, eligible=elig)) != base
    renamed = MarketPanel(
        asset_ids=tuple(reversed(panel.asset_ids)),
        **{n: getattr(panel, n) for n in ("open", "high", "low", "close", "volume", "session_index", "eligible")},
    )
    assert content_digest(renamed) != base
    as64 = _copy_panel(panel, close=np.array(panel.close, dtype=np.float64))
    assert content_digest(as64) != base


def test_panel_chunk_id_fields():
    base = dict(
        data_version="data-v1",
        universe_version="univ-v1",
        week_start=date(2024, 3, 4),
        asset_ids=("A", "B"),
        calendar_version="XNYS-exchange_calendars-1.0",
        dtype="float32",
        content="c" * 64,
    )
    ref = panel_chunk_id(**base)
    assert HEX64.match(ref)
    assert panel_chunk_id(**base) == ref
    changed = dict(
        data_version="data-v2",
        universe_version="univ-v2",
        week_start=date(2024, 3, 11),
        asset_ids=("A", "C"),
        calendar_version="XNYS-exchange_calendars-2.0",
        dtype="float64",
        content="d" * 64,
    )
    assert set(changed) == set(base)
    for name, value in changed.items():
        assert panel_chunk_id(**{**base, name: value}) != ref, name


def test_indicator_key_inputs():
    ref = indicator_key("pid", 20, "atr", {"length": 14}, "code")
    assert HEX64.match(ref)
    assert indicator_key("pid", 20, "atr", {"length": 14.0}, "code") == ref  # 14 and 14.0 are one setting
    for other in (
        indicator_key("pid2", 20, "atr", {"length": 14}, "code"),
        indicator_key("pid", 20, "ema", {"length": 14}, "code"),
        indicator_key("pid", 20, "atr", {"length": 15}, "code"),
        indicator_key("pid", 20, "atr", {"length": 14}, "code2"),
    ):
        assert other != ref


def test_dedupe_rate_on_1000_random_configs(tmp_path):
    (path,) = generate_synthetic_panel(tmp_path / "panel", 10, _sessions("2024-03-04", "2024-03-08"), seed=0)
    panel = open_panel_chunk(path)
    configs = list(enumerate_configs())
    picks = np.random.default_rng(0).choice(len(configs), size=1000, replace=False)
    pid = _panel_id(panel)
    atr = indicator_key(pid, WARMUP, "atr", {"length": ATR_LENGTH}, code_version())
    ind = IndicatorCache(panel)
    hashes = [mask_hash(signal_mask(panel, configs[int(i)], ind).signal, atr) for i in picks]

    report = dedupe_report(hashes)
    assert report.total == 1000
    assert report.unique == len(set(hashes))
    assert report.unique < report.total
    assert report.rate == 1 - report.unique / report.total

    payload = {
        "panel": {"symbols": 10, "sessions": 5, "seed": 0},
        "configs_drawn": 1000,
        "distinct_config_ids": len({config_id(configs[int(i)]) for i in picks}),
        "total": report.total,
        "unique": report.unique,
        "rate": report.rate,
    }
    out = tmp_path / "dedupe.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nD6b dedupe report ({out.name}): {json.dumps(payload, sort_keys=True)}")


def test_dedupe_report_rate_edges():
    assert DedupeReport(total=0, unique=0).rate == 0.0
    assert dedupe_report([]) == DedupeReport(total=0, unique=0)
    assert dedupe_report(["a", "a", "b", "a"]) == DedupeReport(total=4, unique=2)
    assert dedupe_report(["a", "a", "b", "a"]).rate == 0.5
    assert CacheStats().dedupe_rate == 0.0


def test_cutoff_guard_still_raises(cache, monkeypatch):
    chunks = _weeks(3)
    cutoff = chunks[-1].end - timedelta(days=1)
    mask_for = MaskFor(_context())

    def no_read(*args, **kwargs):
        raise AssertionError("cache read before the cutoff guard")

    monkeypatch.setattr(cache, "get_alias", no_read)
    monkeypatch.setattr(cache, "get_path", no_read)
    before = dataclasses.asdict(cache.stats)
    with pytest.raises(LookaheadError):
        cache.get_window_dedup(_context(), "ns:cfg", chunks, cutoff, mask_for)
    assert mask_for.mask_calls == []
    assert dataclasses.asdict(cache.stats) == before


def test_get_window_dedup_rejects_replay_with_wrong_sessions(cache):
    chunk = _week(0)

    def bad(c: Chunk) -> MaskedReplay:
        return MaskedReplay(mask_hash="h", replay=lambda: _fake_result(_week(1), "h"))

    with pytest.raises(ValueError, match="sessions"):
        cache.get_window_dedup(_context(), "ns:cfg", [chunk], chunk.end, bad)
    assert cache.get_alias(_context(), chunk, "ns:cfg") is None


# ---------------------------------------------------------------------------
# key components


def _write_pkg(root: Path, name: str, body_b: str) -> None:
    pkg = root / name
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("X = 1\n", encoding="utf-8")
    (pkg / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sub" / "b.py").write_text(body_b, encoding="utf-8")


def test_code_version_changes_with_source(tmp_path, monkeypatch):
    name = "d6b_codever_pkg"
    _write_pkg(tmp_path, name, "Y = 2\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    # find_spec of a submodule imports the package; undo that at teardown. setitem records the key as
    # absent (so teardown deletes it) and delitem leaves it absent for the test.
    monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, name)

    v1 = code_version((name,))
    assert HEX16.match(v1)
    assert code_version((name,)) == v1
    single = code_version((f"{name}.a",))
    assert single != v1

    (tmp_path / name / "sub" / "b.py").write_text("Y = 3\n", encoding="utf-8")
    v2 = code_version((name,))
    assert v2 != v1
    assert code_version((f"{name}.a",)) == single  # a.py unchanged

    (tmp_path / name / "a.py").write_text("X = 10\n", encoding="utf-8")
    assert code_version((f"{name}.a",)) != single

    with pytest.raises(ValueError):
        code_version(("d6b_no_such_module_xyz",))


def test_code_version_default_modules_resolve():
    assert "shpe_research.features.momentum" in CODE_MODULES
    v = code_version()
    assert HEX16.match(v)
    assert code_version(CODE_MODULES) == v


def test_account_hash_stable_and_sensitive():
    base = account_hash(AccountConfig())
    assert HEX16.match(base)
    assert account_hash(AccountConfig()) == base
    assert account_hash(replace(AccountConfig(), cost_bps=20)) != base
    cfg = AccountConfig()
    nested = replace(cfg, exit=replace(cfg.exit, initial_stop_atr=cfg.exit.initial_stop_atr + 0.5))
    assert account_hash(nested) != base


def test_calendar_version():
    v = calendar_version()
    assert v.startswith("XNYS-exchange_calendars-")
    assert len(v) > len("XNYS-exchange_calendars-")


def test_chunk_spans():
    days = [date(2024, 3, 4) + timedelta(days=d) for d in range(12)]
    sessions = [d for d in days if d.weekday() < 5]
    assert len(sessions) == 10
    weeks = chunk_spans(sessions, "chunk")
    assert weeks == [ChunkSpan(tuple(sessions[:5])), ChunkSpan(tuple(sessions[5:]))]
    per_session = chunk_spans(sessions, "session")
    assert per_session == [ChunkSpan((s,)) for s in sessions]
    assert chunk_spans([], "chunk") == []
    with pytest.raises(ValueError):
        chunk_spans(sessions, "month")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        chunk_spans([sessions[1], sessions[0]], "chunk")


def test_chunk_spans_split_on_iso_week_with_holiday():
    # Good Friday 2024-03-29 is closed: the week has four sessions.
    sessions = [d.date() for d in _sessions("2024-03-25", "2024-04-05")]
    spans = chunk_spans(sessions, "chunk")
    assert [len(s.sessions) for s in spans] == [4, 5]
    assert all(s.sessions[0].weekday() == 0 for s in spans)


def test_code_version_test_leaves_no_module_behind():
    """Runs after ``test_code_version_changes_with_source`` (file order): its package import is undone."""
    assert "d6b_codever_pkg" not in sys.modules
