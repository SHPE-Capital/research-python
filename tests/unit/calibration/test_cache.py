"""Tests for D6, the week-chunk evaluation cache (SQLite WAL)."""

from __future__ import annotations

import dataclasses
import multiprocessing
import pickle
import sqlite3
import zlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from shpe_research.calibration.cache import (
    CacheContext,
    CacheStats,
    Chunk,
    ChunkKey,
    ChunkResult,
    EvalCache,
    LookaheadError,
    WindowResult,
)
from shpe_research.calibration.score import score_j
from shpe_research.config.momentum import CacheSection, ScoreSection

BUSY_TIMEOUT_S = CacheSection().busy_timeout_s
DOWNSIDE_COEFF = ScoreSection().downside_coeff
FIRST_MONDAY = date(2024, 1, 1)


def _context(**overrides) -> CacheContext:
    base = dict(
        data_version="data-v1",
        universe_version="univ-v1",
        execution_model="fast",
        account_hash="acct-1",
        code_version="code-1",
        warmup_sessions=CacheSection().warmup_sessions,
    )
    base.update(overrides)
    return CacheContext(**base)


def _week(i: int, panel_digest: str | None = None) -> Chunk:
    monday = FIRST_MONDAY + timedelta(weeks=i)
    sessions = tuple(monday + timedelta(days=d) for d in range(5))
    return Chunk(sessions=sessions, panel_digest=panel_digest or f"panel-{i}")


def _weeks(n: int) -> list[Chunk]:
    return [_week(i) for i in range(n)]


def _fake_result(chunk: Chunk, salt: str) -> ChunkResult:
    """A deterministic fake replay result for ``chunk`` (depends on salt and the chunk identity)."""
    seed = zlib.crc32(f"{salt}|{chunk.sessions}|{chunk.panel_digest}".encode())
    rng = np.random.default_rng(seed)
    n = len(chunk.sessions)
    returns = rng.normal(0.0, 0.01, size=n)
    entries = tuple(sorted((int(rng.integers(0, n)), f"A{int(rng.integers(0, 9))}", False) for _ in range(2)))
    return ChunkResult(sessions=chunk.sessions, daily_returns=returns, entries=entries, ambiguous_bars=1)


class CountingCompute:
    """A ``compute`` callback that records every chunk it was asked to replay."""

    def __init__(self, config_id: str) -> None:
        self.config_id = config_id
        self.calls: list[Chunk] = []

    def __call__(self, chunk: Chunk) -> ChunkResult:
        self.calls.append(chunk)
        return _fake_result(chunk, self.config_id)


@pytest.fixture
def cache(tmp_path: Path):
    c = EvalCache(tmp_path / "cache.sqlite", busy_timeout_s=BUSY_TIMEOUT_S)
    yield c
    c.close()


def test_round_trip_bytes():
    sessions = _week(0).sessions
    r = ChunkResult(
        sessions=sessions,
        daily_returns=np.array([0.01, np.nan, -0.02, 0.0, 1e-12]),
        entries=((0, "AAPL", False), (2, "MSFT", True), (2, "NVDA", False)),
        ambiguous_bars=3,
    )
    blob = r.to_bytes()
    back = ChunkResult.from_bytes(blob)
    assert back.to_bytes() == blob
    assert back.sessions == sessions
    assert np.isnan(back.daily_returns[1])
    np.testing.assert_array_equal(back.daily_returns, r.daily_returns)
    assert back.daily_returns.dtype == np.float64
    assert back.entries == r.entries
    assert back.ambiguous_bars == 3
    assert r.to_bytes() == blob  # deterministic


def test_chunk_result_rejects_bad_shapes():
    sessions = _week(0).sessions
    with pytest.raises(ValueError):
        ChunkResult(sessions=sessions, daily_returns=np.zeros(4), entries=(), ambiguous_bars=0)
    with pytest.raises(ValueError):
        ChunkResult(sessions=sessions, daily_returns=np.zeros(5), entries=((5, "A", False),), ambiguous_bars=0)
    with pytest.raises(ValueError):
        ChunkResult(
            sessions=sessions, daily_returns=np.zeros(5), entries=((2, "B", False), (1, "A", False)), ambiguous_bars=0
        )


def test_chunk_requires_increasing_sessions():
    with pytest.raises(ValueError):
        Chunk(sessions=(), panel_digest="p")
    with pytest.raises(ValueError):
        Chunk(sessions=(date(2024, 1, 2), date(2024, 1, 2)), panel_digest="p")


def test_key_digest_is_hex_and_stable():
    key = ChunkKey(context=_context(), chunk=_week(0), config_id="cfg-a")
    d = key.digest()
    assert len(d) == 64 and int(d, 16) >= 0
    assert ChunkKey(context=_context(), chunk=_week(0), config_id="cfg-a").digest() == d
    assert len(_context().digest()) == 64


def test_hit_for_weight_only_change(cache):
    chunks = _weeks(4)
    compute = CountingCompute("cfg-a")
    first = cache.get_window(_context(), "cfg-a", chunks, chunks[-1].end, compute)
    assert len(compute.calls) == len(chunks)
    hits_before = cache.stats.hits

    second = cache.get_window(_context(), "cfg-a", chunks, chunks[-1].end, compute)
    assert len(compute.calls) == len(chunks)  # 0 new computes
    assert cache.stats.hits - hits_before == len(chunks)
    assert cache.stats.hits == len(chunks)

    n = len(first.sessions)
    uniform = np.full(n, 1.0 / n)
    age = np.arange(n - 1, -1, -1, dtype=np.float64)
    recency = 2.0 ** (-age / 5.0)
    j_uniform = score_j(first.daily_returns, uniform, downside_coeff=DOWNSIDE_COEFF)
    j_recency = score_j(second.daily_returns, recency, downside_coeff=DOWNSIDE_COEFF)
    assert j_uniform != j_recency
    np.testing.assert_array_equal(first.daily_returns, second.daily_returns)


KEY_COMPONENTS = [
    "data_version",
    "universe_version",
    "execution_model",
    "account_hash",
    "code_version",
    "warmup_sessions",
    "sessions",
    "panel_digest",
    "config_id",
]


def _drop_middle_session(chunk: Chunk) -> Chunk:
    s = chunk.sessions
    return Chunk(sessions=s[:2] + s[3:], panel_digest=chunk.panel_digest)


@pytest.mark.parametrize("component", KEY_COMPONENTS)
def test_miss_for_each_key_component(cache, component):
    chunks = _weeks(3)
    context = _context()
    config_id = "cfg-a"
    compute = CountingCompute(config_id)
    cache.get_window(context, config_id, chunks, chunks[-1].end, compute)
    cache.get_window(context, config_id, chunks, chunks[-1].end, compute)
    assert len(compute.calls) == len(chunks)

    if component == "warmup_sessions":
        context = dataclasses.replace(context, warmup_sessions=context.warmup_sessions + 1)
    elif component == "execution_model":
        context = dataclasses.replace(context, execution_model="oracle")
    elif component in {"data_version", "universe_version", "account_hash", "code_version"}:
        context = dataclasses.replace(context, **{component: getattr(context, component) + "-changed"})
    elif component == "sessions":
        chunks = [_drop_middle_session(c) for c in chunks]
        assert all(c.start == o.start and c.end == o.end for c, o in zip(chunks, _weeks(3), strict=True))
    elif component == "panel_digest":
        chunks = [dataclasses.replace(c, panel_digest=c.panel_digest + "-changed") for c in chunks]
    elif component == "config_id":
        config_id = "cfg-b"

    cache.get_window(context, config_id, chunks, chunks[-1].end, compute)
    assert len(compute.calls) == 2 * len(chunks)


def test_extend_window_one_week_computes_one_chunk_per_config(cache):
    configs = ["cfg-a", "cfg-b", "cfg-c"]
    computes = {c: CountingCompute(c) for c in configs}
    four = _weeks(4)
    for c in configs:
        cache.get_window(_context(), c, four, four[-1].end, computes[c])
    assert sum(len(x.calls) for x in computes.values()) == 12

    five = _weeks(5)
    for c in configs:
        window = cache.get_window(_context(), c, five, five[-1].end, computes[c])
        assert len(window.sessions) == 25
    assert sum(len(x.calls) for x in computes.values()) == 15
    for c in configs:
        assert computes[c].calls[-1] == five[-1]


def test_cutoff_guard_raises_before_read(cache, monkeypatch):
    chunks = _weeks(3)
    cutoff = chunks[-1].end - timedelta(days=1)  # last chunk ends on cutoff + 1 day
    compute = CountingCompute("cfg-a")
    reads: list[ChunkKey] = []
    real_get = cache.get

    def spy_get(key):
        reads.append(key)
        return real_get(key)

    monkeypatch.setattr(cache, "get", spy_get)
    before = dataclasses.asdict(cache.stats)
    with pytest.raises(LookaheadError):
        cache.get_window(_context(), "cfg-a", chunks, cutoff, compute)
    assert compute.calls == []
    assert reads == []
    assert dataclasses.asdict(cache.stats) == before


def test_chunk_ending_on_cutoff_allowed(cache):
    chunks = _weeks(2)
    compute = CountingCompute("cfg-a")
    window = cache.get_window(_context(), "cfg-a", chunks, chunks[-1].end, compute)
    assert window.sessions[-1] == chunks[-1].end
    assert len(compute.calls) == 2


def test_get_window_rejects_overlapping_or_unordered_chunks(cache):
    compute = CountingCompute("cfg-a")
    a, b = _week(0), _week(1)
    overlap = Chunk(sessions=(a.end, b.start), panel_digest="x")
    for chunks in ([b, a], [a, overlap], [a, a], []):
        with pytest.raises(ValueError):
            cache.get_window(_context(), "cfg-a", chunks, b.end, compute)
    assert compute.calls == []


def test_get_window_rejects_compute_with_wrong_sessions(cache):
    chunk = _week(0)

    def bad_compute(c: Chunk) -> ChunkResult:
        return _fake_result(_week(1), "x")

    with pytest.raises(ValueError):
        cache.get_window(_context(), "cfg-a", [chunk], chunk.end, bad_compute)
    assert cache.stats.writes == 0


def test_counters(cache):
    expected_bytes = 0
    for config_id in ("cfg-a", "cfg-b"):
        chunks = _weeks(3)
        cache.get_window(_context(), config_id, chunks, chunks[-1].end, CountingCompute(config_id))
        expected_bytes += sum(len(_fake_result(c, config_id).to_bytes()) for c in chunks)
    for config_id in ("cfg-a", "cfg-b"):
        chunks = _weeks(3)
        cache.get_window(_context(), config_id, chunks, chunks[-1].end, CountingCompute(config_id))
    four = _weeks(4)
    cache.get_window(_context(), "cfg-a", four, four[-1].end, CountingCompute("cfg-a"))
    expected_bytes += len(_fake_result(four[-1], "cfg-a").to_bytes())

    assert cache.stats.hits == 9
    assert cache.stats.misses == 7
    assert cache.stats.writes == 7
    assert cache.stats.bytes_written == expected_bytes > 0
    assert cache.stats.hit_rate == 9 / 16
    assert CacheStats().hit_rate == 0.0


def test_put_is_idempotent(cache):
    chunk = _week(0)
    key = ChunkKey(context=_context(), chunk=chunk, config_id="cfg-a")
    first = _fake_result(chunk, "cfg-a")
    cache.put(key, first)
    cache.put(key, first)
    cache.put(key, _fake_result(chunk, "other"))  # ignored: the first payload stays
    assert cache.stats.writes == 1
    assert cache.stats.bytes_written == len(first.to_bytes())
    got = cache.get(key)
    assert got is not None and got.to_bytes() == first.to_bytes()


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "nested" / "cache.sqlite"
    chunk = _week(0)
    key = ChunkKey(context=_context(), chunk=chunk, config_id="cfg-a")
    a = EvalCache(path, busy_timeout_s=BUSY_TIMEOUT_S)
    a.put(key, _fake_result(chunk, "cfg-a"))
    a.close()
    b = EvalCache(path, busy_timeout_s=BUSY_TIMEOUT_S)
    try:
        got = b.get(key)
        assert got is not None and got.to_bytes() == _fake_result(chunk, "cfg-a").to_bytes()
        assert b.stats.hits == 1
    finally:
        b.close()


def test_eval_cache_is_not_picklable(cache):
    with pytest.raises(TypeError):
        pickle.dumps(cache)


# --- multiprocess ------------------------------------------------------------

MP_KEYS = 75


def _mp_key(i: int) -> tuple[ChunkKey, ChunkResult]:
    chunk = _week(i)
    return ChunkKey(context=_context(), chunk=chunk, config_id="cfg-mp"), _fake_result(chunk, "cfg-mp")


def _mp_writer(path: str, indices: list[int], barrier, errors) -> None:
    """Module-level spawn target: open the (fresh) cache together with the other writer, write the given keys."""
    try:
        cache = EvalCache(path, busy_timeout_s=BUSY_TIMEOUT_S)
        barrier.wait(timeout=60)
        for i in indices:
            key, result = _mp_key(i)
            cache.put(key, result)
            got = cache.get(key)
            assert got is not None and got.to_bytes() == result.to_bytes()
        cache.close()
    except BaseException as exc:  # noqa: BLE001 - relayed to the parent test
        errors.put(f"{type(exc).__name__}: {exc}")
        raise


def test_wal_mode_and_multiprocess(tmp_path):
    path = tmp_path / "mp.sqlite"
    assert not path.exists()  # both writers race to create the file, enable WAL and create the table

    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    errors = ctx.Queue()
    procs = [
        ctx.Process(target=_mp_writer, args=(str(path), list(range(0, 50)), barrier, errors)),
        ctx.Process(target=_mp_writer, args=(str(path), list(range(25, 75)), barrier, errors)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    messages = []
    while not errors.empty():
        messages.append(errors.get())
    assert messages == []
    assert [p.exitcode for p in procs] == [0, 0]

    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()
    cache = EvalCache(path, busy_timeout_s=BUSY_TIMEOUT_S)
    for i in range(MP_KEYS):
        key, expected = _mp_key(i)
        got = cache.get(key)
        assert got is not None, i
        assert got.to_bytes() == expected.to_bytes()
    cache.close()
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM paths").fetchone()[0] == MP_KEYS
    finally:
        conn.close()


# --- window slicing ----------------------------------------------------------


def test_window_slice_reindexes_entries():
    a, b = _week(0), _week(1)
    part_a = ChunkResult(sessions=a.sessions, daily_returns=np.arange(5.0), entries=(), ambiguous_bars=1)
    part_b = ChunkResult(
        sessions=b.sessions, daily_returns=np.arange(5.0, 10.0), entries=((2, "AAPL", False),), ambiguous_bars=2
    )
    window = WindowResult.concat([part_a, part_b])
    assert len(window.sessions) == 10
    assert window.entries == ((7, "AAPL", False),)
    assert window.ambiguous_bars == 3

    tail = window.slice_sessions(window.sessions[5:])
    assert tail.sessions == window.sessions[5:]
    assert tail.entries == ((2, "AAPL", False),)
    np.testing.assert_array_equal(tail.daily_returns, np.arange(5.0, 10.0))

    head = window.slice_sessions(window.sessions[:5])
    assert head.entries == ()

    with pytest.raises(ValueError):
        window.slice_sessions(window.sessions[0:2] + window.sessions[3:5])
    with pytest.raises(ValueError):
        window.slice_sessions([date(2030, 1, 1)])
    with pytest.raises(ValueError):
        window.slice_sessions([])


def test_concat_rejects_unordered_parts():
    a, b = _week(0), _week(1)
    pa = ChunkResult(sessions=a.sessions, daily_returns=np.zeros(5), entries=(), ambiguous_bars=0)
    pb = ChunkResult(sessions=b.sessions, daily_returns=np.zeros(5), entries=(), ambiguous_bars=0)
    with pytest.raises(ValueError):
        WindowResult.concat([pb, pa])
    with pytest.raises(ValueError):
        WindowResult.concat([])
