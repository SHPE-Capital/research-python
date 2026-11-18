"""D6 evaluation cache: exact daily return paths per (context, chunk, config).

A candidate config's replay over one chunk (an ISO week by default) is stored
once and reused. Because search/validation replays use a fixed reference equity,
a session's return does not depend on other sessions, so a window is the plain
concatenation of its chunks: an expanding window only computes the new week, and
a weight change reuses every path (weights are never part of a key).

Storage is one SQLite file in WAL mode, table ``paths(key TEXT PRIMARY KEY,
payload BLOB NOT NULL)``. Several processes may read and write it at once:
each process opens its own connection (lazily, and again after a pid change),
writes are single ``INSERT OR IGNORE`` statements so concurrent writers of the
same key are harmless, and SQLite's own locking replaces any temp-file plus
``os.replace`` scheme (which fails on Windows while another process holds the
file open). The file must live on a local disk: WAL does not work over network
file systems.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TypeVar

import numpy as np

from shpe_research.calibration.acceptance import Entry

_KEY_SCHEMA = "d6.config.v1"
_T = TypeVar("_T")
_BUSY_CODES = (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
_RETRY_FIRST_SLEEP_S = 0.005  # mechanism detail, not a protocol tunable: the total wait is busy_timeout_s
_RETRY_MAX_SLEEP_S = 0.1


class LookaheadError(RuntimeError):
    """A chunk containing a session after the decision cutoff was requested."""


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_increasing(sessions: Sequence[date], what: str) -> None:
    if len(sessions) == 0:
        raise ValueError(f"{what} must be non-empty")
    for prev, cur in zip(sessions, sessions[1:], strict=False):
        if not cur > prev:
            raise ValueError(f"{what} must be strictly increasing, got {prev} then {cur}")


def _check_entries(entries: Sequence[Entry], n_sessions: int) -> tuple[Entry, ...]:
    out = tuple((int(pos), str(asset), bool(unresolved)) for pos, asset, unresolved in entries)
    for pos, _, _ in out:
        if not 0 <= pos < n_sessions:
            raise ValueError(f"entry session position {pos} outside [0, {n_sessions})")
    if list(out) != sorted(out):
        raise ValueError("entries must be sorted")
    return out


def _as_returns(daily_returns: np.ndarray, n_sessions: int) -> np.ndarray:
    r = np.array(daily_returns, dtype=np.float64)  # private copy
    if r.ndim != 1 or r.shape[0] != n_sessions:
        raise ValueError(f"daily_returns must be 1-D of length {n_sessions}, got shape {r.shape}")
    r.flags.writeable = False
    return r


@dataclass(frozen=True)
class CacheContext:
    """Everything outside the chunk and config that changes a replayed return path.

    Attributes:
        data_version: Identity of the bar data snapshot.
        universe_version: Identity of the universe schedule.
        execution_model: Replay kernel name, ``"fast"`` or ``"oracle"``.
        account_hash: Hash of the replay account settings.
        code_version: Identity of the replay code.
        warmup_sessions: Length of the warm-up prefix supplied before each chunk
            (``CacheSection.warmup_sessions``).
    """

    data_version: str
    universe_version: str
    execution_model: str
    account_hash: str
    code_version: str
    warmup_sessions: int

    def __post_init__(self) -> None:
        if self.warmup_sessions < 0:
            raise ValueError("warmup_sessions must be >= 0")

    def digest(self) -> str:
        """Return the 64-hex SHA-256 of the canonical JSON of the fields.

        Inputs: the six fields. Output: a lowercase hex string, stable across
        processes and platforms. Pure; no timing dependence.
        """
        return _sha256_hex(_canonical_json(asdict(self)))


@dataclass(frozen=True)
class Chunk:
    """One cache unit: a run of consecutive sessions and the panel it was replayed on.

    Attributes:
        sessions: Non-empty, strictly increasing session dates.
        panel_digest: Identity of the assembled warm-up prefix plus chunk panel;
            changing any input bar must change it.
    """

    sessions: tuple[date, ...]
    panel_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sessions", tuple(self.sessions))
        _check_increasing(self.sessions, "Chunk.sessions")

    @property
    def start(self) -> date:
        """First session of the chunk."""
        return self.sessions[0]

    @property
    def end(self) -> date:
        """Last session of the chunk; the chunk is usable once this session has closed."""
        return self.sessions[-1]


@dataclass(frozen=True)
class ChunkKey:
    """Cache key of one config's replay over one chunk in one context (weights excluded)."""

    context: CacheContext
    chunk: Chunk
    config_id: str

    def digest(self) -> str:
        """Return the 64-hex SHA-256 key for the ``paths`` table.

        Inputs: context fields, chunk start, end, sessions and panel digest,
        and ``config_id``. Output: a lowercase hex string. Pure.
        """
        payload = {
            "schema": _KEY_SCHEMA,
            "context": asdict(self.context),
            "chunk": {
                "start": self.chunk.start.isoformat(),
                "end": self.chunk.end.isoformat(),
                "sessions": [s.isoformat() for s in self.chunk.sessions],
                "panel_digest": self.chunk.panel_digest,
            },
            "config_id": self.config_id,
        }
        return _sha256_hex(_canonical_json(payload))


@dataclass(frozen=True, eq=False)
class ChunkResult:
    """Exact replay output of one config over one chunk.

    Attributes:
        sessions: The chunk's sessions.
        daily_returns: float64 daily net returns, one per session (read-only copy;
            NaN is stored as is).
        entries: ``(session offset in chunk, asset_id, unresolved)``, sorted.
        ambiguous_bars: Count of bars where the fill was ambiguous.
    """

    sessions: tuple[date, ...]
    daily_returns: np.ndarray
    entries: tuple[Entry, ...]
    ambiguous_bars: int

    def __post_init__(self) -> None:
        sessions = tuple(self.sessions)
        _check_increasing(sessions, "ChunkResult.sessions")
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "daily_returns", _as_returns(self.daily_returns, len(sessions)))
        object.__setattr__(self, "entries", _check_entries(self.entries, len(sessions)))
        object.__setattr__(self, "ambiguous_bars", int(self.ambiguous_bars))

    def to_bytes(self) -> bytes:
        """Serialize deterministically.

        Output: UTF-8 canonical JSON header (sessions, entries, ambiguous_bars),
        ``b"\\n"``, then ``daily_returns`` as little-endian float64 bytes.
        Same result → same bytes on every platform. Pure.
        """
        header = {
            "sessions": [s.isoformat() for s in self.sessions],
            "entries": [list(e) for e in self.entries],
            "ambiguous_bars": self.ambiguous_bars,
        }
        return _canonical_json(header).encode("utf-8") + b"\n" + self.daily_returns.astype("<f8").tobytes()

    @classmethod
    def from_bytes(cls, blob: bytes) -> ChunkResult:
        """Inverse of ``to_bytes``.

        Input: bytes produced by ``to_bytes``. Output: an equal ``ChunkResult``.
        Raises ValueError if the blob is malformed. Pure.
        """
        head, sep, body = blob.partition(b"\n")
        if not sep:
            raise ValueError("ChunkResult blob has no header separator")
        header = json.loads(head.decode("utf-8"))
        returns = np.frombuffer(body, dtype="<f8").astype(np.float64)
        return cls(
            sessions=tuple(date.fromisoformat(s) for s in header["sessions"]),
            daily_returns=returns,
            entries=tuple((int(p), str(a), bool(u)) for p, a, u in header["entries"]),
            ambiguous_bars=int(header["ambiguous_bars"]),
        )


@dataclass(frozen=True, eq=False)
class WindowResult:
    """A config's return path over a run of consecutive sessions (one or more chunks).

    Attributes:
        sessions: Strictly increasing sessions of the window.
        daily_returns: float64 daily net returns, one per session.
        entries: ``(session position in the window, asset_id, unresolved)``, sorted.
        ambiguous_bars: Total ambiguous bars.
    """

    sessions: tuple[date, ...]
    daily_returns: np.ndarray
    entries: tuple[Entry, ...]
    ambiguous_bars: int

    def __post_init__(self) -> None:
        sessions = tuple(self.sessions)
        _check_increasing(sessions, "WindowResult.sessions")
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "daily_returns", _as_returns(self.daily_returns, len(sessions)))
        object.__setattr__(self, "entries", _check_entries(self.entries, len(sessions)))
        object.__setattr__(self, "ambiguous_bars", int(self.ambiguous_bars))

    @classmethod
    def concat(cls, parts: Sequence[ChunkResult]) -> WindowResult:
        """Join chunk results in order into one window.

        Input: non-empty chunk results whose sessions are strictly increasing
        across parts. Output: the window; entry offsets are shifted to window
        positions and ambiguous bars summed. Raises ValueError otherwise. Pure;
        the caller guarantees every part ends at or before the cutoff.
        """
        if len(parts) == 0:
            raise ValueError("concat needs at least one part")
        sessions: list[date] = []
        entries: list[Entry] = []
        for part in parts:
            offset = len(sessions)
            sessions.extend(part.sessions)
            entries.extend((offset + pos, asset, unresolved) for pos, asset, unresolved in part.entries)
        return cls(
            sessions=tuple(sessions),
            daily_returns=np.concatenate([p.daily_returns for p in parts]),
            entries=tuple(entries),
            ambiguous_bars=sum(p.ambiguous_bars for p in parts),
        )

    def slice_sessions(self, sessions: Sequence[date]) -> WindowResult:
        """Return the sub-window over ``sessions``.

        Input: a non-empty, contiguous run of this window's sessions. Output:
        the matching returns and the entries inside it, re-indexed to the
        sub-window. ``ambiguous_bars`` is carried over unchanged (it is not
        stored per session). Raises ValueError if ``sessions`` is empty or not
        a contiguous run of this window. Pure.
        """
        wanted = tuple(sessions)
        if not wanted:
            raise ValueError("slice_sessions needs at least one session")
        try:
            lo = self.sessions.index(wanted[0])
        except ValueError:
            raise ValueError(f"session {wanted[0]} is not in the window") from None
        hi = lo + len(wanted)
        if self.sessions[lo:hi] != wanted:
            raise ValueError("slice_sessions requires a contiguous run of the window's sessions")
        return WindowResult(
            sessions=wanted,
            daily_returns=self.daily_returns[lo:hi],
            entries=tuple((pos - lo, asset, u) for pos, asset, u in self.entries if lo <= pos < hi),
            ambiguous_bars=self.ambiguous_bars,
        )


@dataclass
class CacheStats:
    """Per-instance cache counters (process-local; not part of any deterministic output)."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    bytes_written: int = 0

    @property
    def hit_rate(self) -> float:
        """``hits / (hits + misses)``, or 0.0 before any lookup."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _check_chunks(chunks: Sequence[Chunk], cutoff: date) -> None:
    """Validate a window's chunks before any cache read.

    Inputs: the chunks in window order and the decision cutoff session (the
    last completed session). Raises ValueError if ``chunks`` is empty, or not
    strictly increasing and non-overlapping; raises LookaheadError if any chunk
    contains a session after ``cutoff``. A chunk ending on the cutoff is allowed.
    """
    if len(chunks) == 0:
        raise ValueError("a window needs at least one chunk")
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        if not cur.start > prev.end:
            raise ValueError(f"chunks must be strictly increasing and non-overlapping: {prev.end} then {cur.start}")
    late = [c for c in chunks if c.end > cutoff]
    if late:
        raise LookaheadError(f"chunk ending {late[0].end} is after the cutoff session {cutoff}")


class EvalCache:
    """Process-safe store of chunk return paths in one SQLite WAL file.

    Not picklable: each worker process constructs its own instance from the path.
    """

    def __init__(self, path: str | Path, *, busy_timeout_s: float) -> None:
        """Bind the cache to ``path`` without opening it.

        Inputs: the SQLite file path (parent directories are created on first
        use) and the lock wait in seconds (``CacheSection.busy_timeout_s``).
        """
        if busy_timeout_s < 0:
            raise ValueError("busy_timeout_s must be >= 0")
        self.path = Path(path)
        self.busy_timeout_s = float(busy_timeout_s)
        self.stats = CacheStats()
        self._conn: sqlite3.Connection | None = None
        self._pid: int | None = None

    def __reduce__(self):
        raise TypeError("EvalCache is not picklable; construct one per process from its path")

    def _retry_busy(self, op: Callable[[], _T]) -> _T:
        """Run ``op``, retrying on SQLITE_BUSY/LOCKED until ``busy_timeout_s`` has elapsed.

        SQLite skips its busy handler in some lock states (switching a fresh
        file to WAL while another process creates it, a stale WAL snapshot),
        so ``busy_timeout`` alone does not cover every concurrent case.
        """
        deadline = time.monotonic() + self.busy_timeout_s
        pause = _RETRY_FIRST_SLEEP_S
        while True:
            try:
                return op()
            except sqlite3.OperationalError as exc:
                remaining = deadline - time.monotonic()
                code = getattr(exc, "sqlite_errorcode", None)
                if code is None or (code & 0xFF) not in _BUSY_CODES or remaining <= 0:
                    raise
                time.sleep(min(pause, remaining))
                pause = min(2 * pause, _RETRY_MAX_SLEEP_S)

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=self.busy_timeout_s, isolation_level=None)
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RuntimeError(f"could not enable WAL on {self.path} (got {mode!r})")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("CREATE TABLE IF NOT EXISTS paths (key TEXT PRIMARY KEY, payload BLOB NOT NULL)")
        except BaseException:
            conn.close()
            raise
        return conn

    def _connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._conn is not None and self._pid != pid:
            # Inherited from a parent process (fork): abandon it, never use or close it here.
            self._conn = None
        if self._conn is None:
            self._conn = self._retry_busy(self._open)
            self._pid = pid
        return self._conn

    def get(self, key: ChunkKey) -> ChunkResult | None:
        """Look up one chunk path.

        Input: the key. Output: the stored result, or None. Counts a hit or a
        miss in ``stats``. Timing is the caller's concern: ``get_window`` applies
        the cutoff guard before calling this.
        """
        digest = key.digest()
        row = self._retry_busy(
            lambda: self._connection().execute("SELECT payload FROM paths WHERE key = ?", (digest,)).fetchone()
        )
        if row is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return ChunkResult.from_bytes(bytes(row[0]))

    def put(self, key: ChunkKey, result: ChunkResult) -> None:
        """Store one chunk path with ``INSERT OR IGNORE`` (idempotent; the first payload wins).

        Inputs: the key and its result. Counts ``writes`` and ``bytes_written``
        only when a row was actually inserted. Safe to call from several
        processes for the same key.
        """
        payload = result.to_bytes()
        digest = key.digest()
        cur = self._retry_busy(
            lambda: self._connection().execute(
                "INSERT OR IGNORE INTO paths (key, payload) VALUES (?, ?)", (digest, payload)
            )
        )
        if cur.rowcount == 1:
            self.stats.writes += 1
            self.stats.bytes_written += len(payload)

    def get_window(
        self,
        context: CacheContext,
        config_id: str,
        chunks: Sequence[Chunk],
        cutoff: date,
        compute: Callable[[Chunk], ChunkResult],
    ) -> WindowResult:
        """Return a config's return path over ``chunks``, computing only missing chunks.

        Inputs: the replay context, the config id, the window's chunks in order,
        the decision cutoff session (last completed session) and a ``compute``
        callback that replays one chunk. Output: the concatenated window.

        Timing: before any read, the chunks are validated and LookaheadError is
        raised if any chunk contains a session after ``cutoff``. Then, per
        chunk, a cache hit is used as is; a miss calls ``compute(chunk)``,
        checks its sessions equal the chunk's (ValueError otherwise) and stores it.
        """
        _check_chunks(chunks, cutoff)
        parts: list[ChunkResult] = []
        for chunk in chunks:
            key = ChunkKey(context=context, chunk=chunk, config_id=config_id)
            result = self.get(key)
            if result is None:
                result = compute(chunk)
                if result.sessions != chunk.sessions:
                    raise ValueError(f"compute returned sessions that do not match the chunk starting {chunk.start}")
                self.put(key, result)
            parts.append(result)
        return WindowResult.concat(parts)

    def close(self) -> None:
        """Close this process's connection (a later call reopens it lazily)."""
        if self._conn is not None and self._pid == os.getpid():
            self._conn.close()
        self._conn = None
        self._pid = None
