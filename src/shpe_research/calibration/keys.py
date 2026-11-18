"""D6b cache-key spec: code, account, calendar, panel, indicator and mask identities.

Every value that can change a replayed return path has a stable hex identity
here, so the evaluation cache (``calibration.cache``) can key on content
instead of on names:

- ``code_version``: a content hash of the source files that determine a return
  path (not the git commit, so unrelated commits keep the cache valid).
- ``account_hash``: the replay account settings.
- ``calendar_version``: the trading calendar implementation.
- ``content_digest`` / ``panel_chunk_id``: the assembled warm-up prefix plus
  chunk panel; changing any bar changes both.
- ``indicator_key``: one indicator array on one panel (adoptable by a future
  persisted indicator store; none is built here).
- ``mask_hash``: one signal mask plus the ATR array it is replayed with. Configs
  with equal mask hashes in the same context and chunk replay identically, so
  each unique mask is replayed once.

All functions are pure and deterministic across processes and platforms.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from shpe_research.config.momentum import Granularity
from shpe_research.data.panel import week_chunks
from shpe_research.trade.account_types import AccountConfig, MarketPanel

CODE_MODULES: tuple[str, ...] = (
    "shpe_research.trade.account", "shpe_research.trade.account_fast", "shpe_research.trade.account_types",
    "shpe_research.trade.fill_proxy", "shpe_research.trade.funded_sizing", "shpe_research.trade.exits",
    "shpe_research.trade.exit_package", "shpe_research.features.momentum",
    "shpe_research.strategies.momentum.breakout.signals", "shpe_research.strategies.momentum.breakout.indicator_cache",
    "shpe_research.strategies.momentum.breakout.grid",
)  # fmt: skip

_CALENDAR_NAME = "XNYS"
_CALENDAR_PACKAGE = "exchange_calendars"
_SHORT_HEX = 16
_PANEL_FIELDS = ("open", "high", "low", "close", "volume", "eligible")


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_json(obj: object) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _module_files(module: str) -> list[tuple[str, Path]]:
    """``(relative path, file)`` pairs whose bytes define ``module``; a package lists every ``*.py`` under it."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError) as exc:
        raise ValueError(f"cannot locate module {module!r}: {exc}") from None
    if spec is None:
        raise ValueError(f"cannot locate module {module!r}")
    if spec.submodule_search_locations is not None:
        files: dict[str, Path] = {}
        for root in map(Path, spec.submodule_search_locations):
            for f in root.rglob("*.py"):
                files.setdefault(f.relative_to(root).as_posix(), f)
        if not files:
            raise ValueError(f"package {module!r} has no .py files")
        return sorted(files.items())
    if not spec.has_location or spec.origin is None:
        raise ValueError(f"module {module!r} has no source file")
    origin = Path(spec.origin)
    return [(origin.name, origin)]


def code_version(modules: Sequence[str] = CODE_MODULES) -> str:
    """Return a 16-hex content hash of the source files that determine a return path.

    Inputs: dotted module names, in order. A plain module contributes its source
    file; a package contributes every ``*.py`` under it, sorted by relative path.
    Each file is hashed as (module name, relative path, byte length, bytes) with
    CRLF normalized to LF, so a Windows checkout and a Linux checkout agree.
    Output: the first 16 hex characters of the SHA-256. Raises ValueError if a
    module cannot be located or has no source. Locating a submodule imports its
    parent package (``importlib.util.find_spec``); the hashed files are read from
    disk, so an edit after import is seen. No timing dependence.
    """
    h = hashlib.sha256()
    for module in modules:
        for rel, path in _module_files(module):
            body = path.read_bytes().replace(b"\r\n", b"\n")
            header = _canonical_json({"module": module, "path": rel, "size": len(body)})
            h.update(header.encode("utf-8") + b"\n" + body)
    return h.hexdigest()[:_SHORT_HEX]


def account_hash(cfg: AccountConfig) -> str:
    """Return a 16-hex hash of the replay account settings.

    Input: an ``AccountConfig`` (nested exit and fill packages included). Output:
    the first 16 hex characters of the SHA-256 of the canonical JSON of
    ``dataclasses.asdict(cfg)``. Equal configs give equal hashes; any field
    change gives a different one. Pure; no timing dependence.
    """
    return _sha256_json(dataclasses.asdict(cfg))[:_SHORT_HEX]


def calendar_version() -> str:
    """Return the trading-calendar identity, ``"XNYS-exchange_calendars-<installed version>"``.

    No inputs. Output: a string naming the calendar and the installed
    ``exchange_calendars`` version, since holiday and early-close tables change
    between releases. Reads package metadata only; no timing dependence.
    """
    return f"{_CALENDAR_NAME}-{_CALENDAR_PACKAGE}-{importlib.metadata.version(_CALENDAR_PACKAGE)}"


def _le_contiguous(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    return np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))


def content_digest(panel: MarketPanel) -> str:
    """Return the 64-hex SHA-256 of a panel's full content.

    Input: a ``MarketPanel`` (memory-mapped arrays are fine; they are streamed,
    not copied when already C-contiguous little-endian). Output: a hash over the
    asset ids, shape, every array's dtype, ``session_index`` and the
    open/high/low/close/volume/eligible bytes in C order. Changing any bar,
    eligibility flag, asset id or dtype changes it. Pure; no timing dependence
    (the caller passes the assembled warm-up prefix plus chunk panel).
    """
    arrays = {"session_index": panel.session_index, **{n: getattr(panel, n) for n in _PANEL_FIELDS}}
    header = {
        "schema": "d6b.panel_content.v1",
        "asset_ids": list(panel.asset_ids),
        "shape": [panel.n_symbols, panel.n_minutes],
        "dtypes": {n: np.asarray(a).dtype.newbyteorder("<").str for n, a in arrays.items()},
        "order": list(arrays),
    }
    h = hashlib.sha256(_canonical_json(header).encode("utf-8") + b"\n")
    for arr in arrays.values():
        h.update(memoryview(_le_contiguous(arr)).cast("B"))
    return h.hexdigest()


def panel_chunk_id(
    *,
    data_version: str,
    universe_version: str,
    week_start: date,
    asset_ids: Sequence[str],
    calendar_version: str,
    dtype: str,
    content: str,
) -> str:
    """Return the 64-hex identity of an assembled warm-up prefix plus chunk panel.

    Inputs: the data snapshot version, universe schedule version, first session
    of the chunk, the panel's asset ids in row order, the calendar version, the
    array dtype name and the panel's ``content_digest``. Output: the SHA-256 of
    their canonical JSON; changing any argument changes it. Pure; no timing
    dependence. Used as ``Chunk.panel_digest``.
    """
    return _sha256_json(
        {
            "schema": "d6b.panel_chunk.v1",
            "data_version": data_version,
            "universe_version": universe_version,
            "week_start": week_start.isoformat(),
            "asset_ids": list(asset_ids),
            "calendar_version": calendar_version,
            "dtype": dtype,
            "content": content,
        }
    )


def _canon_number(v: int | float) -> int | float:
    """Collapse an integral float to int so ``14`` and ``14.0`` key identically."""
    if isinstance(v, bool) or not isinstance(v, int | float | np.integer | np.floating):
        raise TypeError(f"indicator params must be numbers, got {v!r}")
    f = float(v)
    return int(f) if f.is_integer() else f


def indicator_key(
    panel_id: str, warmup_sessions: int, name: str, params: Mapping[str, int | float], code_version: str
) -> str:
    """Return the 64-hex key of one indicator array on one panel.

    Inputs: the panel identity (``panel_chunk_id``), the warm-up prefix length in
    sessions, the indicator name, its parameters (integral floats are treated as
    ints) and the code version. Output: the SHA-256 of their canonical JSON.
    Callers key the ATR array as
    ``indicator_key(chunk.panel_digest, warmup, "atr", {"length": ATR_LENGTH}, code_version)``.
    Pure; no timing dependence.
    """
    return _sha256_json(
        {
            "schema": "d6b.indicator.v1",
            "panel_id": panel_id,
            "warmup_sessions": int(warmup_sessions),
            "name": name,
            "params": {str(k): _canon_number(v) for k, v in params.items()},
            "code_version": code_version,
        }
    )


def mask_hash(signal: np.ndarray, atr_key: str) -> str:
    """Return the 64-hex identity of one signal mask replayed with one ATR array.

    Inputs: a bool signal array (symbol x minute) and the ATR array's
    ``indicator_key``. Output: the SHA-256 over the schema tag, the shape, the
    ATR key and ``np.packbits`` of the array in C order (memory layout does not
    matter; shape does). Raises ValueError if ``signal`` is not bool. Pure; no
    timing dependence.
    """
    sig = np.asarray(signal)
    if sig.dtype != np.bool_:
        raise ValueError(f"signal must be a bool array, got {sig.dtype}")
    header = _canonical_json({"schema": "d6b.mask.v1", "shape": list(sig.shape), "atr_key": atr_key})
    packed = np.packbits(np.ascontiguousarray(sig), axis=None)
    return hashlib.sha256(header.encode("utf-8") + b"\n" + packed.tobytes()).hexdigest()


@dataclass(frozen=True)
class ChunkSpan:
    """A planned cache unit: a non-empty run of strictly increasing sessions."""

    sessions: tuple[date, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sessions", tuple(self.sessions))
        if not self.sessions:
            raise ValueError("ChunkSpan.sessions must be non-empty")


def chunk_spans(sessions: Sequence[date], granularity: Granularity) -> list[ChunkSpan]:
    """Split sessions into cache units.

    Inputs: strictly increasing session dates and the cache granularity
    (``CacheSection.granularity``). Output: ``"chunk"`` → one span per ISO week
    (``data.panel.week_chunks`` semantics); ``"session"`` → one span per
    session, each replayed with exactly ``warmup_sessions`` prefix sessions so a
    session's path is identical whatever window it is reached from. An empty
    input gives ``[]``. Raises ValueError for unordered sessions or an unknown
    granularity. Pure; no timing dependence.
    """
    days = tuple(sessions)
    for prev, cur in zip(days, days[1:], strict=False):
        if not cur > prev:
            raise ValueError(f"sessions must be strictly increasing, got {prev} then {cur}")
    if granularity == "session":
        return [ChunkSpan((d,)) for d in days]
    if granularity == "chunk":
        if not days:
            return []
        weeks = week_chunks(pd.DatetimeIndex([pd.Timestamp(d) for d in days]))
        return [ChunkSpan(tuple(ts.date() for ts in wk)) for wk in weeks]
    raise ValueError(f"unknown granularity {granularity!r}; expected 'chunk' or 'session'")


@dataclass(frozen=True)
class DedupeReport:
    """How many replays mask-hash dedup saves over a set of configs.

    Attributes:
        total: Number of configs (mask hashes) considered.
        unique: Number of distinct mask hashes, i.e. replays actually needed.
    """

    total: int
    unique: int

    @property
    def rate(self) -> float:
        """Fraction of replays avoided, ``1 - unique / total`` (0.0 when ``total == 0``)."""
        return 1.0 - self.unique / self.total if self.total else 0.0


def dedupe_report(mask_hashes: Sequence[str]) -> DedupeReport:
    """Summarize dedup over a list of mask hashes.

    Input: one mask hash per config (duplicates expected). Output: the total
    count and the number of distinct hashes. Pure; no timing dependence.
    """
    return DedupeReport(total=len(mask_hashes), unique=len(set(mask_hashes)))
