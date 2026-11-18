"""D7 window planning: decision cutoff, search and validation windows, activation week, panel assembly.

Windows are defined in sessions, cache units in chunks (ISO weeks, or single sessions with
``cache.granularity = "session"``). At a cutoff (the last archive session on or before ``as_of``):

- validation is the last ``validation.sessions`` sessions up to and including the cutoff;
- search runs from the first chunk whose first session has at least ``min_prior_sessions`` earlier
  archive sessions (every chunk gets its full warm-up prefix and the similarity descriptors get
  their lookback) to the session before validation starts;
- the chunk straddling the search/validation boundary is one cache unit listed in both span lists
  and replayed once; each stage keeps its own sessions with ``WindowResult.slice_sessions``;
- a cutoff before the end of its ISO week truncates the last chunk at the cutoff;
- activation is the first full ISO week after the cutoff (the platform changes config on weekends).

``Archive.build_panel`` assembles one chunk's panel from the A6 week directories with exactly
``warmup_sessions`` earlier sessions in front, nothing after the chunk's last session, and the
universe week's membership AND-ed into ``eligible``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from shpe_research.calibration.keys import ChunkSpan, chunk_spans
from shpe_research.config.momentum import Granularity
from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import FIELDS, list_chunks, open_panel_chunk, read_meta
from shpe_research.trade.account_types import MarketPanel

_DAYS_PER_WEEK = 7


class InsufficientHistoryError(ValueError):
    """The archive does not hold enough sessions before a requested point."""


@dataclass(frozen=True)
class UniverseWeek:
    """The tradable members of one ISO week.

    Attributes:
        week_start: The ISO Monday of the week (callers pass ``iso_monday(...)``).
        members: ``(asset_id, symbol)`` pairs, sorted by asset_id, unique.
        version: Identity of the universe schedule that produced the week.
    """

    week_start: date
    members: tuple[tuple[str, str], ...]
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple((str(a), str(s)) for a, s in self.members))
        if self.week_start.weekday() != 0:
            raise ValueError(f"UniverseWeek.week_start must be a Monday, got {self.week_start}")
        ids = [a for a, _ in self.members]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise ValueError("UniverseWeek.members must be sorted by asset_id and unique")


def iso_monday(d: date) -> date:
    """Return the Monday of ``d``'s ISO week (``d - timedelta(days=d.weekday())``). Pure; no timing dependence."""
    return d - timedelta(days=d.weekday())


@dataclass(frozen=True)
class _ChunkInfo:
    path: Path
    sessions: tuple[date, ...]
    first: int  # archive position of the chunk's first session


class Archive:
    """Read-only view of an A6 chunk archive (one directory per week under ``root``).

    Attributes:
        root: The archive directory.
        sessions: Every archive session in order.
        asset_ids: The archive's row order (identical in every chunk).
    """

    def __init__(self, root: Path, chunks: tuple[_ChunkInfo, ...], asset_ids: tuple[str, ...]) -> None:
        """Bind already-read chunk metadata (use ``Archive.open``).

        Inputs: the archive directory, each chunk's metadata in session order and the asset ids.
        Builds the session index and the session -> chunk map; reads no bar. Timing: metadata only.
        """
        self.root = root
        self._chunks = chunks
        self.asset_ids = asset_ids
        self.sessions: tuple[date, ...] = tuple(s for c in chunks for s in c.sessions)
        self._pos = {s: i for i, s in enumerate(self.sessions)}
        self._chunk_of = np.repeat(np.arange(len(chunks)), [len(c.sessions) for c in chunks])

    @classmethod
    def open(cls, root: str | Path) -> Archive:
        """Index the chunk directories under ``root`` (``list_chunks`` / ``read_meta``; no array is read).

        Input: the archive root. Output: an ``Archive``. Raises ValueError if there are no chunks, the
        chunks disagree on asset ids, or the sessions are not strictly increasing across chunks.
        """
        root = Path(root)
        paths = list_chunks(root)
        if not paths:
            raise ValueError(f"no A6 chunks under {root}")
        chunks: list[_ChunkInfo] = []
        asset_ids: tuple[str, ...] | None = None
        n = 0
        for p in paths:
            meta = read_meta(p)
            ids = tuple(meta["asset_ids"])
            if asset_ids is None:
                asset_ids = ids
            elif ids != asset_ids:
                raise ValueError(f"chunk {p.name} has different asset_ids from the first chunk")
            sessions = tuple(date.fromisoformat(s) for s in meta["sessions"])
            chunks.append(_ChunkInfo(p, sessions, n))
            n += len(sessions)
        all_sessions = [s for c in chunks for s in c.sessions]
        for prev, cur in zip(all_sessions, all_sessions[1:], strict=False):
            if not cur > prev:
                raise ValueError(f"archive sessions must be strictly increasing, got {prev} then {cur}")
        assert asset_ids is not None
        return cls(root, tuple(chunks), asset_ids)

    def position(self, session: date) -> int:
        """Return the archive position of ``session`` (0 = first). Raises ValueError if absent. No timing dependence."""
        try:
            return self._pos[session]
        except KeyError:
            raise ValueError(f"session {session} is not in the archive") from None

    def build_panel(
        self, span: ChunkSpan, warmup_sessions: int, membership: frozenset[str] | None
    ) -> tuple[MarketPanel, int]:
        """Assemble the warm-up prefix plus chunk panel for ``span``.

        Inputs: a span of consecutive archive sessions, the prefix length in sessions
        (``cache.warmup_sessions``) and the asset ids that are members of the span's ISO week
        (None = no restriction). Output: ``(panel, n_prefix)`` where the panel holds exactly
        ``warmup_sessions`` archive sessions before ``span.sessions[0]`` and the span's sessions,
        columns end with the last minute of ``span.sessions[-1]``, rows are ``asset_ids`` and
        ``eligible`` is AND-ed with membership (prefix included; prefix sessions never trade). Arrays
        keep the archive dtype; ``session_index`` is the chunks' stored global numbering.

        Timing: nothing after ``span.sessions[-1]`` is read into the panel. Raises
        InsufficientHistoryError if fewer than ``warmup_sessions`` earlier sessions exist and
        ValueError if the span is not a contiguous run of archive sessions.
        """
        if warmup_sessions < 0:
            raise ValueError("warmup_sessions must be >= 0")
        lo = self.position(span.sessions[0])
        hi = lo + len(span.sessions)
        if self.sessions[lo:hi] != span.sessions:
            raise ValueError(f"span starting {span.sessions[0]} is not a contiguous run of archive sessions")
        if lo < warmup_sessions:
            raise InsufficientHistoryError(
                f"span starting {span.sessions[0]} has {lo} earlier archive sessions, needs {warmup_sessions}"
            )
        first = lo - warmup_sessions
        pieces: dict[str, list[np.ndarray]] = {name: [] for name in (*FIELDS, "eligible", "session_index")}
        for k in range(int(self._chunk_of[first]), int(self._chunk_of[hi - 1]) + 1):
            info = self._chunks[k]
            panel = open_panel_chunk(info.path)
            sidx = np.asarray(panel.session_index)
            starts = np.flatnonzero(np.r_[True, sidx[1:] != sidx[:-1]])
            if len(starts) != len(info.sessions):
                raise ValueError(
                    f"chunk {info.path.name}: session_index has {len(starts)} sessions, meta lists {len(info.sessions)}"
                )
            bounds = np.r_[starts, len(sidx)]
            a = max(first, info.first) - info.first
            b = min(hi, info.first + len(info.sessions)) - info.first
            cols = slice(int(bounds[a]), int(bounds[b]))
            for name in pieces:
                arr = np.asarray(getattr(panel, name))
                pieces[name].append(arr[cols] if arr.ndim == 1 else arr[:, cols])
        arrays = {name: np.concatenate(parts, axis=-1) for name, parts in pieces.items()}
        if membership is not None:
            member = np.fromiter((a in membership for a in self.asset_ids), dtype=np.bool_, count=len(self.asset_ids))
            arrays["eligible"] = arrays["eligible"] & member[:, None]
        out = MarketPanel(asset_ids=self.asset_ids, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
        return out, warmup_sessions


@dataclass(frozen=True)
class WindowPlan:
    """Search, validation and activation windows at one cutoff.

    Attributes:
        cutoff: The last completed archive session (decision time).
        search_sessions: The expanding search window, ending the session before validation starts.
        validation_sessions: The last ``validation.sessions`` sessions, ending on the cutoff.
        activation_sessions: Calendar sessions of the first ISO week after the cutoff.
        search_spans: Cache units covering the search sessions.
        validation_spans: Cache units covering the validation sessions. The unit straddling the
            validation start is listed in both span lists; it is replayed once and each stage takes
            its own sessions with ``WindowResult.slice_sessions``. ``ambiguous_bars`` is not stored
            per session, so a stage's count is the total over the stage's own spans (straddling unit
            included whole): an upper bound for the stage. Replay each stage with its own span list;
            never slice a window replayed over both stages' spans and read its ``ambiguous_bars``.
    """

    cutoff: date
    search_sessions: tuple[date, ...]
    validation_sessions: tuple[date, ...]
    activation_sessions: tuple[date, ...]
    search_spans: tuple[ChunkSpan, ...]
    validation_spans: tuple[ChunkSpan, ...]


def lock_cutoff(archive: Archive, as_of: date) -> date:
    """Return the last archive session on or before ``as_of`` (the decision cutoff).

    Inputs: the archive and the run date. Output: a session date; data through its close is known at
    decision time. Raises InsufficientHistoryError if the archive starts after ``as_of``.
    """
    known = [s for s in archive.sessions if s <= as_of]
    if not known:
        raise InsufficientHistoryError(f"archive starts {archive.sessions[0]}, after as_of {as_of}")
    return known[-1]


def activation_sessions(cutoff: date) -> tuple[date, ...]:
    """Return the calendar sessions of the first ISO week starting after ``cutoff``.

    Input: the cutoff session. Output: the next ISO week's sessions from the exchange calendar (not
    the archive, which holds no future data). A mid-week cutoff skips the rest of its week. Raises
    ValueError if that week has no sessions.
    """
    monday = iso_monday(cutoff) + timedelta(days=_DAYS_PER_WEEK)
    days = get_calendar().sessions(monday, monday + timedelta(days=_DAYS_PER_WEEK - 1))
    if len(days) == 0:
        raise ValueError(f"the ISO week starting {monday} has no sessions")
    return tuple(d.date() for d in days)


def plan_window(
    archive: Archive, as_of: date, *, validation_sessions: int, min_prior_sessions: int, granularity: Granularity
) -> WindowPlan:
    """Plan the search and validation windows and their cache units at the cutoff for ``as_of``.

    Inputs: the archive, the run date, ``validation.sessions``, the minimum number of archive
    sessions before the first search chunk (callers pass
    ``max(cache.warmup_sessions, weights.descriptor_lookback + 1)``) and ``cache.granularity``.
    Output: a ``WindowPlan`` (see the module docstring for the rules).

    Timing: every planned session is on or before the cutoff; the activation week is after it.
    Raises InsufficientHistoryError if there are fewer than ``validation_sessions`` sessions up to
    the cutoff or no qualifying search chunk before validation starts.
    """
    if validation_sessions < 1:
        raise ValueError("validation_sessions must be >= 1")
    if min_prior_sessions < 0:
        raise ValueError("min_prior_sessions must be >= 0")
    cutoff = lock_cutoff(archive, as_of)
    hist = archive.sessions[: archive.position(cutoff) + 1]
    if len(hist) < validation_sessions:
        raise InsufficientHistoryError(f"{len(hist)} sessions up to {cutoff}, validation needs {validation_sessions}")
    val_lo = len(hist) - validation_sessions
    spans = chunk_spans(hist, granularity)
    start = next(
        (archive.position(sp.sessions[0]) for sp in spans if archive.position(sp.sessions[0]) >= min_prior_sessions),
        None,
    )
    if start is None or start >= val_lo:
        raise InsufficientHistoryError(
            f"no {granularity} with {min_prior_sessions} prior sessions starts before validation ({hist[val_lo]})"
        )
    search = hist[start:val_lo]
    validation = hist[val_lo:]
    used = [sp for sp in spans if sp.sessions[-1] >= search[0]]
    return WindowPlan(
        cutoff=cutoff,
        search_sessions=search,
        validation_sessions=validation,
        activation_sessions=activation_sessions(cutoff),
        search_spans=tuple(sp for sp in used if sp.sessions[0] <= search[-1]),
        validation_spans=tuple(sp for sp in used if sp.sessions[-1] >= validation[0]),
    )
