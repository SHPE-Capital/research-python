"""Entry filters (T3): drop candidates a strategy should not take, with a reason per row.

Filters read precomputed columns of ``frame`` (the candidate rows joined to feature columns, same
index as the candidates) and never compute features. A filter spec is a mapping
``{"filter_type": <key>, **params}`` (``"type"`` is accepted as an alias), the form stored in
``StrategySpec.filters``. ``build_filter`` turns one into a filter object through ``FILTER_REGISTRY``.

Timing: every input must already be known at the candidate's ``decision_ts``. Time windows use the
New York wall-clock time of ``decision_ts``; blackout calendars only use events whose ``known_ts``
is at or before ``decision_ts``.

Missing data: a missing column raises ``KeyError`` naming the filter and the column. A NaN value in
a present column fails the filter (the candidate is skipped), because "not known" cannot pass a
floor or a band.

======================  ==========================================  =================================
key                     params                                      skips when
======================  ==========================================  =================================
``time_window``         ``start``, ``end`` ("HH:MM", New York)      decision time not in [start, end)
``min_liquidity``       ``column``, ``min``                         column < min
``max_spread``          ``column`` (spread in bps), ``max_bps``     column > max_bps
``regime``              ``column``, ``min``, ``max`` (each opt.)    column outside [min, max]
``blackout``            ``calendar``, ``sessions_before``,          inside an event window
                        ``sessions_after``
``min_rvol``            ``column``, ``min``                         column < min
``meta_score``          ``column``, ``min_probability``             column < min_probability
======================  ==========================================  =================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from shpe_research.data.calendar import NY, get_calendar

FILTER_REGISTRY: dict[str, type] = {}

#: Named event calendars (``earnings``, ``fomc``) registered by the caller with ``register_event_calendar``.
EVENT_CALENDARS: dict[str, pd.DataFrame] = {}


def register_filter(cls: type) -> type:
    """Class decorator: register a filter class under its ``filter_type`` key."""
    key = cls.filter_type  # type: ignore[attr-defined]
    if key in FILTER_REGISTRY:
        raise ValueError(f"filter_type {key!r} is already registered")
    FILTER_REGISTRY[key] = cls
    return cls


def register_event_calendar(name: str, events: pd.DataFrame) -> None:
    """Make an event table available to ``blackout`` under ``name`` (e.g. ``"fomc"``).

    ``events`` has ``event_date`` (New York date), optional ``symbol`` (absent or NaN = applies to all
    symbols) and optional ``known_ts`` (UTC time the date became public; absent = always known, which
    is only safe for dates scheduled far ahead, such as FOMC meetings).
    """
    EVENT_CALENDARS[name] = events


def _column(frame: pd.DataFrame, filter_type: str, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"filter {filter_type!r}: column {column!r} is not in the frame")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _hhmm(value: str) -> int:
    h, m = str(value).split(":")
    return int(h) * 60 + int(m)


@register_filter
@dataclass(frozen=True)
class TimeWindow:
    """Keep decisions whose New York time of day is in ``[start, end)``."""

    filter_type: ClassVar[str] = "time_window"
    start: str = "09:30"
    end: str = "16:00"

    def __post_init__(self) -> None:
        if _hhmm(self.start) >= _hhmm(self.end):
            raise ValueError(f"time_window needs start < end, got {self.start!r} and {self.end!r}")

    def mask(self, candidates: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
        """True where ``decision_ts`` (UTC) falls in the window in New York time."""
        ny = pd.to_datetime(candidates["decision_ts"], utc=True).dt.tz_convert(NY)
        minute = ny.dt.hour * 60 + ny.dt.minute
        return (minute >= _hhmm(self.start)) & (minute < _hhmm(self.end))


@register_filter
@dataclass(frozen=True)
class MinLiquidity:
    """Keep rows where ``column`` (dollar volume or ADV, known at decision time) is at least ``min``."""

    filter_type: ClassVar[str] = "min_liquidity"
    column: str
    min: float

    def mask(self, candidates: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
        return _column(frame, self.filter_type, self.column) >= self.min


@register_filter
@dataclass(frozen=True)
class MaxSpread:
    """Keep rows where ``column`` (quoted spread in basis points) is at most ``max_bps``."""

    filter_type: ClassVar[str] = "max_spread"
    column: str
    max_bps: float

    def mask(self, candidates: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
        return _column(frame, self.filter_type, self.column) <= self.max_bps


@register_filter
@dataclass(frozen=True)
class Regime:
    """Keep rows where the regime measure in ``column`` is within ``[min, max]`` (either bound optional)."""

    filter_type: ClassVar[str] = "regime"
    column: str
    min: float | None = None
    max: float | None = None

    def mask(self, candidates: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
        v = _column(frame, self.filter_type, self.column)
        ok = v.notna()
        if self.min is not None:
            ok &= v >= self.min
        if self.max is not None:
            ok &= v <= self.max
        return ok


@register_filter
@dataclass(frozen=True)
class MinRvol:
    """Keep rows where relative volume in ``column`` is at least ``min``."""

    filter_type: ClassVar[str] = "min_rvol"
    column: str
    min: float

    def mask(self, candidates: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
        return _column(frame, self.filter_type, self.column) >= self.min


@register_filter
@dataclass(frozen=True)
class MetaScore:
    """Keep rows where the meta-model probability in ``column`` is at least ``min_probability``."""

    filter_type: ClassVar[str] = "meta_score"
    column: str
    min_probability: float

    def mask(self, candidates: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
        return _column(frame, self.filter_type, self.column) >= self.min_probability


def _load_events(calendar: Any) -> pd.DataFrame:
    if isinstance(calendar, pd.DataFrame):
        events = calendar
    elif isinstance(calendar, str) and calendar in EVENT_CALENDARS:
        events = EVENT_CALENDARS[calendar]
    elif isinstance(calendar, str | Path) and Path(calendar).is_file():
        path = Path(calendar)
        events = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    else:
        raise ValueError(
            f"blackout calendar {calendar!r} is not a DataFrame, a file path or a registered calendar "
            f"{sorted(EVENT_CALENDARS)}; register it with register_event_calendar()"
        )
    if "event_date" not in events.columns:
        raise KeyError("filter 'blackout': the calendar has no 'event_date' column")
    return events


@register_filter
@dataclass(frozen=True, eq=False)
class Blackout:
    """Skip decisions within ``sessions_before`` / ``sessions_after`` trading sessions of an event.

    ``calendar`` is an events table (see ``register_event_calendar``), a registered name, or a
    csv/parquet path. An event only counts when its ``known_ts`` is at or before the decision's
    ``decision_ts``, so a date announced later cannot block earlier decisions. An event date that
    is not a trading session maps to the next session.
    """

    filter_type: ClassVar[str] = "blackout"
    calendar: Any
    sessions_before: int = 0
    sessions_after: int = 0
    _events: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_events", _load_events(self.calendar))

    def mask(self, candidates: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
        """True where the decision is outside every known event window."""
        ev = self._events
        dts = pd.to_datetime(candidates["decision_ts"], utc=True)
        day = dts.dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
        cal = get_calendar()
        sessions = cal.sessions(cal.first_session, cal.last_session)
        cand_pos = pd.Series(sessions.searchsorted(day.to_numpy(), side="left"), index=candidates.index)
        ev_pos = sessions.searchsorted(pd.to_datetime(ev["event_date"]).dt.normalize().to_numpy(), side="left")
        ev_sym = ev["symbol"].astype(object).to_numpy() if "symbol" in ev.columns else np.full(len(ev), None)
        if "known_ts" in ev.columns:
            ev_known = pd.to_datetime(ev["known_ts"], utc=True).dt.tz_localize(None).to_numpy("datetime64[ns]")
        else:
            ev_known = np.full(len(ev), np.datetime64("NaT", "ns"))
        blocked = pd.Series(False, index=candidates.index)
        for sym, idx in candidates.groupby("symbol", sort=False).groups.items():
            sel = pd.isna(ev_sym) | (ev_sym == sym)
            if not sel.any():
                continue
            p = cand_pos.loc[idx].to_numpy()[:, None]
            t = dts.loc[idx].dt.tz_localize(None).to_numpy("datetime64[ns]")[:, None]
            ep = ev_pos[sel][None, :]
            known = np.isnat(ev_known[sel])[None, :] | (ev_known[sel][None, :] <= t)
            hit = known & (p >= ep - self.sessions_before) & (p <= ep + self.sessions_after)
            blocked.loc[idx] = hit.any(axis=1)
        return ~blocked


def build_filter(spec: dict[str, Any] | Any) -> Any:
    """Build a filter from a spec mapping ``{"filter_type": key, **params}``; objects pass through.

    Raises ``ValueError`` listing the registered keys for an unknown ``filter_type``.
    """
    if not isinstance(spec, dict):
        return spec
    params = dict(spec)
    key = params.pop("filter_type", None) or params.pop("type", None)
    params.pop("type", None)
    if key not in FILTER_REGISTRY:
        raise ValueError(f"unknown filter_type {key!r}; registered: {sorted(FILTER_REGISTRY)}")
    return FILTER_REGISTRY[key](**params)


def apply_filters(
    candidates: pd.DataFrame, frame: pd.DataFrame, filters: Sequence[Any], return_all: bool = False
) -> Any:
    """Run ``filters`` in order and return ``(keep, skip_reason)``, both indexed like ``candidates``.

    Inputs: ``candidates`` (a CandidateFrame, needs ``symbol`` and ``decision_ts``), ``frame``
    (the same rows, same index, joined to precomputed feature columns), and filter specs or
    objects. ``keep`` is bool; ``skip_reason`` is the ``filter_type`` of the first failing filter
    and ``""`` for kept rows. Every filter runs on every row, so with ``return_all=True`` a third
    Series (``skip_reasons_all``) holds the list of every failing filter per row, in filter order.
    Everything must be known at ``decision_ts``; a missing column raises ``KeyError``.
    """
    if not frame.index.equals(candidates.index):
        raise ValueError("frame must have the same index as candidates")
    n = len(candidates)
    keep = pd.Series(True, index=candidates.index, dtype=bool)
    reason = pd.Series("", index=candidates.index, dtype=object)
    names: list[str] = []
    fails: list[np.ndarray] = []
    for f in (build_filter(s) for s in filters):
        ok = f.mask(candidates, frame).reindex(candidates.index).fillna(False).astype(bool).to_numpy()
        names.append(f.filter_type)
        fails.append(~ok)
        newly = keep.to_numpy() & ~ok
        reason[newly] = f.filter_type
        keep &= ok
    if return_all:
        arr = np.array(fails).T if fails else np.zeros((n, 0), dtype=bool)
        all_reasons = pd.Series(
            [[nm for nm, hit in zip(names, row, strict=True) if hit] for row in arr],
            index=candidates.index,
            dtype=object,
        )
        return keep, reason, all_reasons
    return keep, reason

