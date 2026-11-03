"""Decision samplers: which bars become dataset rows (C1b).

Every sampler is a frozen dataclass registered with `@register_sampler`, built from config as
`{type: <key>, ...params}`, and run with `sampler.sample(bars)`. It returns a CandidateFrame
(contracts/candidates.py): one row per decision, sorted by `decision_ts` then `symbol`, fresh RangeIndex.

Timing rules shared by all samplers in this package:

* `decision_ts` is the `ts_available` of a bar that exists for that symbol, never a bar's `ts`.
* A decision uses only bars with `ts_available <= decision_ts`. Session opens and closes come from the
  exchange calendar (a fixed schedule, not data), so truncating the bars never changes an earlier decision.
* Only regular-session bars are used. Missing minutes are skipped; nothing is forward-filled.
* `side` is `long` unless a sampler takes a `side` param; the label then reads as a signed forward return.

This module holds the shared helpers and the clock-based samplers (`every_n_minutes`, `session_open`,
`session_close`) and the `candidates` sampler. Importing it also imports `thresholds` and `breakouts`, so
their samplers register too.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import yaml

from shpe_research.contracts.base import UTC_NS
from shpe_research.contracts.candidates import CandidateFrame, validate_candidates
from shpe_research.data.calendar import get_calendar, session_date

# The sampler registry lives in pipeline/, which sits above events in the layer order (C1l), so the decorator is
# resolved at import time without a static import. Move the registry down or allow the edge in C1l to remove this.
register_sampler = importlib.import_module("shpe_research.pipeline.registry").register_sampler

_MINUTE = pd.Timedelta(minutes=1)
_DAILY_BAR = pd.Timedelta(hours=2)  # a bar longer than this (in ts_available - ts) is a daily bar
_LEVEL_COLUMNS = ["symbol", "decision_ts"]


def session_clock(bars: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Regular-session bars sorted by (symbol, ts), with the calendar clock added, and whether they are daily.

    Inputs: bars per contracts/bars.py. Adds `_day` (New York session date), `_open` and `_close` (UTC calendar
    times of that session, early closes included) and `_minute` (minutes from the open to the bar's
    `ts_available`, 1 for the first regular minute bar). These are schedule facts, known before the session
    starts. Daily bars (ts = open, ts_available = close) get the same columns.
    """
    missing = {"symbol", "ts", "ts_available", "session"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars are missing columns {sorted(missing)}")
    reg = bars[bars["session"] == "regular"].sort_values(["symbol", "ts"], kind="stable")
    reg.attrs = {}  # fixture truth in attrs must not travel into the candidates
    if reg.empty:
        return reg.assign(_day=pd.NaT, _open=pd.NaT, _close=pd.NaT, _minute=float("nan")), False
    daily = bool((reg["ts_available"] - reg["ts"]).median() > _DAILY_BAR)
    day = session_date(reg["ts"])
    schedule = get_calendar().schedule(day.min(), day.max())
    opens = pd.Series(schedule["open"].reindex(day.to_numpy()).to_numpy(), index=reg.index).astype(UTC_NS)
    closes = pd.Series(schedule["close"].reindex(day.to_numpy()).to_numpy(), index=reg.index).astype(UTC_NS)
    reg = reg.assign(_day=day, _open=opens, _close=closes)
    reg["_minute"] = (reg["ts_available"] - reg["_open"]) / _MINUTE
    return reg, daily


def strategy_id(sampler: Any) -> str:
    """`<sampler_type>:<8-hex hash of its params>`, stable across runs."""
    payload = json.dumps(asdict(sampler), sort_keys=True, default=str)
    return f"{sampler.sampler_type}:{hashlib.sha1(payload.encode()).hexdigest()[:8]}"


def as_candidates(rows: pd.DataFrame, sampler: Any, side: str = "long") -> CandidateFrame:
    """Finish a sampler's output: add side, strategy_id and action, sort, reindex and validate.

    `rows` holds `symbol`, `decision_ts` (a bar's ts_available) and optional candidate columns.
    """
    out = rows.assign(side=side, strategy_id=strategy_id(sampler), action="enter")
    out = out.drop_duplicates(_LEVEL_COLUMNS).sort_values(["decision_ts", "symbol"], kind="stable")
    out = out.reset_index(drop=True)
    if out.empty:
        out = out.astype({"decision_ts": UTC_NS})
    validate_candidates(out)
    return out


class Sampler:
    """Base for samplers: `sample(bars)` plus per-call param overrides."""

    sampler_type: ClassVar[str]

    def sample(self, bars: pd.DataFrame, **params: Any) -> CandidateFrame:
        """Decisions in `bars`; keyword params override the configured ones for this call."""
        return (replace(self, **params) if params else self)._sample(bars)  # type: ignore[type-var]

    def _sample(self, bars: pd.DataFrame) -> CandidateFrame:
        raise NotImplementedError


@register_sampler
@dataclass(frozen=True)
class EveryNMinutes(Sampler):
    """One decision per symbol every `n` regular-session minutes (daily bars: every bar).

    Decision times are the `ts_available` of bars whose minutes since the open is a multiple of `n`.
    `minutes` is the C0 name for `n` and is kept so existing configs and `mask` keep working.
    """

    sampler_type: ClassVar[str] = "every_n_minutes"
    n: int | None = None
    minutes: int | None = None

    def __post_init__(self) -> None:
        value = self.n if self.n is not None else self.minutes
        value = 1 if value is None else int(value)
        if value < 1:
            raise ValueError("every_n_minutes needs n >= 1")
        object.__setattr__(self, "n", value)
        object.__setattr__(self, "minutes", value)

    def mask(self, decision_ts: pd.Series) -> pd.Series:
        """True for decision times that are a multiple of `n` minutes after the session open."""
        from shpe_research.features.session.minute_of_session import minutes_since_open

        return (minutes_since_open(decision_ts) % int(self.n)) == 0  # type: ignore[arg-type]

    def _sample(self, bars: pd.DataFrame) -> CandidateFrame:
        reg, daily = session_clock(bars)
        keep = reg if daily else reg[(reg["_minute"] % int(self.n)) == 0]  # type: ignore[arg-type]
        return as_candidates(keep.rename(columns={"ts_available": "decision_ts"})[_LEVEL_COLUMNS], self)


@register_sampler
@dataclass(frozen=True)
class SessionOpen(Sampler):
    """One decision per symbol per session at 09:30 + `offset_minutes`.

    The decision is the `ts_available` of the bar that starts at that time (offset 0: the 09:31 timestamp). A
    session without that bar yields no row. Needs intraday bars.
    """

    sampler_type: ClassVar[str] = "session_open"
    offset_minutes: int = 0

    def _sample(self, bars: pd.DataFrame) -> CandidateFrame:
        reg, daily = session_clock(bars)
        if daily:
            raise ValueError("session_open needs intraday bars")
        keep = reg[reg["_minute"] == int(self.offset_minutes) + 1]
        return as_candidates(keep.rename(columns={"ts_available": "decision_ts"})[_LEVEL_COLUMNS], self)


@register_sampler
@dataclass(frozen=True)
class SessionClose(Sampler):
    """One decision per symbol per session, `offset_minutes` before the calendar close.

    Intraday: the bar whose `ts_available` is the session close minus the offset (default 1: the second-to-last
    minute, so an order can still be sent at the close). The close is the calendar's, early closes included, so
    it is known before the session starts. A session without that bar yields no row. Daily bars: the bar's own
    `ts_available`, which is the close.
    """

    sampler_type: ClassVar[str] = "session_close"
    offset_minutes: int = 1

    def _sample(self, bars: pd.DataFrame) -> CandidateFrame:
        reg, daily = session_clock(bars)
        keep = reg if daily else reg[(reg["_close"] - reg["ts_available"]) == self.offset_minutes * _MINUTE]
        return as_candidates(keep.rename(columns={"ts_available": "decision_ts"})[_LEVEL_COLUMNS], self)


@register_sampler
@dataclass(frozen=True)
class Candidates(Sampler):
    """Run a registered strategy's `CandidateGenerator` and return its validated frame.

    `strategy` is a path to a YAML strategy config, or the mapping itself: `{type: <strategy_type>, ...params}`
    or a StrategySpec-shaped `{strategy_type: ..., params: {...}}`. The generator owns its timing rules; this
    sampler only validates the frame and sorts it.
    """

    sampler_type: ClassVar[str] = "candidates"
    strategy: Any = None

    def sample(self, bars: pd.DataFrame, context: pd.DataFrame | None = None, **params: Any) -> CandidateFrame:
        """The strategy's candidates for `bars`; `context` is passed to it as its context-symbol bars."""
        me = replace(self, **params) if params else self
        return me._run(bars, context)

    def _sample(self, bars: pd.DataFrame) -> CandidateFrame:
        return self._run(bars, None)

    def _run(self, bars: pd.DataFrame, context: pd.DataFrame | None) -> CandidateFrame:
        if self.strategy is None:
            raise ValueError("the candidates sampler needs a `strategy` (config path or mapping)")
        spec = self.strategy
        if isinstance(spec, str | Path):
            spec = yaml.safe_load(Path(spec).read_text())
        spec = dict(spec)
        if "strategy_type" in spec:
            spec = {"type": spec["strategy_type"], **spec.get("params", {})}
        # strategies sits above events in the layer order, so it is resolved at run time, not imported.
        registry = importlib.import_module("shpe_research.strategies.registry")
        frame = registry.build_strategy(spec).generate(bars, context)
        validate_candidates(frame)
        return frame.sort_values(["decision_ts", "symbol"], kind="stable").reset_index(drop=True)


# Register the other samplers of this package whichever module is imported first.
from shpe_research.events import breakouts, thresholds  # noqa: E402,F401
