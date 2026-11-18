"""Shared test helpers for the calibration tests (Tasks 9-12).

Everything here is module-level and picklable so spawn workers can unpickle it.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import numpy as np

from shpe_research.calibration.cache import WindowResult
from shpe_research.calibration.keys import ChunkSpan
from shpe_research.calibration.replay import ChunkReplayer, ReplayCounters, ReplayRequest, ReplayResponse
from shpe_research.calibration.window import UniverseWeek
from shpe_research.strategies.momentum.breakout.grid import Config

# Cache-state fields (Global Constraints): they depend on what the cache already held, so every
# identity test removes them before comparing. Artifact ``runtime``, ``evaluations.actual``,
# ``StepRecord.actual_evaluations``, audit ``cache_hit`` flags and ``actual`` counts, and the
# ``run.json`` ``chunk_replays`` / ``cache_stats``.
_CACHE_STATE_KEYS = frozenset({"runtime", "actual", "actual_evaluations", "cache_hit", "chunk_replays", "cache_stats"})
_CACHE_STATE_KINDS = frozenset({"replay_counters"})


class AllMembersUniverse:
    """A static universe: every week's members are all ``asset_ids`` (symbol = asset id)."""

    def __init__(self, asset_ids: Sequence[str], version: str = "static-test") -> None:
        self.asset_ids = tuple(sorted(asset_ids))
        self.version = version

    def week(self, week_start: date) -> UniverseWeek:
        """Members of the ISO week starting ``week_start`` (a Monday): every asset."""
        return UniverseWeek(week_start=week_start, members=tuple((a, a) for a in self.asset_ids), version=self.version)


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _CACHE_STATE_KEYS and k not in _CACHE_STATE_KINDS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj if not (isinstance(v, dict) and v.get("kind") in _CACHE_STATE_KINDS)]
    return obj


def strip_cache_state(obj: dict) -> dict:
    """Return a deep copy of ``obj`` without the cache-state fields named in Global Constraints.

    Removes, at any depth, the keys ``runtime``, ``actual`` (``evaluations.actual`` and the audit
    status ``actual`` count), ``actual_evaluations`` (``StepRecord``), ``cache_hit``,
    ``chunk_replays``, ``cache_stats`` and ``replay_counters``, and drops list items that are
    ``{"kind": "replay_counters", ...}`` audit lines. The input is not modified.
    """
    return _strip(copy.deepcopy(obj))


def short_breakout(config: Config) -> bool:
    """Predicate: the config's breakout lookback is the shortest grid value (10), i.e. the most trades."""
    return config.as_dict()["breakout.n"] == 10


def every_config(config: Config) -> bool:
    """Predicate: always True."""
    return True


class PlantedEdgeReplayer:
    """Wraps a real ``ChunkReplayer`` and adds a fixed ``delta`` to the daily returns of chosen configs.

    ``run`` delegates to the wrapped replayer (so the cache, counters and entries stay real) and, for
    each request whose config satisfies ``predicate``, returns the response with ``delta`` added to
    the window's daily returns on ``sessions`` (every session when None). Trade counts and entries are
    untouched, so the acceptance floors stay meaningful. Wrappers nest (plant a gain and a loss).
    """

    def __init__(
        self,
        inner: ChunkReplayer | PlantedEdgeReplayer,
        predicate: Callable[[Config], bool],
        delta: float,
        sessions: Sequence[date] | None = None,
    ) -> None:
        self.inner, self.predicate, self.delta = inner, predicate, delta
        self.sessions = None if sessions is None else frozenset(sessions)

    @property
    def counters(self) -> ReplayCounters:
        """The wrapped replayer's counters."""
        return self.inner.counters

    @property
    def by_strategy(self) -> dict[str, ReplayCounters]:
        """The wrapped replayer's per-strategy counters."""
        return self.inner.by_strategy

    def _plant(self, window: WindowResult) -> WindowResult:
        if self.sessions is None:
            mask = np.ones(len(window.sessions), dtype=bool)
        else:
            mask = np.array([s in self.sessions for s in window.sessions], dtype=bool)
        return dataclasses.replace(window, daily_returns=window.daily_returns + self.delta * mask)

    def run(self, requests: Sequence[ReplayRequest], spans: Sequence[ChunkSpan], cutoff: date) -> list[ReplayResponse]:
        """Delegate to the wrapped replayer, then plant ``delta`` into the matching responses."""
        out = self.inner.run(requests, spans, cutoff)
        return [
            dataclasses.replace(resp, window=self._plant(resp.window)) if self.predicate(req.config) else resp
            for req, resp in zip(requests, out, strict=True)
        ]
