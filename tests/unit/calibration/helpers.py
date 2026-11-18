"""Shared test helpers for the calibration tests (Tasks 9-12).

Everything here is module-level and picklable so spawn workers can unpickle it.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from datetime import date
from typing import Any

from shpe_research.calibration.window import UniverseWeek

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
