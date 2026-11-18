"""D7 weekly universe providers: which assets the calibration may trade in each ISO week.

A provider is a picklable object with a ``version`` (part of every cache key) and ``week(monday)``.
``StaticUniverseProvider`` returns the same members every week; the real point-in-time schedule
replaces it when MO-A5 lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

from shpe_research.calibration.window import Archive, UniverseWeek


class UniverseProvider(Protocol):
    """Weekly universe schedule. Implementations must be picklable (spawn workers receive them)."""

    version: str

    def week(self, week_start: date) -> UniverseWeek:
        """Members of the ISO week starting ``week_start`` (a Monday), known before that week opens."""
        ...


class StaticUniverseProvider:
    """The same members every week.

    TODO(MO-A5): replace with the data/weekly_universe.py schedule.
    """

    def __init__(self, members: Sequence[tuple[str, str]], version: str) -> None:
        """Bind the members.

        Inputs: ``(asset_id, symbol)`` pairs (any order; asset ids unique) and the schedule version.
        No timing dependence: the members are fixed, so no week can reveal later information.
        """
        self.members: tuple[tuple[str, str], ...] = tuple(sorted((str(a), str(s)) for a, s in members))
        if len({a for a, _ in self.members}) != len(self.members):
            raise ValueError("StaticUniverseProvider members must have unique asset ids")
        self.version = version

    @classmethod
    def from_archive(cls, archive: Archive, version: str = "static-archive-v1") -> StaticUniverseProvider:
        """Every archive asset, with symbol = asset_id.

        Input: an opened archive (only its asset ids are read). Output: a provider whose every week
        holds all the archive's assets. Survivorship caveat: an archive built from today's listings
        carries that bias until MO-A5 supplies the point-in-time schedule.
        """
        return cls([(a, a) for a in archive.asset_ids], version)

    def week(self, week_start: date) -> UniverseWeek:
        """Return the members for the ISO week starting ``week_start`` (must be a Monday)."""
        return UniverseWeek(week_start=week_start, members=self.members, version=self.version)
