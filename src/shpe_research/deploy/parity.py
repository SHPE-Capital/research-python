"""Research/platform ledger reconciliation (C1a stub; T7 implements it)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParityReport:
    """Result of comparing a research ledger with platform round trips.

    `passed` only with zero research-only, zero platform-only and zero mismatches.
    """

    matched: int = 0
    research_only: list[Any] = field(default_factory=list)
    platform_only: list[Any] = field(default_factory=list)
    mismatches: list[Any] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (self.research_only or self.platform_only or self.mismatches)


def compare_ledgers(research: Any, platform_trades: Any) -> ParityReport:
    """Match trades on (symbol, direction, entry time) and report differences."""
    raise NotImplementedError("TODO(T7): compare_ledgers")
