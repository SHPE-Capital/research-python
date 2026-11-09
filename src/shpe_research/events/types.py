"""Event-layer types shared with strategies."""

from __future__ import annotations

from typing import ClassVar, Protocol

import pandas as pd

from shpe_research.contracts.candidates import CandidateFrame

__all__ = ["CandidateFrame", "CandidateGenerator"]


class CandidateGenerator(Protocol):
    """Turns bars into candidate decisions, using only data available at each decision_ts."""

    strategy_type: ClassVar[str]

    def generate(self, bars: pd.DataFrame, context: pd.DataFrame | None = None) -> CandidateFrame:
        """Candidate rows for `bars`; `context` holds optional context-symbol bars (C1d)."""
        ...
