"""`fixture` source: deterministic synthetic bars, every `synthetic_bars` kwarg passed through unchanged."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.sources.base import register_source


@register_source
class FixtureSource:
    source_type: ClassVar[str] = "fixture"

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def load(self, symbols: list[str] | None, start: Any, end: Any, timeframe: str = "1Min") -> pd.DataFrame:
        """Synthetic 1-minute bars; `symbols`/`start`/`end` override the constructor kwargs when not None.

        `timeframe` is ignored (fixtures are minute bars). `ts_available` is the bar close, as everywhere.
        """
        params = dict(self.kwargs)
        for name, value in (("symbols", symbols), ("start", start), ("end", end)):
            if value is not None:
                params[name] = value
        return synthetic_bars(**params)
