"""`alpaca` source (live REST via `data/alpaca.py`) and `store` source (the local store D1 fills)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import pandas as pd

from shpe_research.contracts.bars import validate_bars
from shpe_research.data.sources.base import ny_window, register_source


@register_source
class AlpacaSource:
    source_type: ClassVar[str] = "alpaca"

    def __init__(self, feed: str = "sip", adjustment: str = "raw", client_factory: Callable[[], Any] | None = None):
        self.feed = feed
        self.adjustment = adjustment
        self._client_factory = client_factory

    def load(self, symbols: list[str] | None, start: Any, end: Any, timeframe: str = "1Min") -> pd.DataFrame:
        """Bars downloaded from Alpaca (needs ALPACA_DATA_KEY/SECRET). `symbols`, `start`, `end` are required.

        `ts_available` is the bar close (minute) or session close (daily), set by `normalize_bars`. SIP requests
        are pulled back from the last 15 minutes. Prefer a snapshot for anything you want to reproduce.
        """
        if not symbols or start is None or end is None:
            raise ValueError("the alpaca source needs symbols, start and end")
        from shpe_research.data.alpaca import AlpacaDataClient, sip_safe_end
        from shpe_research.data.bars import normalize_bars

        lo, hi = ny_window(start, end)
        if self.feed == "sip":
            hi = sip_safe_end(hi)
        client = self._client_factory() if self._client_factory else AlpacaDataClient()
        try:
            frames = [
                normalize_bars(client.bars(s.upper(), lo, hi, timeframe, self.adjustment, self.feed), s.upper(),
                               timeframe)
                for s in symbols
            ]
        finally:
            client.close()
        bars = pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"], kind="stable")
        bars = bars.reset_index(drop=True)
        validate_bars(bars)
        return bars


@register_source
class StoreSource:
    source_type: ClassVar[str] = "store"

    def __init__(self, root: str | None = None, feed: str = "sip", adjustment: str = "raw"):
        self.root = root
        self.feed = feed
        self.adjustment = adjustment

    def load(self, symbols: list[str] | None, start: Any, end: Any, timeframe: str = "1Min") -> pd.DataFrame:
        """Bars already downloaded into the local store; `symbols`, `start`, `end` are required."""
        if not symbols or start is None or end is None:
            raise ValueError("the store source needs symbols, start and end")
        from shpe_research.data.store import DataStore

        return DataStore(self.root).read_bars([s.upper() for s in symbols], start, end, timeframe=timeframe,
                                              feed=self.feed, adjustment=self.adjustment)
