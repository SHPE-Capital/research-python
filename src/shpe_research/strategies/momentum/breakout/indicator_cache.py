"""B5: per-panel indicator and signal cache.

Each distinct indicator array (ATR14, ADX(a) triple, preceding high per n, EMA per length, session
VWAP, RVOL ratio per w) and each derived boolean signal (per distinct parameter setting) is computed
once per ``MarketPanel`` and reused across every config that shares it. ``hits``/``misses`` count
lookups per kind so tests can show reuse. Array dtype follows the panel (float32 panels give float32
indicators); signals are bool arrays. Numeric indicator values stay reachable via ``indicator``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from shpe_research.features.momentum.breakout import breakout_signal_from_high, preceding_high
from shpe_research.features.momentum.clock_rvol import rvol_ratio, rvol_signal_from_ratio
from shpe_research.features.momentum.ema_trend import ema, ema_trend_signal
from shpe_research.features.momentum.readiness import readiness_masks
from shpe_research.features.momentum.vwap_band import session_vwap, vwap_band_signal
from shpe_research.features.momentum.wilder import ATR_LENGTH, adx_di, adx_signal, atr
from shpe_research.trade.account_types import MarketPanel


class IndicatorCache:
    """Memoises indicator arrays and signal masks for one panel."""

    def __init__(self, panel: MarketPanel) -> None:
        self.panel = panel
        self._store: dict[tuple[Any, ...], Any] = {}
        self.hits: Counter[str] = Counter()
        self.misses: Counter[str] = Counter()

    def _get(self, key: tuple[Any, ...], compute: Callable[[], Any]) -> Any:
        kind = str(key[0])
        if key in self._store:
            self.hits[kind] += 1
            return self._store[key]
        self.misses[kind] += 1
        val = compute()
        self._store[key] = val
        return val

    # numeric indicators -------------------------------------------------
    def atr14(self) -> np.ndarray:
        p = self.panel
        return self._get(("atr", ATR_LENGTH), lambda: atr(p.high, p.low, p.close, ATR_LENGTH))

    def adx(self, a: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.panel
        return self._get(("adx", int(a)), lambda: adx_di(p.high, p.low, p.close, int(a)))

    def prec_high(self, n: int) -> np.ndarray:
        return self._get(("prec_high", int(n)), lambda: preceding_high(self.panel.high, int(n)))

    def ema(self, length: int) -> np.ndarray:
        return self._get(("ema", int(length)), lambda: ema(self.panel.close, int(length)))

    def vwap(self) -> np.ndarray:
        p = self.panel
        return self._get(("vwap",), lambda: session_vwap(p.high, p.low, p.close, p.volume, p.session_index))

    def rvol(self, w: int) -> np.ndarray:
        p = self.panel
        return self._get(("rvol", int(w)), lambda: rvol_ratio(p.volume, p.session_index, int(w)))

    def indicator(self, name: str, *param: int) -> Any:
        """Accessor for logs: ``indicator("ema", 20)``, ``indicator("adx", 14)``, ``indicator("atr")``."""
        getters: dict[str, Callable[..., Any]] = {
            "atr": lambda *_: self.atr14(),
            "adx": self.adx,
            "prec_high": self.prec_high,
            "ema": self.ema,
            "vwap": lambda *_: self.vwap(),
            "rvol": self.rvol,
        }
        return getters[name](*param)

    # readiness ----------------------------------------------------------
    def readiness(self) -> dict[str, np.ndarray]:
        p = self.panel
        return self._get(
            ("readiness",),
            lambda: readiness_masks(p.high, p.low, p.close, p.volume, p.session_index, atr14=self.atr14()),
        )

    # boolean signals ----------------------------------------------------
    def breakout(self, n: int, b: float) -> np.ndarray:
        return self._get(
            ("sig_breakout", int(n), float(b)),
            lambda: breakout_signal_from_high(self.panel.close, self.prec_high(n), self.atr14(), float(b)),
        )

    def adx_sig(self, a: int, q: float) -> np.ndarray:
        def f() -> np.ndarray:
            adx, p, m = self.adx(a)
            return adx_signal(adx, p, m, float(q))

        return self._get(("sig_adx", int(a), float(q)), f)

    def ema_sig(self, fast: int, slow: int, d: float) -> np.ndarray:
        return self._get(
            ("sig_ema", int(fast), int(slow), float(d)),
            lambda: ema_trend_signal(self.ema(fast), self.ema(slow), self.atr14(), float(d)),
        )

    def rvol_sig(self, w: int, v: float) -> np.ndarray:
        return self._get(("sig_rvol", int(w), float(v)), lambda: rvol_signal_from_ratio(self.rvol(w), float(v)))

    def vwap_sig(self, e: float) -> np.ndarray:
        return self._get(
            ("sig_vwap", float(e)),
            lambda: vwap_band_signal(self.panel.close, self.vwap(), self.atr14(), float(e)),
        )

    # stats --------------------------------------------------------------
    @property
    def distinct_arrays(self) -> int:
        return sum(self.misses.values())

    @property
    def total_lookups(self) -> int:
        return sum(self.hits.values()) + sum(self.misses.values())
