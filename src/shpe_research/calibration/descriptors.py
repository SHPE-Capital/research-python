"""D7 market descriptors for similarity weighting.

A provider returns one descriptor row per session. The row for session ``s`` (``rows``) uses data
through the session before ``s`` only, so it is known when ``s`` opens; ``current(cutoff)`` uses data
through the cutoff session inclusive (known at decision time). Robust scaling of the rows is fitted
on search rows only, by the caller (``descriptor_scaling``).

``PanelProxyDescriptors`` stands in for the real market descriptors (MO-A7) using only the archive:
an equal-weight return of the eligible universe replaces SPY.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

import numpy as np

from shpe_research.calibration.window import Archive
from shpe_research.data.panel import list_chunks, open_panel_chunk, read_meta

N_DESCRIPTORS = 3


class DescriptorProvider(Protocol):
    """Per-session market descriptors known before each session opens."""

    version: str

    def rows(self, sessions: Sequence[date]) -> np.ndarray:
        """``(n, k)`` rows; the row for session ``s`` uses data through the session before ``s``."""
        ...

    def current(self, cutoff: date) -> np.ndarray:
        """``(k,)`` row using data through the ``cutoff`` session inclusive."""
        ...


def _last_finite(block: np.ndarray) -> np.ndarray:
    """Last finite value of each row of ``block`` (NaN where a row has none)."""
    finite = np.isfinite(block)
    has = finite.any(axis=1)
    idx = block.shape[1] - 1 - np.argmax(finite[:, ::-1], axis=1)
    return np.where(has, block[np.arange(block.shape[0]), idx], np.nan)


class PanelProxyDescriptors:
    """Three descriptors from the archive alone (k = 3).

    With ``r_t`` = the mean over symbols eligible in session ``t`` (and with a finite daily close in
    ``t`` and ``t - 1``) of ``close_t / close_{t-1} - 1`` (the SPY proxy; daily close = the session's
    last finite minute close), the descriptors at session ``i`` over the last ``lookback`` returns
    ``r_{i-lookback+1} .. r_i`` are:

    - ``d1 = (prod(1 + r) - 1) / std(r, ddof=1)`` (trend per unit of volatility),
    - ``d2 = std(r, ddof=1)``,
    - ``d3`` = share of symbols eligible in ``i`` with ``lookback`` finite daily closes whose close at
      ``i`` is above the mean of those closes (breadth).

    A row is all-NaN when fewer than ``lookback`` returns exist (``i < lookback``); a single
    descriptor is NaN when its inputs are empty (no eligible symbol) or the volatility is zero.

    TODO(MO-A7): replace with data/market_descriptors.py.
    """

    def __init__(self, archive: Archive, *, lookback: int, through: date) -> None:
        """Bind the provider; the descriptor table is built lazily on the first ``rows``/``current`` call.

        Inputs: the archive, ``weights.descriptor_lookback`` (>= 2: a sample standard deviation needs
        two returns) and ``through``, the last archive session the provider may read (a calibration
        passes its cutoff after the reserved-period guard has accepted it; a walk-forward its last
        cutoff). Construction reads no bar. The first query reads the daily closes and eligibility up
        to ``through`` once: chunks starting after ``through`` are never opened, and in the chunk
        holding ``through`` only the columns up to its last minute are read (the chunk is
        memory-mapped and sliced before conversion). Each precomputed row ``i`` uses archive sessions
        ``<= i`` only. Raises ValueError here if ``lookback < 2`` or ``through`` is not an archive
        session.
        """
        if lookback < 2:
            raise ValueError("PanelProxyDescriptors needs lookback >= 2 (sample std of the returns)")
        self.archive = archive
        self.lookback = lookback
        self.through = through
        self._last = archive.position(through)
        self.version = f"panel-proxy-v1-lookback{lookback}"
        self._table_cache: np.ndarray | None = None

    @property
    def _table(self) -> np.ndarray:
        """The ``(sessions through through, 3)`` descriptor table, built once on first use."""
        if self._table_cache is None:
            self._table_cache = self._build_table()
        return self._table_cache

    def _build_table(self) -> np.ndarray:
        """Read the daily closes and eligibility up to ``through`` and compute the descriptors."""
        archive, through = self.archive, self.through
        closes, eligible, sessions = [], [], []
        for path in list_chunks(archive.root):
            chunk_sessions = [date.fromisoformat(s) for s in read_meta(path)["sessions"]]
            if chunk_sessions[0] > through:
                break
            keep = sum(1 for s in chunk_sessions if s <= through)
            panel = open_panel_chunk(path)
            sidx = np.asarray(panel.session_index)
            starts = np.r_[np.flatnonzero(np.r_[True, sidx[1:] != sidx[:-1]]), len(sidx)]
            if len(starts) - 1 != len(chunk_sessions):
                raise ValueError(f"chunk {path.name}: session_index does not match its meta sessions")
            end = int(starts[keep])
            close = np.asarray(panel.close[:, :end], dtype=np.float64)
            elig = np.asarray(panel.eligible[:, :end])
            for a, b in zip(starts[:keep], starts[1 : keep + 1], strict=True):
                closes.append(_last_finite(close[:, a:b]))
                eligible.append(elig[:, a:b].any(axis=1))
            sessions.extend(chunk_sessions[:keep])
        if tuple(sessions) != archive.sessions[: self._last + 1] or len(closes) != len(sessions):
            raise ValueError("archive chunks do not match the archive's session index")
        return self._descriptors(np.array(closes).T, np.array(eligible).T, self.lookback)

    @staticmethod
    def _descriptors(c: np.ndarray, e: np.ndarray, lookback: int) -> np.ndarray:
        """``(n_sessions, 3)`` descriptors; row ``i`` uses columns ``<= i`` of ``c`` and ``e`` only."""
        n = c.shape[1]
        r = np.full(n, np.nan)
        for t in range(1, n):
            ok = e[:, t] & np.isfinite(c[:, t]) & np.isfinite(c[:, t - 1])
            if ok.any():
                r[t] = np.mean(c[ok, t] / c[ok, t - 1] - 1.0)
        out = np.full((n, N_DESCRIPTORS), np.nan)
        for i in range(lookback, n):
            w = r[i - lookback + 1 : i + 1]
            if np.all(np.isfinite(w)):
                sd = float(np.std(w, ddof=1))
                out[i, 1] = sd
                if sd > 0:
                    out[i, 0] = (np.prod(1.0 + w) - 1.0) / sd
            win = c[:, i - lookback + 1 : i + 1]
            ok = e[:, i] & np.all(np.isfinite(win), axis=1)
            if ok.any():
                out[i, 2] = np.mean(c[ok, i] > win[ok].mean(axis=1))
        return out

    def rows(self, sessions: Sequence[date]) -> np.ndarray:
        """Descriptor rows for ``sessions`` (archive sessions).

        Input: session dates. Output: ``(n, 3)`` float64; the row for ``s`` is the descriptors at the
        archive session before ``s`` (all-NaN for the first archive session or a short history).
        Timing: known before ``s`` opens. Raises ValueError for a session not in the archive or whose
        previous session is after ``through``.
        """
        out = np.full((len(sessions), N_DESCRIPTORS), np.nan)
        for k, s in enumerate(sessions):
            pos = self.archive.position(s) - 1
            if pos > self._last:
                raise ValueError(f"descriptor row for {s} needs data after through={self.through}")
            if pos >= 0:
                out[k] = self._table[pos]
        return out

    def current(self, cutoff: date) -> np.ndarray:
        """Descriptors through the ``cutoff`` session inclusive.

        Input: an archive session (the decision cutoff). Output: ``(3,)`` float64. Timing: known at
        the cutoff's close. Raises ValueError for a session not in the archive or after ``through``.
        """
        pos = self.archive.position(cutoff)
        if pos > self._last:
            raise ValueError(f"current({cutoff}) needs data after through={self.through}")
        return self._table[pos].copy()
