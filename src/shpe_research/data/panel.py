"""A6: memory-mappable market panel chunks (symbol x minute, float32), one directory per week.

Layout of a chunk directory ``<root>/w<first-session-date>/``:
    open.npy high.npy low.npy close.npy volume.npy   float32, shape (n_symbols, n_minutes), C-contiguous
    eligible.npy                                      bool, same shape
    session_index.npy                                 int32, (n_minutes,), global session number per column
    minute_ts.npy                                     int64, (n_minutes,), UTC ns of each bar start
    meta.json                                         asset_ids, sessions, dtype, shapes

Columns are the regular-session minute grid (``TradingCalendar.regular_minutes``), so early closes have
210 columns instead of 390. Missing bars are NaN prices and zero volume. ``open_panel_chunk`` memory-maps
every array read-only (no copy), so many workers share the page cache and each reads only what it replays.

Scale (float32): 5 fields x 4 B + 1 B eligible = 21 B per symbol-minute, so a 1,000-symbol week is ~41 MB
and 1,000 symbols x 10 years ~ 21 GB: build and map per week.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from shpe_research.data.calendar import get_calendar
from shpe_research.data.store import DataStore

FIELDS = ("open", "high", "low", "close", "volume")
DTYPE = np.float32
META = "meta.json"

# source(sessions, minute_ts_ns, session_index_local) -> {"open","high","low","close","volume": f32, "eligible": bool}
ChunkSource = Callable[[pd.DatetimeIndex, np.ndarray, np.ndarray], dict[str, np.ndarray]]


def week_chunks(sessions: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    """Split sessions into ISO-week groups (Monday-Sunday), in order."""
    idx = pd.DatetimeIndex(sessions)
    if len(idx) == 0:
        return []
    key = idx.tz_localize(None).to_period("W-SUN") if idx.tz is not None else idx.to_period("W-SUN")
    return [idx[key == k] for k in pd.unique(key)]


def session_grid(sessions: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(minute_ts_ns int64, local session ordinal int32) for the regular-session grid of ``sessions``."""
    cal = get_calendar()
    parts = [cal.regular_minutes(d) for d in sessions]
    if not parts:
        raise ValueError("no sessions")
    ts = np.concatenate([p.asi8 for p in parts]).astype(np.int64)
    sess = np.concatenate([np.full(len(p), i, np.int32) for i, p in enumerate(parts)])
    return ts, sess


def chunk_dir_name(first_session: pd.Timestamp) -> str:
    return f"w{pd.Timestamp(first_session).date().isoformat()}"


def build_panel_chunk(
    out_root: str | Path,
    asset_ids: Sequence[str],
    sessions: pd.DatetimeIndex,
    source: ChunkSource,
    first_session_number: int = 0,
) -> Path:
    """Build one chunk directory (a week, or any session range) from ``source``; returns its path.

    Written to a temp directory and renamed, so a crash never leaves a half-written chunk.
    ``first_session_number`` offsets ``session_index`` so numbers stay global across chunks.
    """
    sessions = pd.DatetimeIndex(sessions)
    asset_ids = tuple(asset_ids)
    if len(set(asset_ids)) != len(asset_ids):
        raise ValueError("asset_ids must be unique")
    ts, local = session_grid(sessions)
    arrays = source(sessions, ts, local)
    shape = (len(asset_ids), len(ts))
    for name in FIELDS:
        if arrays[name].shape != shape:
            raise ValueError(f"source returned {name} with shape {arrays[name].shape}, expected {shape}")
    elig = arrays["eligible"]
    if elig.shape != shape or elig.dtype != np.bool_:
        raise ValueError("source eligible must be a bool array of shape (n_symbols, n_minutes)")

    final = Path(out_root) / chunk_dir_name(sessions[0])
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in FIELDS:
        np.save(tmp / f"{name}.npy", np.ascontiguousarray(arrays[name], dtype=DTYPE))
    np.save(tmp / "eligible.npy", np.ascontiguousarray(elig))
    np.save(tmp / "session_index.npy", (local + first_session_number).astype(np.int32))
    np.save(tmp / "minute_ts.npy", ts)
    meta = {
        "asset_ids": list(asset_ids),
        "sessions": [d.date().isoformat() for d in sessions],
        "first_session_number": first_session_number,
        "n_symbols": shape[0],
        "n_minutes": shape[1],
        "dtype": np.dtype(DTYPE).name,
    }
    (tmp / META).write_text(json.dumps(meta), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return final


def read_meta(path: str | Path) -> dict:
    return json.loads((Path(path) / META).read_text(encoding="utf-8"))


def _market_panel_cls():
    # trade sits above data in the layered architecture; resolve the type dynamically instead of importing it.
    return importlib.import_module("shpe_research.trade.account_types").MarketPanel


def open_panel_chunk(path: str | Path):
    """Memory-map a chunk read-only and return a ``MarketPanel`` over the maps (no copy)."""
    path = Path(path)
    meta = read_meta(path)
    arr = {n: np.load(path / f"{n}.npy", mmap_mode="r") for n in (*FIELDS, "eligible", "session_index")}
    return _market_panel_cls()(
        asset_ids=tuple(meta["asset_ids"]),
        open=arr["open"], high=arr["high"], low=arr["low"], close=arr["close"], volume=arr["volume"],
        session_index=arr["session_index"], eligible=arr["eligible"],
    )


def open_minute_ts(path: str | Path) -> np.ndarray:
    return np.load(Path(path) / "minute_ts.npy", mmap_mode="r")


def list_chunks(root: str | Path) -> list[Path]:
    """Chunk directories under ``root`` in chronological order."""
    return sorted(p for p in Path(root).glob("w*") if (p / META).exists())


# ---------------------------------------------------------------- DataStore loader

def store_source(
    store: DataStore,
    symbols: Sequence[str],
    *,
    feed: str = "sip",
    adjustment: str = "raw",
    eligible_fn: Callable[[pd.DatetimeIndex, Sequence[str]], np.ndarray] | None = None,
) -> ChunkSource:
    """ChunkSource reading 1Min bars from the DataStore parquet layout, aligned to the session grid.

    Bars off the regular grid are dropped. ``eligible_fn(sessions, symbols)`` returns a bool
    (n_symbols, n_sessions) schedule (e.g. from the A4 universe); default is "has a bar that session".
    """
    symbols = list(symbols)

    def source(sessions: pd.DatetimeIndex, minute_ts: np.ndarray, local: np.ndarray) -> dict[str, np.ndarray]:
        bars = store.read_bars(symbols, sessions[0], sessions[-1], "1Min", feed, adjustment)
        shape = (len(symbols), len(minute_ts))
        out = {f: np.full(shape, np.nan, DTYPE) for f in FIELDS[:4]}
        out["volume"] = np.zeros(shape, DTYPE)
        row = pd.Series(np.arange(len(symbols)), index=symbols).reindex(bars["symbol"]).to_numpy()
        ts = pd.DatetimeIndex(bars["ts"]).asi8
        col = np.minimum(np.searchsorted(minute_ts, ts), len(minute_ts) - 1)
        ok = (minute_ts[col] == ts) & ~np.isnan(row)
        r, c = row[ok].astype(np.int64), col[ok]
        for f in FIELDS:
            out[f][r, c] = bars[f].to_numpy()[ok].astype(DTYPE)
        if eligible_fn is None:
            has = np.zeros(shape, np.int32)
            has[r, c] = 1
            starts = np.flatnonzero(np.r_[True, np.diff(local) != 0])
            per_sess = np.add.reduceat(has, starts, axis=1) > 0
        else:
            per_sess = np.asarray(eligible_fn(sessions, symbols), bool)
        out["eligible"] = np.ascontiguousarray(per_sess[:, local])
        return out

    return source


def build_from_store(store: DataStore, out_root: str | Path, symbols: Sequence[str], start, end, **kw) -> list[Path]:
    """Build one chunk per ISO week for sessions start..end from the DataStore; returns chunk paths."""
    sessions = get_calendar().sessions(start, end)
    src = store_source(store, symbols, **kw)
    return [
        build_panel_chunk(out_root, symbols, wk, src, first_session_number=int(sessions.get_loc(wk[0])))
        for wk in week_chunks(sessions)
    ]


# ---------------------------------------------------------------- ranking helper

def daily_dollar_volume(panel) -> np.ndarray:
    """(n_symbols, n_sessions) sum of close*volume per session; missing bars count as zero. float64."""
    sidx = np.asarray(panel.session_index)
    starts = np.flatnonzero(np.r_[True, np.diff(sidx) != 0])
    out = np.empty((panel.n_symbols, len(starts)), np.float64)
    step = 256  # bound the float64 temporary
    for lo in range(0, panel.n_symbols, step):
        sl = slice(lo, lo + step)
        dv = np.nan_to_num(np.asarray(panel.close[sl], np.float64)) * np.asarray(panel.volume[sl], np.float64)
        out[sl] = np.add.reduceat(dv, starts, axis=1)
    return out


def median_dollar_volume_20d(panel, prior_daily: np.ndarray | None = None, window: int = 20) -> np.ndarray:
    """(n_symbols, n_minutes) float32 median daily dollar volume over the ``window`` sessions before each session.

    No lookahead: a minute in session s sees sessions s-window..s-1 only. ``prior_daily`` is
    (n_symbols, k) daily dollar volume of the sessions preceding this chunk (oldest first, e.g. the
    previous chunks' ``daily_dollar_volume``). Sessions with fewer than ``window`` prior sessions are NaN.
    """
    daily = daily_dollar_volume(panel)
    n_sym, n_sess = daily.shape
    hist = daily if prior_daily is None else np.concatenate([np.asarray(prior_daily, np.float64), daily], axis=1)
    k = hist.shape[1] - n_sess
    med = np.full((n_sym, n_sess), np.nan, np.float32)
    for s in range(n_sess):
        end = k + s  # exclusive: sessions strictly before s
        if end >= window:
            med[:, s] = np.median(hist[:, end - window:end], axis=1)
    sidx = np.asarray(panel.session_index)
    local = np.cumsum(np.r_[0, np.diff(sidx) != 0]).astype(np.int64)
    return np.ascontiguousarray(med[:, local])


# ---------------------------------------------------------------- synthetic generator

def intraday_volume_profile(n_minutes: int) -> np.ndarray:
    """U-shaped minute volume fractions (sum 1): heavy open and close, quiet midday."""
    t = np.arange(n_minutes, dtype=np.float64)
    prof = 1.0 + 3.0 * np.exp(-t / 25.0) + 2.0 * np.exp(-(n_minutes - 1 - t) / 25.0)
    return prof / prof.sum()


def synth_asset_ids(n: int) -> list[str]:
    return [f"S{i:04d}" for i in range(n)]


class SyntheticPanel:
    """Deterministic synthetic panel: geometric random walk + U-shaped volume, generated chunk by chunk.

    Daily open levels come from one seeded (n_symbols x n_sessions) walk (small); intraday paths from RNGs
    seeded by (seed, session number), so a chunk is reproducible regardless of build order.
    """

    def __init__(
        self,
        n_symbols: int,
        sessions: pd.DatetimeIndex,
        seed: int = 0,
        minute_vol: float = 0.0005,
        daily_vol: float = 0.015,
        missing_rate: float = 0.001,
        min_eligible_dv: float = 10e6,
    ):
        self.asset_ids = synth_asset_ids(n_symbols)
        self.sessions = pd.DatetimeIndex(sessions)
        self.seed, self.minute_vol, self.missing_rate = seed, minute_vol, missing_rate
        rng = np.random.default_rng([seed, 0])
        p0 = np.exp(rng.uniform(np.log(5), np.log(500), n_symbols))
        self.base_shares = np.exp(rng.uniform(np.log(5e4), np.log(2e7), n_symbols))  # shares per day
        rets = rng.normal(0.0, daily_vol, (n_symbols, len(self.sessions)))
        self.level = p0[:, None] * np.exp(np.cumsum(rets, axis=1) - rets)  # open level of each session
        self.eligible_sym = p0 * self.base_shares >= min_eligible_dv

    def source(self) -> ChunkSource:
        def src(sessions: pd.DatetimeIndex, minute_ts: np.ndarray, local: np.ndarray) -> dict[str, np.ndarray]:
            n = len(self.asset_ids)
            shape = (n, len(minute_ts))
            o, h, lo, c, v = (np.empty(shape, DTYPE) for _ in range(5))
            gi0 = int(self.sessions.get_loc(sessions[0]))
            for i in range(len(sessions)):
                cols = np.flatnonzero(local == i)
                m = len(cols)
                s = slice(cols[0], cols[-1] + 1)
                rng = np.random.default_rng([self.seed, 1, gi0 + i])
                steps = rng.normal(0.0, self.minute_vol, (n, m))
                path = self.level[:, gi0 + i, None] * np.exp(np.cumsum(steps, axis=1))
                prev = np.concatenate([self.level[:, gi0 + i, None], path[:, :-1]], axis=1)
                wick = np.abs(rng.normal(0.0, self.minute_vol / 2, (2, n, m)))
                o[:, s], c[:, s] = prev, path
                h[:, s] = np.maximum(prev, path) * (1 + wick[0])
                lo[:, s] = np.minimum(prev, path) * (1 - wick[1])
                day_scale = np.exp(rng.normal(0.0, 0.3, (n, 1)))
                noise = np.exp(rng.normal(0.0, 0.5, (n, m)))
                v[:, s] = np.round(self.base_shares[:, None] * day_scale * intraday_volume_profile(m) * noise)
            miss = np.random.default_rng([self.seed, 2, gi0]).random(shape) < self.missing_rate
            for a in (o, h, lo, c):
                a[miss] = np.nan
            v[miss] = 0
            elig = np.ascontiguousarray(np.broadcast_to(self.eligible_sym[:, None], shape))
            return {"open": o, "high": h, "low": lo, "close": c, "volume": v, "eligible": elig}

        return src


def generate_synthetic_panel(
    out_root: str | Path,
    n_symbols: int,
    sessions: pd.DatetimeIndex,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Write week chunks for ``n_symbols`` x ``sessions`` to disk, one week in RAM at a time."""
    sessions = pd.DatetimeIndex(sessions)
    gen = SyntheticPanel(n_symbols, sessions, seed)
    src = gen.source()
    weeks = week_chunks(sessions)
    paths = []
    for k, wk in enumerate(weeks):
        paths.append(build_panel_chunk(out_root, gen.asset_ids, wk, src, int(sessions.get_loc(wk[0]))))
        if progress:
            progress(k + 1, len(weeks))
    return paths
