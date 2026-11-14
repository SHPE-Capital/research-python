"""A6: panel chunks are grid-aligned, memory-mapped without copy, deterministic, and match the DataStore."""

from __future__ import annotations

import numpy as np
import pandas as pd
import psutil
import pytest

from shpe_research.data.calendar import get_calendar
from shpe_research.data.fixtures import synthetic_bars
from shpe_research.data.panel import (
    FIELDS,
    build_from_store,
    build_panel_chunk,
    daily_dollar_volume,
    generate_synthetic_panel,
    list_chunks,
    median_dollar_volume_20d,
    open_panel_chunk,
    week_chunks,
)
from shpe_research.data.store import BarUnit, DataStore, period_window
from shpe_research.trade.account_types import MarketPanel

THANKSGIVING_WEEK = ("2025-11-24", "2025-11-28")  # Fri 28th is a 13:00 early close


def _sessions(a: str, b: str) -> pd.DatetimeIndex:
    return get_calendar().sessions(a, b)


def test_week_chunks_and_early_close_grid(tmp_path):
    sessions = _sessions(*THANKSGIVING_WEEK)
    assert len(sessions) == 4  # Thanksgiving closed
    (path,) = generate_synthetic_panel(tmp_path, 5, sessions)
    p = open_panel_chunk(path)
    counts = np.bincount(np.asarray(p.session_index) - p.session_index[0])
    assert list(counts) == [390, 390, 390, 210]
    assert [len(w) for w in week_chunks(_sessions("2025-11-17", "2025-12-05"))] == [5, 4, 5]


def test_open_is_memmap_readonly_and_contiguous(tmp_path):
    (path,) = generate_synthetic_panel(tmp_path, 8, _sessions("2024-03-04", "2024-03-08"))
    p = open_panel_chunk(path)
    assert isinstance(p, MarketPanel)
    for name in (*FIELDS, "eligible", "session_index"):
        a = getattr(p, name)
        assert isinstance(a, np.memmap)
        assert not a.flags.writeable
        assert a.flags.c_contiguous
    assert all(getattr(p, f).dtype == np.float32 for f in FIELDS)
    with pytest.raises(ValueError):
        p.close[0, 0] = 1.0
    # two opens share the same file pages, nothing is copied into a private array
    q = open_panel_chunk(path)
    assert np.shares_memory(p.close, p.close[:, 5:])
    assert np.array_equal(p.close, q.close, equal_nan=True)


def test_open_does_not_copy_rss(tmp_path):
    n = 800
    (path,) = generate_synthetic_panel(tmp_path, n, _sessions("2024-03-04", "2024-03-08"))
    size = sum(f.stat().st_size for f in path.glob("*.npy"))  # ~30 MB
    proc = psutil.Process()
    before = proc.memory_info().rss
    p = open_panel_chunk(path)
    grew = proc.memory_info().rss - before
    assert grew < size * 0.25, f"open grew RSS by {grew} for {size} bytes of data"
    assert p.n_symbols == n


def test_synthetic_is_deterministic_and_sane(tmp_path):
    sess = _sessions("2024-03-04", "2024-03-15")
    a = generate_synthetic_panel(tmp_path / "a", 20, sess, seed=3)
    b = generate_synthetic_panel(tmp_path / "b", 20, sess, seed=3)
    c = generate_synthetic_panel(tmp_path / "c", 20, sess, seed=4)
    assert len(a) == 2 and [x.name for x in list_chunks(tmp_path / "a")] == ["w2024-03-04", "w2024-03-11"]
    pa, pb, pc = (open_panel_chunk(x[1]) for x in (a, b, c))
    assert np.array_equal(pa.close, pb.close, equal_nan=True)
    assert not np.array_equal(pa.close, pc.close, equal_nan=True)
    assert pa.session_index[0] == 5  # global numbering continues across chunks
    ok = ~np.isnan(pa.close)
    assert (pa.high[ok] >= np.maximum(pa.open, pa.close)[ok]).all()
    assert (pa.low[ok] <= np.minimum(pa.open, pa.close)[ok]).all()
    assert (pa.volume[~ok] == 0).all() and (pa.volume[ok] >= 0).all()
    day = pa.volume[:, :390].astype(np.float64).sum(axis=0)
    assert day[:30].sum() > 2 * day[180:210].sum()  # U-shape: open much heavier than midday


def test_builder_rejects_bad_source(tmp_path):
    sess = _sessions("2024-03-04", "2024-03-04")
    bad = {f: np.zeros((2, 3), np.float32) for f in FIELDS} | {"eligible": np.ones((2, 3), bool)}
    with pytest.raises(ValueError, match="shape"):
        build_panel_chunk(tmp_path, ["A", "B"], sess, lambda s, t, i: bad)
    with pytest.raises(ValueError, match="unique"):
        build_panel_chunk(tmp_path, ["A", "A"], sess, lambda s, t, i: bad)
    assert not list(tmp_path.glob("w*"))  # nothing half-written


def test_store_loader_matches_store_values(tmp_path):
    store = DataStore(tmp_path / "store")
    symbols = ["AAA", "BBB"]
    minute = synthetic_bars(symbols=symbols, start="2026-09-01", end="2026-09-30")
    for sym in symbols:
        grp = minute[minute["symbol"] == sym].reset_index(drop=True)
        unit = BarUnit("sip", "1Min", "raw", sym, "2026-09")
        store.write_bar_unit(unit, grp, *period_window("2026-09"), complete=True)
    store.bars_manifest.flush()
    paths = build_from_store(store, tmp_path / "panel", symbols, "2026-09-01", "2026-09-30")
    assert len(paths) == len(week_chunks(_sessions("2026-09-01", "2026-09-30")))
    p = open_panel_chunk(paths[0])
    wk = minute[minute["ts"].isin(pd.DatetimeIndex(np.load(paths[0] / "minute_ts.npy")).tz_localize("UTC"))]
    for i, sym in enumerate(symbols):
        rows = wk[wk["symbol"] == sym]
        ts = pd.DatetimeIndex(rows["ts"]).asi8
        cols = np.searchsorted(np.load(paths[0] / "minute_ts.npy"), ts)
        for f in FIELDS:
            assert np.array_equal(getattr(p, f)[i, cols], rows[f].to_numpy().astype(np.float32))
    assert p.eligible.all()


def test_store_loader_missing_data_raises(tmp_path):
    from shpe_research.data.store import StoreError

    with pytest.raises(StoreError):
        build_from_store(DataStore(tmp_path / "empty"), tmp_path / "panel", ["AAA"], "2026-09-01", "2026-09-04")


def test_median_dollar_volume_no_lookahead(tmp_path):
    sess = _sessions("2024-03-04", "2024-04-12")  # 6 weeks, 30 sessions
    paths = generate_synthetic_panel(tmp_path, 6, sess, seed=1)
    prior = None
    meds = []
    for path in paths:
        p = open_panel_chunk(path)
        meds.append(median_dollar_volume_20d(p, prior))
        d = daily_dollar_volume(p)
        prior = d if prior is None else np.concatenate([prior, d], axis=1)
    full = np.concatenate(meds, axis=1)
    n_min = full.shape[1]
    assert full.dtype == np.float32 and full.shape == (6, n_min)
    # Session 20 sees exactly sessions 0..19 and nothing from session 20 or later
    ts = np.concatenate([np.load(x / "minute_ts.npy") for x in paths])
    sidx = np.concatenate([np.load(x / "session_index.npy") for x in paths])
    daily = prior
    col = np.flatnonzero(sidx == 20)[0]
    assert np.allclose(full[:, col], np.median(daily[:, 0:20], axis=1), rtol=1e-6)
    assert np.isnan(full[:, np.flatnonzero(sidx == 19)[0]]).all()
    assert len(ts) == n_min
    assert not np.isnan(full[:, np.flatnonzero(sidx == 25)]).any()
