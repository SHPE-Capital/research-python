"""Tests for the D7 market-descriptor proxy (similarity weighting inputs)."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from shpe_research.calibration.descriptors import PanelProxyDescriptors
from shpe_research.calibration.window import Archive
from shpe_research.config.momentum import WeightsSection
from shpe_research.data.panel import list_chunks, open_panel_chunk, read_meta

LOOKBACK = WeightsSection().descriptor_lookback
TAMPER_POS = 60  # archive position of the first tampered session (well past the lookback)


@pytest.fixture(scope="module")
def archive(synth_archive_root: Path) -> Archive:
    return Archive.open(synth_archive_root)


@pytest.fixture(scope="module")
def provider(archive: Archive) -> PanelProxyDescriptors:
    return PanelProxyDescriptors(archive, lookback=LOOKBACK)


def _tampered_archive(src: Path, dst: Path, first: date) -> Archive:
    """Copy the archive and scale every price of session ``first`` and later by 1.3 (volume untouched)."""
    shutil.copytree(src, dst)
    for chunk in list_chunks(dst):
        sessions = [date.fromisoformat(s) for s in read_meta(chunk)["sessions"]]
        sidx = np.load(chunk / "session_index.npy")
        local = np.cumsum(np.r_[0, sidx[1:] != sidx[:-1]])
        cols = np.array([sessions[k] >= first for k in local])
        if not cols.any():
            continue
        for name in ("open", "high", "low", "close"):
            arr = np.load(chunk / f"{name}.npy")
            arr[:, cols] *= np.float32(1.3)
            np.save(chunk / f"{name}.npy", arr)
    return Archive.open(dst)


def test_rows_use_previous_session_only(archive, provider, synth_archive_root, tmp_path) -> None:
    s = archive.sessions[TAMPER_POS]
    tampered = PanelProxyDescriptors(_tampered_archive(synth_archive_root, tmp_path / "t", s), lookback=LOOKBACK)
    upto = archive.sessions[LOOKBACK + 1 : TAMPER_POS + 1]  # every finite row through session s
    np.testing.assert_array_equal(provider.rows(upto), tampered.rows(upto))
    nxt = [archive.sessions[TAMPER_POS + 1]]
    assert np.all(np.isfinite(provider.rows(nxt)))
    assert not np.array_equal(provider.rows(nxt), tampered.rows(nxt))


def test_current_includes_cutoff(archive, provider, synth_archive_root, tmp_path) -> None:
    s = archive.sessions[TAMPER_POS]
    nxt = archive.sessions[TAMPER_POS + 1]
    np.testing.assert_array_equal(provider.current(s), provider.rows([nxt])[0])
    assert provider.current(s).shape == (3,)
    tampered = PanelProxyDescriptors(_tampered_archive(synth_archive_root, tmp_path / "t", s), lookback=LOOKBACK)
    assert not np.array_equal(provider.current(s), tampered.current(s))
    prev = archive.sessions[TAMPER_POS - 1]
    np.testing.assert_array_equal(provider.current(prev), tampered.current(prev))


def test_short_history_is_nan(archive, provider) -> None:
    rows = provider.rows(archive.sessions[: LOOKBACK + 2])
    assert rows.shape == (LOOKBACK + 2, 3)
    assert np.all(np.isnan(rows[: LOOKBACK + 1]))  # row p uses positions <= p - 1; needs lookback returns
    assert np.all(np.isfinite(rows[LOOKBACK + 1]))
    assert np.all(np.isnan(provider.current(archive.sessions[LOOKBACK - 1])))
    assert np.all(np.isfinite(provider.current(archive.sessions[LOOKBACK])))


def test_values_match_definition(archive, provider) -> None:
    """Independent recomputation of d1, d2, d3 at one cutoff from the raw chunks."""
    closes, elig = [], []
    for chunk in list_chunks(archive.root):
        panel = open_panel_chunk(chunk)
        sidx = np.asarray(panel.session_index)
        for k in np.unique(sidx):
            cols = sidx == k
            block = np.asarray(panel.close)[:, cols]
            last = np.full(block.shape[0], np.nan)
            for i in range(block.shape[0]):
                finite = block[i][np.isfinite(block[i])]
                if finite.size:
                    last[i] = finite[-1]
            closes.append(last)
            elig.append(np.asarray(panel.eligible)[:, cols].any(axis=1))
    c = np.array(closes, dtype=np.float64).T  # (n_symbols, n_sessions)
    e = np.array(elig).T
    pos = 80
    r = []
    for t in range(pos - LOOKBACK + 1, pos + 1):
        ok = e[:, t] & np.isfinite(c[:, t]) & np.isfinite(c[:, t - 1])
        r.append(np.mean(c[ok, t] / c[ok, t - 1] - 1.0))
    r = np.array(r)
    sd = np.std(r, ddof=1)
    d1 = (np.prod(1.0 + r) - 1.0) / sd
    win = c[:, pos - LOOKBACK + 1 : pos + 1]
    ok = e[:, pos] & np.all(np.isfinite(win), axis=1)
    d3 = np.mean(c[ok, pos] > win[ok].mean(axis=1))
    np.testing.assert_allclose(provider.current(archive.sessions[pos]), [d1, sd, d3], rtol=1e-12, atol=1e-15)
