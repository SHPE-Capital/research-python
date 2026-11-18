"""Tests for D7 window planning: cutoff, search/validation windows, activation week, panel assembly."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from shpe_research.calibration.keys import ChunkSpan
from shpe_research.calibration.window import (
    Archive,
    InsufficientHistoryError,
    UniverseWeek,
    WindowPlan,
    activation_sessions,
    iso_monday,
    lock_cutoff,
    plan_window,
)
from shpe_research.config.momentum import MomentumProtocol, ResolvedProtocol
from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import list_chunks, open_panel_chunk

DEFAULT = MomentumProtocol()


def _min_prior(p: MomentumProtocol) -> int:
    return max(p.cache.warmup_sessions, p.weights.descriptor_lookback + 1)


def _plan(archive: Archive, as_of: date, p: MomentumProtocol = DEFAULT, granularity=None) -> WindowPlan:
    return plan_window(
        archive,
        as_of,
        validation_sessions=p.validation.sessions,
        min_prior_sessions=_min_prior(p),
        granularity=granularity or p.cache.granularity,
    )


@pytest.fixture(scope="module")
def archive(synth_archive_root: Path) -> Archive:
    return Archive.open(synth_archive_root)


def test_archive_open_lists_sessions_and_assets(archive: Archive) -> None:
    cal = tuple(d.date() for d in get_calendar().sessions("2024-01-02", "2024-06-28"))
    assert archive.sessions == cal
    assert archive.asset_ids == tuple(f"S{i:04d}" for i in range(10))


def test_lock_cutoff(archive: Archive) -> None:
    assert lock_cutoff(archive, date(2024, 6, 1)) == date(2024, 5, 31)  # Saturday -> Friday
    assert lock_cutoff(archive, date(2024, 5, 31)) == date(2024, 5, 31)
    with pytest.raises(InsufficientHistoryError):
        lock_cutoff(archive, date(2023, 12, 29))


def test_iso_monday() -> None:
    assert iso_monday(date(2024, 5, 29)) == date(2024, 5, 27)
    assert iso_monday(date(2024, 5, 27)) == date(2024, 5, 27)
    assert iso_monday(date(2024, 6, 2)) == date(2024, 5, 27)


def test_validation_is_last_n_sessions(archive: Archive) -> None:
    plan = _plan(archive, date(2024, 6, 1))
    hist = tuple(s for s in archive.sessions if s <= date(2024, 5, 31))
    assert plan.cutoff == date(2024, 5, 31)
    assert len(plan.validation_sessions) == 63
    assert plan.validation_sessions[-1] == plan.cutoff
    assert plan.validation_sessions == hist[-63:]
    # search ends on the session before validation starts, contiguously
    joined = plan.search_sessions + plan.validation_sessions
    lo = archive.sessions.index(joined[0])
    assert joined == archive.sessions[lo : lo + len(joined)]


def test_search_start_rule(archive: Archive) -> None:
    plan = _plan(archive, date(2024, 6, 1))
    first = plan.search_sessions[0]
    idx = archive.sessions.index(first)
    assert idx >= 21  # max(cache.warmup_sessions = 20, descriptor_lookback + 1 = 21) prior archive sessions
    assert plan.search_spans[0].sessions[0] == first
    assert iso_monday(first) == iso_monday(archive.sessions[idx])
    # it starts an ISO week and the previous week does not qualify
    assert archive.sessions[idx - 1] < iso_monday(first)
    prev_week_start = min(s for s in archive.sessions if iso_monday(s) == iso_monday(archive.sessions[idx - 1]))
    assert archive.sessions.index(prev_week_start) < 21
    assert first == date(2024, 2, 5)


def test_spans_cover_both_stages_and_straddle(archive: Archive) -> None:
    plan = _plan(archive, date(2024, 6, 1))
    search_cover = [s for sp in plan.search_spans for s in sp.sessions]
    val_cover = [s for sp in plan.validation_spans for s in sp.sessions]
    assert set(plan.search_sessions) <= set(search_cover)
    assert set(plan.validation_sessions) <= set(val_cover)
    assert all(s <= plan.cutoff for s in val_cover)
    v0 = plan.validation_sessions[0]
    if iso_monday(v0) == iso_monday(plan.search_sessions[-1]):
        assert plan.search_spans[-1] == plan.validation_spans[0]
        assert plan.search_spans[-1].sessions[0] < v0 <= plan.search_spans[-1].sessions[-1]


def test_activation_is_next_iso_week(archive: Archive) -> None:
    expected = tuple(date(2024, 6, d) for d in (3, 4, 5, 6, 7))
    assert activation_sessions(date(2024, 5, 31)) == expected
    assert _plan(archive, date(2024, 6, 1)).activation_sessions == expected


def test_midweek_cutoff_truncates_last_chunk(archive: Archive) -> None:
    plan = _plan(archive, date(2024, 5, 29))
    assert plan.cutoff == date(2024, 5, 29)
    assert plan.validation_spans[-1].sessions == (date(2024, 5, 28), date(2024, 5, 29))  # 27th is Memorial Day
    assert plan.activation_sessions[0] == date(2024, 6, 3)
    assert plan.activation_sessions == tuple(date(2024, 6, d) for d in (3, 4, 5, 6, 7))


def test_insufficient_history(archive: Archive) -> None:
    with pytest.raises(InsufficientHistoryError):
        _plan(archive, date(2024, 1, 31))


def test_small_protocol_window(archive: Archive, small_protocol: ResolvedProtocol) -> None:
    plan = _plan(archive, date(2024, 6, 1), small_protocol.protocol)
    assert len(plan.validation_sessions) == 10
    assert plan.search_sessions[0] == date(2024, 2, 5)


def test_build_panel_prefix_and_membership(archive: Archive, synth_archive_root: Path) -> None:
    week = tuple(s for s in archive.sessions if iso_monday(s) == date(2024, 3, 4))
    span = ChunkSpan(week[:3])  # truncated span: columns must stop after its last session
    members = frozenset(archive.asset_ids[:4])
    panel, n_prefix = archive.build_panel(span, 20, members)
    assert n_prefix == 20
    cal = get_calendar()
    idx = archive.sessions.index(span.sessions[0])
    used = archive.sessions[idx - 20 : idx + 3]
    assert panel.n_minutes == sum(len(cal.regular_minutes(d)) for d in used)
    assert len(np.unique(np.asarray(panel.session_index))) == 23
    assert panel.asset_ids == archive.asset_ids
    elig = np.asarray(panel.eligible)
    out = [i for i, a in enumerate(panel.asset_ids) if a not in members]
    assert not elig[out].any()
    # the tail equals the raw chunk's first three sessions
    raw_dir = next(p for p in list_chunks(synth_archive_root) if p.name == "w2024-03-04")
    raw = open_panel_chunk(raw_dir)
    n_tail = sum(len(cal.regular_minutes(d)) for d in span.sessions)
    np.testing.assert_array_equal(np.asarray(panel.close)[:, -n_tail:], np.asarray(raw.close)[:, :n_tail])
    inside = [i for i, a in enumerate(panel.asset_ids) if a in members]
    np.testing.assert_array_equal(elig[inside, -n_tail:], np.asarray(raw.eligible)[inside, :n_tail])
    assert panel.close.dtype == np.float32

    full, _ = archive.build_panel(span, 20, None)
    np.testing.assert_array_equal(np.asarray(full.eligible)[:, -n_tail:], np.asarray(raw.eligible)[:, :n_tail])


def test_build_panel_insufficient_prefix(archive: Archive) -> None:
    span = ChunkSpan(tuple(s for s in archive.sessions if iso_monday(s) == date(2024, 1, 22)))
    with pytest.raises(InsufficientHistoryError):
        archive.build_panel(span, 20, None)


def test_build_panel_rejects_unknown_or_gappy_span(archive: Archive) -> None:
    with pytest.raises(ValueError):
        archive.build_panel(ChunkSpan((date(2024, 3, 4), date(2024, 3, 6))), 20, None)
    with pytest.raises(ValueError):
        archive.build_panel(ChunkSpan((date(2024, 7, 1),)), 20, None)


def test_session_granularity_spans(archive: Archive) -> None:
    plan = _plan(archive, date(2024, 6, 1), granularity="session")
    assert all(len(sp.sessions) == 1 for sp in plan.search_spans + plan.validation_spans)
    assert tuple(sp.sessions[0] for sp in plan.search_spans) == plan.search_sessions
    assert tuple(sp.sessions[0] for sp in plan.validation_spans) == plan.validation_sessions
    assert archive.sessions.index(plan.search_sessions[0]) == 21


def test_universe_week_validates() -> None:
    UniverseWeek(date(2024, 3, 4), (("A", "a"), ("B", "b")), "v")
    with pytest.raises(ValueError):
        UniverseWeek(date(2024, 3, 5), (("A", "a"),), "v")  # not a Monday
    with pytest.raises(ValueError):
        UniverseWeek(date(2024, 3, 4), (("B", "b"), ("A", "a")), "v")  # not sorted
