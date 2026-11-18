"""Session-scoped calibration fixtures shared by Tasks 9-12."""

from __future__ import annotations

from pathlib import Path

import pytest

from shpe_research.calibration.window import Archive
from shpe_research.config.momentum import ResolvedProtocol, load_protocol
from shpe_research.data.calendar import get_calendar
from shpe_research.data.panel import generate_synthetic_panel
from tests.unit.calibration.helpers import AllMembersUniverse

SMALL_PROTOCOL_OVERRIDES = (
    "search.budget=24",
    "search.initial_samples=12",
    "search.population=6",
    "search.proposals_per_round=24",
    "search.evals_per_round=6",
    "search.evals_split=[2,2,2]",
    "search.surrogate_min_results=8",
    "search.finalists_per_seed=3",
    "search.seeds=[1,2]",
    "surrogate.max_iter=20",
    "surrogate.min_samples_leaf=2",
    "validation.sessions=10",
    "acceptance.search_min_trades=5",
    "acceptance.search_min_sessions=3",
    "acceptance.validation_min_trades=2",
    "acceptance.validation_min_sessions=2",
    "acceptance.max_drawdown=0.5",
    "acceptance.max_single_stock_share=1.0",
    "walkforward.reserved_start=2030-01-01",
)


@pytest.fixture(scope="session")
def synth_archive_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A6 week chunks: 10 synthetic symbols over the 2024-01-02 .. 2024-06-28 sessions, seed 7."""
    root = tmp_path_factory.mktemp("synth_archive")
    generate_synthetic_panel(root, 10, get_calendar().sessions("2024-01-02", "2024-06-28"), seed=7)
    return root


@pytest.fixture(scope="session")
def small_protocol() -> ResolvedProtocol:
    """The default protocol with a small search, short validation and loose acceptance floors."""
    return load_protocol(None, list(SMALL_PROTOCOL_OVERRIDES))


@pytest.fixture(scope="session")
def all_members_universe(synth_archive_root: Path) -> AllMembersUniverse:
    """A static universe whose every week holds every archive asset."""
    return AllMembersUniverse(Archive.open(synth_archive_root).asset_ids, version="static-test")
