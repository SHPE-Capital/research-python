"""Paths and settings shared by every command.

Settings come from environment variables, optionally loaded from a git-ignored
`.env` at the repo root (see `.env.example`). Nothing here holds secrets in
code; keys are read only when a download actually needs them.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env", override=False)


def data_dir() -> Path:
    """Root of the local market-data store (default ./data, git-ignored)."""
    configured = os.environ.get("RESEARCH_DATA_DIR", "").strip()
    return Path(configured) if configured else REPO_ROOT / "data"


def runs_dir() -> Path:
    """Where experiment runs write reports and model artifacts (git-ignored)."""
    return REPO_ROOT / "runs"


def snapshots_dir() -> Path:
    """Where frozen dataset snapshots are built (git-ignored)."""
    return REPO_ROOT / "snapshots"


def requests_per_minute() -> int:
    """Alpaca request budget for this process; the free plan allows 200/min."""
    return int(os.environ.get("ALPACA_DATA_REQUESTS_PER_MINUTE", "190") or 190)


@lru_cache(maxsize=1)
def git_commit() -> str | None:
    """The research repo's current commit, recorded in every artifact; None outside git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None
