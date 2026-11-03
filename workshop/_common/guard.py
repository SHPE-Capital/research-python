"""The real-data guard: cells that need the frozen snapshot run only on request."""

from __future__ import annotations

import os

REAL_DATA_ENV = "SHPE_WORKSHOP_REAL_DATA"


def real_data_enabled() -> bool:
    """True when ``SHPE_WORKSHOP_REAL_DATA=1`` is set. Use it as ``if real_data_enabled():``.

    Notebooks run on the synthetic fixture by default (offline, no keys); a cell that
    reads the snapshot must sit behind this check.
    """
    return os.environ.get(REAL_DATA_ENV, "").strip() == "1"


def require_real_data() -> None:
    """Raise ``RuntimeError`` unless real data is enabled. For helpers that only make sense on the snapshot."""
    if not real_data_enabled():
        raise RuntimeError(f"this step needs the workshop snapshot; set {REAL_DATA_ENV}=1 to run it")
