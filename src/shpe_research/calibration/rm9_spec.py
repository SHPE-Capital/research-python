"""Named profiles; only the demo and pilot reduce the dimensions and evidence floors."""

from __future__ import annotations

import hashlib
from pathlib import Path

from shpe_research.data.rm9_io import read

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = REPO / "studies/momentum/rm9-weekly-calibration/protocol.json"


def spec(mode):
    if mode not in ("demo", "pilot", "full"):
        raise ValueError("profile must be demo, pilot or full")
    p = read(PROTOCOL)
    if mode != "full":
        p.update(
            start="2023-01-09",
            end="2023-03-31",
            universe=9,
            history=4,
            validation=3,
            primary=2,
            quote_draws_per_regime=1,
            quote_symbol_samples=1,
            quote_min_windows=3,
            initial_ladder=[18],
            weekly_ladder=[2],
            finalists=4,
            surrogate_trees=3,
            surrogate_iter=24,
            pool=64,
            bootstrap_n=400,
            min_validation_trades=3,
        )
        p["accept"] = {**p["accept"], "min_observations": 10, "min_ess": 8}
    if mode == "demo":
        p["cap_hours"] = 2.0
    return dict(p, mode=mode)


def source_id():
    """Content identity covers uncommitted source, protocol and dependency lock, portable across CRLF checkouts."""
    h = hashlib.sha256()
    files = sorted((REPO / "src/shpe_research").rglob("*.py")) + [
        REPO / "scripts/momentum/rm9.py",
        PROTOCOL,
        REPO / "uv.lock",
    ]
    for f in files:
        h.update(f.relative_to(REPO).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()
