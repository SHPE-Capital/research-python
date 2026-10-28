"""D4: reproducible stock universe across liquidity and volatility conditions.

    uv run shpe data universe --config configs/universe/workshop_v0.yaml --download

Follows Christopher's ORB plan (section 3), generalized by config:

1. Measure every candidate over the `window_sessions` sessions before `as_of`,
   from split-adjusted daily bars:
       liquidity  L = median(close x volume)        (dollar-volume proxy)
       volatility S = std of daily log returns
       price      median close
   plus, when the coverage screen is on, the share of 09:30-11:00 minutes with
   a 1-minute bar (from D2's audit).
2. Screen: complete history, minimum price, minimum dollar volume, minimum
   morning coverage. Every exclusion is recorded with its reason.
3. Split eligible names into liquidity bins, then volatility bins within each
   liquidity bin. Sample `per_cell` names per cell with a fixed seed, never
   looking at strategy results, under a per-sector cap.
4. Split each cell's names into development and held-out with the same seed.

The manifest lists every candidate with its measurements, cell and status
(selected, excluded, not_sampled, sector_capped), so the selection can be
audited and reproduced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from shpe_research.config import data_dir, git_commit
from shpe_research.contracts.version import CONTRACT_VERSION
from shpe_research.data.audit import audit_bars
from shpe_research.data.calendar import get_calendar
from shpe_research.data.store import DataStore, StoreError

log = logging.getLogger(__name__)


@dataclass
class UniverseConfig:
    name: str
    candidates: Path
    as_of: str
    window_sessions: int = 60
    feed: str = "sip"
    screens: dict[str, Any] = field(default_factory=dict)
    cells: dict[str, Any] = field(default_factory=dict)
    sector_cap: int | None = 15
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> UniverseConfig:
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        raw["candidates"] = (path.parent / raw["candidates"]).resolve()
        return cls(**raw)

    @property
    def window(self) -> pd.DatetimeIndex:
        return get_calendar().previous_sessions(self.as_of, self.window_sessions)

    def cell(self, key: str, default: int) -> int:
        return int(self.cells.get(key, default))


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df["symbol"] = df["symbol"].str.strip().str.upper()
    return df.drop_duplicates("symbol").reset_index(drop=True)


def measure(cfg: UniverseConfig, symbols: list[str], store: DataStore) -> pd.DataFrame:
    """Per-symbol window measurements; `missing` explains symbols without the data to measure them."""
    window = cfg.window
    start, end = window[0], window[-1]
    rows = []
    for symbol in symbols:
        row: dict[str, Any] = {"symbol": symbol}
        try:
            daily = store.read_bars([symbol], start, end, timeframe="1Day", feed=cfg.feed, adjustment="split")
        except StoreError:
            row["missing"] = "no split-adjusted daily bars stored for the window"
            rows.append(row)
            continue
        daily = daily[daily["session"] == "regular"]
        row["sessions"] = len(daily)
        if len(daily) > 1:
            row["median_close"] = float(daily["close"].median())
            row["median_dollar_volume"] = float((daily["close"] * daily["volume"]).median())
            row["volatility"] = float(np.log(daily["close"]).diff().std())
        rows.append(row)
    metrics = pd.DataFrame(rows)
    for col in ("missing", "sessions", "median_close", "median_dollar_volume", "volatility"):
        if col not in metrics:
            metrics[col] = np.nan
    if cfg.screens.get("min_morning_coverage") is not None:
        metrics["morning_coverage"] = [_morning_coverage(store, cfg, s, start, end) for s in metrics["symbol"]]
    return metrics


def _morning_coverage(store: DataStore, cfg: UniverseConfig, symbol: str, start, end) -> float:
    try:
        minute = store.read_bars([symbol], start, end, timeframe="1Min", feed=cfg.feed, adjustment="raw")
    except StoreError:
        return np.nan
    if minute.empty:
        return 0.0
    return float(audit_bars(minute, start, end).symbols["morning_coverage"].iloc[0])


def screen(cfg: UniverseConfig, metrics: pd.DataFrame) -> pd.Series:
    """Exclusion reason per candidate ("" when eligible), checked in a fixed order."""
    s = cfg.screens
    reasons = pd.Series("", index=metrics.index, dtype=object)
    missing = metrics["missing"].notna()
    reasons[missing] = metrics.loc[missing, "missing"]

    def mark(mask: pd.Series, reason: str) -> None:
        reasons[(reasons == "") & mask] = reason

    mark(metrics["sessions"] < cfg.window_sessions, f"incomplete daily history (< {cfg.window_sessions} sessions)")
    if s.get("min_median_close") is not None:
        mark(metrics["median_close"] < s["min_median_close"], f"median close below ${s['min_median_close']}")
    if s.get("min_median_dollar_volume") is not None:
        mark(metrics["median_dollar_volume"] < s["min_median_dollar_volume"],
             f"median dollar volume below ${s['min_median_dollar_volume']:,.0f}")
    if s.get("min_morning_coverage") is not None:
        coverage = metrics["morning_coverage"]
        mark(coverage.isna(), "no 1-minute bars stored for the coverage screen")
        mark(coverage < s["min_morning_coverage"], f"morning coverage below {s['min_morning_coverage']:.0%}")
    return reasons


def assign_cells(cfg: UniverseConfig, eligible: pd.DataFrame) -> pd.DataFrame:
    """liquidity_bin (0 = lowest), volatility_bin within it, and a "L{i}-V{j}" cell label."""
    n_liq, n_vol = cfg.cell("liquidity_bins", 2), cfg.cell("volatility_bins", 3)
    out = eligible.copy()
    out["liquidity_bin"] = pd.qcut(out["median_dollar_volume"].rank(method="first"), n_liq, labels=False)
    out["volatility_bin"] = out.groupby("liquidity_bin")["volatility"].transform(
        lambda v: pd.qcut(v.rank(method="first"), min(n_vol, len(v)), labels=False)
    )
    liquidity = out["liquidity_bin"].astype(int).astype(str)
    out["cell"] = "L" + liquidity + "-V" + out["volatility_bin"].astype(int).astype(str)
    return out


def sample_cells(cfg: UniverseConfig, celled: pd.DataFrame) -> pd.DataFrame:
    """status (selected / not_sampled / sector_capped) and role (dev / held_out) for each eligible name."""
    rng = np.random.default_rng(cfg.seed)
    per_cell = cfg.cell("per_cell", 10)
    held_ratio = cfg.cell("held_out_per_cell", 2) / per_cell
    out = celled.assign(status="not_sampled", role="")
    sector_counts: dict[str, int] = {}
    for cell in sorted(out["cell"].unique()):
        names = sorted(out.loc[out["cell"] == cell, "symbol"])
        chosen = []
        for symbol in rng.permutation(names):
            if len(chosen) == per_cell:
                break
            sector = out.loc[out["symbol"] == symbol, "sector"].iloc[0]
            if cfg.sector_cap is not None and sector_counts.get(sector, 0) >= cfg.sector_cap:
                out.loc[out["symbol"] == symbol, "status"] = "sector_capped"
                continue
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            chosen.append(symbol)
        n_held = int(round(len(chosen) * held_ratio))
        order = rng.permutation(chosen)
        for i, symbol in enumerate(order):
            out.loc[out["symbol"] == symbol, ["status", "role"]] = ["selected", "held_out" if i < n_held else "dev"]
    return out


@dataclass
class UniverseResult:
    manifest: dict[str, Any]
    table: pd.DataFrame  # every candidate with measurements, status and role

    @property
    def selected(self) -> pd.DataFrame:
        return self.table[self.table["status"] == "selected"]

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.manifest, sort_keys=False, allow_unicode=True))
        return path


def build_universe(cfg: UniverseConfig, store: DataStore | None = None) -> UniverseResult:
    store = store or DataStore()
    candidates = load_candidates(cfg.candidates)
    metrics = candidates.merge(measure(cfg, candidates["symbol"].tolist(), store), on="symbol", how="left")
    metrics["exclusion"] = screen(cfg, metrics)
    eligible = metrics[metrics["exclusion"] == ""]
    table = metrics.assign(status="excluded", role="", cell="")
    if not eligible.empty:
        sampled = sample_cells(cfg, assign_cells(cfg, eligible))
        table = table.set_index("symbol")
        table.update(sampled.set_index("symbol")[["status", "role", "cell"]])
        table = table.reset_index()
    return UniverseResult(manifest=_manifest(cfg, table), table=table)


def _manifest(cfg: UniverseConfig, table: pd.DataFrame) -> dict[str, Any]:
    window = cfg.window
    per_cell = cfg.cell("per_cell", 10)
    cells = []
    for cell, group in table[table["cell"] != ""].groupby("cell"):
        selected = group[group["status"] == "selected"]
        cells.append({"cell": cell, "eligible": len(group), "selected": len(selected),
                      "shortfall": max(0, per_cell - len(selected)),
                      "dev": sorted(selected.loc[selected["role"] == "dev", "symbol"]),
                      "held_out": sorted(selected.loc[selected["role"] == "held_out", "symbol"])})
    columns = ["symbol", "sector", "status", "role", "cell", "exclusion", "sessions", "median_close",
               "median_dollar_volume", "volatility", *(["morning_coverage"] if "morning_coverage" in table else [])]
    records = [{k: _plain(v) for k, v in row.items()} for row in table[columns].to_dict("records")]
    selected = table[table["status"] == "selected"]
    return {
        "name": cfg.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "contract_version": CONTRACT_VERSION,
        "git_commit": git_commit(),
        "as_of": cfg.as_of,
        "window": {"sessions": len(window), "first": str(window[0].date()), "last": str(window[-1].date())},
        "feed": cfg.feed,
        "seed": cfg.seed,
        "screens": cfg.screens,
        "cells_config": cfg.cells,
        "sector_cap": cfg.sector_cap,
        "candidates_file": cfg.candidates.name,
        "counts": {
            "candidates": len(table),
            "excluded": int((table["status"] == "excluded").sum()),
            "eligible": int((table["status"] != "excluded").sum()),
            "selected": len(selected),
            "dev": int((selected["role"] == "dev").sum()),
            "held_out": int((selected["role"] == "held_out").sum()),
        },
        "cells": cells,
        "dev": sorted(selected.loc[selected["role"] == "dev", "symbol"]),
        "held_out": sorted(selected.loc[selected["role"] == "held_out", "symbol"]),
        "candidates": records,
    }


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def default_output(cfg: UniverseConfig) -> Path:
    return data_dir() / "universes" / f"{cfg.name}.yaml"


def required_downloads(cfg: UniverseConfig) -> list[dict[str, Any]]:
    """The `download_bars` calls a build needs (daily split-adjusted, plus 1-minute raw for the coverage screen)."""
    symbols = load_candidates(cfg.candidates)["symbol"].tolist()
    window = cfg.window
    calls = [{"symbols": symbols, "start": window[0], "end": window[-1], "timeframe": "1Day", "adjustment": "split"}]
    if cfg.screens.get("min_morning_coverage") is not None:
        calls.append({"symbols": symbols, "start": window[0], "end": window[-1], "timeframe": "1Min",
                      "adjustment": "raw"})
    return calls
