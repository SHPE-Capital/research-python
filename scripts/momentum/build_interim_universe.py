# ruff: noqa: E501
"""Interim symbol list for the A1 staged pulls (300x2, 500x5) until A4/A5 exist.

    uv run python scripts/momentum/build_interim_universe.py --download
    uv run python scripts/momentum/build_interim_universe.py --assets-json <out>/assets_2026-11-17.json   # offline rebuild

1. Candidates: every US equity asset Alpaca lists (active and inactive), from one /v2/assets call, filtered to
   exchange-listed common stock by exchange, symbol shape and name rules. The raw snapshot and its sha256 are saved
   so the list can be rebuilt offline.
2. Measure and select with D4 (`data/universe.py`, unchanged): split-adjusted daily bars over the config window,
   price and dollar-volume screens, then 3 liquidity x 3 volatility cells sampled with a fixed seed.
3. Write the D4 manifest and `symbols_500.txt` (one ticker per line, `#` provenance header, plus SPY).

Interim on purpose: names are typed by regex, there is no sector cap (Alpaca has no GICS sector), and selecting at
one `as_of` date favors names alive then. A4 (weekly eligibility) and A5 (weekly 3x3 schedule) replace this.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from shpe_research.config import data_dir
from shpe_research.data.alpaca import AlpacaDataClient, AlpacaError
from shpe_research.data.store import DataStore, sha256_file
from shpe_research.data.universe import UniverseConfig, UniverseResult, build_universe, required_downloads

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs" / "universe" / "momentum_interim_v0.yaml"
TRADING_URLS = ("https://api.alpaca.markets/v2", "https://paper-api.alpaca.markets/v2")
BENCHMARK = "SPY"
SYMBOLS_FILE = "symbols_500.txt"

LISTED_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}
SYMBOL_SHAPE = re.compile(r"^[A-Z]{1,5}$")  # drops share classes (BRK.B), warrants/units with separators
NOT_COMMON = re.compile(
    r"\bETF\b|\bETN\b|\bFund\b|\bWarrants?\b|\bRights?\b|\bUnits?\b|Preferred|\bNotes?\b|\bBonds?\b|%"
    r"|Acquisition Corp|\bDebentures?\b|\bTrust,? Series\b|iShares|SPDR|ProShares|Direxion|WisdomTree|VanEck|Global X",
    re.IGNORECASE,
)
# ARCA and BATS listings are mostly exchange-traded products whose names need not say "ETF"; elsewhere "Trust"
# is common in bank and REIT names, so it only counts on these venues.
PRODUCT_ON_ETP_VENUE = re.compile(r"\bTrust\b|\bShares\b|\bIndex\b|Invesco|Vanguard", re.IGNORECASE)


def fetch_assets(client: AlpacaDataClient) -> list[dict]:
    """Every US equity asset, active and inactive (one unpaged response)."""
    return client._get("/assets", {"asset_class": "us_equity", "status": "all"})  # noqa: SLF001 - trading API, not a bars call


def fetch_assets_any(transport=None) -> list[dict]:
    """Try live then paper trading endpoints; a paper account's keys only work on the paper one."""
    last: AlpacaError | None = None
    for url in TRADING_URLS:
        try:
            with AlpacaDataClient(base_url=url, transport=transport) as client:
                return fetch_assets(client)
        except AlpacaError as err:
            if err.status not in (401, 403):
                raise
            last = err
    raise last  # type: ignore[misc]


def filter_common_stock(assets: list[dict]) -> pd.DataFrame:
    """Exchange-listed common stock candidates; columns symbol, sector, exchange, status, name."""
    rows = []
    for a in assets:
        symbol = str(a.get("symbol", "")).strip().upper()
        name = str(a.get("name") or "")
        exchange = str(a.get("exchange") or "").upper()
        status = str(a.get("status") or "").lower()
        if a.get("class", "us_equity") != "us_equity" or exchange not in LISTED_EXCHANGES:
            continue
        if status == "active" and not a.get("tradable", False):
            continue
        if not SYMBOL_SHAPE.match(symbol) or NOT_COMMON.search(name):
            continue
        if exchange in {"ARCA", "BATS"} and PRODUCT_ON_ETP_VENUE.search(name):
            continue
        rows.append({"symbol": symbol, "sector": "unknown", "exchange": exchange, "status": status, "name": name})
    df = pd.DataFrame(rows, columns=["symbol", "sector", "exchange", "status", "name"])
    # Prefer the active row when a ticker appears twice (a reused symbol).
    df = df.assign(_inactive=df["status"] != "active").sort_values(["symbol", "_inactive"], kind="stable")
    return df.drop_duplicates("symbol").drop(columns="_inactive").reset_index(drop=True)


def load_or_fetch_assets(out: Path, assets_json: Path | None) -> tuple[list[dict], Path]:
    if assets_json is None:
        assets = fetch_assets_any()
        assets_json = out / f"assets_{datetime.now(UTC).date()}.json"
        assets_json.write_text(json.dumps(assets, sort_keys=True))
    return json.loads(Path(assets_json).read_text()), Path(assets_json)


def symbols_text(result: UniverseResult, cfg: UniverseConfig, assets_json: Path, n_candidates: int) -> str:
    counts = result.manifest["counts"]
    symbols = sorted({*result.selected["symbol"], BENCHMARK})
    header = [
        f"# interim universe {cfg.name} (scripts/momentum/build_interim_universe.py); A4/A5 replace this",
        f"# as_of {cfg.as_of}, window {cfg.window_sessions} sessions, feed {cfg.feed}, seed {cfg.seed}",
        f"# screens {json.dumps(cfg.screens, sort_keys=True)}, cells {json.dumps(cfg.cells, sort_keys=True)}",
        f"# assets {assets_json.name} sha256 {sha256_file(assets_json)}, {n_candidates} candidates",
        f"# eligible {counts['eligible']}, selected {counts['selected']}, plus {BENCHMARK} (descriptor only)",
        f"# git {result.manifest['git_commit']}",
    ]
    return "\n".join([*header, *symbols]) + "\n"


def build(
    config: Path,
    out: Path,
    assets: list[dict],
    assets_json: Path,
    store: DataStore | None = None,
    download: bool = False,
    workers: int = 8,
) -> tuple[UniverseResult, Path]:
    out.mkdir(parents=True, exist_ok=True)
    candidates = filter_common_stock(assets)
    candidates_csv = out / "candidates_interim.csv"
    candidates.to_csv(candidates_csv, index=False)
    print(f"{len(assets)} assets -> {len(candidates)} common-stock candidates ({candidates_csv})")

    cfg = replace(UniverseConfig.from_yaml(config), candidates=candidates_csv)
    if download:
        from shpe_research.data.bars import download_bars

        for call in required_downloads(cfg):
            print(download_bars(**call, feed=cfg.feed, workers=workers, store=store))
    result = build_universe(cfg, store)
    manifest_path = result.write(out / f"{cfg.name}.yaml")
    symbols_path = out / SYMBOLS_FILE
    symbols_path.write_text(symbols_text(result, cfg, assets_json, len(candidates)))

    counts = result.manifest["counts"]
    print(f"{counts['candidates']} candidates, {counts['excluded']} excluded, {counts['eligible']} eligible, "
          f"{counts['selected']} selected")
    for cell in result.manifest["cells"]:
        short = f", SHORT {cell['shortfall']}" if cell["shortfall"] else ""
        print(f"  {cell['cell']}: {cell['selected']} of {cell['eligible']} eligible{short}")
    print(f"manifest {manifest_path}\nsymbols  {symbols_path} ({counts['selected']} + {BENCHMARK})")
    return result, symbols_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=None, help="default <data dir>/universes/<config name>/")
    ap.add_argument("--assets-json", type=Path, default=None, help="reuse a saved /v2/assets snapshot (no API call)")
    ap.add_argument("--download", action="store_true", help="fetch the daily bars the measurement needs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    out = args.out or data_dir() / "universes" / UniverseConfig.from_yaml(args.config).name
    out.mkdir(parents=True, exist_ok=True)
    assets, assets_json = load_or_fetch_assets(out, args.assets_json)
    result, _ = build(args.config, out, assets, assets_json, download=args.download, workers=args.workers)
    return 0 if not any(c["shortfall"] for c in result.manifest["cells"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
