"""Command-line entry point: `uv run shpe <command>`.

Commands are thin wrappers; the logic lives in the library modules so
notebooks and tests call the same code. Each `_add_*` function registers one
command group.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _symbols(value: str) -> list[str]:
    return [s.strip().upper() for s in value.split(",") if s.strip()]


# ---------------------------------------------------------------- experiments


def _run(args: argparse.Namespace) -> int:
    from shpe_research.experiments.runner import ExperimentConfig, run_experiment

    result = run_experiment(ExperimentConfig.from_yaml(args.config), out_root=args.out)
    print((result.run_dir / "report.md").read_text())
    print(f"Run written to {result.run_dir}")
    return 0


def _add_run(sub) -> None:
    p = sub.add_parser("run", help="run one experiment config end to end")
    p.add_argument("config", type=Path, help="experiment YAML, e.g. configs/experiments/hello_model.yaml")
    p.add_argument("--out", type=Path, default=None, help="runs directory (default ./runs)")
    p.set_defaults(handler=_run)


# ---------------------------------------------------------------- fixtures


def _fixtures(args: argparse.Namespace) -> int:
    from shpe_research.contracts.bars import BarMetadata, bars_to_table
    from shpe_research.data.fixtures import synthetic_bars

    bars = synthetic_bars(symbols=args.symbols, start=args.start, end=args.end, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    meta = BarMetadata(feed="synthetic", timeframe="1Min", adjustment="raw", source="fixture")
    pq.write_table(bars_to_table(bars, meta), args.out)
    print(f"Wrote {len(bars):,} synthetic bars for {bars['symbol'].nunique()} symbols to {args.out}")
    return 0


def _add_fixtures(sub) -> None:
    from shpe_research.config import data_dir
    from shpe_research.data.fixtures import DEFAULT_SYMBOLS

    p = sub.add_parser("fixtures", help="write deterministic synthetic bars for offline work")
    p.add_argument("--out", type=Path, default=data_dir() / "fixtures" / "synthetic_1min.parquet")
    p.add_argument("--symbols", type=_symbols, default=list(DEFAULT_SYMBOLS))
    p.add_argument("--start", default="2026-03-02")
    p.add_argument("--end", default="2026-08-28")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(handler=_fixtures)


# ---------------------------------------------------------------- data


def _symbol_args(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", type=_symbols, help="comma-separated, e.g. SPY,KO")
    group.add_argument("--symbols-file", type=Path, help="text/CSV file; first column is the symbol")


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return args.symbols
    lines = [ln.split(",")[0].strip().upper() for ln in args.symbols_file.read_text().splitlines()]
    return [s for s in lines if s and s != "SYMBOL" and not s.startswith("#")]


def _data_bars(args: argparse.Namespace) -> int:
    from shpe_research.data.bars import download_bars

    summary = download_bars(
        _resolve_symbols(args), args.start, args.end, timeframe=args.timeframe, adjustment=args.adjustment,
        feed=args.feed, workers=args.workers, force=args.force,
    )
    print(summary)
    return 1 if summary.failed else 0


def _data_verify(args: argparse.Namespace) -> int:
    from shpe_research.data.store import DataStore

    problems = DataStore().verify()
    print("\n".join(problems) if problems else "All stored files match their manifest checksums.")
    return 1 if problems else 0


def _data_audit(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from shpe_research.config import data_dir
    from shpe_research.data.audit import audit_bars
    from shpe_research.data.store import DataStore

    bars = DataStore().read_bars(_resolve_symbols(args), args.start, args.end, timeframe=args.timeframe,
                                 feed=args.feed, adjustment=args.adjustment)
    report = audit_bars(bars, args.start, args.end, timeframe=args.timeframe)
    out = args.out or data_dir() / "audits" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report.write(out)
    print(report.to_markdown())
    print(f"Audit written to {out}")
    return 1 if len(report.errors) else 0


def _data_ticks(args: argparse.Namespace) -> int:
    from shpe_research.data.ticks import download_ticks

    summary = download_ticks(
        _resolve_symbols(args), args.start, args.end, kinds=tuple(args.kinds.split(",")), feed=args.feed,
        session=args.session, start_time=args.start_time, end_time=args.end_time, workers=args.workers,
        force=args.force,
    )
    print(summary)
    return 1 if summary.failed else 0


def _data_tick_summary(args: argparse.Namespace) -> int:
    from shpe_research.config import data_dir
    from shpe_research.data.store import DataStore
    from shpe_research.data.ticks import minute_summaries

    symbols = _resolve_symbols(args)
    store = DataStore()
    quotes = store.read_ticks("quotes", symbols, args.start, args.end, args.feed)
    trades = store.read_ticks("trades", symbols, args.start, args.end, args.feed)
    summary = minute_summaries(quotes, trades)
    out = args.out or data_dir() / "tick_summaries" / f"{'-'.join(symbols)}_{args.start}_{args.end}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out, index=False)
    print(f"{len(quotes):,} quotes and {len(trades):,} trades -> {len(summary):,} minute rows written to {out}")
    return 0


def _data_universe(args: argparse.Namespace) -> int:
    from shpe_research.data.bars import download_bars
    from shpe_research.data.universe import UniverseConfig, build_universe, default_output, required_downloads

    cfg = UniverseConfig.from_yaml(args.config)
    if args.download:
        for call in required_downloads(cfg):
            print(download_bars(**call, feed=cfg.feed, workers=args.workers))
    result = build_universe(cfg)
    path = result.write(args.out or default_output(cfg))
    counts = result.manifest["counts"]
    print(f"{counts['candidates']} candidates, {counts['excluded']} excluded, {counts['selected']} selected "
          f"({counts['dev']} dev, {counts['held_out']} held out)")
    for cell in result.manifest["cells"]:
        short = f", short {cell['shortfall']}" if cell["shortfall"] else ""
        print(f"  {cell['cell']}: {cell['selected']} of {cell['eligible']} eligible{short}")
    print(f"Manifest written to {path}")
    return 0


def _add_data(sub) -> None:
    data = sub.add_parser("data", help="download, audit and package market data")
    dsub = data.add_subparsers(dest="data_command", metavar="<data command>", required=True)

    p = dsub.add_parser("bars", help="download Alpaca bars into the local store (D1)")
    _symbol_args(p)
    p.add_argument("--from", dest="start", required=True, help="first New York date, YYYY-MM-DD")
    p.add_argument("--to", dest="end", required=True, help="last New York date, YYYY-MM-DD (inclusive)")
    p.add_argument("--timeframe", default="1Min", choices=["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"])
    p.add_argument("--adjustment", default="raw", choices=["raw", "split", "dividend", "all"])
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--force", action="store_true", help="refetch units even if already stored")
    p.set_defaults(handler=_data_bars)

    p = dsub.add_parser("verify", help="re-check stored files against manifest checksums")
    p.set_defaults(handler=_data_verify)

    p = dsub.add_parser("audit", help="coverage, duplicates, invalid prices and split gaps in stored bars (D2)")
    _symbol_args(p)
    p.add_argument("--from", dest="start", required=True)
    p.add_argument("--to", dest="end", required=True)
    p.add_argument("--timeframe", default="1Min", choices=["1Min", "5Min", "15Min", "30Min", "1Hour"])
    p.add_argument("--adjustment", default="raw")
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--out", type=Path, default=None, help="report directory (default <data dir>/audits/<time>)")
    p.set_defaults(handler=_data_audit)

    p = dsub.add_parser("ticks", help="download quotes and/or trades into the local store (D3)")
    _symbol_args(p)
    p.add_argument("--from", dest="start", required=True)
    p.add_argument("--to", dest="end", required=True)
    p.add_argument("--kinds", default="quotes,trades", help="quotes, trades, or quotes,trades")
    p.add_argument("--session", default="regular", choices=["regular", "extended"])
    p.add_argument("--start-time", default=None, help="narrow each day's window, New York HH:MM")
    p.add_argument("--end-time", default=None, help="narrow each day's window, New York HH:MM")
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--force", action="store_true")
    p.set_defaults(handler=_data_ticks)

    p = dsub.add_parser("tick-summary", help="per-minute spread, imbalance and signed volume from stored ticks (D3)")
    _symbol_args(p)
    p.add_argument("--from", dest="start", required=True)
    p.add_argument("--to", dest="end", required=True)
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(handler=_data_tick_summary)

    p = dsub.add_parser("universe", help="seeded liquidity x volatility universe with dev/held-out split (D4)")
    p.add_argument("--config", type=Path, required=True, help="e.g. configs/universe/workshop_v0.yaml")
    p.add_argument("--download", action="store_true", help="first download the bars the measurements need")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", type=Path, default=None, help="manifest path (default <data dir>/universes/<name>.yaml)")
    p.set_defaults(handler=_data_universe)


# ---------------------------------------------------------------- main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shpe", description="SHPE Capital research tools")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _add_run(sub)
    _add_fixtures(sub)
    _add_data(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
