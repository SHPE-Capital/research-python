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
