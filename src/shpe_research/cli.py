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


# ---------------------------------------------------------------- main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shpe", description="SHPE Capital research tools")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _add_run(sub)
    _add_fixtures(sub)
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
