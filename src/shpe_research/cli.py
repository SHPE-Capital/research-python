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


def _sweep(args: argparse.Namespace) -> int:
    from shpe_research.experiments.sweep import (
        default_tracking_uri,
        execute_sweep,
        select_best,
        summarize,
        trial_count,
    )

    uri = args.tracking_uri or default_tracking_uri()
    runs, sweep = execute_sweep(args.config, out_root=args.out, tracking_uri=uri)
    print(f"{len(runs)} runs; selection metric {sweep.metric} ({sweep.direction}), validation only")
    for row in summarize(runs, sweep.metric):
        print(f"  {row['config']}: mean {row['mean']:.5f} sd {row['sd']:.5f} over {row['n_seeds']} seed(s)")
    try:
        best = select_best(runs, sweep.metric, sweep.direction)
        print(f"selected: {best['config']} (mean {best['mean']:.5f})")
    except ValueError as exc:
        print(f"no selection: {exc}")
    print(f"trials recorded for {sweep.base.name!r}: {trial_count(uri, sweep.base.name)}")
    return 0


def _add_sweep(sub) -> None:
    p = sub.add_parser("sweep", help="run a config grid and log every run to MLflow (M2)")
    p.add_argument("config", type=Path, help="sweep YAML, e.g. configs/experiments/sweeps/hello_sweep.yaml")
    p.add_argument("--out", type=Path, default=None, help="runs directory (default ./runs)")
    p.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (default ./mlruns)")
    p.set_defaults(handler=_sweep)


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


def _series(value: str) -> tuple[tuple[str, str], ...]:
    pairs = [item.split(":") for item in value.split(",") if item]
    return tuple((tf, adj) for tf, adj in pairs)


def _snapshot_build(args: argparse.Namespace) -> int:
    from shpe_research.data.snapshot import build_snapshot, load_snapshot

    folder = build_snapshot(
        args.name, args.start, args.end, symbols=args.symbols, universe=args.universe,
        extra_symbols=args.extra_symbols, series=args.series, feed=args.feed,
    )
    snap = load_snapshot(folder)
    rows = sum(f["rows"] for f in snap.manifest["files"])
    print(f"Snapshot {snap.name}: {len(snap.symbols())} symbols, {rows:,} rows, fingerprint {snap.fingerprint()[:12]}")
    print(f"Written to {folder} and {folder.with_suffix('.zip')}")
    print("Members-only: it contains Alpaca market data. Do not commit or publish it.")
    return 0


def _snapshot_verify(args: argparse.Namespace) -> int:
    from shpe_research.data.snapshot import load_snapshot, verify_snapshot

    problems = verify_snapshot(args.path)
    if problems:
        print("\n".join(problems))
        return 1
    snap = load_snapshot(args.path)
    print(f"Snapshot {snap.name} verified: {len(snap.manifest['files'])} files, fingerprint {snap.fingerprint()[:12]}")
    return 0


def _add_snapshot(dsub) -> None:
    snap = dsub.add_parser("snapshot", help="build or verify a frozen dataset snapshot (D5)")
    ssub = snap.add_subparsers(dest="snapshot_command", metavar="<snapshot command>", required=True)

    p = ssub.add_parser("build", help="copy bars from the store into snapshots/<name>/ and a zip")
    p.add_argument("--name", required=True, help="immutable; use a new name for a new version")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--universe", type=Path, help="D4 manifest; its dev and held-out names become the symbols")
    source.add_argument("--symbols", type=_symbols)
    p.add_argument("--extra-symbols", type=_symbols, default=[], help="context symbols such as SPY")
    p.add_argument("--from", dest="start", required=True)
    p.add_argument("--to", dest="end", required=True)
    p.add_argument("--series", type=_series, default="1Min:raw,1Day:split", help="timeframe:adjustment list")
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.set_defaults(handler=_snapshot_build)

    p = ssub.add_parser("verify", help="check a snapshot folder or zip against its checksums")
    p.add_argument("path", type=Path)
    p.set_defaults(handler=_snapshot_verify)


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

    _add_snapshot(dsub)


# ---------------------------------------------------------------- momentum


def _date(value: str):
    from datetime import date

    return date.fromisoformat(value)


def _momentum_cache_path(p) -> Path:
    """``cache.path``, else ``<compute.scratch_dir or runs_dir()/momentum-scratch>/momentum_cache.sqlite``."""
    from shpe_research.config import runs_dir

    if p.cache.path:
        return Path(p.cache.path)
    scratch = Path(p.compute.scratch_dir) if p.compute.scratch_dir else runs_dir() / "momentum-scratch"
    return scratch / "momentum_cache.sqlite"


def _momentum_calibrate(args: argparse.Namespace) -> int:
    from shpe_research.calibration.cache import EvalCache
    from shpe_research.calibration.descriptors import PanelProxyDescriptors
    from shpe_research.calibration.reserved import ReservedGuard, ReservedPeriodError
    from shpe_research.calibration.scheduler import run_strategies
    from shpe_research.calibration.strategies import load_strategies
    from shpe_research.calibration.universe import StaticUniverseProvider
    from shpe_research.calibration.weekly import CalibrationInputs, IncompleteRunError, guarded_cutoff
    from shpe_research.calibration.window import Archive, InsufficientHistoryError
    from shpe_research.config.momentum import load_protocol

    rp = load_protocol(args.protocol, args.set)
    p = rp.protocol
    archive = Archive.open(args.panel_root)  # metadata only
    try:
        # The guard runs before any bar is read; the descriptors read nothing after the cutoff.
        cutoff = guarded_cutoff(archive, args.cutoff, ReservedGuard.from_protocol(rp, args.protocol_hash))
        cache = EvalCache(_momentum_cache_path(p), busy_timeout_s=p.cache.busy_timeout_s)
        try:
            inputs = CalibrationInputs(
                rp=rp,
                archive=archive,
                universe=StaticUniverseProvider.from_archive(archive),
                descriptors=PanelProxyDescriptors(archive, lookback=p.weights.descriptor_lookback, through=cutoff),
                cache=cache,
                data_version=args.data_version,
                out_root=args.out,
                expected_hash=args.protocol_hash,
            )
            outcomes, _ = run_strategies(inputs, args.cutoff, args.policy, load_strategies(rp))
        finally:
            cache.close()
    except (IncompleteRunError, ReservedPeriodError, InsufficientHistoryError) as exc:  # incl. StaleArchiveError
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for outcome in outcomes.values():
        print(f"{outcome.status}: {outcome.run_dir / 'weekly_config.json'}")
    return 0


def _momentum_walkforward(args: argparse.Namespace) -> int:
    """``shpe momentum walkforward``: run or resume the D8 walk-forward.

    Exit codes: 0 on ``complete`` or ``archive_end``; 2 with a message on stderr for
    IncompleteRunError (rerunning the same command resumes), InsufficientHistoryError (including
    StaleArchiveError: fix the archive or the first cutoff) and ReservedPeriodError. A
    ReservedPeriodError while ``reserved_start`` is set is a stop at the reserved boundary, not a
    failure: every earlier step is recorded and rerunning goes no further unless the protocol is
    frozen and ``--protocol-hash`` matches. Exit 2 keeps the code the D8 brief fixes for it.
    """
    from shpe_research.calibration.cache import EvalCache
    from shpe_research.calibration.descriptors import PanelProxyDescriptors
    from shpe_research.calibration.reserved import ReservedGuard, ReservedPeriodError
    from shpe_research.calibration.strategies import load_strategies
    from shpe_research.calibration.universe import StaticUniverseProvider
    from shpe_research.calibration.walkforward import run_walkforward, weekly_cutoffs
    from shpe_research.calibration.weekly import CalibrationInputs, IncompleteRunError
    from shpe_research.calibration.window import Archive, InsufficientHistoryError
    from shpe_research.config.momentum import load_protocol

    overrides = list(args.set)
    if args.policies:  # an override, so a frozen protocol refuses it (frozen runs set policies in the YAML)
        overrides.append(f"walkforward.policies=[{','.join(args.policies)}]")
    rp = load_protocol(args.protocol, overrides)
    p = rp.protocol
    strategies = load_strategies(rp)
    if len(strategies) != 1:
        raise ValueError(f"walkforward needs exactly one strategy in search.strategies, got {len(strategies)}")
    archive = Archive.open(args.panel_root)  # metadata only
    guard = ReservedGuard.from_protocol(rp, args.protocol_hash)
    try:
        # The guard runs before any bar is read: the descriptors read nothing past the last cutoff
        # the guard allows (later cutoffs are refused by run_walkforward before they are calibrated).
        guard.require_set()
        cutoffs = weekly_cutoffs(archive, args.first_cutoff, args.weeks)
        if not cutoffs:
            raise ValueError(f"the archive has no session in the ISO week of {args.first_cutoff}")
        guard.check_calibration(cutoffs[0])
        through = max(c for c in cutoffs if guard.allows_calibration(c))
        cache = EvalCache(_momentum_cache_path(p), busy_timeout_s=p.cache.busy_timeout_s)
        try:
            inputs = CalibrationInputs(
                rp=rp,
                archive=archive,
                universe=StaticUniverseProvider.from_archive(archive),
                descriptors=PanelProxyDescriptors(archive, lookback=p.weights.descriptor_lookback, through=through),
                cache=cache,
                data_version=args.data_version,
                out_root=args.out / "artifacts",
                expected_hash=args.protocol_hash,
            )
            result = run_walkforward(inputs, args.first_cutoff, args.weeks, args.out, strategy=strategies[0])
        finally:
            cache.close()
    except ReservedPeriodError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if guard.reserved_start is not None:
            print(
                f"stopped at the reserved-period boundary ({guard.reserved_start}): this is a boundary stop, "
                f"not a failure; the steps before it are recorded in {args.out}. Rerunning goes no further "
                "unless the protocol is frozen and --protocol-hash matches.",
                file=sys.stderr,
            )
        return 2
    except IncompleteRunError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("rerun the same command to resume", file=sys.stderr)
        return 2
    except InsufficientHistoryError as exc:  # incl. StaleArchiveError
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"{result.stopped}: {len(result.records)} steps; daily account series in {result.out_dir / 'daily'}")
    return 0


def _policies(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def _add_momentum(sub) -> None:
    from shpe_research.config import runs_dir
    from shpe_research.config.momentum import DEFAULT_PROTOCOL_PATH

    group = sub.add_parser("momentum", help="momentum weekly calibration (stream D)")
    msub = group.add_subparsers(dest="momentum_command", metavar="<momentum command>", required=True)

    p = msub.add_parser("calibrate", help="calibrate one weekend: one cutoff -> one 0-A weekly config (D7)")
    p.add_argument("--panel-root", type=Path, required=True, help="A6 week-chunk archive directory")
    p.add_argument("--cutoff", type=_date, required=True, help="as-of date YYYY-MM-DD (last session on or before)")
    p.add_argument("--policy", required=True, choices=["uniform", "recency", "similarity"])
    p.add_argument("--data-version", required=True, help="identity of the data snapshot (cache key, artifact)")
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH, help="0-G protocol YAML")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="protocol override (repeatable)")
    p.add_argument("--out", type=Path, default=runs_dir() / "momentum" / "calibrate", help="run directories root")
    p.add_argument("--protocol-hash", default=None, metavar="HEX", help="unlocks the reserved period if frozen")
    p.set_defaults(handler=_momentum_calibrate)

    p = msub.add_parser("walkforward", help="walk-forward over consecutive weekends with forward accounts (D8)")
    p.add_argument("--panel-root", type=Path, required=True, help="A6 week-chunk archive directory")
    p.add_argument("--first-cutoff", type=_date, required=True, help="first cutoff YYYY-MM-DD (its ISO week)")
    p.add_argument("--weeks", type=int, required=True, help="number of consecutive weeks")
    p.add_argument("--data-version", required=True, help="identity of the data snapshot (cache key, artifact)")
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH, help="0-G protocol YAML")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="protocol override (repeatable)")
    p.add_argument("--out", type=Path, default=runs_dir() / "momentum" / "walkforward", help="walk-forward out dir")
    p.add_argument("--protocol-hash", default=None, metavar="HEX", help="unlocks the reserved period if frozen")
    p.add_argument(
        "--policies", type=_policies, default=None, help="comma-separated subset (override of walkforward.policies)"
    )
    p.set_defaults(handler=_momentum_walkforward)


# ---------------------------------------------------------------- main


def _export(args: argparse.Namespace) -> int:
    """`shpe export <run_dir>/model`: write <run_dir>/export/ with model.onnx, manifest.json, golden.parquet.

    Golden rows are the first N validation-split rows of the dataset rebuilt
    from <run_dir>/config.yaml.
    """
    from shpe_research.contracts.model import load_model
    from shpe_research.experiments.runner import ExperimentConfig, build_dataset
    from shpe_research.experiments.sources import load_bars
    from shpe_research.models.export_onnx import export_onnx

    model_dir = Path(args.model_dir)
    run_dir = model_dir.parent
    cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml")
    golden = build_dataset(cfg, load_bars(cfg.data)).subset("validation").X.head(args.golden_rows)
    out = export_onnx(load_model(model_dir), golden, run_dir / "export", tolerance=args.tolerance)
    print(f"Exported to {out}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shpe", description="SHPE Capital research tools")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _add_run(sub)
    _add_sweep(sub)
    _add_fixtures(sub)
    _add_data(sub)
    _add_momentum(sub)
    from shpe_research.scaffold.new import _add_new

    _add_new(sub)
    from shpe_research.models.export_onnx import _add_export

    _add_export(sub, _export)
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
