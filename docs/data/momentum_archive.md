# Momentum archive: manifest, pull procedure and limitations (task A1)

Status: tooling built and unit-tested with mocked HTTP. **The pilot pull has not been run**: `ALPACA_DATA_KEY` and
`ALPACA_DATA_SECRET` were not set in the environment and no `.env` exists. The measured sections below are
placeholders to fill from `<root>/reports/pilot_summary.json` after the run.

## How to run

```
uv run python scripts/momentum/pull_archive.py --stage pilot --root "<store root>" --symbols SPY,AAPL --start 2024-03-04 --end 2024-03-05 --dry-run
uv run python scripts/momentum/pull_archive.py --stage pilot --root "<store root>"
```

Store root used here: `C:\Users\juanc\Local Development\SHPE Capital\momentum-data`.

Stages: `pilot` (built-in pool, 2 years ending 2025-10-31 plus 3 months warm-up), `300x2`, `500x5`, `1000x10`
(these need `--symbols-file`). The script refuses to start if the planned size exceeds 10 GB (`--max-gb`) or would
leave under 20 GB free. It first probes SPY minute bars on both feeds and stops if SIP is refused.

## Layout and manifest

- Files: `<root>/bars/feed=sip/timeframe=1Min/adjustment=raw/symbol=X/period=YYYY-MM.parquet`, the same layout and
  Parquet schema as `data/store.py`, so `DataStore` can read them. Raw (unadjusted) prices; corporate actions are A3.
- Manifest: `<root>/archive_manifest.sqlite` (WAL). One row per symbol-month unit with rows, regular-session rows,
  expected regular minutes (early closes counted), coverage, first/last bar, fetched window, sha256, bytes,
  requests, feed and stage. `year_coverage()` gives the per symbol-year view. Tables `probes` (entitlement and
  delisted history) and `runs` (summary per run). `store.py`'s JSON manifest is unchanged.
- Resume: a complete unit whose window covers the wanted one and whose file exists is skipped; failed units write
  nothing and are retried next run.
- Logs: `<root>/logs/<stage>.log`; summary: `<root>/reports/<stage>_summary.json`.

## Pilot pool

About 58 liquid US common stocks, SPY (descriptor only), and 16 delisted or renamed names chosen from memory
(not verified against a listing database): SPLK, JNPR, ANSS, HES, WBA, DFS, PXD, MRO, CTLT, SRCL, SAVE, SQ, ALTR
(delisted or renamed inside the window), and ATVI, SIVB, TWTR (delisted before the window, to test how far back
Alpaca keeps minute bars). The pool is the full list in `pull_archive.py`.

## Measured results (pilot)

Not yet measured. To be recorded: coverage per symbol-year, 429 behaviour and requests per minute (mean and peak),
bars per request, bytes per row and total, wall time, SIP vs IEX entitlement, delisted verdict.

**Delisted coverage verdict:** pending the pilot run.
