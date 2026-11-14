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

## Measured results (pilot, PAID Alpaca plan, run 2026-11-13)

All numbers below come from a **paid** Alpaca account. The planned switch to the free plan can change the rate
limit, SIP history access and stream limits, so re-measure after the switch before trusting any pull projection.

| Item | Measured |
|---|---|
| Window | 2023-08-01 to 2025-10-31 (3 months warm-up plus 2 years), feed SIP |
| Pool | 77 symbols (58 liquid, SPY, 18 delisted/renamed probes) |
| Units | 2,079 symbol-months, 0 failed, 0 skipped |
| Rows | 19,253,795 minute bars |
| Wall time | 881 s (14.7 min) |
| Requests | 2,790 HTTP requests, all 200, **0 HTTP 429** |
| Request rate | mean 190/min, peak 191 per 60 s. This is our own throttle (`--rpm 190`), not Alpaca's ceiling, so the real paid-plan limit is **not yet found** |
| Bars per OK request | 6,901 (the plan assumed 10,000) |
| Bytes on disk | 789 MB (Parquet), 41.0 bytes per row, about 4.1 MB per full symbol-year |
| Bytes received | 2.01 GB over the wire (about 2.5x the on-disk size) |
| Entitlement | SIP 200 and IEX 200 on SPY (SIP returned about 2.1x the bars of IEX) |

### Extrapolation to later stages (linear in symbol-years, paid plan, 190 req/min)

Measured rate: about 5.1 s per symbol-year, about 4.1 MB per symbol-year. Pool is 3 x N symbols, plus warm-up.

| Stage | Symbol-years | Pull time | Disk |
|---|---|---|---|
| 300 x 2 years (900 symbols) | about 2,000 | about 2.9 h | about 8 GB |
| 500 x 5 years (1,500 symbols) | about 7,900 | about 11 h | about 32 GB |
| 1,000 x 10 years (3,000 symbols) | about 30,800 | about 44 h | about 126 GB |

The plan assumed about 16 h for 2 billion rows and about 43 GB for a 2,000-symbol, 10-year pool. Measured
bytes per row and bars per request are worse than assumed: the 1,000 x 10 stage needs about 40 h of pulling and
more disk than this machine has free (about 75 GB), before any panel build. Either raise the request rate if the
paid plan allows it (not yet tested), shrink the pool or history, or compress/convert to float32 panels per week
chunk and delete the Parquet after conversion.

### Delisted coverage verdict

**Sufficient for the pilot, with one caveat.** Minute bars exist for every probed delisted symbol back to
2016-01-04 (ATVI, SIVB, TWTR, JNPR, SPLK, HES, PXD, MRO, DFS, WBA, CTLT, SRCL and others) and end at the delisting
date. TWTR (delisted Oct 2022) and SIVB (March 2023) both return full history. Caveat: **renamed tickers stop at
the rename** (for example SQ ends 2025-01-18, when Block moved to XYZ), so continuity across a ticker change needs
the stable-asset-ID work in A3; the pull alone does not provide it. Pool membership of delisted names was
chosen from memory; the real candidate pool must be built from a point-in-time listing source, not from today's
symbols.
