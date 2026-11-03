# C1g: Data sources

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 1 | Int | C1a, C1h | C1m, studies that need non-Alpaca data | `repo-architect` (review: lead) |

## Goal

Move data loading behind a registry of sources, so a config can name Alpaca
bars, a local file, FRED series or EDGAR filings the same way, and every
source lands in a snapshot.

## Owns (exclusive)

- `src/shpe_research/data/sources/__init__.py`, `sources/base.py`, `sources/alpaca.py`,
  `sources/file.py`, `sources/fixture.py`, `sources/fred.py`, `sources/edgar.py`
- `src/shpe_research/experiments/sources.py` (reduced to re-exports)
- `tests/unit/data/test_sources.py`

## Reads

- `src/shpe_research/experiments/sources.py` (current `_fixture`, `_alpaca`, `_snapshot`)
- `src/shpe_research/data/alpaca.py`, `data/snapshot.py`, `data/verify.py` (D5)
- [adr/0011-pluggable-data-sources-security-master.md](../../adr/0011-pluggable-data-sources-security-master.md)

## Interface

- `@register_source` keyed on `source_type`; `build_source(cfg)` pops `type`.
- `Source.load(symbols, start, end, timeframe) -> pd.DataFrame` returning the
  bar contract (or, for `fred`/`edgar`, a frame with `ts`, `ts_available`,
  `series`, `value`).

| Key | Params | Notes |
|---|---|---|
| `fixture` | every `synthetic_bars` kwarg, passed through unchanged | Includes C1o's new options. Replaces the hand-picked kwargs in `experiments/sources._fixture` |
| `alpaca` | `feed`, `adjustment` | Wraps `data/alpaca.py` |
| `snapshot` | `path` | Verifies the checksum (D5) before loading |
| `file` | `path`, `format` (`parquet`\|`csv`) | Validates against the bar contract |
| `fred` | `series`, `release_lag` | `ts_available` = release date plus `release_lag`, never the observation date |
| `edgar` | `form`, `cik` | `ts_available` = filing acceptance time |

## Behaviour

- `experiments/sources.py` keeps its public functions as thin re-exports so
  existing configs and tests keep working.
- `fred` and `edgar` never set `ts_available` to the period the value
  describes. Network calls are mocked in tests.
- The fixture source passes every kwarg through; adding a C1o option needs no
  change here.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/data/test_sources.py` | `test_registry_keys` | All six keys resolve; an unknown key lists them |
| | `test_fixture_passthrough` | `{type: fixture, momentum: {...}}` reaches `synthetic_bars` unchanged |
| | `test_fred_release_lag` | Mocked series: `ts_available` equals release date plus lag |
| | `test_snapshot_checksum_checked` | A tampered snapshot raises |
| | `test_existing_configs_load` | Every config in `configs/` still loads |

## Out of scope

Security master (C1m), downloading quotes and trades (D3).

## Done when

Tests pass and the existing configs and goldens are unchanged.
