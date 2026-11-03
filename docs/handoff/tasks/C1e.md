# C1e: Daily frequency alongside minute bars

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 1 | Int | C1a, C1o | SM3, W3, W5, W8 | `repo-architect` (review: `leakage-auditor`) |

## Goal

Run the same pipeline on daily bars (`1Day`) so cross-sectional and
volatility studies work, without changing minute-bar behaviour.

## Owns (exclusive)

- `src/shpe_research/data/resample.py` (minute → daily)
- `src/shpe_research/features/daily/__init__.py`, `features/daily/returns.py` (`past_return_days`, `momentum_12_1`), `features/daily/har.py` (`har_rv`)
- `tests/unit/data/test_resample.py`, `tests/unit/features/test_daily_features.py`

## Reads

- `src/shpe_research/contracts/bars.py` (`DAILY = "1Day"`, `TIMEFRAMES`, `timeframe_delta`)
- `src/shpe_research/data/alpaca.py`, `data/snapshot.py`
- [design/pipeline.md](../../design/pipeline.md) (frequency and era)

## Interface

- Config: `data.frequency: minute | daily` (C1h adds the field; default `minute`).
- `resample_daily(bars) -> bars`: one row per symbol and regular session.
  `ts` = the session date at 09:30 New York; `ts_available` = the session's
  last regular bar's `ts_available` (16:00, or 13:00 on early-close days).
  OHLC from regular-session bars, `volume` summed, `vwap` volume-weighted,
  `session = "regular"`.
- Daily features (registered with `register_feature`):

| Key | Params | Value at session d |
|---|---|---|
| `past_return_days` | `days` | close_d / close_{d−days} − 1 |
| `momentum_12_1` | `skip_days=21`, `lookback_days=252` | close_{d−skip} / close_{d−lookback} − 1 |
| `har_rv` | `bar_minutes=5`, `components=(1, 5, 22)` | Columns `rv_d`, `rv_w`, `rv_m`: daily RV of session d and its 5- and 22-session means (Corsi 2009). Needs the minute bars, so it reads `context["minute_bars"]` |

## Behaviour

- A daily row's features use only sessions ≤ d. Labels start at d+1.
- `era` for daily rows is the ISO week (C1a).
- Early-close sessions are full rows; their `ts_available` is the early close.
- Sessions with no regular bars produce no row.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/data/test_resample.py` | `test_daily_ohlc_hand` | Hand-built two sessions resample to the exact OHLCV |
| | `test_ts_and_ts_available` | `ts` is 09:30 NY in UTC; `ts_available` is the last regular bar's |
| `tests/unit/features/test_daily_features.py` | `test_momentum_12_1_hand` | Value on a known path |
| | `test_har_rv_planted` | On C1o `vol_clustering{persistence: 0.95}`, `rv_d` autocorrelation at lag 1 > 0.5 |
| | `test_no_lookahead[key]` | `assert_no_lookahead` passes for each daily feature |
| | `test_daily_pipeline_runs` | `frequency: daily` with `session_close` decisions and `forward_return` runs end to end on C1o daily mode |

## Out of scope

Point-in-time universes (C1m), daily data download beyond resampling (C1g).

## Done when

Tests pass and a daily config runs on the fixture with the hello model.
