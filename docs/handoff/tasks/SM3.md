# SM3: Cross-sectional 12-1 momentum

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 2 | Int | C1d, C1e (T9 for portfolio runs) | W3, W8, momentum studies | `strategy-builder` (review: `leakage-auditor`) |

## Goal

The classic daily cross-sectional momentum signal: rank stocks by the return
from 12 months ago to 1 month ago and hold the winners against the losers
(Jegadeesh and Titman, 1993). It is the daily, cross-sectional counterpart to
the intraday ORB.

## Owns (exclusive)

- `src/shpe_research/strategies/momentum/xs_momentum.py`
- `configs/strategies/momentum/xs_momentum.yaml`
- `tests/unit/strategies/momentum/test_xs_momentum.py`

## Reads

- [C1d.md](C1d.md) (cross-sectional transforms), [C1e.md](C1e.md) (`momentum_12_1`, daily frequency)
- [T9.md](T9.md) (portfolio construction)
- [teams/momentum.md](../../teams/momentum.md)

## Interface

Registered as `xs_momentum`. Params: `lookback_days` (252), `skip_days` (21),
`rebalance` (`monthly`\|`weekly`), `transform` (`rank`\|`zscore`),
`min_price` (5.0), `min_adv_usd`. Output: one `CandidateFrame` row per
`(rebalance date, symbol)` with a `score` column; side comes from T9.

## Behaviour

- Requires `data.frequency: daily` (C1e); raises otherwise.
- Scores use `momentum_12_1` and the C1d transform grouped by `decision_ts`.
- Rebalance dates are the last trading day of each period; `decision_ts` is
  that day's close `ts_available`.
- Symbols below the price or ADV floor get no score (reported).

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/strategies/momentum/test_xs_momentum.py` | `test_passes_check_strategy` | C1k check passes |
| | `test_requires_daily` | Minute config raises |
| | `test_skip_month` | Score at t ignores the last `skip_days` returns |
| | `test_planted_effect` | On `synthetic_bars(daily={"years": 3, "n_symbols": 50, "xs_momentum": ...})` the rank IC is positive (seeded) |

## Out of scope

Weights and turnover (T9), the ranker model (W3).

## Done when

Tests pass and `configs/strategies/momentum/xs_momentum.yaml` runs on the daily fixture.
