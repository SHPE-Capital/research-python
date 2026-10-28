# Task board

Each task is sized for one member working a few hours a week. Every row lists
the files to work in and what "done" means. Stub files contain a
`TODO(<id>)` docstring with the full spec, and most tasks have a skipped
acceptance test in `tests/todo/` that your PR un-skips and makes pass.
[CONTRIBUTING.md](../CONTRIBUTING.md) explains how to claim one.

**Only C0 blocked other work, and it is done.** Every open task can start now
against the synthetic fixture data (`shpe_research.data.fixtures.synthetic_bars`).
Real data becomes available as the data track lands.

Levels follow the master doc's ladder: **Beg** = can write basic programs,
**Int** = can organize modules and tests, **Adv** = comfortable with
statistics or systems design.

## Lead

| ID | Task | Files | Status |
|---|---|---|---|
| C0 | Contracts v0, repo skeleton, fixture data, hello model end to end | `contracts/`, `data/fixtures.py`, `experiments/runner.py` | Done |

## Data

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| D1 | Alpaca bars downloader into the Parquet store | Int | C0 | `data/alpaca.py`, `data/store.py`, `data/bars.py` | Re-running is a no-op; manifest has checksums | Done |
| D2 | Calendar, session tags, data audit | Beg–Int | C0 | `data/calendar.py`, `data/audit.py` | Audit report for a month of data | Done |
| D3 | Quotes/trades downloader plus minute summaries | Int | D1 | `data/ticks.py` | One week for 5 symbols | In progress (Juan) |
| D4 | Universe builder: liquidity x volatility cells, seeded dev/held-out split | Beg | D1, D2 | `data/universe.py`, `configs/universe/` | Reproducible manifest | In progress (Juan) |
| D5 | Frozen, versioned workshop dataset | Beg | D1, D4 | `data/snapshot.py` | Build, verify and load a snapshot | In progress (Juan) |
| D6 | Corporate actions: confirm split gaps flagged by the audit | Beg–Int | D1, D2 | `data/corporate_actions.py` | `tests/todo/test_experiments_todo.py::test_corporate_actions` | Open |

## Features

One feature per person is a good first PR. Each must pass `assert_no_lookahead`.

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| F1 | Relative volume and the intraday volume curve | Beg–Int | | `features/relative_volume.py` | `test_relative_volume` | Open |
| F2 | Distance from session VWAP | Beg | | `features/vwap_distance.py` | `test_vwap_distance` | Open |
| F3 | Realized volatility | Beg | | `features/realized_vol.py` | `test_realized_vol` | Open |
| F4 | Opening-range width and breakout distance | Beg–Int | | `features/opening_range.py` | `test_opening_range` | Open |
| F5 | Regime set: variance ratio, autocorrelation, efficiency ratio | Int | | `features/regime.py` | `test_regime_features` | Open |
| F6 | Order-flow features from tick minute summaries | Int | D3 | `features/order_flow.py` | `test_order_flow` | Open |
| F7 | Pair spread z-score, hedge ratio and half-life | Int | F8 (shared contract extension) | `features/cross_asset.py` | `test_pair_spread` | Open |
| F8 | Market (SPY) return at decision time | Int | F7 (shared contract extension) | `features/cross_asset.py` | `test_market_return` | Open |

Acceptance tests: `tests/todo/test_features_todo.py`.

## Labels and validation

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| L1 | Forward returns at several horizons, cost- and volatility-adjusted variants | Beg | | `labels/multi_horizon.py` | `test_multi_horizon` | Open |
| L2 | Exit-policy (stop/target/time) labels using the platform's bar fill rules | Int | | `labels/exit_policy.py` | `test_exit_policy`, plus reconciliation with the platform on 5 sessions | Open |
| V1 | Purged walk-forward splitter with embargo | Int | | `validation/purged.py` | `test_purged_walk_forward` | Open |

## Evaluation

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| E1 | IC decay, turnover, cost-adjusted P&L, by-group breakdowns | Int | L1 (for decay) | `evaluation/trading.py` | `test_trading_metrics`; written into every report | Open |
| E2 | Block bootstrap, deflated Sharpe, probability of backtest overfitting | Adv | | `evaluation/stats.py` | `test_stats` | Open |

## Models and experiments

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| M1a | Random forest | Beg | | `models/random_forest.py` | `test_m1_models[random_forest]` | Open |
| M1b | LightGBM | Beg–Int | | `models/gradient_boosting.py` | `test_m1_models[lightgbm]` | Open |
| M1c | Small neural network (chronological early stopping) | Int | | `models/mlp.py` | `test_m1_models[mlp]` | Open |
| M2 | Config sweeps, seed ensembles, MLflow registry | Int | | `experiments/sweep.py` | `test_sweep`; every run visible in MLflow | Open |
| M3 | ONNX export with golden parity rows for the platform | Int | | `models/export_onnx.py` | `test_onnx_export` | Open |

## Platform

These live in the trading-platform repo. See [platform-tasks.md](platform-tasks.md).

| ID | Task | Level | Depends on |
|---|---|---|---|
| P1 | Candidate/feature log: every strategy candidate, taken or skipped | Int (TS) | |
| P2 | ONNX selector wrapper around any strategy, plus parity test | Adv (TS) | M3, P1 |
| P3 | Replay data path: filter, loader from candidates, fills and bars | Beg–Int (TS) | P1 |
| P4 | Replay decision-viewer UI | Int (frontend) | P3 |
| P5 | Read-only Models tab | Int (frontend) | M3 |

## Research studies

Each study is team-owned and reports in the E1 format. Folders live in
[studies/](../studies).

| ID | Study | Team | Builds on |
|---|---|---|---|
| R1 | ORB trade selectors A–D (Christopher's plan) | Momentum | F1, F4, F8, L2, M1, D4 |
| R2 | Reversal meta-labeling | Reversals | F2, F3, F5, L2 |
| R3 | Pair selection by clustering, then spread entry meta-labeling | Stat arb | F7, D4 |
| R4 | Order-flow imbalance and short-horizon midpoint prediction | Microstructure | D3, F6 |
| R5 | Transaction cost analysis: paper fills vs. backtest assumptions | Any | Platform broker ledger |
