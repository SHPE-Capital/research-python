# Task board

Each task is sized for one member working a few hours a week. Every row lists
the files to work in and what "done" means. Stub files contain a
`TODO(<id>)` docstring with the full spec, and most tasks have a skipped
acceptance test in `tests/todo/` that your PR un-skips and makes pass.
[CONTRIBUTING.md](../CONTRIBUTING.md) explains how to claim one.

**Only C0 blocked other work, and it is done.** Every open task in the
original tracks (D, F, L, V, E, M) can start now against the synthetic fixture
data (`shpe_research.data.fixtures.synthetic_bars`). Real data becomes
available as the data track lands.

The architecture (C1), trade toolkit (T), strategy (S) and workshop (W) tracks
run in waves. Each has a handoff packet in [handoff/tasks/](handoff/tasks/);
[handoff/README.md](handoff/README.md) explains worktrees, ownership, waves and
merge order. **Wave 0 is a lead chore:** the migration PR that moves files to
the target tree in [design/architecture.md](design/architecture.md), then C1a.
File paths below are the paths after that PR.

Levels follow the master doc's ladder: **Beg** = can write basic programs,
**Int** = can organize modules and tests, **Adv** = comfortable with
statistics or systems design.

Paths in the Files column are relative to `src/shpe_research/` unless they
start with `tests/`, `configs/`, `workshop/` or another top-level folder.

## Lead

| ID | Task | Files | Status |
|---|---|---|---|
| C0 | Contracts v0, repo skeleton, fixture data, hello model end to end | `contracts/`, `data/fixtures.py`, `experiments/runner.py` | Done |
| Migration PR | Move files to the target tree (Wave 0, before C1a); no behaviour change | see [handoff/README.md](handoff/README.md#wave-0-the-migration-pr) | Done |

## Data

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| D1 | Alpaca bars downloader into the Parquet store | Int | C0 | `data/alpaca.py`, `data/store.py`, `data/bars.py` | Re-running is a no-op; manifest has checksums | Done |
| D2 | Calendar, session tags, data audit | Beg–Int | C0 | `data/calendar.py`, `data/audit.py` | Audit report for a month of data | Done |
| D3 | Quotes/trades downloader plus minute summaries | Int | D1 | `data/ticks.py` | One week for 5 symbols | Done |
| D4 | Universe builder: liquidity x volatility cells, seeded dev/held-out split | Beg | D1, D2 | `data/universe.py`, `configs/universe/` | Reproducible manifest | Done |
| D5 | Frozen, versioned workshop dataset | Beg | D1, D4 | `data/snapshot.py` | Build, verify and load a snapshot | Done |
| D6 | Corporate actions: confirm split gaps flagged by the audit | Beg–Int | D1, D2 | `data/corporate_actions.py` | `tests/todo/test_experiments_todo.py::test_corporate_actions` | Open |

## Features

One feature per person is a good first PR. Each must pass `assert_no_lookahead`.
F1–F6 and F7/F8 have short pointer packets that record what the new layout
changes; the spec stays in the stub docstring.

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| [F1](handoff/tasks/F1.md) | Relative volume and the intraday volume curve | Beg–Int | | `features/volume/relative_volume.py` | `test_relative_volume` | Open |
| [F2](handoff/tasks/F2.md) | Distance from session VWAP | Beg | | `features/volume/vwap_distance.py` | `test_vwap_distance` | Open |
| [F3](handoff/tasks/F3.md) | Realized volatility | Beg | | `features/volatility/realized_vol.py` | `test_realized_vol` | Open |
| [F4](handoff/tasks/F4.md) | Opening-range width and breakout distance | Beg–Int | | `features/session/opening_range.py` | `test_opening_range`, for `range_minutes` 5, 15 and 30 | Open |
| [F5](handoff/tasks/F5.md) | Regime set: variance ratio, autocorrelation, efficiency ratio | Int | | `features/regime/regime.py` | `test_regime_features` | Open |
| [F6](handoff/tasks/F6.md) | Order-flow features from tick minute summaries | Int | D3 | `features/microstructure/order_flow.py` | `test_order_flow` | Open |
| [F7](handoff/tasks/F7-F8.md) | Pair spread z-score, hedge ratio and half-life | Int | C1d, SS2, SS3 | `features/cross_asset/pair_spread.py` | `test_pair_spread` | Open |
| [F8](handoff/tasks/F7-F8.md) | Market (SPY) return and β at decision time | Int | C1d | `features/cross_asset/market_return.py` | `test_market_return` | Open |

Acceptance tests: `tests/todo/test_features_todo.py`.

## Labels and validation

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| L1 | Forward returns at several horizons, cost- and volatility-adjusted variants | Beg | | `labels/multi_horizon.py` | `test_multi_horizon` | Open |
| [L2](handoff/tasks/L2.md) | Exit-policy (stop/target/time) labels using the platform's bar fill rules. Superseded in part by T1: L2 is now a thin wrapper over `trade/` | Int | T1, C1c | `labels/exit_policy.py` | Shim tests pass, plus reconciliation with the platform on 5 sessions | Open |
| [V1](handoff/tasks/V1.md) | Purged walk-forward splitter with embargo | Int | | `validation/purged.py` | `test_purged_walk_forward` | Open |

## Evaluation

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| [E1](handoff/tasks/E1.md) | IC decay, turnover, cost-adjusted P&L, by-group breakdowns | Int | C1a; L1 (for decay) | `evaluation/trading.py` | `test_trading_metrics`; written into every report | Open |
| [E2](handoff/tasks/E2.md) | Block bootstrap, deflated Sharpe, probability of backtest overfitting | Adv | | `evaluation/stats.py` | `test_stats` | Open |

## Models and experiments

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| M1a | Random forest | Beg | | `models/random_forest.py` | `test_m1_models[random_forest]` | Open |
| M1b | LightGBM | Beg–Int | | `models/gradient_boosting.py` | `test_m1_models[lightgbm]` | Open |
| M1c | Small neural network (chronological early stopping) | Int | | `models/mlp.py` | `test_m1_models[mlp]` | Open |
| M2 | Config sweeps, seed ensembles, MLflow registry | Int | | `experiments/sweep.py` | `test_sweep`; every run visible in MLflow | Open |
| [M3](handoff/tasks/M3.md) | ONNX export with golden parity rows for the platform | Int | | `models/export_onnx.py` | `test_onnx_export` | Open |

## Architecture (C1)

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| [C1a](handoff/tasks/C1a.md) | Pipeline stages, registries and package skeleton | Adv | ADRs 0002–0005, migration PR | `pipeline/`, `experiments/runner.py`, `contracts/candidates.py`, `events/types.py`, `trade/types.py`, empty package skeletons | Registry tests pass; hello golden unchanged | Done |
| [C1b](handoff/tasks/C1b.md) | Events and candidates layer | Int–Adv | C1a, C1o | `contracts/candidates.py`, `events/samplers.py`, `events/thresholds.py`, `events/breakouts.py` | `decisions: {type: session_close}` runs end to end | Open |
| [C1c](handoff/tasks/C1c.md) | Label registry | Int | C1a; T1 for `triple_barrier` | `labels/base.py`, `labels/realized_var.py`, `labels/triple_barrier.py` | Label tests pass; hello golden unchanged | Open |
| [C1d](handoff/tasks/C1d.md) | Context symbols and cross-sectional transforms | Int–Adv | C1a, C1o | `features/base.py` (`context`), `transforms/` | `context_symbols: [SPY]` config runs; contract bump reviewed | Open |
| [C1e](handoff/tasks/C1e.md) | Daily frequency alongside minute bars | Int | C1a, C1o | `data/resample.py`, `features/daily/` | Daily config runs with the hello model | Open |
| [C1f](handoff/tasks/C1f.md) | Fit schemes | Adv | C1a, V1 | `validation/schemes.py` | Hello model runs with each scheme; manifest records folds | Open |
| [C1g](handoff/tasks/C1g.md) | Pluggable data sources | Int | C1a, C1h | `data/sources/` | Source tests pass; configs and goldens unchanged | Open |
| [C1h](handoff/tasks/C1h.md) | Typed config schema and `StrategySpec` | Int | C1a | `config/` | Runner loads through `load_config`; golden unchanged | Open |
| [C1i](handoff/tasks/C1i.md) | Baselines, permutation importance and by-era report sections | Int | C1a, E1 | `evaluation/baselines.py`, `evaluation/importance.py`, `evaluation/by_era.py` | Three sections render in the hello report | Open |
| [C1j](handoff/tasks/C1j.md) | `shpe new` scaffolding framework | Beg–Int | C1a, C1k | `scaffold/`, `cli.py` (`new`) | `shpe new feature returns/demo --dry-run` prints the expected paths | Open |
| [C1k](handoff/tasks/C1k.md) | Contributor check kits | Int | C1a, C1o | `testing/checks.py`, `testing/fixtures.py` | Check-kit tests pass | Open |
| [C1l](handoff/tasks/C1l.md) | Import-layer enforcement | Beg–Int | C1a, migration PR | `pyproject.toml` (`[tool.importlinter]`), `.github/workflows/ci.yml` | `lint-imports` passes in CI | Open |
| [C1m](handoff/tasks/C1m.md) | Security master and point-in-time universe | Int–Adv | C1g, D4, D5 | `data/security_master.py`, `data/universe.py` (`as_of`) | `universe: {type: point_in_time}` config runs | Open |
| [C1n](handoff/tasks/C1n.md) | Documentation site | Beg | C1a; best after most Wave 2 tasks | `mkdocs.yml`, `docs/api/` | `mkdocs build --strict` passes in CI | Open |
| [C1o](handoff/tasks/C1o.md) | Synthetic fixture options with planted effects | Int–Adv | C1a, migration PR | `data/fixtures.py` | Every effect recovered by its test; defaults unchanged | Open |

## Trade toolkit (T)

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| [T1](handoff/tasks/T1.md) | Exit policies and trade simulator | Adv | C1a, C1o | `trade/types.py`, `trade/exits.py`, `trade/simulate.py` | Simulator and fill tests pass; hello golden unchanged | Open |
| [T2](handoff/tasks/T2.md) | Sizing rules | Int | C1a | `trade/sizing.py` | `simulate_trades(..., sizing=...)` works | Open |
| [T3](handoff/tasks/T3.md) | Entry filters | Beg–Int | C1a | `trade/filters.py` | `StrategySpec.filters` builds through the registry | Open |
| [T4](handoff/tasks/T4.md) | Risk limits | Int | C1a | `trade/risk.py` | `simulate_trades(..., risk=...)` rejects with reasons | Open |
| [T5](handoff/tasks/T5.md) | Cost model | Int | C1a; R5 for calibration | `trade/costs.py` | T1 uses `CostModel` for `cost` | Open |
| [T6](handoff/tasks/T6.md) | Trade analysis: holding, excursions, surfaces | Int–Adv | T1, E1 | `analysis/` | Team guides' holding and stop steps run on fixtures | Open |
| [T7](handoff/tasks/T7.md) | Platform config export and parity check | Adv | C1h, T1, T2, T4, T5 | `deploy/platform_config.py`, `deploy/parity.py`, `tests/integration/golden/` | SR1 and SS4 pass their goldens; commission finding recorded | Open |
| [T8](handoff/tasks/T8.md) | Strategy spec card and strategy template | Beg–Int | C1h, C1j, T7 | `scaffold/templates/strategy/`, `strategies/card.py` | SR1 has a rendered card | Open |
| [T9](handoff/tasks/T9.md) | Portfolio construction | Int–Adv | C1a, C1d, T5 | `portfolio/construct.py` | W8 builds its portfolios through `construct.py` | Open |

## Strategy families (S)

Strategy packages are `strategies/momentum/`, `strategies/reversal/` and
`strategies/statarb/`. The team guides in [teams/](teams/README.md) say which
study uses which task.

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| [SM1](handoff/tasks/SM1.md) | Opening-range breakout candidate generator | Int | C1b, T1 | `strategies/momentum/orb.py`, `configs/strategies/momentum/orb.yaml` | Config runs end to end on the fixture | Open |
| [SM2](handoff/tasks/SM2.md) | EMA crossover baseline | Beg | C1a | `strategies/momentum/ema_cross.py`, `configs/strategies/momentum/ema_cross.yaml` | Config runs on the fixture | Open |
| [SM3](handoff/tasks/SM3.md) | Cross-sectional 12-1 momentum | Int | C1d, C1e | `strategies/momentum/xs_momentum.py`, `configs/strategies/momentum/xs_momentum.yaml` | Config runs on the daily fixture | Open |
| [SR1](handoff/tasks/SR1.md) | Bar-fade baseline (platform parity) | Int | C1a, T1 | `strategies/reversal/bar_fade.py`, `configs/strategies/reversal/bar_fade.yaml` | All `minute_reversal` goldens pass | Open |
| [SR2](handoff/tasks/SR2.md) | Stretch fade | Int–Adv | T1, F2, F3 | `strategies/reversal/stretch.py`, `configs/strategies/reversal/stretch_fade.yaml` | Config runs end to end | Open |
| [SR3](handoff/tasks/SR3.md) | Residual fade | Adv | F7/F8, T1 | `strategies/reversal/residual.py`, `configs/strategies/reversal/residual_fade.yaml` | SR1–SR3 comparable on one fixture | Open |
| [SS1](handoff/tasks/SS1.md) | Pair selection | Adv | SS2, SS3 | `strategies/statarb/selection.py` | W7 can call each method | Open |
| [SS2](handoff/tasks/SS2.md) | Hedge ratios | Int | C1a | `features/cross_asset/pair_math.py` (hedge section), `strategies/statarb/hedge.py` | Hedge tests pass; no copied estimators | Done |
| [SS3](handoff/tasks/SS3.md) | Ornstein-Uhlenbeck fit and half-life | Int | C1a | `features/cross_asset/pair_math.py` (OU section), `strategies/statarb/ou.py` | OU tests pass; T6 delegates to `fit_ou` | Open |
| [SS4](handoff/tasks/SS4.md) | Pairs candidate generator (platform parity) | Adv | SS2, SS3, T1 | `strategies/statarb/pairs.py`, `configs/strategies/statarb/pairs.yaml` | All `pairs_trading` goldens pass | Open |

## Workshop build (W)

Example folders are `workshop/NN_slug/`, tests are
`tests/workshop/test_NN_slug.py`, and solutions live on the
`workshop-solutions` branch. Specs for each example are in
[workshop/examples/](workshop/examples/).

| ID | Task | Level | Depends on | Files | Done when | Status |
|---|---|---|---|---|---|---|
| [W0](handoff/tasks/W0.md) | Workshop infrastructure | Int | C1o | `workshop/README.md`, `workshop/_common/`, `tests/workshop/`, `pyproject.toml` (`workshop` group) | `uv sync --group workshop` works; harness test passes | Open |
| [W1](handoff/tasks/W1.md) | Leakage lab | Beg–Int | W0, C0, D4, D5 | `workshop/01_leakage_lab/` | Tests pass on the solutions branch | Open |
| [W2](handoff/tasks/W2.md) | Intraday model comparison | Int | W0, M1a–M1c, F1, F2, F3, F5, C1f or V1 | `workshop/02_intraday_model_comparison/` | Notebook runs with `--extra ml` | Open |
| [W3](handoff/tasks/W3.md) | Cross-sectional ranker | Int–Adv | W0, C1a, C1b, C1d, C1e, L1, V1, E1, M1b, SM3 | `workshop/03_cross_sectional_ranker/`, `models/ridge.py`, `models/identity.py` | Five configs run from the notebook | Open |
| [W4](handoff/tasks/W4.md) | Meta-labeling the opening-range breakout | Adv | W0, SM1, F1–F4, F8, T1, C1b, C1c, M1a, M1c, E1, E2 | `workshop/04_meta_labeling_orb/` | Notebook compares selectors A–D in one table | Open |
| [W5](handoff/tasks/W5.md) | Volatility forecast | Int | W0, F3, C1e, C1c, C1f, M1b, E2 | `workshop/05_volatility_forecast/` | `ex05_har_vs_gbm.py` prints the comparison table | Open |
| [W6](handoff/tasks/W6.md) | Overfitting casino | Int | W0, T1, C1b, E2, M2, T6, T5 | `workshop/06_overfitting_casino/` | Notebook runs under 5 minutes | Open |
| [W7](handoff/tasks/W7.md) | Unsupervised pairs | Adv | W0, C1o, SS1–SS4, F7, T1, T5, E2, M2 | `workshop/07_unsupervised_pairs/` | Notebook prints the method comparison table | Open |
| [W8](handoff/tasks/W8.md) | Portfolio construction | Adv | W0, C1d, C1e, C1o, F3, F8, T4, T5, E1, E2, T9 | `workshop/08_portfolio_construction/` | Every portfolio built through `portfolio/construct.py` | Open |
| [W9](handoff/tasks/W9.md) | Trade management | Adv | W0, T1–T7, SM1, SR1, SR2, SS3, SS4, F3, E2, M2 | `workshop/09_trade_management/` | Notebook ends with a filled spec card section per team | Open |

## Platform

These live in the trading-platform repo. See [platform-tasks.md](platform-tasks.md).

| ID | Task | Level | Depends on |
|---|---|---|---|
| P1 | Candidate/feature log: every strategy candidate, taken or skipped | Int (TS) | |
| P2 | ONNX selector wrapper around any strategy, plus parity test | Adv (TS) | M3, P1 |
| P3 | Replay data path: filter, loader from candidates, fills and bars | Beg–Int (TS) | P1 |
| P4 | Replay decision-viewer UI | Int (frontend) | P3 |
| P5 | Read-only Models tab | Int (frontend) | M3 |
| P6 | Wire the vol-scaled and Kelly position sizers into the orchestrator | Int (TS) | T2 (evidence for the defaults) |
| P7 | Reversal strategy family (stretch and residual fade, σ exits) | Adv (TS) | SR1–SR3, T7 |
| P8 | Momentum ORB strategy | Adv (TS) | SM1, T7, P6 |
| P9 | Pairs v7: Kalman hedge, half-life max hold, stop rule, earnings blackout, spread-vol sizing | Adv (TS) | SS2–SS4, T7 |

## Research studies

Each study is team-owned and reports in the E1 format. Folders live in
[studies/](../studies), under `studies/<team>/<id>-<slug>/`. Moving the
existing folders there is part of the Wave 0 migration PR (a lead chore).
Existing R1–R5 keep their IDs; new team studies use `RM<n>` (momentum),
`RR<n>` (reversals) and `RS<n>` (stat arb), as listed in the team guides:
[momentum](teams/momentum.md), [reversals](teams/reversals.md),
[stat-arb](teams/stat-arb.md).

| ID | Study | Team | Builds on | Folder |
|---|---|---|---|---|
| R1 | ORB trade selectors A–D (Christopher's plan) | Momentum | F1, F4, F8, L2, M1, D4 | `studies/momentum/r1-orb-selectors/` |
| R2 | Reversal meta-labeling | Reversals | F2, F3, F5, L2 | `studies/reversals/r2-reversal-metalabel/` |
| R3 | Pair selection by clustering, then spread entry meta-labeling | Stat arb | F7, D4 | `studies/stat-arb/r3-pairs-clustering/` |
| R4 | Order-flow imbalance and short-horizon midpoint prediction | Reversals | D3, F6 | `studies/reversals/r4-microstructure-ofi/` |
| R5 | Transaction cost analysis: paper fills vs. backtest assumptions | Any | Platform broker ledger | `studies/shared/r5-tca/` |
