# Platform parity

A research result only matters if the trading platform would have traded the
same way. This page describes how a `StrategySpec` becomes a platform strategy
config, how research and platform backtests are reconciled, and how ONNX
meta-models are checked. Decision:
[ADR 0009](../adr/0009-research-platform-parity.md). Tasks:
[T7](../handoff/tasks/T7.md) (export and parity), [M3](../handoff/tasks/M3.md)
(ONNX export), P2 (ONNX selector, in [platform-tasks.md](../platform-tasks.md)).

The platform repo (`trading-platform`) is the reference. Research mirrors it;
it never asks the platform to mirror research. Where they differ, this page says
which side moves and under which task.

## What is compared

| Layer | Research side | Platform side |
|---|---|---|
| Decisions | `CandidateFrame` from the strategy's `CandidateGenerator` | `StrategySignal`s (P1 adds a candidate log of taken and skipped candidates) |
| Fills | `simulate_trades(...) -> TradeLedger` with the platform `FillModel` | `core/execution/fillModel.ts`, `BACKTEST_ENGINE_VERSION = 2` |
| Round trips | `TradeLedger` rows | `RoundTrip` in `core/analytics/tradeLedger.ts` (`symbol`, `direction`, `qty`, `entryTs`, `exitTs`, `entryPrice`, `exitPrice`, `pnl`, `commission`, `holdingMs`) |
| Risk | `RiskLimits` (`trade/risk.py`) | `RiskConfig` / `StrategyRiskBudget`, merged with `BACKTEST_RISK_CONFIG` |
| Model scores | `Model.predict_proba` in Python | ONNX model in onnxruntime-node (P2) |

## Export: `StrategySpec` to platform config

`export_platform_config(spec)` (T7, `deploy/platform_config.py`) takes a
validated `StrategySpec` and returns two JSON objects:

- a strategy config matching the platform's `BaseStrategyConfig` plus the
  family's own fields;
- a `BacktestConfig` fragment: `fillModel`, `slippageBps`, `commissionPerShare`,
  `riskConfig`, `dataGranularity: "bar"`, and the date range.

It refuses, with a message naming the field, any spec that uses something the
platform cannot run. Refusal is the point: a spec that exports is a spec the
platform can reproduce.

Research specs use snake_case and seconds or bar counts; the platform uses
camelCase and milliseconds. The tables below give the target names. The
`StrategySpec` field names are fixed in the T7 and S* packets; the platform
names are taken from the code.

### Shared fields (`BaseStrategyConfig`, `backend/src/types/strategy.ts`)

| Platform field | From the spec | Notes |
|---|---|---|
| `id` | not exported | Assigned by the platform |
| `name` | `spec.name` | |
| `type` | family, see below | `StrategyType` |
| `symbols` | `spec.universe.symbols` | Point-in-time list at export date |
| `rollingWindowMs` | family-specific window | |
| `maxPositionSizeUsd` | `spec.sizing.max_position_usd` | |
| `cooldownMs` | `spec.entry.cooldown` | |
| `enabled` | `true` | |
| `executionAlgo` | `"market"` | Only value research models |
| `sizerType` | `spec.sizing.rule` | Only `fixed_notional` until P6; others refused |
| `riskBudget` | `spec.risk` | `StrategyRiskBudget` names, see [trade-layer.md](trade-layer.md) |
| `sharpeConvention` | `"daily"` | Research reports daily Sharpe |

### Stat arb pairs (SS4) to `PairsStrategyConfig`, type `"pairs_trading"`

Source: `backend/src/strategies/pairs/pairsTypes.ts` and `pairsConfig.ts`.

| Platform field | Default | From the spec | Notes |
|---|---|---|---|
| `leg1Symbol`, `leg2Symbol` | — | `spec.pair.leg1`, `spec.pair.leg2` | `symbols` = both legs |
| `hedgeRatioMethod` | `"fixed"` | `spec.hedge.method` | `fixed` or `rolling_ols`. Kalman and TLS refused until P9 |
| `fixedHedgeRatio` | `1` | `spec.hedge.ratio` | |
| `olsWindowMs` | `14_400_000` | `spec.hedge.window` | |
| `olsRecalcIntervalBars` | `5` | `spec.hedge.recalc_every_bars` | |
| `rollingWindowMs` | `3_600_000` | `spec.zscore.window` | Spread mean and std window |
| `minObservations` | `30` | `spec.zscore.min_observations` | |
| `entryZScore` | `2` | `spec.entry.z` | Enter when \|z\| > threshold |
| `exitZScore` | `0.5` | `spec.exit.z` | `exit_reason = exit` |
| `stopLossZScore` | `4` | `spec.exit.stop_z` | `exit_reason = stop_loss`; signal-based, not intrabar |
| `maxHoldingTimeMs` | `86_400_000` | `spec.exit.max_hold` | `exit_reason = max_hold_exit`. Whole ms, at least `60_000`, at most one year (`validatePairsConfig`) |
| `cointSignificanceLevel` | `0.05` | `spec.filters.coint_alpha` | 0.01, 0.05 or 0.10. Ignored when the hedge is fixed |
| `cointHysteresis` | `0.15` | `spec.filters.coint_hysteresis` | Ignored when the hedge is fixed |
| `tradeNotionalUsd` | `5_000` | `spec.sizing.notional_per_leg` | Leg 2 gets notional × hedge ratio |
| `cooldownMs` | `60_000` | `spec.entry.cooldown` | |
| `priceSource` | `"mid"` | always `"mid"` | Ignored by the strategy since v5; bar close is used. Exported only so the config validates |

Not exportable until P9: Kalman or TLS hedge, max hold derived from OU
half-life, any other stop rule, earnings blackout, spread-volatility sizing.

### Reversals bar fade (SR1) to `MinuteReversalConfig`, type `"minute_reversal"`

Source: `backend/src/strategies/minuteReversal/`.

| Platform field | Default | From the spec | Notes |
|---|---|---|---|
| `symbols` | 10 large caps | `spec.universe.symbols` | |
| `qtyPerTrade` | `1` | `spec.sizing.qty` | Fixed shares |
| `maxPositionQty` | `1` | `spec.sizing.max_position_qty` | Caps \|position\| |
| `maxBarGapMs` | `180_000` | `spec.filters.max_bar_gap` | Skip when the gap to the previous bar is larger |
| `regularHoursOnly` | `true` | `spec.filters.regular_hours_only` | |
| `rollingWindowMs` | `60_000` | fixed | |
| `cooldownMs` | `0` | fixed | |
| `maxPositionSizeUsd` | `10_000` | `spec.sizing.max_position_usd` | |

The platform strategy has no exit rule: a position closes only when the next
opposite signal flips it. So an exportable SR1 spec has no stop, target or max
hold, and `session_end` is off. Stretch (SR2) and residual (SR3) fades are
refused until P7 adds a parameterised reversal family.

### Momentum

There is no exportable momentum family yet. The platform `momentum` type exists
in `StrategyType`, but `strategies/momentum/index.ts` is a stub. ORB (SM1)
exports after P8. EMA crossover (SM2) and cross-sectional momentum (SM3) are
research baselines; SM3 is daily and runs outside the intraday platform.

## Parity procedure

T7 implements this as `deploy/parity.py`, which returns a `ParityReport`. The
`platform-parity` agent ([.claude/agents/platform-parity.md](../../.claude/agents/platform-parity.md))
runs it.

1. **Export.** `export_platform_config(spec)` writes the strategy config and
   backtest fragment to `<run_dir>/export/`.
2. **Pick golden sessions** (below) and freeze their bars. Both sides read the
   same bars: the research snapshot exports them, or both read the same Alpaca
   range and the checksums are compared first.
3. **Run the platform backtest** with the exported config, `dataGranularity:
   "bar"`, the same date range, and `engineVersion` 2. Save its round trips,
   fills and signals as JSON.
4. **Run the research side**: candidates, filters, `simulate_trades` with the
   same `FillModel`, `CostModel` and `RiskLimits`.
5. **Match** trades by (`symbol`, `direction`, entry time). Compare:

| Check | Tolerance | Failure means |
|---|---|---|
| Set of entry decisions | exact | Strategy logic, filter or warm-up differs |
| Entry and exit timestamps | exact, at bar resolution | Decision timing or exit priority differs |
| `exit_reason` | exact, after mapping (see [trade-layer.md](trade-layer.md)) | Exit priority differs |
| Quantity | exact | Sizing or rounding differs |
| Fill prices | 1e-6 relative | Fill formula differs (spread, slippage, stop price) |
| Commission | 1e-6 absolute | Commission model differs |
| P&L per trade | 1e-6 relative to notional | Follows from the above |
| Session P&L | 1e-6 relative to capital | Follows from the above |

6. **Report.** `ParityReport` lists matched trades, research-only trades,
   platform-only trades and field mismatches, each with the first bar where the
   two sides diverged. Parity passes only with zero unmatched trades and no
   tolerance breach.

Tolerances are tight on purpose. Both sides run deterministic arithmetic on the
same bars, so any gap beyond floating-point noise is a logic difference. Do not
widen a tolerance to make a run pass; find the cause.

### Golden sessions

Each exportable family keeps at least five golden sessions, chosen to exercise
every code path:

| Session type | Exercises |
|---|---|
| Ordinary session | Entry, normal exit |
| Large gap at the open | Stops filling at the open, not at the stop |
| Early close (half day) | DAY expiry; the platform uses 16:00 NY with no early-close calendar, so research must match |
| Session with missing bars or a zero-volume bar | `maxBarGapMs`, zero-volume rejection |
| Session where risk limits bind | `RiskLimits` rejection order and reason codes |

For pairs, add a session where the cointegration gate flips. Golden sessions,
their bars and the expected `ParityReport` live in
`tests/integration/golden/<family>/` and run in CI, so a change on either side
that breaks parity fails a test. When the platform bumps
`BACKTEST_ENGINE_VERSION`, the goldens are regenerated and the change is
recorded in the family's spec card.

### Known differences to resolve before parity

| Difference | Resolution |
|---|---|
| L2 stub defaults (1 bp slippage, 2 bps spread, no commission, additive) differ from `DEFAULT_FILL_MODEL` (5 bps slippage, 2 bps half-spread, $0.005/share, multiplicative) | T1 adopts the platform values and formula |
| `backtest-execution.md` says a stop fills "at the stop"; `fillModel.ts` fills at max(open, stop) for a buy stop and then applies spread and slippage | Follow the code. Flagged for the platform docs |
| Research `session_end` exit has no platform counterpart | Off for every exported spec |
| Platform DAY orders expire 16:00 NY even on half days | Research uses the same rule for parity runs |
| Platform sizers other than fixed notional return 0 | Non-fixed sizing refused until P6 |

## ONNX meta-models (M3, P2)

A meta-model scores candidates; the platform takes a candidate only if the
score clears a threshold. The model must give the same score in both places.

**Export (M3).** `export_onnx(model, golden_rows, out_dir, tolerance=1e-5)` in
`models/export_onnx.py` writes:

| File | Contents |
|---|---|
| `model.onnx` | The model including its fitted scaler and any other fit-on-train processor, so the platform feeds raw features |
| `manifest.json` | Feature names in input order, their dtypes, the threshold, the label, the contract version, the `ModelManifest` id |
| `golden.parquet` | About 200 validation rows: feature values and the Python score for each |

The planned command is `shpe export <run_dir>/model`, which writes to
`<run_dir>/export/`. Export fails if onnxruntime in Python does not reproduce
the Python scores within the tolerance.

**Platform check (P2).** The selector wrapper loads `model.onnx` in
onnxruntime-node, runs the golden rows, and fails its parity test if any score
differs by more than 1e-5.

**Feature parity is the harder half.** Matching scores on golden rows proves the
model is ported; it does not prove the platform computes the same feature
values live. The P1 candidate log records the platform's feature values for
every candidate. Parity for a meta-model therefore has two checks:

1. Model: golden rows, within 1e-5.
2. Features: for the golden sessions, research feature values against the P1
   log, per feature, within 1e-6 relative. A feature the platform cannot
   compute is not allowed in an exportable model.

## Where parity sits in the lifecycle

Parity is the gate between "study accepted" and "paper trading" in
[strategy-lifecycle.md](../research/strategy-lifecycle.md). A strategy goes to
paper only with a passing `ParityReport` attached to its spec card. After paper
trading, R5 compares paper fills with backtest fills and recalibrates the
`CostModel` (T5).
