# Trade layer: strategies, trade and analysis

This page covers the three packages that turn a signal into trades:
`strategies/` (what to trade and when), `trade/` (how a trade is filled, sized,
filtered, limited, costed and exited) and `analysis/` (what the trades tell
you). It explains why labels reuse the trade simulator, and how each piece
mirrors the trading platform so research results can be checked against
platform backtests.

Decisions: [ADR 0004](../adr/0004-events-candidates-unit-of-decision.md),
[ADR 0005](../adr/0005-strategies-and-shared-trade-toolkit.md),
[ADR 0009](../adr/0009-research-platform-parity.md). Task packets with the
exact signatures: [T1](../handoff/tasks/T1.md) through
[T8](../handoff/tasks/T8.md). For the research method that uses these tools,
see [research/trade-management.md](../research/trade-management.md).

## Responsibilities

| Package | Owns | Does not own |
|---|---|---|
| `strategies/` | `StrategySpec` (Pydantic, loaded from `configs/strategies/*.yaml`), `@register_strategy`, one `CandidateGenerator` per family, family-specific estimators (hedge ratios, OU fits, stretch measures) | Fills, sizing, risk, costs: those come from `trade/` |
| `trade/` | `ExitPolicy`, `FillModel`, `simulate_trades(...) -> TradeLedger`, `SizingRule`, entry filters, `RiskLimits`, `CostModel` | Which symbols or times to trade |
| `analysis/` | Diagnostics over a `TradeLedger` or a `CandidateFrame`: holding-period curve, OU half-life, MFE/MAE, stop and target curves, parameter surfaces | Pass/fail decisions; those belong to the study and its pre-registration |

Team code lives in `strategies/momentum/`, `strategies/reversal/` and
`strategies/statarb/`. These three are independent (import-linter contract in
[architecture.md](architecture.md#layers-and-import-rules)). Anything two teams
need moves down into `trade/`, `features/` or `analysis/`.

## Data flow

```
StrategySpec ──► CandidateGenerator(bars, features) ──► CandidateFrame
                                                          │
                     optional meta-model score ──► trade/filters (take / skip)
                                                          │
                                         trade/sizing (SizingRule) ──► qty
                                                          │
                                         trade/risk (RiskLimits) ──► accept / reject + reason
                                                          │
         bars + ExitPolicy + FillModel + CostModel ──► trade/simulate.simulate_trades
                                                          │
                                                     TradeLedger ──► analysis/, evaluation/, deploy/parity
```

### CandidateFrame

One row per decision a strategy proposes. Required columns: `symbol`,
`decision_ts`, `side` (`long` / `short`), `strategy_id`. Optional columns carry
what the generator knew at `decision_ts` (trigger value, z-score, stretch,
counterpart symbol and hedge ratio for pairs, a stop or target level). The
column schema lives in `contracts/candidates.py`, so `trade/` can use it
without importing `events/`; `events/types.py` re-exports it next to the
`CandidateGenerator` protocol (C1a stub, filled by C1b). A candidate is not a trade:
filters, sizing and risk can still skip it, and every skip is recorded with a
reason so a study can count them.

### TradeLedger

One row per trade (a pair is one trade with two legs). Columns include
`symbol`, `strategy_id`, `side`, `decision_ts`, `entry_ts`, `entry_price`,
`qty`, `exit_ts`, `exit_price`, `exit_reason`, `gross_return`, `cost`,
`net_return`, `label_end_ts`, and `mfe` / `mae` when requested. The column
schema lives in `trade/types.py` (C1a stub, filled by T1). `label_end_ts` equals `exit_ts` so the
ledger can feed purged validation directly.

## Why labels reuse the simulator

`labels/triple_barrier` (C1c) is a thin wrapper: it calls
`trade/simulate.simulate_trades` on the candidates and returns `net_return`,
`exit_reason` and `label_end_ts` as the label. There is exactly one
implementation of "what happens after you enter".

Reasons:

1. **Labels and backtests cannot drift.** If the label simulator and the
   backtest simulator differed (say, in how a gap through a stop fills), a
   meta-model would be trained on outcomes the strategy never earns.
2. **Platform parity is checked once.** T1's reconciliation against the platform
   on golden sessions covers both labels and backtests.
3. **Stops, targets and costs are label parameters.** Changing the stop
   distance in a meta-labeling study changes the label, as it should. The
   `ExitPolicy` is part of the label config and is recorded in the run manifest.

This is why `trade/` sits at rank 3 in the layer table, below `labels/`.
`labels/exit_policy.py` (L2) is absorbed by T1: `ExitPolicy` moves to
`trade/exits.py` and `simulate_exits` becomes `simulate_trades` in
`trade/simulate.py`. The L2 acceptance test (`test_exit_policy`) moves with it.

## Fill semantics (mirrors the platform bar model)

Source: `trading-platform/docs/architecture/backtest-execution.md` and
`trading-platform/backend/src/core/execution/fillModel.ts`. The platform stamps
these semantics as `BACKTEST_ENGINE_VERSION` (currently 2). `FillModel` in
research records the engine version it mirrors, and the parity check refuses to
compare against a platform result with a different version.

| Rule | Platform behaviour | Research `FillModel` |
|---|---|---|
| Decision time | Each bar batch is processed at its end (`barEndMs`); the 10:00 bar is usable at 10:01 | `decision_ts` = the bar's `ts_available` |
| Market order price | Reference price (default `next_open`) × (1 + `halfSpreadBps`/1e4) × (1 + `slippageBps`/1e4) for buys; × (1 − …) × (1 − …) for sells | Same formula, multiplicative, same parameter names in snake_case |
| Which bar fills | Only a bar that ends after the order was placed | The first bar with `ts >= decision_ts`; a missing minute means the next bar that exists |
| Zero-volume bar | Rejected: no liquidity | Rejected; ledger row gets `exit_reason = unfilled` for an entry |
| Volume cap | `volumeParticipationCap` (default 0.1) of bar volume; partial fills on by default | Same; partial fills reduce `qty` and are flagged |
| Stop order | Rests until a regular-session bar's range reaches the stop; triggers at the stop, or at the bar open if the bar gapped through; then fills as a market order from that trigger price with spread and slippage | Same; the stop is checked on bars after the entry fill, regular session only |
| Limit order | Synthetic touch check against the next open; rejected if not marketable | Same; resting limits are not modelled in the bar model |
| Stop and target in the same bar | No intrabar path in minute bars | Assume the stop fills first (conservative) |
| DAY orders | Expire 16:00 New York; no holiday or early-close calendar | Mirror for parity: an early-close session has no bars after 13:00, so a DAY stop simply stops being checked; flag such trades |
| Commission | `commissionPerShare` (default 0.005), accounting only | `CostModel` adds it to `cost` |

Platform defaults in `DEFAULT_FILL_MODEL` are `slippageBps: 5`,
`halfSpreadBps: 2`, `commissionPerShare: 0.005`,
`volumeParticipationCap: 0.1`, `allowPartialFills: true`. The current
`labels/exit_policy.py` stub defaults (`half_spread_bps=1.0`,
`slippage_bps=2.0`, `commission_per_share=0.0`) and its additive wording do not
match. T1 adopts the platform defaults and the multiplicative formula.

The platform also has a quote model (latency 250 ms by default, 5 s quote
lookback, 5 s maximum wait). Research does not mirror it; the bar model is the
parity target. R5 (`studies/shared/r5-tca/`) measures how far quote-model and
paper fills sit from bar-model fills, and T5 turns that into a `CostModel`.

### Signal exits vs. order exits

There are two kinds of exit and they fill differently:

- **Order exits** are resting orders: a price stop or a price target. They are
  checked inside each later bar and fill at the trigger price (or the gap open).
- **Signal exits** are decisions made at a bar's end from strategy state: a
  z-score stop, a reversion exit, a max-hold timer, an opposite signal. They
  fill at the next bar's open like any market order.

The platform pairs strategy uses only signal exits: its `stopLossZScore` is a
z-score test at evaluation time, not a resting stop order. `ExitPolicy` keeps
the two kinds separate so a pairs replication does not accidentally fill a
z-score stop intrabar.

## exit_reason vocabulary

The platform pairs strategy labels exits with `meta.signalType` in
`exit | stop_loss | max_hold_exit` and checks them in that priority order:
stop loss, then max hold, then normal exit (`pairsStrategy.ts`,
`_checkExitSignals`). The trigger label is `z_score_<signalType>`. The minute
reversal strategy has no exit labels: it flips or flattens on the next opposite
bar.

Research uses one vocabulary that is a superset of the platform's:

| `exit_reason` | Meaning | Platform equivalent | Old L2 name |
|---|---|---|---|
| `stop_loss` | Price stop filled, or a signal stop (z-score, adverse move) fired | `stop_loss` | `stop` |
| `take_profit` | Price target filled | none (platform has no price target yet) | `target` |
| `exit` | Strategy's normal exit signal (reversion to mean, opposite signal) | `exit` | — |
| `max_hold_exit` | Holding-time limit reached | `max_hold_exit` | `time` |
| `session_end` | Flattened at the session close | none; see below | `session_end` |
| `unfilled` | Entry never filled (zero volume, limit not marketable, no next bar) | order rejected | `unfilled` |

When two exits fire at the same decision time, use the platform's priority:
`stop_loss`, then `max_hold_exit`, then `exit`. `session_end` is research-only.
Neither platform strategy above flattens at the close by itself, so for a parity
run the `ExitPolicy` must have the session-end flatten turned off unless the
platform strategy being replicated does it.

## Sizing

Platform source: `backend/src/core/sizing/` and
`docs/architecture/position-sizing-layer.md`.

- The platform interface is `IPositionSizer.computeQty(PositionSizerParams)`,
  where the params carry `symbol`, `direction`, `estimatedPrice`, `portfolio`,
  `symbolState` and `strategyConfig`. It returns a non-negative integer.
- `SizerType` is `fixed_notional | volatility_scaled | kelly_fractional | equal_weight`.
- `fixedNotionalSizer` returns `floor(maxPositionSizeUsd / estimatedPrice)`.
- `volatilityScaledSizer` and `kellyFractionalSizer` are scaffolds that return 0.
- **The sizers are not wired into the orchestrator yet (P6).** Today each
  strategy emits `qty` itself: the pairs strategy from `tradeNotionalUsd`, the
  minute reversal strategy from `qtyPerTrade`.

Research `SizingRule` (T2) implements fixed notional, fixed quantity,
risk-per-trade (stop distance), inverse volatility, probability to size, and
fractional Kelly. Every rule returns whole shares, floored, like the platform.
Only `fixed_notional` and the strategy-native rules (`qtyPerTrade`,
`tradeNotionalUsd`) are exportable today. The others are research results that
justify P6; `export_platform_config` refuses to export them until P6 lands
(see [platform-parity.md](platform-parity.md)).

## Risk

Platform source: `backend/src/core/risk/riskEngine.ts`, `backend/src/types/risk.ts`,
`docs/architecture/risk-layer.md`.

`RiskLimits` (T4) uses the snake_case form of the platform's `RiskConfig` and
`StrategyRiskBudget` names, so a limit means the same thing in both repos:

| Platform | Research `RiskLimits` |
|---|---|
| `maxPositionSizeUsd` | `max_position_size_usd` |
| `maxNotionalExposureUsd` | `max_notional_exposure_usd` |
| `orderCooldownMs` | `order_cooldown_ms` |
| `staleQuoteThresholdMs` | `stale_quote_threshold_ms` |
| `allowShortSelling` | `allow_short_selling` |
| `maxIntradayDrawdownPct` | `max_intraday_drawdown_pct` |
| `maxConcentrationPct` | `max_concentration_pct` |
| `cashReservePct` | `cash_reserve_pct` |
| `maxGrossExposurePct` | `max_gross_exposure_pct` |
| `maxNetExposurePct` | `max_net_exposure_pct` |
| `StrategyRiskBudget.maxCapitalPct` | `strategy_max_capital_pct` |
| `StrategyRiskBudget.maxOrderNotionalPct` | `strategy_max_order_notional_pct` |
| `StrategyRiskBudget.maxOpenOrders` | `strategy_max_open_orders` |

Checks run in the platform's order and reject with the platform's reason codes:
`KILL_SWITCH`, `MARKET_CLOSED`, `STALE_QUOTE`, `ORDER_COOLDOWN`,
`MAX_POSITION_SIZE`, `MAX_NOTIONAL_EXPOSURE`, `SHORT_SELLING_DISALLOWED`,
`CASH_RESERVE`, `INTRADAY_DRAWDOWN`, `CONCENTRATION_LIMIT`, then the strategy
budget checks `STRATEGY_BUDGET`, `STRATEGY_ORDER_NOTIONAL`, `MAX_OPEN_ORDERS`.
The first failing check wins. `KILL_SWITCH` and `STALE_QUOTE` have no meaning on
historical bars and always pass in research; they stay in the list so reason
codes line up. Portfolio-level checks (gross, net, drawdown) run per bar like
the platform's `checkPortfolio()`.

## Filters and costs

- **Filters (T3)** take a `CandidateFrame` plus feature columns and return a
  boolean keep mask and a skip reason per row: time-of-day windows, liquidity
  and spread floors, regime gates, event blackouts (earnings, FOMC), and a
  meta-model score threshold. Filters read precomputed columns; they never
  compute features, which keeps `trade/` below `features/` in the layer order.
- **Costs (T5)**: `CostModel` turns a fill into `cost` in currency and basis
  points: half-spread, slippage, commission, and optionally a size-dependent
  impact term. The default reproduces the platform bar-model defaults. A
  calibrated model from R5 replaces it for any result a team presents.

## Analysis (T6)

`analysis/` reads ledgers and candidates and produces tables and figures.
It never changes a strategy. Planned diagnostics:

| Diagnostic | Question it answers | Typical team |
|---|---|---|
| Holding-period curve | Mean net return by exit horizon, with block-bootstrap bands | All |
| OU half-life | How fast a spread or stretch reverts; sets a max hold | Stat arb, reversals |
| MFE / MAE | How far trades go for and against you before exit; places stops and targets | All |
| Stop and target curves | Net return and hit rate across a grid of stop and target distances | Momentum, reversals |
| Parameter surface | Two-parameter grid with the trial count recorded | All |

Every grid search in `analysis/` reports how many configurations it tried, so
the study can apply the deflated Sharpe ratio (E2). See
[research/research-protocol.md](../research/research-protocol.md).
