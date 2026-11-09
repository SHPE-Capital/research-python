# 0009. Research-to-platform parity

- Status: Proposed
- Date: 2026-11-02
- Tasks: T7, P1, P2, P5, P7, P8, P9, R5

## Context

Strategies that pass research here are meant to run on the trading platform
(`C:\Users\juanc\Local_Dev\trading-platform`, TypeScript). The platform has its
own strategy configs (`PairsStrategyConfig`, `MinuteReversalConfig`), its own
fill model, risk engine and backtest runner (`runtime/backtest.ts` scripts,
`BacktestConfig`, `RoundTrip` output). A research result that the platform
cannot reproduce is not usable: differences in fills, session handling or
parameter meaning would move live results away from what was studied. The
platform momentum strategy is a TODO stub, so not every research spec has a
platform counterpart today.

## Decision

The platform is the reference implementation. Research matches it, not the
other way round.

- `export_platform_config(spec: StrategySpec)` (T7, in `deploy/`) returns a
  platform strategy config and a `BacktestConfig` fragment. It refuses, with a
  reason, any spec that uses a feature the platform does not implement yet
  (for example Kalman hedge ratios before P9, SR2/SR3 before P7, any momentum
  spec before P8). It never drops a field silently.
- A `ParityReport` compares research `TradeLedger` rows with platform
  `RoundTrip` rows for the same sessions. Decisions, timestamps, exit reason
  and quantity must match exactly; prices, P&L and commission within 1e-6.
- Golden sessions (ordinary day, gap open, early close, missing or
  zero-volume bar, binding risk limit, and a cointegration-gate flip for pairs)
  are stored under `tests/integration/golden/<family>/` and regenerated when
  the platform `BACKTEST_ENGINE_VERSION` changes.
- ONNX models (M3, P2) are checked against Python predictions at 1e-5; feature
  values against the platform candidate log (P1) at 1e-6.
- A strategy goes to paper trading only after its parity report passes. R5
  uses paper fills to recalibrate the cost model (T5).

Details: [platform-parity.md](../design/platform-parity.md).

## Consequences

- Research fill and session rules copy the platform, including behaviour that
  may look odd (stops fill at `max(open, stop)` before spread and slippage;
  `minuteReversal` and pairs never flatten at the close, so research exits for
  exported specs run with `session_end` off).
- Each new research capability needs either a platform task or a refusal in
  the exporter. The refusal list doubles as the platform backlog.
- Golden files must be regenerated and reviewed when the platform engine
  changes; CI fails until they are.
- Specs that research only (no export) remain allowed; they are marked
  non-exportable in reports.

## Alternatives considered

- **Research as the reference, platform ports it.** Puts the burden on the
  platform team for every research change and loses the tested live code path.
- **Call the platform backtester from Python for every study.** Exact by
  construction, but slow, needs Node in every research environment and blocks
  fast iteration on labels and exits.
- **Compare summary metrics only (Sharpe, trade count).** Hides offsetting
  errors; trade-level comparison catches them.
