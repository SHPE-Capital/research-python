# 0005. Strategies plus a shared trade toolkit

- Status: Proposed
- Date: 2026-11-02
- Tasks: T1–T8, SM1–SM3, SR1–SR3, SS1–SS4

## Context

Three teams (momentum, reversals, stat arb) will each build strategy families.
Every family needs the same trade machinery: exits (stop, target, max hold,
signal exit), fills, sizing, entry filters, risk limits, costs and trade
diagnostics. The platform already implements this machinery once
(`core/execution/fillModel.ts`, the RiskEngine, the sizers). The L2 stub
(`labels/exit_policy.py`) started a second copy inside labels, with fill
defaults that differ from the platform's.

## Decision

We will split strategy code into two layers:

- `strategies/<team>/` (rank 5): team-owned `CandidateGenerator` families that
  decide when and which side. They own their signal logic and nothing else.
- `trade/` (rank 3): one shared toolkit used by every team. `trade/exits.py`
  (`ExitPolicy`), `trade/simulate.py` (`simulate_trades(...) -> TradeLedger`,
  using the platform `FillModel` rules), `trade/sizing.py` (`SizingRule`),
  `trade/filters.py`, `trade/risk.py` (`RiskLimits`, with names matching the
  platform RiskEngine in snake_case) and `trade/costs.py` (`CostModel`). Trade
  diagnostics go in `analysis/` (rank 5).

Each family is described by a `StrategySpec` (Pydantic, YAML in
`configs/strategies/`) that names its generator and its exit, sizing, filter,
risk and cost choices. `labels/triple_barrier` calls `simulate_trades`, so a
label and a backtest of the same spec agree by construction. T1 absorbs L2's
simulator and adopts the platform's fill defaults.

Details: [trade-layer.md](../design/trade-layer.md).

## Consequences

- One fill implementation to keep in parity with the platform, not one per
  team.
- Teams compare stops, holding periods and sizing with the same code, so
  cross-team results are comparable.
- `trade/` changes affect every team. They need review from the lead and a
  passing parity run.
- The independence contract (ADR 0002) stops a team from depending on another
  team's code; shared needs go to `trade/` or `analysis/`.

## Alternatives considered

- **Each team owns its full stack.** Fast at first, then three fill models,
  three sizing conventions and results that cannot be compared.
- **Exits and sizing inside each strategy class, as the platform strategies
  do.** Mirrors the platform, but prevents studying an exit rule across
  strategies, which is a main research question for every team.
- **Use an existing backtester (Backtrader, vectorbt, Zipline).** Their fill
  semantics differ from the platform's and would need to be overridden anyway;
  parity is easier with a small simulator that copies the platform rules.
