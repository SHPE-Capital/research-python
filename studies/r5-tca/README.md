# R5: Transaction cost analysis

**Owner:** open to any team.

**Question:** how far are the platform's actual paper fills from what the
backtest assumes (next-bar open ± half spread ± slippage bps)?

**Inputs:** fills from the platform's broker ledger (paper runs), the same
runs' backtests, and D3 quotes at each order's arrival time.

**Method:** implementation shortfall per order: fill price vs. the mid at
decision time, split into spread, delay and market impact. Compare against the
backtest's assumed costs by symbol, liquidity cell (D4) and time of day.

**Why it matters:** the club's cost assumptions (`slippageBps`,
`halfSpreadBps`) are currently guesses. Every study's cost-adjusted results
inherit them.

**TODO(R5):**
- [ ] Export fills and orders from the platform (coordinate with whoever owns P1).
- [ ] Shortfall decomposition and calibrated cost assumptions per liquidity cell.
