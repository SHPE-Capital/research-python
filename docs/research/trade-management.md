# Trade management

How to decide what happens after an entry signal: how long to hold, where to
stop, when to take profit, how much to trade, when not to trade, and what
limits apply. The entry signal is often the smaller part of a strategy's
result; these choices decide whether a weak edge survives costs.

The tools are in the shared trade toolkit:

| Decision | Library | Task |
|---|---|---|
| Exits and simulation | `trade/types.py` (`ExitPolicy`), `trade/exits.py`, `trade/simulate.py` (`simulate_trades(...) -> TradeLedger`) | T1 |
| Sizing | `trade/sizing.py` (`SizingRule`) | T2 |
| Filters | `trade/filters.py` | T3 |
| Risk limits | `trade/risk.py` (`RiskLimits`) | T4 |
| Costs | `trade/costs.py` (`CostModel`, `FillModel`) | T5 |
| Diagnostics | `analysis/`: horizon curve, OU half-life, MFE/MAE, stop curves, parameter surfaces | T6 |

A `TradeLedger` has one row per trade with columns `entry_ts`, `exit_ts`,
`side`, `entry_px`, `exit_px`, `qty`, `exit_reason`, `gross_pnl`, `costs`,
`net_pnl`, `mfe`, `mae`, `r_multiple`.

The step-by-step workflow is in the
[`trade-management-analyst`](../../.claude/agents/trade-management-analyst.md)
agent. Every choice here is a trial and follows the
[research protocol](research-protocol.md#3-trial-counting).

## Order of decisions

Decide in this order. Each decision changes the trades the next one sees.

1. Cost hurdle: is there anything to manage?
2. Holding period
3. Stops
4. Targets and trailing exits
5. Filters and cooldowns
6. Sizing
7. Risk limits

At every step, keep the **baseline**: no filter, no stop, no target, fixed size,
time exit only. Report every variant next to it.

## Costs and break-even

### Platform fill defaults

Research must cost trades the way the platform fills them. The platform's
`DEFAULT_FILL_MODEL` (`trading-platform/backend/src/core/execution/fillModel.ts`):

| Setting | Default | Meaning |
|---|---|---|
| `referencePrice` | `next_open` | Market orders fill at the open of the next bar after placement |
| `halfSpreadBps` | 2 | Buys pay the reference plus half spread, sells receive it minus half spread |
| `slippageBps` | 5 | A further adverse shift on every fill |
| `volumeParticipationCap` | 0.1 | At most 10% of a bar's volume fills |
| `allowPartialFills` | true | The rest of the order stays open |
| `commissionPerShare` | 0.005 | Per share, per fill |

Bar fill rules that matter for exits:

- A stop triggers when the bar's range reaches it. It fills from the stop
  price, or from the open if the bar gapped through the stop, and then pays the
  same spread and slippage as a market order.
- Limit orders (targets) fill only if the price after spread and slippage still crosses the limit.
- When a stop and a target are both touched in the same bar, the stop wins.
- Orders only fill against a bar that ends after the order was placed.

### Round-trip cost

Per side, spread plus slippage is 2 + 5 = 7 bps. A round trip is **14 bps**,
plus $0.01 per share of commission. Commission in bps is `100 × 0.01 / price`:
5 bps at $20, 1 bp at $100.

| Price | Spread + slippage | Commission | Round trip |
|---|---|---|---|
| $20 | 14 bps | 5 bps | 19 bps |
| $50 | 14 bps | 2 bps | 16 bps |
| $100 | 14 bps | 1 bp | 15 bps |
| $300 | 14 bps | 0.3 bp | 14.3 bps |

A pairs trade has two legs and pays this on both.

The research `ExitPolicy` stub in `labels/exit_policy.py` currently defaults
to `half_spread_bps=1.0`, `slippage_bps=2.0`, `commission_per_share=0.0`, which
is about 6 bps per round trip. That is cheaper than the platform. Until T5
calibrates costs from R5, use the platform defaults above in every study and
show 2× costs as a stress case.

### Break-even

- **Edge per trade:** mean gross return per trade must exceed the round-trip cost.
  Plot mean gross bps by signal-strength bucket against the cost line; buckets
  below it are not tradable whatever the exits.
- **Break-even hit rate:** with average win W, average loss L (both positive, in
  bps, before costs) and round-trip cost c, `p* = (L + c) / (W + L)`.
  For W = L = 20 bps and c = 16 bps, p* = 0.90 – a fade that wins 60% of the
  time with 20 bp moves loses money.
- **Turnover:** for signals traded every bar, net return ≈ gross − cost × trades.
  The E1 `cost_adjusted` table shows the net at 0, 2, 5 and 10 bps per side.

## Holding period

### Horizon curve

For all candidates, compute the mean (and median) forward return from entry to
each minute after entry, out to the session close. Plot net of the round-trip
cost (T6).

- **Momentum:** the curve rises, then flattens. Exit where it flattens.
- **Mean reversion:** the curve rises fast, then fades as the reversion completes
  or the stretch keeps going. Exit near the turn.
- Choose a **plateau**, not the single best minute.

### IC decay

E1's `ic_decay` gives the information coefficient of the signal against
returns at several horizons. Where the IC halves is a natural first guess for
the time exit.

### OU half-life (mean-reverting signals)

Model the deviation `x` (spread, z-score, distance from VWAP) as
an Ornstein–Uhlenbeck process. Fit an AR(1) on bars:

```
x_t = c + φ · x_{t−1} + ε_t
hl  = −ln 2 / ln φ          (in bars; 0 < φ < 1)
```

Equivalently, regress `Δx_t` on `x_{t−1}` to get λ = φ − 1 and
`hl = −ln 2 / ln(1 + λ)`. SS3 implements this.

- φ ≥ 1 means no mean reversion in the window; do not trade the pair or signal.
- Use a time exit of 1–3 half-lives. Longer holds mostly add risk from the
  reversion failing.
- For the platform `PairsStrategy`, `maxHoldingTimeMs` (default 24 h) can be set
  from the half-life once P9 adds it as a rule.
- Re-estimate on a rolling window; a half-life that drifts upward is a warning.

## Stops

### Units

Set stops in units that scale with the symbol and the day:

| Unit | Formula | Use |
|---|---|---|
| ATR multiple | `k × ATR(n)` on the bar frequency traded | General |
| Horizon volatility | `k × σ_bar × sqrt(h)` for an h-bar hold | Ties stop width to holding period |
| Structural | Opening-range midpoint, prior low, z-score level | When the structure defines the trade (ORB, pairs) |

Fixed dollar or fixed bps stops behave differently on every symbol and day.

### MAE analysis

Maximum adverse excursion (Sweeney 1997) is the worst point a trade reached
before it closed. The `TradeLedger` records it as `mae`.

1. Plot MAE for trades that ended as winners and for losers (with no stop).
2. Winners usually cluster at small MAE. A stop just beyond where most winners
   turn around cuts losers without cutting many winners.
3. If winners and losers have the same MAE distribution, a stop cannot separate
   them and will only add costs.

### Compare against no stop

Always run "no stop" (time exit only). A stop helps when it raises the net
return or cuts tail loss enough to pay for the trades it turns into losses.
Report: mean net bps, hit rate, worst trade, worst day, max drawdown and 5% CVaR,
for each stop width and for no stop.

### Why mean-reversion stops differ

A momentum trade that goes against you is evidence the trade was wrong. A
mean-reversion trade that goes against you is, at first, a better entry: the
deviation got bigger. Tight stops on reversion trades sell at the worst point
and usually lower the mean return.

- For reversion, use **disaster stops**: wide (for example 3–4σ of the deviation,
  or the platform pairs default `stopLossZScore: 4` against `entryZScore: 2`),
  meant to cap tails and regime breaks, not to improve the average.
- Judge reversion stops on tail loss and drawdown, not on mean return.
- Pair them with a time exit from the half-life. Time is often the better stop.

## Targets and trailing exits

### R-multiples

R is the initial risk: entry price minus stop price. The ledger's `r_multiple`
is net P&L divided by R. Targets are set as multiples of R (2R, 3R), so they
scale with the stop.

### MFE analysis

Maximum favourable excursion (`mfe`) is the best point a trade reached. Compare
MFE with the final result:

- Trades that reach +2R and close at a loss suggest a target or a trailing stop.
- If most of the return comes from a few trades with very large MFE, a fixed
  target cuts the right tail. This is typical of momentum.

### Exit variants

| Exit | Rule | Typical fit |
|---|---|---|
| Time only | Exit after h bars | Baseline for all |
| Fixed target | Exit at +kR | Reversion, where the move has a natural end |
| Trailing (chandelier) | Stop at highest high since entry − k × ATR (longs) | Momentum, to keep the tail |
| Breakeven | Move stop to entry after +1R | Cuts losers that were winners; costs some winners |
| Signal exit | Exit when the signal reverses or the deviation returns (pairs `exitZScore: 0.5`) | Reversion and pairs |

Test each against time-only. Remember the same-bar rule: if the stop and target
are both in one bar's range, the stop fills.

## Filters, regime and cooldowns

Filters remove candidates before sizing (T3). Add one at a time; each is a trial.

| Filter | Examples |
|---|---|
| Time of day | No entries in the first 5 minutes or last 15 minutes; ORB entries only until 11:00 |
| Liquidity and spread | Minimum dollar volume; maximum quoted spread (from D3 tick summaries) |
| Regime | Volatility regime (realised vol percentile), SPY trend state, variance ratio or efficiency ratio of the symbol |
| Events | Earnings blackout, index rebalance days, half days, FOMC afternoons |
| Data quality | Bar gap longer than allowed (platform `minuteReversal` uses `maxBarGapMs` of 3 min); stale quotes |

**Cooldowns.** After an exit or a stop-out, a symbol may be blocked for a
period. Measure the return of the second entry on the same symbol and day
against first entries. If second entries are worse, add a cooldown. The
platform has a global `orderCooldownMs` (per strategy and symbol) and
`PairsStrategy` has `cooldownMs` (default 60 s). Note that the backtest risk
config sets `orderCooldownMs: 0`.

## Sizing

Compare sizing rules in this order (T2). Each is a trial.

| Step | Rule | Formula | Notes |
|---|---|---|---|
| 1 | Fixed notional | `qty = floor(notional / price)` | What the platform does today (`fixedNotionalSizer` with `maxPositionSizeUsd`) |
| 2 | Risk per trade | `qty = equity × risk% / (stop_distance × price)`, stop distance as a fraction of price | Every trade risks the same amount if stopped. Needs a stop |
| 3 | Inverse volatility | `qty ∝ target_vol / (σ × price)` | Equal risk contribution without a stop |
| 4 | Probability → size | `z = (p − 0.5) / sqrt(p(1 − p))`, `size = 2Φ(z) − 1` (López de Prado 2018, ch. 10) | From a meta-label probability p. Size 0 at p = 0.5 |
| 5 | Fractional Kelly | Binary: `f* = p − (1 − p)/b` (b = win/loss ratio). Continuous: `f* = μ / σ²`. Use ¼ to ½ of f* | Kelly maximises long-run growth only if p, b, μ, σ are known. They are estimated, so full Kelly over-bets. Correlated positions make it worse |

**Bunched signals.** Intraday signals arrive together (the open, macro news,
sector moves). Positions opened at the same time are correlated, so the
portfolio risks more than the sum of per-trade risks suggests. Options:

- cap the number of concurrent positions
- scale each new position by `1/sqrt(n_open)`
- size on the portfolio's total risk rather than per trade

Report P&L by the number of simultaneous open positions to see if bunching hurts.

The platform has `volatilityScaledSizer` and `kellyFractionalSizer` as scaffolds
that return 0, and none of the sizers is connected to the orchestrator yet.
Task P6 connects them. Until then, anything beyond fixed notional needs a
platform task before it can trade.

## Risk limits

`RiskLimits` (T4) uses the platform's `RiskConfig` names so the research
simulation rejects the same orders the platform would. The platform's
`RiskEngine` checks each order in this order and rejects at the first failure:

| Order | Check (rejection code) | Config field | Default |
|---|---|---|---|
| 1 | `KILL_SWITCH` | `killSwitchActive` | false |
| 2 | `MARKET_CLOSED` | market hours | – |
| 3 | `STALE_QUOTE` | `staleQuoteThresholdMs` | 10,000 |
| 4 | `ORDER_COOLDOWN` (per strategy and symbol) | `orderCooldownMs` | env; 0 in backtests |
| 5 | `MAX_POSITION_SIZE` | `maxPositionSizeUsd` | env |
| 6 | `MAX_NOTIONAL_EXPOSURE` | `maxNotionalExposureUsd` | env |
| 7 | `SHORT_SELLING_DISALLOWED` | `allowShortSelling` | true |
| 8 | `CASH_RESERVE` | `cashReservePct` | 0.05 |
| 9 | `INTRADAY_DRAWDOWN` | `maxIntradayDrawdownPct` | 0.05 |
| 10 | `CONCENTRATION_LIMIT` | `maxConcentrationPct` | 0.30 |

Also in the platform:

- Strategy budgets: `maxOrderNotionalPct`, with rejections `STRATEGY_BUDGET`,
  `STRATEGY_ORDER_NOTIONAL` and `MAX_OPEN_ORDERS`.
- Portfolio checks: optional `maxGrossExposurePct` and `maxNetExposurePct`, plus
  the intraday drawdown. A breach can trigger the kill switch and flatten positions.
- Buffers: `gapBufferBps` (20) and `spreadBufferBps` (5).

In a study, report how often each limit binds and how much P&L it changed. A
limit that binds often is part of the strategy and belongs in the spec card.

## Honest tuning

- Tune on validation, dev symbols only. Open test and held-out symbols once.
- Count every grid point (stop widths × targets × holds × sizers × filters)
  in the trial count passed to `deflated_sharpe`.
- Show the full parameter surface and choose the centre of a plateau.
- Use volatility, half-life or structural units so parameters transfer.
- Use platform fill rules and costs; never cheaper unless R5 calibrates them.
- Report the baseline (no stop, no target, fixed size) next to the final choice.

## Typical behaviour by family

These are starting hypotheses to test, not results. Each team guide in
[docs/teams/](../teams/) turns them into studies.

| Choice | Momentum (ORB, EMA cross, 12-1) | Reversals (bar fade, stretch, residual) | Stat arb (pairs) |
|---|---|---|---|
| Holding period | Until the horizon curve flattens; intraday ORB up to the session or a 30–90 min time exit; daily 12-1 holds about a month | Short: minutes to the turn of the horizon curve | 1–3 OU half-lives; cap by `maxHoldingTimeMs` |
| Stop | Structural (range midpoint, R1's default) or 1–2 ATR; expected to help | Wide disaster stop only; tight stops expected to hurt the mean | Z-score disaster stop (default 4 vs. entry 2) plus a time stop |
| Target | Usually none; trailing stop to keep the right tail | Return to the anchor (VWAP, EMA, previous close) or a fixed R | Signal exit when z falls below `exitZScore` |
| Sizing | Risk per trade from the stop; inverse vol | Inverse vol; bunching control at the open | Spread-volatility sizing; equal dollar or hedge-ratio legs |
| Filters | Trend or regime filter; avoid the first minutes; volume confirmation | Avoid news and earnings; spread filter; avoid strong trend days | Cointegration still holds (`cointSignificanceLevel` with hysteresis); earnings blackout |
| Cost sensitivity | Moderate: fewer, larger moves | High: many small moves; costs often decide the result | High: two legs, four fills per trade |
| Main risk | Whipsaw in range-bound markets | Trend days; news gaps through the stop | Relationship break; one leg gaps |

## References

- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 3 (triple-barrier labels, meta-labeling) and 10 (bet sizing).
- Sweeney, J. (1997). *Maximum Adverse Excursion: Analyzing Price Fluctuations for Trading Management*. Wiley.
- Gatev, E., Goetzmann, W. N. and Rouwenhorst, K. G. (2006). Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3).
- Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1).
