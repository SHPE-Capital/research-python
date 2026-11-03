# Session 5: From signal to strategy

| | |
|---|---|
| Level | Int |
| Duration | 90 min |
| Prerequisites | Session 4 |
| Data | Fixture; the snapshot for the R1 demonstration |
| Code used | `labels/exit_policy.py` (L2, absorbed by T1), `features/session/opening_range.py` (F4), `studies/momentum/r1-orb-selectors/` |
| Ends with a PR for | T1 (exit simulator), one of T2–T6, or an S-task for your team (SM1, SR1, SS3, …) |

## Why this session

A forward return at a fixed horizon is not a trade. A trade has an entry
price you might not get, a stop, a target, a time limit, a session close, a
size and costs. Many strategies that rank well on forward returns lose money
once these are added. Many good strategies come from improving **which trades
to take** rather than predicting prices.

## Learning objectives

1. Explain the difference between a signal, a candidate and a trade.
2. Describe an exit policy (stop, target, time, session end) and how bar
   data fills it, including when a stop and a target are both touched in one
   bar.
3. Explain meta-labeling: a rule proposes trades; a model decides which to
   take and how large.
4. Name the trade-management decisions every strategy needs (holding
   period, stops, targets, sizing, filters, risk limits, costs) and where
   each lives in the repo.

## Background

### Signal, candidate, trade

| Term | Example | Repo |
|---|---|---|
| Signal | Score from a model, or "close above the opening-range high" | `features/`, `models/` |
| Candidate | One concrete decision point: symbol, time, side, rule that proposed it | `events/` (C1b), `strategies/` (SM1, SR1, SS4) |
| Trade | A candidate that was taken, with fills, exit reason and net return | `trade/simulate.py` (T1) |

### Exit policies as labels

`labels/exit_policy.py` (L2, being absorbed into `trade/exits.py` and
`trade/simulate.py` by T1) defines:

```python
ExitPolicy(max_minutes=30, stop_distance=..., target_distance=...,
           side="long", half_spread_bps=1.0, slippage_bps=2.0,
           commission_per_share=0.0)
```

The cost defaults shown are the current stub's and are too cheap (about 6 bps
per round trip). T1 replaces them with the platform `DEFAULT_FILL_MODEL`
values: `half_spread_bps=2.0`, `slippage_bps=5.0`, `commission_per_share=0.005`
(about 14 bps plus commission). Use those in exercises.

`simulate_exits` returns, per candidate, `entry_ts`, `entry_price`,
`exit_ts`, `exit_price`, `exit_reason` (`stop`, `target`, `time`,
`session_end`, `unfilled`) and `net_return` after costs, plus `label_end_ts`.

Fill rules must match the trading platform's backtester, or research and the
platform will disagree. The packet for T1
([handoff/tasks/T1.md](../../handoff/tasks/T1.md)) and
[design/trade-layer.md](../../design/trade-layer.md) give the rules. The
important one: when a bar's range touches both the stop and the target,
bar data cannot tell which came first, so the simulator assumes the stop
(the conservative choice).

### Triple-barrier labels and meta-labeling

An exit policy with a stop, a target and a time limit is a "triple barrier"
(López de Prado, 2018, ch. 3). The label is which barrier was hit first, or the
net return. Meta-labeling (same book, ch. 3) keeps an existing rule for the
side and trains a model to predict whether the rule's trade will be
profitable. The model then filters trades and, through its probability, sizes
them.

### The trade-management checklist

| Decision | Question | Where |
|---|---|---|
| Holding period | How long does the edge last? | T6 horizon curve, IC decay (E1) |
| Stop | Where is the idea wrong? | T1 exits, T6 stop curves, MFE/MAE |
| Target | Is there a natural reversion or extension level? | T1, T6 |
| Sizing | How much per trade, and does model confidence matter? | T2 |
| Filters | When not to trade: time of day, spread, regime, earnings | T3 |
| Risk limits | Max positions, exposure, daily loss | T4 (names match the platform's RiskEngine) |
| Costs | Spread, slippage, commission | T5, calibrated from R5 |

[research/trade-management.md](../../research/trade-management.md) covers each
in depth. [Example 9](../examples/09-trade-management.md) is the full session.

## Walkthrough

1. **From signal to candidates.** On the fixture, turn the hello model's score
   into candidates: go long when the score is in the top decile of its day so
   far. Count how many candidates per day and per symbol.

2. **Simulate exits by hand.** For three candidates, step through the bars
   after entry on screen: entry at the next bar's open, stop 0.3% below,
   target 0.5% above, 30-minute limit. Write down which barrier hit first.
   Then show one bar that touches both.

3. **Fixed horizon vs. exit policy.** Compare the mean 30-minute forward return
   of the candidates with the mean `net_return` from the exit policy (using
   the T1 simulator if merged, otherwise the solution branch's reference).
   The difference is costs, fill timing and the barriers.

4. **R1 as the worked example.** Open
   [studies/momentum/r1-orb-selectors/README.md](../../../studies/momentum/r1-orb-selectors/README.md).
   R1 keeps one ORB entry rule (first completed bar after the opening range
   that closes above its high, through 11:00) and compares four selectors:
   A takes every trade, B is a rule (relative volume ≥ 1.2 and SPY above its
   open), C is a random forest on net trade return, D is an MLP. That is
   meta-labeling. If P4 (the Replay viewer) is available, open the R1 trade
   ledger in it and look at five trades the model skipped.

5. **Sizing from probability.** Show how T2's `prob→size` maps a calibrated
   probability to a position size, and why an uncalibrated score cannot be
   used this way.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | For 10 hand-picked candidates, label each with the barrier hit first. Compare with the simulator | Disagreements should only be the both-touched bars |
| Core | Change the stop from 0.3% to 0.1%. What happens to the hit rate and to the mean net return? | More stops, a higher share of losers, and more trades end early; the net effect is the thing to measure |
| Stretch | Add costs of 0, 2, 5 bps per side and find where the candidate set's mean net return reaches zero | The break-even cost; compare with session 4's |
| Stretch | Write the trade-management checklist for your team's main strategy, with a first guess and how you would test each item | Use your team guide |
| Challenge | Implement T1 or start your team's S-task (SM1, SR1 or SS3) | Acceptance tests in the packet |

## Discussion

- Why might a model with a worse AUC give a better strategy?
- A stop improves the backtest's Sharpe ratio. What else should you check
  before believing it?
- Which trade-management decisions can be learned from data, and which should
  be fixed in advance?

## Pitfalls

- Entering at the close of the signal bar. The decision happens at
  `ts_available`; the earliest realistic fill is the next bar's open.
- Optimising stop and target on the same data used to report results.
- Ignoring the session close: an exit policy with a 30-minute limit at 15:45
  exits at the close, with reason `session_end`.

## Connection to team strategies

| Team | Strategy this session's ideas apply to |
|---|---|
| [Momentum](../../teams/momentum.md) | ORB (SM1) with meta-labeling; [example 4](../examples/04-meta-labeling-orb.md) |
| [Reversals](../../teams/reversals.md) | Bar fade (SR1) and stretch fade (SR2) with exit policies; R2 reversal meta-labeling |
| [Stat arb](../../teams/stat-arb.md) | Pairs (SS4): entry and exit on spread z-scores, half-life time stops (SS3) |

## Wrap-up PR

Claim T1 (Int), one of T2–T6, or your team's first S-task.
