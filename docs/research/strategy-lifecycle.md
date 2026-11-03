# Strategy lifecycle

Every strategy moves through the same stages, from idea to retirement. Each
stage has a gate: the criteria that must hold to move on, the evidence that
shows it, and who signs off. A strategy can stop at any gate. Stopping is a
result, and it is written up like any other result.

```
idea → hypothesis → spec card v0 → pre-registered study → validation
     → test + held-out symbols → platform parity → paper → small live
     → monitoring → retirement
```

## Roles

| Role | Who | Job at the gates |
|---|---|---|
| Author | The team member(s) running the strategy | Produce the evidence |
| Team lead | Lead of the owning team | Approves scope and moves between stages |
| Reviewer | Someone from another team, plus a `leakage-auditor` run | Challenges the evidence; must not have worked on the study |
| Platform owner | Maintainer of the trading-platform repo | Signs off parity and anything that touches live config |
| Risk owner | Whoever sets the club's live risk budget | Signs off paper → live and live size changes |

One person can hold more than one role, except that the reviewer is never the author.

## Stages and gates

### 1. Idea

A paragraph: what the strategy trades, when, and why it might work.

| Gate criterion | Evidence |
|---|---|
| A mechanism is named: who is on the other side of the trade, and why they accept a worse price (risk transfer, liquidity provision, slow information diffusion, behavioural bias, a constraint) | One paragraph in the spec card's "Rationale" |
| The data exists in the store or a planned source (C1g) | A data source and date range |
| Back-of-envelope edge is above costs | Expected move per trade versus the round trip at platform fill defaults (about 14 bps plus commission; see [trade-management.md](trade-management.md#costs-and-break-even)) |

Sign-off: team lead.

### 2. Hypothesis and rationale

Turn the idea into a statement that can be false.

| Gate criterion | Evidence |
|---|---|
| A testable claim with a sign and a horizon ("stocks stretched more than 2σ below VWAP return toward VWAP within 30 minutes, net of costs") | Spec card "Hypothesis" |
| The null result is named ("mean net return per trade ≤ 0") | Spec card |
| Related literature or prior studies are listed, including failures | Links |

Sign-off: team lead.

### 3. Spec card v0

Fill in [spec-card-template.md](spec-card-template.md) as far as you can.
Unknown parameters are marked "to be studied", with the study that will set them.

| Gate criterion | Evidence |
|---|---|
| Universe, entry, exits, sizing, risk limits and costs each have a value or a named study | Spec card v0 committed |
| Kill criteria are written now, before any results | Spec card "Kill criteria" |
| A baseline is named (for example `minuteReversal` for reversals, "take every breakout" for ORB) | Spec card |

Sign-off: team lead.

### 4. Pre-registered study

Create the study folder with the [study template](study-template.md) and fill
in the pre-registration. Follow the [research protocol](research-protocol.md).

| Gate criterion | Evidence |
|---|---|
| Pre-registration committed before any validation result exists | `git log` on the study README |
| Train, validation and test dates fixed; dev/held-out symbols fixed from the D4 universe builder | README |
| The parameter grid, the selection metric and the baselines are listed | README |
| The trial budget is stated | README |

Sign-off: team lead and reviewer (the reviewer checks the design, not results).

### 5. Validation

Develop on train and validation periods, dev symbols only. Tune trade
management with the `trade-management-analyst` workflow.

| Gate criterion | Evidence |
|---|---|
| Net of costs at platform defaults, the strategy beats its baselines on validation | E1 report, paired block bootstrap |
| The result holds at 2× costs, or the cost sensitivity is stated | E1 `cost_adjusted` |
| The chosen parameters sit on a plateau of the parameter surface | Surface figures (T6) |
| The result is not carried by one symbol, one month or one universe cell | E1 `by_group` |
| The trial count is recorded | Study README |
| Configuration frozen and committed | Commit hash |

Sign-off: team lead. Then the reviewer runs `leakage-auditor` on the code and configs.

### 6. Test and held-out symbols

Open the test period and the held-out symbols once, with the frozen configuration.

| Gate criterion | Evidence |
|---|---|
| Test was opened once; the frozen commit predates the test run | Run directory timestamps and git history |
| The deflated Sharpe ratio at the recorded trial count is reported (E2) | `deflated_sharpe` output |
| Net performance on test and held-out symbols is consistent with validation (same sign, confidence interval overlaps) | E1 report |
| Kill criteria are not already met on test | Spec card check |

Sign-off: reviewer. If test fails, the study is written up as negative and the
strategy stops here. Do not re-tune and re-open test; a second look makes the
result validation-only.

### 7. Platform parity

Export the frozen `StrategySpec` with `export_platform_config` and reconcile
with the platform backtest (T7, `platform-parity` agent).

| Gate criterion | Evidence |
|---|---|
| Every spec parameter maps to a platform config field, or a P-task exists for it | Field mapping table |
| On at least 5 sessions (including a gap day, a half day, a gap-through stop, and a same-bar stop/target tie) trades match within the stated tolerance | `ParityReport` |
| Every difference is explained | `ParityReport` |

Sign-off: platform owner.

### 8. Paper trading

Run the exported config on the platform in paper mode.

| Gate criterion | Evidence |
|---|---|
| At least 20 sessions and enough trades to measure the mean (set the number in the spec card; 100 is a reasonable default for intraday) | Paper trade log |
| Realised slippage and spread are within the cost model's assumption, or the cost model is updated (T5, R5) | TCA against `CostModel` |
| Paper P&L per trade is inside the test-period confidence interval | Comparison table |
| No risk-engine rejections that research did not expect | Platform logs by rejection code |
| Kill criteria not met | Spec card check |

Sign-off: team lead and risk owner.

### 9. Small live

Live with a size well below the eventual target, for example 10–25% of the
planned `maxPositionSizeUsd`.

| Gate criterion | Evidence |
|---|---|
| Same checks as paper, on live fills | Live log, TCA |
| Kill criteria monitored daily | Monitoring report |
| Size increases follow a written schedule, each signed by the risk owner | Spec card change log |

Sign-off: risk owner for each size step.

### 10. Monitoring

Runs for as long as the strategy is live.

| What | How often | Action threshold |
|---|---|---|
| Net P&L per trade versus the test-period distribution | Weekly | Below the test-period 5th percentile over a rolling window |
| Realised costs versus `CostModel` | Weekly | More than 25% above the model |
| Trade frequency, hit rate, average holding time | Weekly | Outside the test-period range |
| Feature drift (input distributions versus training) | Monthly | Any feature shifts enough to put most rows outside the training range |
| Risk-engine rejections by code | Daily | Any unexpected code |
| Drawdown | Daily | Kill criteria |

Sign-off: team lead reviews weekly; the risk owner sees any threshold breach.

### 11. Retirement

A strategy is retired when a kill criterion is met, when monitoring shows the
edge has gone, or when the team decides the capital is better used elsewhere.

| Step | Evidence |
|---|---|
| Disable in the platform config | Config commit |
| Write a post-mortem: what changed, whether it was foreseeable, what the monitoring caught | Study folder `POSTMORTEM.md` |
| Keep the spec card, configs and study; mark the spec card "retired" | Spec card status |

Sign-off: team lead and risk owner.

## Re-entering the lifecycle

A changed strategy is a new version. Small changes (a cost model update, a new
P-task field with the same behaviour) go back to parity. Changes to entry,
exits, sizing or universe go back to stage 4 with a new pre-registration, and
the old trial count carries forward.
