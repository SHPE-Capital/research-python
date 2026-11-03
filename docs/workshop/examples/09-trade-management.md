# Example 9: Trade management (holding period, stops, targets, sizing per team)

| | |
|---|---|
| **Teams** | All three, side by side. Each team brings its own strategy: momentum brings SM1 ORB, reversals bring SR1/SR2, stat arb brings SS4 pairs. |
| **Level** | Int (the sizing ladder and honest-tuning parts are Adv) |
| **Duration** | 150 minutes: 20 min plenary, 90 min team breakouts, 40 min side-by-side presentation |
| **Prerequisites** | [Example 5](05-volatility-forecast.md) (volatility units), [Example 6](06-overfitting-casino.md) (trial counting), the team's guide in [docs/teams/](../../teams/README.md) |
| **Data** | Synthetic fixture with planted momentum, reversal and cointegrated-pair effects (C1o) for the dry run; then the workshop snapshot (D5), dev symbols of D4 `workshop_v0` |
| **Library tasks used** | T1 exits and simulator, T2 sizing, T3 filters, T4 risk limits, T5 costs, T6 diagnostics, T7 platform export and parity, SM1, SR1/SR2, SS3/SS4, F3, E2, M2 |
| **Build task** | W9 |

## Why this example

Teams spend most of their time on entries. Most of a strategy's outcome is
decided after entry: how long to hold, where to stop out, whether to take
profit, and how big to trade. These choices interact with the edge in ways
that differ by strategy type:

- A **momentum** trade makes money from a minority of large winners. A tight
  stop or a fixed target cuts off exactly those winners.
- A **reversal** trade makes money from many small reversions. It needs to take
  the reversion when it arrives, and a tight stop exits right before the
  reversion happens.
- A **stat-arb** trade has a natural clock: the spread's half-life. Holding much
  longer than that mostly adds the risk that the relationship breaks.

The workshop runs **the same diagnostics** on all three strategies, then puts
the results next to each other. The contrast is the lesson.

**Industry analogue:** systematic desks separate the signal from execution and
trade management, and they test exits with the same rigour as entries. A
typical review asks for the holding-period curve, the MFE/MAE distribution and
the parameter surface for every exit rule before a strategy goes live.

## Learning objectives

1. Read a holding-period curve and choose a holding period with bootstrap bands.
2. Estimate an OU half-life and use it to set a max hold.
3. Use MFE/MAE to place stops and targets from evidence, not habit.
4. Read a stop-efficiency curve, including "no stop", and say whether a stop
   adds value or only reduces variance.
5. Compare sizing rules (fixed, risk-per-trade, inverse volatility, prob→size,
   fractional Kelly) on drawdown and P&L volatility for the same mean.
6. Tune exits honestly: in a fixed order, on validation, with trials counted and
   the whole surface reported.
7. Export the chosen trade management to platform config (T7) and check parity.

## Background

### Order of decisions

Tune one decision at a time, in this order, with the others fixed. The order is
the one in [research/trade-management.md](../../research/trade-management.md):

```
cost hurdle → holding period → stop → target / trailing → filters → sizing → risk limits
```

The cost hurdle comes first: if gross return never clears costs, there is
nothing to manage. Holding period comes next because stop and target distances
only make sense relative to how long the trade lasts. Sizing comes after the
exits because the risk-per-trade rule needs the stop distance. Keep the
baseline (time exit only, no stop, no target, no filter, fixed size) next to
every variant. Every pass adds to the trial count
([Example 6](06-overfitting-casino.md)). Filters and risk limits are outside
this session; the team guides cover them (RM4, RR4).

### Holding-period curve

For each candidate entry, compute the net return if exited after `h` minutes
(or bars), for `h` in a grid, e.g. 1, 2, 5, 10, 15, 30, 60, 120, session close.
Plot the mean net return against `h`, with block-bootstrap bands over sessions
(E2). The peak gives the natural horizon. A flat or falling curve from `h = 1`
means there is no edge to manage. This diagnostic is descriptive, does not
depend on any exit choice, and costs no trials.

### OU half-life

For a mean-reverting quantity `x_t`, such as a pair spread or a reversal
"stretch" in σ units, fit an AR(1) on training data:

```
x_t = c + φ·x_{t−1} + ε_t
half-life hl = −ln 2 / ln φ                (bars; requires 0 < φ < 1)
```

Half of a deviation is expected to be gone after `hl` bars. A max hold of
`k·hl` (k between 2 and 4) is a natural time exit. SS3 implements the fit;
T6's `analysis/holding.py` plots it next to the holding-period curve.

### MFE and MAE

For each trade, measured from entry and in σ units (from [Example 5](05-volatility-forecast.md)):

- **MFE** (maximum favourable excursion): the best unrealised P&L before exit.
- **MAE** (maximum adverse excursion): the worst unrealised P&L before exit.

Plot MAE against final P&L, split by winners and losers.
- If winners rarely have MAE beyond `m·σ`, a stop at a bit more than `m·σ` cuts
  losers without touching winners.
- If winners often have MAE beyond `m·σ`, a stop there turns winners into
  losers.
- The MFE distribution of eventual losers shows how much a target could have saved.

### Stop-efficiency curve

Run the strategy with stop distances `d` in a grid **including ∞ (no stop)**,
and plot net mean return, hit rate, P&L standard deviation and the 5th
percentile of trade P&L against `d`. A stop is worth having if it improves the
mean, or keeps the mean while cutting the tail. A stop that only lowers
variance by lowering the mean is a sizing decision in disguise.

### Targets and trailing stops

- **Fixed target** `t·σ`: suits reversion, where the expected move is bounded.
- **Trailing stop** (chandelier): exit when price falls `k·σ` (or `k·ATR`) from
  the highest price since entry. It suits momentum, where the size of the move
  is unbounded.
- **Signal exit:** exit when the signal says so, e.g. when the z-score crosses
  `exitZScore`. This is the stat-arb default.

### Sizing ladder

All rules return whole shares, floored, like the platform (T2).

| Rule | Quantity | Needs |
|---|---|---|
| Fixed notional | `floor(N / P)` | `N` = `maxPositionSizeUsd` (platform `fixedNotionalSizer`) |
| Risk per trade | `floor(equity · r% / |entry − stop|)` | A stop distance; `r%` e.g. 0.25–0.5% |
| Inverse volatility | `floor(target_risk_usd / (σ̂_h · P))` | A volatility forecast for the trade horizon |
| Probability → size | `size = 2Φ(z) − 1`, `z = (p − 0.5) / sqrt(p(1 − p))`, times a base size | A calibrated probability `p` from a meta-model (Example 4) |
| Fractional Kelly | `f = c · μ / σ²`, with `c ≤ 0.5` | Estimates of mean `μ` and variance `σ²` of trade returns; very noisy |

The probability-to-size rule is from López de Prado, *Advances in Financial
Machine Learning*, ch. 10. Full Kelly maximises long-run growth only when
`μ` and `σ²` are known. Estimation error makes full Kelly too aggressive, so use
a fraction `c ≤ 0.5`.

**Sizing should change risk, not create edge.** Compare rules at the same mean
return, on max drawdown, daily P&L volatility and worst day. If one rule has a
much higher mean, check why: it usually means larger size on a subset that
happened to win.

**Platform status:** the platform's `volatility_scaled` and `kelly_fractional`
sizers are scaffolds that return 0 and are not wired into the orchestrator
(P6). Today strategies emit their own size: `qtyPerTrade` (reversal) and
`tradeNotionalUsd` (pairs). T7 refuses to export the other rules until P6 lands
([design/trade-layer.md](../../design/trade-layer.md)). A sizing result from
this example is evidence for P6, not a deployable config.

### Cost hurdle chart

Plot mean gross return per trade against the holding period, with horizontal
lines for the round-trip cost: 1×, 2× and 3× the T5 default, and the R5-calibrated
cost when available. The platform `DEFAULT_FILL_MODEL` gives about 14 bps per
round trip (half spread 2 bps plus slippage 5 bps per side) before commission.
Any horizon where gross is below the 2× line is not tradeable.

## Setup

### What each team brings

| Team | Strategy | Entry | Exit to analyse | Mean-reverting quantity for half-life |
|---|---|---|---|---|
| Momentum | SM1 ORB | 15-minute opening-range breakout, confirmed by close, entries until 11:00 | Time exit, `k·σ` stop, trailing stop vs. fixed target | None. The team checks that the half-life is undefined or very long |
| Reversals | SR1 bar fade (platform `minuteReversal` baseline) and SR2 stretch fade | Fade a stretch of `k·σ` from the anchor (prev close / VWAP / EMA) | Target at the anchor, disaster stop, time exit | The stretch in σ units |
| Stat arb | SS4 pairs | `entryZScore` 2 | `exitZScore` 0.5, `stopLossZScore` 4, `maxHoldingTimeMs` 24h vs. `k·hl` | The spread |

### The six panels every team produces

| # | Panel | T6 module |
|---|---|---|
| 1 | Holding-period curve with bootstrap bands and the cost hurdle lines | `analysis/holding.py` |
| 2 | OU half-life, with the AR(1) fit and `hl` on the same time axis as panel 1 | `analysis/holding.py` (uses SS3) |
| 3 | MAE vs. final P&L scatter, winners and losers coloured; MFE histogram of losers | `analysis/excursions.py` |
| 4 | Stop-efficiency curve including ∞: mean, hit rate, 5th percentile | `analysis/excursions.py` |
| 5 | Exit-rule surface (stop × target or trailing, or entry z × exit z), with the trial count printed on the figure | `analysis/surfaces.py` |
| 6 | Sizing ladder: daily P&L volatility, max drawdown and mean for each rule | T2 `SizingRule` + E1 |

### Config sketch

One file per team. Shown for momentum; syntax pending T1–T7 and C1h.

```yaml
name: ex09_momentum_orb
seed: 7
data:
  source: snapshot
  snapshot: workshop_v0
  role: dev
strategy:                                 # SM1
  type: orb
  range_minutes: 15
  confirm_rule: close
  entry_cutoff: "11:00"
exit_policy:                              # T1
  max_minutes: {grid: [15, 30, 60, 120, session_end]}       # panel 1 and 5
  stop: {type: sigma, k: {grid: [1.0, 1.5, 2.0, 3.0, null]}} # null = no stop, panel 4
  target: {type: none}                    # compared with trailing below
  trailing: {type: chandelier, k: {grid: [2.0, 3.0, null]}}
  session_end_flatten: true
sizing:                                   # T2
  compare: [fixed_notional, risk_per_trade, inverse_vol, prob_to_size, fractional_kelly]
  fixed_notional_usd: 5000
  risk_per_trade_pct: 0.25
  kelly_fraction: 0.25
risk:                                     # T4, platform names
  maxPositionSizeUsd: 5000
  maxConcentrationPct: 0.30
  maxIntradayDrawdownPct: 0.05
costs: {half_spread_bps: 2.0, slippage_bps: 5.0, commission_per_share: 0.005}   # T5
split: {train: 0.6, validation: 0.2, test: 0.2, gap_days: 1}
report:
  panels: [holding_curve, half_life, excursions, stop_curve, exit_surface, sizing_ladder]
  trials_from: registry                   # M2 -> E2 deflated Sharpe
```

The reversal file swaps in `strategy: {type: stretch_fade, anchor: vwap, k: 2.0}`
and `target: {type: anchor}`. The stat-arb file uses `strategy: {type: pairs}`
with the pair list from [Example 7](07-unsupervised-pairs.md) and exit grids on
`exitZScore`, `stopLossZScore` and `max_hold: {type: half_life_multiple, k: [1, 2, 3, 4]}`.

## Walkthrough

### Plenary (20 min)

Introduce the decision order, the six panels and the honest-tuning rules. Run
all six panels once on the **C1o fixture**, where the answers are known:

| Planted effect (C1o) | Expected panel shapes |
|---|---|
| Momentum: drift continues after a breakout, with fat right tail | Holding curve rises then flattens over tens of minutes. Winners have sizeable MAE. Stop curve: tight stops lower the mean; ∞ or wide is best. Trailing beats a fixed target. |
| Reversal: the AR(5) reversal in the current fixture (`signal_strength` 0.3) | Holding curve peaks within a few bars, then decays. Short half-life. A target at the anchor captures most of the move. Tight stops hurt. |
| Cointegrated pair: OU spread with a known half-life | Fitted `hl` matches the planted value. Net P&L per trade levels off beyond about `2–3·hl` while the variance keeps rising. |

These are properties of the fixture, used to check the code. They are not
claims about real markets.

### Team breakouts (90 min)

Each team runs the six panels on the snapshot, in the decision order:

1. **Cost hurdle and holding period (20 min).** Panel 1 with its cost lines, then panel 2.
   - *Momentum:* where does the curve peak? Is it beyond the 2× cost line?
   - *Reversals:* is the peak gross return above the 2× cost line at any horizon?
     If not, write that down. It is the team's main risk (RR1).
   - *Stat arb:* compare `hl` with the platform's 24h `maxHoldingTimeMs`.
2. **Stops (20 min).** Panels 3 and 4. Choose a stop rule from MAE evidence,
   then confirm it on the stop curve **on validation**. Count the curve's points as trials.
3. **Targets / trailing / signal exit (20 min).** Panel 5. Mark the plateau,
   not the best cell. If there is no plateau, keep the simplest rule.
4. **Sizing (20 min).** Panel 6 with the chosen exits fixed. Compare rules at
   matched mean. For prob→size, use example 4's meta-model if the team has one;
   otherwise skip it.
5. **Write the spec (10 min).** Fill in the trade-management block of the spec
   card ([research/spec-card-template.md](../../research/spec-card-template.md)):
   chosen values, the trial count, the deflated Sharpe on validation, and what
   was **not** tested.

### Side-by-side presentation (40 min)

Each team shows its six panels in a fixed 2 × 3 layout, in the same order, so
the room can compare across teams. Ten minutes per team, then ten minutes of
discussion.

**Hypotheses to test** (stated in advance, confirmed or rejected by the data):

| Team | Hypothesis |
|---|---|
| Momentum | Wide or no stops, no fixed target, a trailing stop or time exit. Stops expressed in volatility units. |
| Reversals | A target at the anchor, a disaster stop only (wide), and a short time exit. Costs decide whether anything is tradeable. |
| Stat arb | A max hold of a few half-lives beats the fixed 24h. The z-score stop matters less than a half-life exit for the losers. |

If a team's result goes against its hypothesis, that is a finding to present, not a
failure.

### Honest-tuning rules

1. Follow the decision order. Fix the others while tuning one. Keep the
   baseline in every table.
2. Choose on **validation**. Open test once. Then check the held-out symbols (D4).
3. Every grid point, including the curves in panels 4 and 5, goes to the M2
   registry, and the total goes into the deflated Sharpe (E2).
4. Report the whole surface, mark the plateau, and choose its centre.
5. A trade-management rule needs an economic reason that matches the strategy
   type (bounded vs. unbounded move, natural clock). No reason, no rule.

### Platform export and parity (after the session)

T7 maps the chosen values to platform config. The team's agent for this is
`platform-parity` (`.claude/agents/`). Then it runs a 5-session parity check:
the same sessions in the research simulator and the platform backtest, compared
trade by trade ([design/platform-parity.md](../../design/platform-parity.md)).

| Research setting | Platform field | Status |
|---|---|---|
| Pairs entry / exit / z-stop | `entryZScore`, `exitZScore`, `stopLossZScore` | Exportable |
| Pairs max hold | `maxHoldingTimeMs` | Exportable as a fixed value; `k·hl` per pair needs P9 |
| Pairs hedge method | `hedgeRatioMethod` (`fixed` or `rolling_ols`) | Exportable; Kalman needs P9 |
| Pairs size | `tradeNotionalUsd` | Exportable; spread-vol sizing needs P9 |
| Reversal size | `qtyPerTrade`, `maxPositionQty` | Exportable |
| Reversal targets, stops, time exit, stretch entry | One parameterised reversal family | Needs P7 |
| ORB entry, stops, trailing | ORB strategy | Needs P8 |
| Risk-per-trade, inverse-vol, Kelly sizing | `SizerType` sizers | Needs P6 |
| Risk limits | `maxPositionSizeUsd`, `maxConcentrationPct`, `maxIntradayDrawdownPct`, … | Exportable (T4 uses the same names) |

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Produce panels 1 and 4 for your team's strategy on the fixture and confirm the planted shape. | Use the T1 simulator with only a time exit for panel 1. Panel 4 adds the stop grid with `null` included. Daily-aggregated P&L for the bootstrap bands. |
| Core | Estimate the half-life of your mean-reverting quantity and compare it with the peak of the holding curve. | SS3 `fit_ou` on training data. For momentum, report `φ` and explain why `hl` is undefined or very long. The two estimates should roughly agree for reversals and stat arb. |
| Stretch | Build the sizing ladder (panel 6) and compare rules at matched mean, by rescaling each rule's P&L series to the same mean before comparing risk. | Risk-per-trade needs the stop from step 2. Inverse vol uses F3's forecast. Kelly with `c = 0.25` on rolling training estimates of `μ` and `σ²`. Report how often each rule hits `maxPositionSizeUsd`. |
| Stretch | Momentum only: trailing stop vs. fixed target on the same entries. Reversals only: target at the anchor vs. at half the stretch. | Same trades, different exits, paired bootstrap of the daily P&L difference. |
| Challenge | Triple-barrier labels from your chosen exits feed a meta-model (example 4). Does prob→size beat fixed size on held-out symbols? | Labels from C1c `triple_barrier` with your chosen stop, target and time. Purged walk-forward (V1). Prob→size needs calibrated probabilities: check a reliability curve first. |
| Challenge | Run the 5-session parity check for the exportable fields. | T7 `ParityReport`. Expect mismatches at session boundaries and on gapped stops first; [design/platform-parity.md](../../design/platform-parity.md) lists the tolerances. |

Full solutions live on the `workshop-solutions` branch under `workshop/09_trade_management/`.

## Discussion questions

1. Why should a momentum strategy and a reversal strategy want opposite exit rules?
2. The stop curve shows "no stop" is best. Would you still run one in live
   trading? What is it protecting against that the backtest does not contain?
3. If a sizing rule raises the mean return, is that edge or luck? How would you tell?
4. Stat arb: does a half-life estimated over 60 sessions apply to the next 20?
5. Which of today's chosen values would you expect to change most on new data,
   and how would you monitor it live?

## Pitfalls

- **Intrabar path assumptions.** Minute bars do not show whether the stop or the
  target hit first. T1 assumes the stop (conservative). Do not change that to
  make results look better.
- **Same-bar fills.** A decision at the close of bar `t` fills at the open of
  bar `t+1`, never at the close of `t`.
- **Gaps through stops.** A stop fills at the open when the bar gaps through it.
  Overnight gaps make this larger, so session-end flattening matters for
  momentum.
- **Tuning on the full sample.** The stop that looks best on all data has seen
  the test set.
- **Trade-level Sharpe.** Trades on the same day are correlated. Use daily P&L
  for significance.
- **Sizing from future volatility.** The inverse-vol rule must use the forecast
  available at entry, not realised volatility over the trade.
- **Ignoring `exit_reason`.** Report the mix of `stop_loss`, `take_profit`,
  `exit`, `max_hold_exit` and `session_end`. A strategy whose profit comes from
  `session_end` exits is a different strategy from the one designed.

## Connection to team strategies

| Team | Guide section | Study IDs this example previews |
|---|---|---|
| Momentum | [teams/momentum.md](../../teams/momentum.md) §5 Trade management for ORB | RM3 exits, RM5 sizing, RM7 cost sensitivity |
| Reversals | [teams/reversals.md](../../teams/reversals.md) §5 Trade management | RR1 cost hurdle, RR5 holding period and exits, RR6 sizing |
| Stat arb | [teams/stat-arb.md](../../teams/stat-arb.md) §5 Trade management | RS4 half-life, RS5 entry/exit z, RS6 stops, RS7 sizing |

The general method is in [research/trade-management.md](../../research/trade-management.md).
The `trade-management-analyst` agent follows the same panel order.

## Builder notes (W9)

| Item | Detail |
|---|---|
| Folder | `workshop/09_trade_management/`: `README.md`, `ex09_panels.py` (shared panel code calling T6), `ex09_momentum.py`, `ex09_reversals.py`, `ex09_statarb.py` (jupytext), `configs/ex09_{momentum,reversals,statarb}_{fixture,snapshot}.yaml`, `panels_template.md` (the 2 × 3 slide layout) |
| Library tasks required | T1, T2, T3, T4, T5, T6 (`holding.py`, `excursions.py`, `surfaces.py`), T7, SM1, SR1, SR2, SS3, SS4, F3, E2, M2 |
| Fixture effect needed | C1o: breakout momentum with fat right tail; the existing AR(5) reversal; a cointegrated OU pair with known half-life |
| Fallback | If SM1/SR2/SS4 are not merged, the notebooks fall back to minimal in-folder candidate generators, flagged in the README, so the session can still run |
| CI check | On the C1o fixture with fixed seed: (a) momentum stop curve has its maximum at the widest stop or ∞; (b) reversal holding curve peaks at ≤ 5 bars; (c) fitted pair half-life within 30% of planted; (d) each panel function reports a trial count; (e) all three notebooks run headless in under 5 minutes each |
| Participant scaffold | Panels 1 and 2 given; panels 3–6 are `# TODO` with the expected fixture shape written as an assert |

## Further reading

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Ch. 10 (bet sizing).
- Avellaneda, M. & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*. (OU residuals and s-score entry/exit.)
- Bailey, D. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*.
- [research/trade-management.md](../../research/trade-management.md)
- [design/trade-layer.md](../../design/trade-layer.md): fill rules, `exit_reason` vocabulary, sizing and risk names.
