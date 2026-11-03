# Example 4: Meta-labeling an opening-range breakout

| | |
|---|---|
| Teams | Momentum (primary); Reversals for the same method on fades (R2) |
| Level | Int–Adv |
| Duration | 120 min (core 90 min) |
| Prerequisites | Sessions 1–5; examples 1 and 2 |
| Data | Fixture with a planted ORB effect (C1o) for the walkthrough; snapshot `role: dev` for the R1 comparison; `role: held_out` opened once by the facilitator |
| Library tasks used | SM1, F1, F2, F3, F4, F8, T1, C1b, C1c (`triple_barrier`), M1a, M1c, E1, E2; optional T2, M2 |
| Build task | W4 |

## Why this example

An opening-range breakout (ORB) rule decides **when and which side**: go long
when price closes above the morning's range. It says nothing about which
breakouts are worth taking. Meta-labeling keeps the rule and trains a second
model to answer a narrower question: given that the rule fired, will this
trade make money after costs? The model then filters trades and, through
its probability, sizes them (López de Prado, 2018, ch. 3 and ch. 10).

This is how ML is often added to an existing discretionary or rule-based
strategy in practice: the rule carries the economic idea, the model learns
when the idea works, and the comparison is always against the rule alone and
against a simple hand-made filter. The study R1
([studies/r1-orb-selectors/README.md](../../../studies/r1-orb-selectors/README.md))
is this example at full size.

## Learning objectives

1. Generate ORB candidates with correct timing: the range is known at 09:45,
   the decision at the breakout bar's `ts_available`, the fill at the next
   bar's open.
2. Label each candidate with a fixed exit policy (stop, time exit, session
   end) through the trade simulator, and explain why the policy is fixed
   before the model is trained.
3. Train and threshold a meta-model; read the precision/recall/trade-count
   tradeoff and choose a threshold on validation only.
4. Turn probabilities into position sizes and check calibration first.
5. Compare four selectors (R1's A–D) with a paired bootstrap and a matched
   random-retention baseline.

## Background

### The primary rule (SM1)

| Item | Definition |
|---|---|
| Opening range | High `H` and low `L` of the 1-minute bars from 09:30 to 09:44 (15 bars). Known at 09:45 |
| Candidate | The **first** completed bar after the range, through 11:00, whose close is above `H`. At most one per symbol per day |
| Decision time | That bar's `ts_available` (its end) |
| Side | Long |
| Entry | Open of the next bar, plus half spread and slippage |

`features/opening_range.py` (F4) computes `H`, `L`, range width and breakout
distance. Its worked example: `H = 100`, `L = 98`, close 100.10 gives a
breakout distance of `(100.10 − 100) / 100 = 0.001`.

### The exit policy and the label

R1 fixes the policy before any model is trained:

| Barrier | R1 setting | Exit reason |
|---|---|---|
| Stop | Range midpoint `(H + L) / 2` | `stop` |
| Target | None in R1 (optional in stretch exercises) | `target` |
| Time | 30 minutes after the entry fill | `time` |
| Session close | Exit at the close if earlier | `session_end` |
| No next bar | No fill | `unfilled` (excluded with an explicit status, never a zero return) |

Fill rules (from `labels/exit_policy.py`, moving to `trade/simulate.py` in
T1): a stop fills at the stop price, or at the bar's open if the bar gapped
through it; if a bar touches both stop and target, assume the stop.

The trade's net return per share:

```
y_net = (exit_fill − entry_fill − fees_per_share) / entry_fill
```

and the meta-label is `y = 1 if y_net > 0 else 0`. Selector C in R1 regresses
`y_net` directly; both versions are covered below.

This is a triple-barrier label (stop, target, time). The point of fixing the
policy first is that the label must not depend on choices made after seeing
the model's results. If the stop is retuned later, the labels change and the
model is retrained on validation-informed labels.

### Precision, recall and trade count

For a threshold `τ` on the model's probability `p`:

| Measure | Definition | Trading meaning |
|---|---|---|
| Precision | winners taken / trades taken | Hit rate of the filtered strategy |
| Recall | winners taken / all winners | Share of the good trades kept |
| Trade count | trades with `p ≥ τ` | Statistical power and capacity |
| Mean net return per trade | mean `y_net` of taken trades | Edge per trade |
| Total net return | sum of `y_net` of taken trades | What the account sees |

Raising `τ` usually raises precision and lowers recall and trade count. The
best `τ` for mean return per trade is often a bad `τ` for total return, and a
`τ` that keeps 20 trades out of 2,000 is not a result. R1 pre-registers a
minimum number of trades before choosing `τ`.

### From probability to size

Two standard mappings (T2 implements both):

```
AFML bet size:     z = (p − 0.5) / sqrt(p (1 − p)),   size = 2 Φ(z) − 1   (0 below p = 0.5)
fractional Kelly:  f = c · (p − (1 − p) / b),   b = mean win / mean loss,   c in (0, 1], e.g. 0.25
```

Both assume `p` is calibrated: among trades with `p = 0.6`, about 60% win.
Tree models and neural nets are often not calibrated. Check a reliability
curve on validation, and calibrate (Platt or isotonic, fitted on validation)
before sizing.

### R1's four selectors

| Version | Selection | Fit |
|---|---|---|
| A | Take every candidate | None |
| B | Take when relative volume ≥ 1.2 and SPY is above its session open | None (hand rule) |
| C | Random forest on `y_net`; take when the prediction exceeds a threshold chosen on validation | M1a, 3 seeds |
| D | Small MLP, same inputs and target | M1c, 3 seeds |

A model that beats A has only shown it is better than nothing. It needs to
beat B, and it needs to beat a **matched random retention**: a random subset
of A's trades of the same size as C's, drawn many times. If C's mean net
return is inside the random-retention distribution, the filter adds nothing
beyond trading less.

## Setup

**Data.**

| Part | Data | Planted effect |
|---|---|---|
| 1–3 | Fixture from C1o, 30+ symbols, about 6 months, plus a context symbol for SPY | Breakouts continue (positive drift for 30 minutes after a close above `H`) only when relative volume is high; otherwise no drift |
| 4 | Snapshot `role: dev` | None known; this is research |
| 5 | Snapshot `role: held_out`, facilitator only | None known |

On the current C0 fixture there is no ORB effect, and the planted 5-minute
reversal works against breakouts. Running parts 1–3 on it is a useful negative
control: **hypothesis:** A's mean net return is negative after costs, and no
selector finds a reliable filter.

**Features at the decision** (R1's list): F1 relative volume, F8 market
return (SPY since open), F4 range width and breakout distance, F3 realized
volatility, F2 VWAP distance, `minute_of_day`.

**Config sketch** (`workshop/04_meta_labeling_orb/configs/orb_c.yaml`):

```yaml
name: orb_meta_rf
seed: 1
data: {source: fixture, symbols: [...], start: "2026-03-02", end: "2026-08-28",
       seed: 7, effects: [orb_rvol]}               # C1o effect name to be fixed by C1o
context: [SPY]                                       # C1d context symbols
candidates: {type: orb_first_breakout,               # SM1 via C1b
             range_minutes: 15, last_entry: "11:00", side: long}
features:
  - {type: relative_volume, window_days: 20}         # F1
  - {type: market_return, symbol: SPY}               # F8
  - {type: opening_range, range_minutes: 15}         # F4
  - {type: realized_vol, window: 30}                 # F3
  - {type: vwap_distance}                            # F2
  - {type: minute_of_day}
label:
  type: triple_barrier                               # C1c, simulated by T1
  stop: range_mid
  target: null
  max_minutes: 30
  costs: {half_spread_bps: 2.0, slippage_bps: 5.0, commission_per_share: 0.005}   # platform DEFAULT_FILL_MODEL
target: value                                        # y_net; "sign" for the classifier variant
split: {train: 0.6, validation: 0.2, test: 0.2}
model: {type: random_forest_reg, params: {n_estimators: 500, max_depth: 4, min_samples_leaf: 50}}
selection: {threshold: chosen_on_validation, min_trades: 200}   # pre-registered
```

Names follow C1b, C1c, C1o and SM1 as merged. The runner currently accepts
only `forward_return` labels and every-N-minutes sampling; this example needs
C1b candidates and the C1c `triple_barrier` label.

**Pre-register** in `workshop/04_meta_labeling_orb/README.md` (or, for R1, in
the study README): dates, the exit policy, B's thresholds, the feature list,
model settings, the threshold rule and the minimum trade count, before any
validation result is looked at.

## Walkthrough

### Part 1: candidates and labels (25 min)

1. Generate candidates on the fixture with SM1. Check timing on three by hand:
   the range uses 09:30–09:44, the breakout bar is after 09:45, the decision is
   at the breakout bar's end, the fill is the next bar's open.
2. Simulate exits with T1. Tabulate `exit_reason` counts and mean `y_net` by
   reason. Check one `stop` trade and one `time` trade bar by bar.
3. Plot `y_net` against relative volume in buckets. **Expected on the C1o
   fixture:** the mean `y_net` rises with relative volume, because that is
   the planted effect. On the C0 fixture, there should be no clear pattern.
4. Report A: trades, hit rate, mean and total `y_net`, on validation only.

### Part 2: rule and model selectors (25 min)

1. B: apply the hand rule. Report the same numbers.
2. C: train the random forest regressor on training candidates, 3 seeds. Plot,
   on validation, precision, mean `y_net` per trade, total `y_net` and trade
   count against the threshold. Pick `τ` by the pre-registered rule.
3. D: the same with the MLP (classifier on `y`, or regressor on `y_net`;
   whichever R1 pre-registered).
4. **Expected on the C1o fixture:** B and C both beat A, because the planted
   effect is conditional on relative volume; C should find roughly the same
   condition as B. Whether C beats B depends on whether the planted condition
   is a sharp threshold (B can match it) or a smooth one (C can do better).
   This is a property of the fixture; the CI band fixes what is expected.

### Part 3: is the filter real? (20 min)

1. **Paired bootstrap.** On validation days, compute daily total `y_net` for
   C and for A (and B). Bootstrap the mean daily difference with blocks of 5
   sessions (E2).
2. **Matched random retention.** Draw 1,000 random subsets of A's trades of
   the same size as C's. Plot the distribution of their mean `y_net` and mark
   C's. C is only useful if it is in the upper tail.
3. **Context ablation.** Retrain C without the SPY feature (F8). If the edge
   disappears, the model is mostly timing the market, not selecting
   breakouts. That is a different strategy, and worth knowing.
4. **Sizing.** Calibrate C's classifier variant on validation, plot the
   reliability curve, and compare equal-size trades with AFML sizing and
   quarter-Kelly. Report total return and the worst day for each.

### Part 4: the snapshot, as R1 (15 min)

1. Run A, B, C, D on the snapshot's dev names with the same frozen configs.
2. Fill R1's table: trades, hit rate, mean and total `y_net` at 0, 2, 5, 10
   bps per side (E1), with bootstrap intervals for C − A and C − B.
3. These are hypotheses for the momentum team's study, not conclusions. Three
   months of dev data on fewer than 50 names gives a few hundred candidates at
   most; check the count before reading anything into the result.

### Part 5: held out, once (5 min, facilitator)

The facilitator runs the single pre-registered configuration on
`role: held_out` and shows the result. Nobody changes anything afterwards.
Discuss how it compares with dev.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Count candidates per symbol per day and confirm there is never more than one | The rule takes the first breakout only |
| Core | Move the decision to the breakout bar's `ts` (its start) instead of `ts_available`. What happens to A's results and why is it wrong? | The model would act before the bar closed; on bars this is a look-ahead of one minute |
| Core | Plot precision and trade count against `τ` for C on validation. At what `τ` does the trade count fall below the pre-registered minimum? | That bounds `τ` from above |
| Stretch | Add a profit target at `H + 2 × (H − L)` and rerun A and C. Did you just tune the policy on validation? | Yes; any policy change must be pre-registered and the comparison rerun from scratch |
| Stretch | Train C as a classifier on `y` and as a regressor on `y_net`. Which ranks trades better by total return? | The regressor uses the size of wins and losses; the classifier ignores it. Measure |
| Stretch | Implement the matched random-retention test as a reusable function for E1 | Small PR |
| Challenge | Implement SM1 with its acceptance test (hand-built bars: range, first breakout only, nothing after 11:00, nothing before 09:45) | Momentum team task |
| Challenge | Apply the same method to a reversal rule (R2): candidates from SR1 or SR2, the same selectors, the same tests | Reversals team study |

## Discussion

- Why fix the exit policy before training the meta-model?
- C beats A but not B. What should the team do?
- C's best threshold keeps 8% of trades with a high hit rate. Why might you
  still choose a lower threshold?
- If the context ablation removes the edge, is the model useless?
- What would make you trust D (the MLP) over C?

## Pitfalls

- Using the breakout bar's close as the entry price.
- Opening range computed with the 09:45 bar included, or used before 09:45.
- Tuning stop, target or time exit on validation and then reporting
  validation results.
- Choosing `τ` on test.
- Sizing with uncalibrated scores.
- Comparing C only with A. Always compare with B and with matched random
  retention.
- Counting `unfilled` candidates as zero-return trades.

## Connection to team guides

| Team | Use |
|---|---|
| [Momentum](../../teams/momentum.md) | SM1 and R1 are this example; the team guide covers the ORB research plan and its trade-management decisions |
| [Reversals](../../teams/reversals.md) | R2 reversal meta-labeling uses the same selectors on fade candidates (SR1, SR2) |
| [Stat arb](../../teams/stat-arb.md) | Meta-labeling applies to spread-entry signals (SS4) with the same tests |

[Example 9](09-trade-management.md) continues with the exit policy itself:
stops, targets, holding period and sizing. See also
[research/trade-management.md](../../research/trade-management.md).

## Builder notes (W4)

| Item | Detail |
|---|---|
| Folder | `workshop/04_meta_labeling_orb/` with `notebook.py`, `exercises.py`, `configs/orb_{a,b,c,d}.yaml`, `README.md` (the pre-registration) |
| Required tasks | SM1 (candidates), F1, F2, F3, F4, F8, T1 (simulator), C1b (candidate sampler), C1c (`triple_barrier` label), M1a, M1c, E1, E2 |
| Optional tasks | T2 (sizing), M2 (tracking), C1d (context symbols, if F8 does not handle SPY itself) |
| Fixture effect | C1o needs an ORB effect: after the first close above `H`, a positive drift over the next 30 minutes whose size depends on relative volume (zero below a level, positive above). Plus a context symbol for SPY. Open question for C1o: sharp threshold or smooth dependence, and default strength |
| CI check | `tests/workshop/test_04_meta_labeling_orb.py`: (1) at most one candidate per symbol per day, none before 09:45 or after 11:00; (2) every candidate's entry is the next bar's open; (3) on the C1o fixture, mean `y_net` is higher in the top relative-volume bucket than in the bottom; (4) C's validation mean `y_net` exceeds A's; (5) on the C0 fixture, C is not above the 95th percentile of matched random retention. Bands set from several seeds when W4 is built |
| Data policy | Part 5 is facilitator-only; the notebook's held-out cell is guarded by an environment variable and documented in [facilitator-guide.md](../facilitator-guide.md) |
| Runtime | Under 5 minutes on a laptop with 3 seeds per model |
| Solutions | `workshop-solutions` branch |

## Further reading

- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. Ch. 3 (triple-barrier labels and meta-labeling), ch. 10 (bet sizing).
- Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell
  System Technical Journal*.
- Repo: [studies/r1-orb-selectors/README.md](../../../studies/r1-orb-selectors/README.md),
  [teams/momentum.md](../../teams/momentum.md),
  [session 5](../sessions/05-signal-to-strategy.md),
  [research/trade-management.md](../../research/trade-management.md).
