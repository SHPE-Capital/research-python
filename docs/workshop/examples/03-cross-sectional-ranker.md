# Example 3: Cross-sectional ranker

| | |
|---|---|
| Teams | Momentum (primary), Reversals; Stat arb for the neutralisation part |
| Level | Int–Adv |
| Duration | 120 min (core 90 min) |
| Prerequisites | Sessions 1–4; examples 1 and 2 |
| Data | Snapshot `role: dev`, daily bars (`1Day`, split-adjusted), for the reversal core; a long-history fixture (C1o) or a long daily history (C1e/C1g) for 12-1 momentum |
| Library tasks used | C1a, C1b, C1d, C1e, L1, V1, E1, M1b, SM3; optional C1i, C1m, E2 |
| Build task | W3 |

## Why this example

Most intraday examples predict one symbol at a time. Much of equity
research instead asks a **relative** question: on each date, which stocks
will do better than the others? The answer is a ranking, the trade is a
long-short portfolio of the top and bottom of the ranking, and the
evaluation is the rank correlation between score and forward return on each
date (the information coefficient, IC).

This example replicates two well-documented signals, short-term reversal and
12-1 momentum, and then combines several signals into one ranking, first by
equal-weighting z-scores and then with a learned model. The industry analogue
is a factor-research pipeline: build signals, transform them cross-sectionally,
evaluate by IC and decile spreads, check turnover and costs, then combine.
ML ranking models in published academic work (for example Gu, Kelly and
Xiu, 2020) sit at the end of this pipeline, not in place of it.

## Learning objectives

1. Build daily signals as features with cross-sectional transforms (rank,
   z-score, winsorising) per date.
2. Evaluate a signal with IC by date, its t-statistic, decile spreads and
   turnover, and say why each matters.
3. Handle labels that span several days: overlap, purging and embargo (V1).
4. Combine signals by equal-weight z-scores, ridge regression and LightGBM,
   and judge whether the learned combination is worth it.

## Background

### Signals

With `P_{i,t}` the split-adjusted close of stock `i` on date `t` and
`r_{i,t} = log(P_{i,t} / P_{i,t−1})`:

| Signal | Formula | Known at | Literature |
|---|---|---|---|
| Short-term reversal (5 days) | `STR_{i,t} = −Σ_{k=0..4} r_{i,t−k}` | close of `t` | Jegadeesh (1990); Lehmann (1990) |
| 12-1 momentum | `MOM_{i,t} = log(P_{i,t−21} / P_{i,t−252})` | close of `t` | Jegadeesh and Titman (1993) |
| Volatility (optional) | standard deviation of `r_{i,t−k}` over 20 days | close of `t` | |
| Distance from 20-day high (optional) | `log(P_{i,t} / max_{k<20} P_{i,t−k})` | close of `t` | |

12-1 momentum skips the most recent month (21 sessions) because the most
recent month tends to reverse. It needs at least 253 sessions of history
before the first usable date.

### Cross-sectional transforms (C1d)

For a raw signal `x_{i,t}` across the `N_t` stocks on date `t`:

```
rank:     u_{i,t} = (rank_t(x_{i,t}) − 0.5) / N_t            # in (0, 1)
z-score:  z_{i,t} = (x_{i,t} − mean_t(x)) / std_t(x), clipped at ±3
neutral:  residual of x_{i,t} on sector dummies (or on beta) within date t
```

Transforms use **only date `t`'s cross-section**. A z-score that uses the
mean over all dates is a leak (example 1, part 2).

### Label

The label is the forward return from the decision to `h` sessions later,
often made relative to the cross-section:

```
y_{i,t} = log(P_{i,t+h} / P_{i,t}) − mean_t(log(P_{·,t+h} / P_{·,t}))
```

If the strategy trades at the next open, measure from the next open, not from
the close at which the signal was computed. With `h > 1`, labels on
consecutive dates overlap: today's 5-day label shares four days with
tomorrow's. That has two consequences:

- **Splits need purging and an embargo.** V1 drops training rows whose
  `[decision_ts, label_end_ts]` overlaps the validation block, plus an
  embargo after it (López de Prado, 2018, ch. 7).
- **The IC t-statistic is overstated** if computed on every date, because
  consecutive IC values share returns. Use non-overlapping dates (every
  `h`-th date) or a block bootstrap with blocks of at least `h` dates.

### Evaluation

| Measure | Formula | What it answers |
|---|---|---|
| IC by date | `IC_t = Spearman(s_{·,t}, y_{·,t})` | Does the ranking order returns on this date? |
| Mean IC, ICIR, t | `mean(IC)`, `mean/std`, `mean/std · √n` over non-overlapping dates | Is it consistent? |
| Decile spread | Per date: mean `y` of the top decile minus bottom decile; then averaged | Is the edge in the tails you would trade? |
| Monotonicity | Mean `y` by decile, pooled across dates within date | Is the ranking informative throughout, or only at one end? |
| Turnover | `½ Σ_i |w_{i,t} − w_{i,t−1}|` for the long-short portfolio | How much trading the signal needs; drives costs |
| Cost-adjusted spread | Spread minus turnover × cost per side × 2 | Does it survive costs? (E1) |

The current report's daily IC (`evaluation/report.py`, `MIN_ROWS_PER_DAY = 5`)
already groups by date, so with one row per stock per date it is the
cross-sectional IC. Its decile table is pooled over all rows, not formed per
date; per-date deciles and turnover come from E1 and SM3.

## Setup

**Data.**

| Part | Source | Why |
|---|---|---|
| 1–2 (reversal) | Snapshot `role: dev`, `timeframe: 1Day`, `adjustment: split` | Identical real data for everyone. About 63 sessions and the dev names (48 if every cell fills); small, which is part of the lesson |
| 3 (momentum) | Long-history fixture with a planted cross-sectional momentum effect (C1o), or C1e/C1g daily history over several years | The snapshot is too short for 12-1 momentum |
| 4 (combination) | Same as part 3 | Needs several signals with enough dates |

A long real history built from today's symbol list has survivorship bias
(example 1, part 4). Until C1m provides a point-in-time security master, say
so in every result built on it.

**Universe.** Snapshot dev names (D4, `configs/universe/workshop_v0.yaml`).
SPY is in the snapshot as a `context` symbol, not a ranked name.

**Config sketch** (`workshop/03_cross_sectional_ranker/configs/str.yaml`):

```yaml
name: xs_str_5d
seed: 1
frequency: daily                                  # C1e
data: {source: snapshot, path: snapshots/workshop-v0.zip, role: dev,
       timeframe: 1Day, adjustment: split}
sampler: {type: every_session_close}              # C1b decision points
features:
  - {type: past_return_days, window: 5, sign: -1, transform: zscore}   # C1d transform
label:
  type: forward_return_days                      # L1 / C1c
  horizon_sessions: 5
  relative_to: cross_section_mean
  entry: next_open
target: value
fit: {scheme: purged_walk_forward, n_folds: 4, embargo_sessions: 5}   # V1 / C1f
model: {type: identity}                           # score = the signal itself; no fitting
report: [ic_by_date, deciles_by_date, turnover]   # C1a report sections, E1
```

Field names depend on C1a–C1e and L1 as merged; the sketch shows what each
part needs. `model: identity` (a pass-through "model" that returns one
feature as the score) is a small addition for W3 if C1a does not provide it.

## Walkthrough

### Part 1: one signal, evaluated properly (30 min)

1. Load the snapshot's dev daily bars. Plot the number of names per date and
   check there are no gaps (D2's `audit_bars` on daily bars).
2. Compute `STR` for each name and date, z-score it per date, and form the
   label `y` with `h = 5`, entry at the next open, demeaned per date.
3. Compute IC by date. Report mean IC, ICIR and the t-statistic **twice**: on
   all dates, and on every fifth date. **Expected:** the all-dates t-statistic
   is larger, by roughly `√5` if consecutive ICs were perfectly correlated.
   Explain why the every-fifth-date number is the honest one.
4. Compute deciles per date and the top-minus-bottom spread. With about 48
   names, a decile is about 5 names, so use quintiles if deciles are too thin.
5. Compute turnover of an equal-weight long-short portfolio (long top
   quintile, short bottom quintile), rebalanced daily and every 5 days.
6. **Expected:** the result is a hypothesis with a wide interval. Three months
   of data on fewer than 50 names cannot establish a 5-day reversal; it can
   show whether the pipeline produces sensible numbers. Write the interval
   from a block bootstrap of the non-overlapping IC values.

### Part 2: costs and holding period (15 min)

1. Repeat part 1 for `h` = 1, 5, 10 (L1 horizons). Plot mean IC and turnover
   against `h`.
2. Apply 2, 5 and 10 bps per side (E1 cost grid). **Expected:** shorter
   horizons have higher turnover, so costs remove more of their spread.
   Reversal signals are known for high turnover; the break-even cost is the
   thing to measure.

### Part 3: replicate 12-1 momentum (20 min)

1. On the long-history data, compute `MOM`, z-score per date, `h = 21`
   (one month), sampled monthly so labels do not overlap.
2. Report mean IC, t-statistic and decile spreads by year.
3. **Expected on a C1o fixture with planted momentum:** a positive mean IC
   whose size matches the planted strength (the fixture's CI test fixes the
   band). **On real data:** a hypothesis; published momentum results come
   from much broader universes over decades, and the workshop universe is
   small, liquid and survivor-biased.
4. Note momentum's known weakness: sharp losses when the market rebounds
   after a decline. Show the worst three months in the sample.

### Part 4: combine signals (25 min)

1. Put `STR`, `MOM`, volatility and distance from the 20-day high on the same
   rows (same dates, same names, same label `h = 5`), each z-scored per date.
2. Three combinations, all fitted only on training folds with V1 purging and
   a 5-session embargo:

   | Combination | Score | Fit |
   |---|---|---|
   | Equal weight | `s = mean_j z_j`, with each signal's sign set from the literature, not from the data | None |
   | Ridge | `s = Σ_j β_j z_j`, `β` from ridge regression of `y` on `z` | Training folds |
   | LightGBM | `lightgbm_reg` (M1b) on the z-scores or ranks | Training folds; early stopping on the last training sessions |

3. Report out-of-fold mean IC, t-statistic on non-overlapping dates, spread,
   and turnover for each. Then the paired block bootstrap of the daily IC
   difference: ridge vs. equal weight, LightGBM vs. equal weight.
4. **Expected:** equal weight is a hard baseline to beat. A learned
   combination that wins on IC may lose after costs because its turnover is
   higher. Ridge's coefficients are a useful check: a negative weight on a
   signal whose sign is well established is more likely noise than discovery.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Compute the IC t-statistic with overlapping and non-overlapping dates for `h = 10` and explain the ratio | Overlap of 9 days makes consecutive ICs highly correlated |
| Core | Replace the z-score with a rank transform. Does the IC change? Does the spread? | Spearman IC is unchanged by any monotone transform of the score; the combination step and the spread can change |
| Core | Remove the per-date demeaning from the label. What happens to the IC? | Little, since the IC is computed within dates; the long-only decile returns now include the market's move |
| Stretch | Neutralise `STR` against sector (or against beta to SPY) before ranking. How do IC and turnover change? | Uses C1d `neutralize`; on a small universe sectors are thin, so expect noise |
| Stretch | Show what happens without purging in part 4: train the ridge on all training rows including those whose labels overlap validation | Validation IC rises slightly; the purged number is the honest one |
| Stretch | Add a ridge model to the registry (`ridge`, `ridge_reg`) following `models/logistic.py` | Small PR; it may already arrive with C1i |
| Challenge | Implement SM3 (`strategies/momentum/xs_momentum.py`): monthly 12-1 momentum, long top quintile, short bottom, with turnover and costs; acceptance test on the C1o fixture | Team task for Momentum |
| Challenge | Run a LightGBM ranker with a ranking objective (`lambdarank`, query = date) and compare with regression on demeaned returns | Ranking objectives match the evaluation; gains on small universes are a hypothesis to test |

## Discussion

- Why evaluate within dates instead of across all rows?
- The snapshot shows a strong reversal IC with a t-statistic of 3 on all
  dates. What do you check before believing it?
- When is an equal-weight combination better than a fitted one? Think about
  the number of dates relative to the number of signals.
- How would you combine an intraday signal (from example 2) with a daily
  ranking signal?

## Pitfalls

- Z-scoring over all dates instead of within each date.
- Using the close at which the signal is computed as the entry price.
- Overlapping labels without purging, and t-statistics on overlapping dates.
- Momentum on survivors: today's listed names over the past five years.
- Deciles of 3 names. Use quintiles or terciles on small universes.
- Choosing signal signs from the same data used to evaluate the combination.

## Connection to team guides

| Team | Use |
|---|---|
| [Momentum](../../teams/momentum.md) | SM3 cross-sectional 12-1 momentum is part 3 turned into a strategy; momentum crash checks |
| [Reversals](../../teams/reversals.md) | Short-term reversal across names is the daily counterpart of the intraday fades (SR1–SR3); compare turnover and cost sensitivity |
| [Stat arb](../../teams/stat-arb.md) | Neutralisation against sector or SPY (C1d) is the first step toward residual and pairs strategies; [example 7](07-unsupervised-pairs.md) continues it |

[Example 8](08-portfolio-construction.md) turns the ranking into weights.

## Builder notes (W3)

| Item | Detail |
|---|---|
| Folder | `workshop/03_cross_sectional_ranker/` with `notebook.py`, `exercises.py`, `configs/str.yaml`, `configs/mom.yaml`, `configs/combine_{eq,ridge,lgbm}.yaml`, `README.md` |
| Required tasks | C1a (report sections), C1b (session-close sampler), C1d (transforms), C1e (daily frequency), L1 (multi-day horizons), V1 (purged walk-forward), E1 (per-date deciles, turnover, costs), M1b (`lightgbm_reg`), SM3 for the challenge |
| Optional tasks | C1i (ridge, baselines), C1m (point-in-time universe), E2 (block bootstrap) |
| Fallback | Until C1a–C1e land, parts 1–2 run in the notebook with pandas on the snapshot's daily bars, using the report's `daily_ic` logic. Mark the fallback cells so they can be replaced |
| Fixture effect | C1o must provide a long daily history (at least 3 years) for at least 50 symbols with a planted cross-sectional momentum effect: each symbol's daily drift depends on its own 12-1 past return with a set strength. Open question for C1o: the exact form and default strength |
| CI check | `tests/workshop/test_03_cross_sectional_ranker.py`: (1) per-date z-scores have mean 0 and std 1 within each date; (2) on the C1o fixture, the momentum IC is positive and within a band set from several seeds; (3) with purging, no training row's `label_end_ts` falls inside the validation block; (4) the non-overlapping t-statistic is smaller than the overlapping one |
| Runtime | Daily data is small; under 2 minutes |
| Solutions | `workshop-solutions` branch |

## Further reading

- Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security
  Returns." *Journal of Finance*.
- Lehmann, B. N. (1990). "Fads, Martingales, and Market Efficiency."
  *Quarterly Journal of Economics*.
- Jegadeesh, N. and Titman, S. (1993). "Returns to Buying Winners and Selling
  Losers: Implications for Stock Market Efficiency." *Journal of Finance*.
- Grinold, R. C. and Kahn, R. N. (1999). *Active Portfolio Management*, 2nd
  ed. McGraw-Hill. (Information coefficient and information ratio.)
- Gu, S., Kelly, B. and Xiu, D. (2020). "Empirical Asset Pricing via Machine
  Learning." *Review of Financial Studies*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. Ch. 7.
- Repo: [teams/momentum.md](../../teams/momentum.md),
  [example 8](08-portfolio-construction.md).
