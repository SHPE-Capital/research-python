# Example 1: Leakage lab

| | |
|---|---|
| Teams | All |
| Level | Beg–Int |
| Duration | 90 min |
| Prerequisites | Sessions 1–4 |
| Data | Fixture (seed 7 and fresh seeds); the workshop snapshot `role: dev` for part 4 |
| Library tasks used | C0 (done), D4 and D5 (done), M1a; optional V1, E2, C1m |
| Build task | W1 |

## Why this example

Every leak in this lab makes a model look better than it is, and none of them
raises an error. The lab plants four of them on purpose, measures how much
each one inflates the result, and names the tool in the repo that catches
it, or admits that no tool does.

The industry analogue is a research firm's pre-publication review: before a
backtest is believed, someone checks the timestamps, the universe, the
scaling and the number of trials. The `leakage-auditor` agent
(`.claude/agents/leakage-auditor.md`) is this repo's version of that review.

## Learning objectives

1. Recognise four leak types: feature look-ahead, preprocessing fit on all
   rows, survivorship in the universe, and selection on the test set
   ("snooping").
2. For each, say which check in the repo catches it: `assert_no_lookahead`,
   the model contract, the D4 universe's `as_of`, or only the research
   protocol.
3. Measure each leak's effect as a difference between a clean run and a
   leaky run on the same rows.
4. Use a fresh fixture seed as untouched data to expose snooping.

## Background

### Four leaks

| Leak | What happens | Typical effect | Caught by |
|---|---|---|---|
| Feature look-ahead | A feature reads rows after its own `ts_available` (centered window, `shift(-k)`, whole-day statistic) | Very large: the feature contains part of the label | `assert_no_lookahead` in `features/base.py` |
| Preprocessing on all rows | A scaler, imputer or PCA is fit on train, validation and test together | Usually small on stationary data, larger with trends and regime changes | `SklearnModel` fits its `StandardScaler` inside `fit()`, on train rows only |
| Survivorship | The universe is chosen with knowledge of the period being tested, e.g. "stocks that are liquid and above $10 today" | Biases returns upward and favours buying past losers | D4 measures screens on sessions strictly before `as_of`; C1m adds a point-in-time security master |
| Snooping | Configurations are compared on the test split, and the best one is reported | Grows with the number of configurations tried | Nothing in code. The research protocol, trial counting (M2), and E2's deflated Sharpe ratio |

### Why the label overlaps a centered window

The hello model's label is the 30-minute forward log return from the
decision bar's close:

```
y_t = log(close_{t+30} / close_t)
```

A centered 11-bar mean of the close at `t` includes `close_{t+1}` …
`close_{t+5}`. Those five minutes are the first part of `y_t`, so the feature
"knows" part of the answer.

### Why survivorship biases results

Take a set of stocks over a window and keep only those that, at the end of
the window, still pass a screen such as "price above $10". The ones that fell
below are removed, so the survivors' average return over the window is
higher than the full set's. Strategies that buy recent losers look especially
good on survivors, because the losers that kept falling were removed.

### Why snooping biases results

If you try `N` configurations with no real difference between them and
report the best test score, the expected best score rises with `N` even
though nothing improved. Session 4 covered the general argument; the deflated
Sharpe ratio (Bailey and López de Prado, 2014) is one correction.

## Setup

**Data.** Fixture seed 7 (the hello model's data) for parts 1–3. Fresh fixture
seeds 101–110 for part 3's out-of-sample check. Part 4 uses a fixture with many
symbols and, as a demonstration, the snapshot's `dev` role.

**Universe.** For part 4, the workshop universe
(`configs/universe/workshop_v0.yaml`): `as_of: "2026-10-01"`, screens
measured over the 60 sessions before `as_of`.

**Config sketch** (`workshop/01_leakage_lab/configs/clean.yaml`):

```yaml
name: leak_clean
seed: 7
data: {source: fixture, symbols: [AAA, BBB, CCC, DDD, EEE, FFF],
       start: "2026-03-02", end: "2026-08-28", seed: 7}
features:
  - {type: past_return, window: 5}
  - {type: past_return, window: 30}
  - {type: minute_of_day}
label: {type: forward_return, horizon_minutes: 30, target: sign, sample_every_minutes: 15}
split: {train: 0.6, validation: 0.2, test: 0.2, gap_days: 0}
model: {type: logistic, params: {C: 1.0}}
```

`leaky_feature.yaml` is the same plus `{type: centered_mean, window: 11}`, a
feature registered inside the notebook.

## Walkthrough

### Part 1: feature look-ahead (20 min)

1. Run the clean config. Note validation AUC and daily IC t-statistic. This
   is the baseline; on the fixture it should be slightly above 0.5.
2. In the notebook, register `CenteredMean`:

   ```python
   @register_feature
   class CenteredMean(Feature):
       feature_type = "centered_mean"
       def __init__(self, window: int): self.window = window
       @property
       def name(self): return f"centered_mean_{self.window}"
       def compute(self, bars):
           c = bars["close"]
           return c.rolling(self.window, center=True).mean() / c - 1
   ```

3. Run `leaky_feature.yaml` via `run_experiment(ExperimentConfig(**cfg))`.
   **Expected:** validation AUC jumps far above the clean run, and the
   logistic coefficient on `centered_mean_11` dominates. On the fixture the
   jump is large enough that nobody would doubt it is a leak; on real data,
   a smaller leak (a one-bar shift, say) looks like a plausible edge.
4. Run `assert_no_lookahead(CenteredMean(11), one_symbol_bars)`. It fails and
   prints the first changed row. Fix the feature (`center=False`) and rerun
   the check and the config.

### Part 2: preprocessing on all rows (15 min)

1. Outside the contract, fit `StandardScaler` on all rows of `dataset.X`,
   then fit `LogisticRegression` on the scaled train rows. Compare
   validation AUC with the contract version.
2. **Expected on the fixture:** almost no difference, because the fixture's
   feature distributions do not drift. This is the point: the leak is real
   but invisible here.
3. Make it visible. Add a slow trend to the fixture's volatility (multiply
   each day's returns by `1 + 0.5 * day_index / n_days` in the notebook),
   and add a non-stationary feature such as the raw price level. Repeat.
   Now the scaler's mean and standard deviation carry information about later
   days. **Hypothesis:** the gap between leaky and clean widens; measure it.
4. Show where the repo prevents it: `SklearnModel.fit` builds a `Pipeline`
   whose scaler only sees `dataset.subset("train")`.

### Part 3: snooping (25 min)

1. Generate 40 configurations: all subsets of 5 candidate features (the
   three in the clean config plus `past_return_15` and `past_return_60`)
   crossed with `C` in `{0.01, 1.0}`, cut to 40 at random with seed 0.
2. Run all 40 on fixture seed 7. Pick:
   - **A**: the best test AUC (snooping),
   - **B**: the best validation AUC (honest selection).
3. Rerun A and B, unchanged, on fixture seeds 101–110. These are new
   histories with the same planted effect.
4. **Expected:** on seed 7, A's test AUC is the highest of the 40 by
   construction. On the fresh seeds, A's advantage over B should shrink to
   about zero, and A's mean AUC should fall back toward the typical
   configuration. The difference between A's seed-7 test AUC and its
   fresh-seed mean is the snooping bias. Report it with the spread over the
   10 seeds, not as one number.
5. Count trials: 40 here. Write it down; it is the `n_trials` that a deflated
   Sharpe ratio would need (E2).

### Part 4: survivorship (25 min)

1. **On the fixture.** Generate 60 symbols over the default window
   (symbol names `S00`…`S59`). Remove the 20% with the worst return over the
   whole window: a stand-in for "no longer listed, or below $10, at the end".
2. Compute a simple strategy on both sets: each day, buy the 10% of names with
   the worst previous-day return, hold one day. Compare the mean daily return
   on all 60 and on the 48 survivors.
3. **Expected:** the survivors' version looks better. The fixture has no
   cross-sectional effect (C1o adds one), so on all 60 the strategy's mean
   daily return should be near zero; the survivors' positive mean is entirely
   bias.
4. **On the snapshot (demonstration).** Open
   `configs/universe/workshop_v0.yaml`. The screens (median close at least
   $10, median dollar volume at least $25M, morning coverage at least 95%)
   are measured over the 60 sessions before `2026-10-01`. The snapshot covers
   2026-07-01 to 2026-09-30, which is the same window. So any backtest on the
   snapshot uses names chosen with knowledge of that period. Discuss: for
   intraday strategies that do not hold overnight this bias is smaller than for
   multi-day strategies, but it is not zero, because the screen favours names
   that stayed liquid and did not collapse. The fix is to measure the screen on
   a window that ends before the backtest starts, and C1m's point-in-time
   security master for listings and delistings.

### Wrap-up (5 min)

Fill in the table from the background with your measured numbers: clean,
leaky, difference, and which check caught it.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Replace `CenteredMean` with a one-bar look-ahead: `close.shift(-1) / close - 1`. How much does AUC rise? Does `assert_no_lookahead` catch it? | Smaller jump than the centered mean, still obvious on the fixture; the prefix check catches it at any cutoff |
| Core | Write a `DayZScore` feature (z-score of the close within the whole session). Measure the AUC change and fix it with an expanding window | The leaky version sees the day's close; the fixed version's first rows each day are NaN |
| Core | In part 3, repeat with 10 and with 100 configurations. How does the snooping bias change? | It grows with the number of configurations; plot bias against N |
| Stretch | Part 4 with "drop the bottom 20%" replaced by "drop names whose price ends below a threshold". Does the bias depend on how you remove names? | Any removal correlated with the window's returns biases results; random removal does not |
| Stretch | Implement a check that fails when a feature's values depend on rows from other symbols (cross-symbol leakage when `groupby` is forgotten) | Recompute on each symbol alone and compare with the joint computation |
| Challenge | Implement V1 purging and show, with a 1-day label on daily fixture bars, how much validation AUC falls when overlapping training rows are purged | Requires C1e for daily bars, or resample minute bars to daily in the notebook |
| Challenge | Use E2's `deflated_sharpe` (or implement it) on part 3's best configuration with `n_trials = 40` | `tests/todo/test_eval_todo.py::test_stats` |

## Discussion

- Which of the four leaks would you find by reading code? Which only by
  measuring?
- The snapshot universe was chosen on the same window as the data. Is that
  acceptable for a workshop? For a strategy you intend to trade?
- After part 3, a teammate says "A's features are still the best on validation
  too". Does that save A?
- What would a leak in the label look like? (Hint: a horizon that crosses
  the session close, or a label built from adjusted prices that use a later
  split factor.)

## Pitfalls

- Concluding from part 2 that scaling on all rows is harmless. The fixture is
  stationary; markets are not.
- Running part 3 with a single fresh seed. One seed's AUC is noisy; use ten.
- Fixing a leak by dropping the feature without rerunning the check. Always
  rerun `assert_no_lookahead` and the config.
- Treating the snapshot survivorship discussion as a measured result. Part 4
  measures the bias on the fixture only.

## Connection to team guides

| Team | Leak to watch for |
|---|---|
| [Momentum](../../teams/momentum.md) | Opening-range features computed before 09:45; RVOL baselines that include today's full volume; 12-1 momentum on a universe of today's survivors |
| [Reversals](../../teams/reversals.md) | VWAP that includes the current bar's future prints; buying losers on a survivor universe (part 4 is exactly this) |
| [Stat arb](../../teams/stat-arb.md) | Hedge ratios and cointegration tests fit on the full sample, then traded in-sample (part 2 in another form) |

## Builder notes (W1)

| Item | Detail |
|---|---|
| Folder | `workshop/01_leakage_lab/` with `notebook.py`, `exercises.py`, `configs/clean.yaml`, `configs/leaky_feature.yaml`, `README.md` |
| Required tasks | C0, D4, D5 merged (done); M1a for the random forest variant of part 3 (optional) |
| Fixture effect | The existing reversal (C0). Part 4 needs `synthetic_bars` to accept 60 symbol names, which it does; no C1o effect required |
| Notebook helpers | `CenteredMean`, `DayZScore`, `NextReturn` registered in the notebook, not in `src/` |
| CI check | `tests/workshop/test_01_leakage_lab.py`: (1) clean validation AUC within a band set from several seeds; (2) leaky AUC exceeds clean by a margin; (3) `assert_no_lookahead` raises on `CenteredMean`; (4) survivors' mean return exceeds the full set's on the fixed seed. Bands are measured when W1 is built and written in the test, not guessed |
| Runtime | Part 3 runs 40 + 20 logistic fits; keep under 2 minutes on a laptop |
| Solutions | `workshop-solutions` branch, `workshop/01_leakage_lab/solutions.py` |

## Further reading

- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. Ch. 7 (cross-validation in finance, purging and embargo).
- Bailey, D. H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
  *Journal of Portfolio Management*.
- Repo: [session 2](../sessions/02-features-without-leakage.md),
  [session 4](../sessions/04-honest-evaluation.md),
  [research-protocol.md](../../research/research-protocol.md).
