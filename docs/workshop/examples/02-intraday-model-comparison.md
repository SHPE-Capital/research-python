# Example 2: Intraday model comparison

| | |
|---|---|
| Teams | All; most useful for Reversals and Momentum |
| Level | Int |
| Duration | 90 min (core); stretch and challenge as take-home |
| Prerequisites | Sessions 3 and 4; example 1 |
| Data | Fixture (seed 7 plus fresh seeds); optionally the snapshot `role: dev` |
| Library tasks used | M1a, M1b, M1c, L1, F1, F2, F3, F5, C1f or V1; optional M2, E1, E2 |
| Build task | W2 |

## Why this example

"Which model is best?" is the first question most people ask and usually the
least important. On tabular minute data with a weak signal, a regularised
linear model is often as good as a random forest, gradient boosting or a
neural network, and differences between them are often smaller than
differences between seeds or between days. This example compares four
model families on **the same features, rows and splits**, and asks whether
any difference survives the uncertainty measured in session 4.

The industry analogue is a model bake-off with a fixed dataset and a fixed
evaluation harness: the harness is shared, so only the model changes, and the
decision depends on whether the improvement is larger than the noise, and
worth the extra complexity to maintain and deploy (M3 exports to ONNX).

## Learning objectives

1. Run four model families from config changes only, through the model
   contract.
2. Report each model's validation result as mean ± spread over seeds, and
   per walk-forward fold.
3. Compare two models with a paired block bootstrap of the daily difference,
   not by eye.
4. Explain why deep sequence models rarely help on tabular minute features,
   and when they might.

## Background

### The four families

| Family | Registry name | Task | What it can represent | Main knobs |
|---|---|---|---|---|
| Logistic regression | `logistic` | done (C0) | Linear in the features | `C` |
| Random forest | `random_forest` | M1a | Interactions and thresholds; averages many deep-ish trees | `n_estimators` (500), `max_depth` (4/8/12), `min_samples_leaf` (20/50/100) |
| Gradient boosting | `lightgbm` | M1b | Interactions and thresholds; adds shallow trees sequentially | `n_estimators`, `learning_rate`, `num_leaves`, early stopping on the last sessions of train |
| Small MLP | `mlp` | M1c | Smooth non-linear functions | hidden sizes, weight decay, up to 200 epochs with patience 20 on a chronological stopping slice |

Tree models set `scale = False`. The MLP and logistic models scale inside
`fit()`.

### What the fixture plants

The fixture's minute returns follow

```
r_t = e_t − (k/5) · (r_{t−1} + … + r_{t−5}),   k = 0.3
```

This effect is **linear** in the last five minutes' returns. So the
hypothesis for the fixture is: no model beats logistic regression by more
than the noise, because there is nothing non-linear to find. A model that
does beat it clearly on validation, but not on fresh seeds, is fitting noise.
Part 3 adds a planted non-linear effect to see whether flexible models then
win.

### Measuring the difference

For two models on the same validation days, compute a daily metric for
each (daily IC, or daily mean of `score`-signed `outcome`), take the daily
difference `d_t = m_t(B) − m_t(A)`, and bootstrap the mean of `d_t` in blocks
of consecutive days (E2's `block_bootstrap`, or a ten-line version in the
notebook). If the interval contains zero, there is no evidence that B is
better. Bootstrapping each model separately and comparing intervals throws
away the pairing and is much less sensitive.

### Sources of variance

| Source | How to vary it |
|---|---|
| Model seed | `seed:` in config: 1, 2, 3 |
| Which days | Walk-forward folds (C1f or V1): e.g. 5 folds of about 25 sessions |
| Which history | Fixture `data.seed`: 7, 8, 9 (fixture only; real data has one history) |

## Setup

**Data.** Fixture seed 7, default dates 2026-03-02 to 2026-08-28, symbols
AAA–FFF. Optionally the snapshot `role: dev` for the take-home.

**Features** (the same for every model):

```yaml
features:
  - {type: past_return, window: 5}
  - {type: past_return, window: 15}
  - {type: past_return, window: 30}
  - {type: minute_of_day}
  - {type: relative_volume, window_days: 20}     # F1
  - {type: vwap_distance}                        # F2
  - {type: realized_vol, window: 30}             # F3
  - {type: regime_set}                           # F5
```

Use the parameter names the merged F-tasks define; the sketch above shows
intent, not final names. If an F-task is not merged, drop its line for every
model. Never let the feature list differ between models.

**Label.** `forward_return`, horizon 30 minutes, `target: sign`,
`sample_every_minutes: 15`. With L1 merged, add horizons 5 and 15 in part 2.

**Config sketch** (`workshop/02_intraday_model_comparison/configs/base.yaml`):

```yaml
name: cmp_base
seed: 1
data: {source: fixture, symbols: [AAA, BBB, CCC, DDD, EEE, FFF],
       start: "2026-03-02", end: "2026-08-28", seed: 7}
features: [...]                     # as above
label: {type: forward_return, horizon_minutes: 30, target: sign, sample_every_minutes: 15}
split: {train: 0.6, validation: 0.2, test: 0.2}
model: {type: logistic, params: {C: 1.0}}
```

Each model is one override file:

```yaml
# rf.yaml
model: {type: random_forest, params: {n_estimators: 500, max_depth: 8, min_samples_leaf: 50}}
# lgbm.yaml
model: {type: lightgbm, params: {n_estimators: 2000, learning_rate: 0.02, num_leaves: 15,
                                 early_stopping_sessions: 10}}
# mlp.yaml
model: {type: mlp, params: {hidden: [32, 16], weight_decay: 0.001}}
```

Parameter names follow each M1 task's implementation; the notebook reads
them from the merged model's docstring.

**Pre-register before running.** Write in the notebook: the four configs,
the seeds, the folds, and the comparison rule ("B replaces logistic only if
the paired bootstrap interval of the daily IC difference is above zero on
validation, across all three fixture seeds"). Hyperparameter search is not
part of this example; one sensible setting per family.

## Walkthrough

### Part 1: one split, three seeds (30 min)

1. Run the four configs with model seeds 1, 2, 3 (12 runs). Check `n_rows`
   is identical in every report; if not, a feature produced different NaNs
   and the comparison is invalid.
2. Fill this table from the reports:

   | Model | Val AUC (mean ± range over seeds) | Val daily IC t-stat | Train AUC | Fit time |
   |---|---|---|---|---|

3. **Expected on the fixture:** all four validation AUCs slightly above 0.5,
   with overlapping ranges. Logistic has zero seed variance (its solver is
   deterministic). The random forest and boosting may show a much higher train
   AUC than validation AUC; that gap is overfitting, not skill.
4. Open the logistic coefficients and the tree models' feature importances.
   **Expected:** `past_return_5m` matters most, with a negative coefficient.
   Features with no planted effect (relative volume, VWAP distance, regime
   set) should get coefficients near zero and low importance. If a flexible
   model gives them high importance, that is fitting noise.

### Part 2: folds and the paired bootstrap (30 min)

1. Switch the fit scheme to walk-forward (C1f) or V1's
   `purged_walk_forward` with 5 folds. Report validation AUC per fold for
   each model. Count on how many folds each model beats logistic.
2. Compute daily IC per model on the pooled out-of-fold predictions. For each
   non-linear model, bootstrap the daily difference against logistic with
   block length 5 sessions and 10,000 resamples.
3. **Expected:** intervals that include zero for all three. Writing "random
   forest 0.527 vs. logistic 0.523" without this interval is the mistake this
   part is meant to prevent.
4. Repeat steps 1–3 on fixture seeds 8 and 9 (`data.seed`). If a model wins on
   one history but not on others, it did not find the planted effect any
   better; it found that history's noise.

### Part 3: plant a non-linear effect (20 min)

1. In the notebook, generate a variant fixture where the reversal only exists
   when the 30-minute realized volatility is above its median: compute
   returns with `k = 0.6` in high-volatility stretches and `k = 0` otherwise.
   Keep the overall average strength close to the original.
2. Rerun logistic and LightGBM, three seeds each, folds as in part 2.
3. **Hypothesis:** LightGBM, which can learn the interaction between
   `past_return_5m` and `realized_vol_30m`, now beats logistic by more than
   the bootstrap interval. Add the interaction term
   `past_return_5m × realized_vol_30m` to the logistic model as a feature and
   check whether that closes the gap.
4. The lesson: flexible models pay off when the signal has a shape that a
   linear model cannot express, and a well-chosen feature often recovers most
   of it.

### Wrap-up (10 min)

Each pair writes the decision as a short paragraph: which model would you
take forward, on what evidence, and what it costs (fit time, export through
M3, explainability).

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Add `min_samples_leaf` 20 vs. 100 for the random forest. Which has the smaller train–validation gap? | Larger leaves regularise more; the gap should shrink, validation may not move |
| Core | Rerun part 1 with `signal_strength: 0`. Do any of the four models "find" something? | They should not. Any model that shows a positive validation edge on pure noise across seeds indicates a pipeline problem |
| Core | Run LightGBM with early stopping on validation instead of the end of train, and compare the reported validation AUC | It rises slightly and is no longer an honest estimate (session 3) |
| Stretch | With L1 merged, repeat part 1 for horizons 5, 15 and 30. Plot AUC against horizon per model | Hypothesis: the edge is strongest at the 5-minute horizon, where the planted effect acts |
| Stretch | Calibrate the best model's probabilities (Platt or isotonic, fit on validation, report on test once). Plot a reliability curve | Calibration matters for sizing (T2, example 4); AUC does not change |
| Stretch | Track all runs with M2 (MLflow) and record `n_trials` for this example | Every run counts, including discarded ones |
| Challenge | Build an LSTM or 1-D CNN on raw sequences of the last 60 minutes' returns and volumes. It needs a sequence dataset adapter that the M1 contract does not have yet (`X` is a flat table). Compare against logistic under the same splits | Hypothesis: no improvement on the fixture, since five lags carry the whole signal; much longer fit time. Write the adapter in the notebook and propose a contract extension as an issue |
| Challenge | Run the comparison on the snapshot `role: dev` with the same four configs. Report the same tables and the paired bootstrap | Do not open `held_out`. Results are hypotheses for the team, not conclusions |

## Discussion

- Logistic and LightGBM are tied on validation. Which would you deploy, and
  why?
- A neural net gives the best validation AUC with seed 3 only. What should
  you report?
- When would you expect sequence models to help in intraday equity
  research? Think about inputs that are not already summarised in features
  (order-book snapshots, many symbols at once, long histories).
- Feature importances disagree between the random forest and LightGBM. Does
  that matter?

## Pitfalls

- Different feature lists or row counts between models.
- Tuning one model for an hour and the others for five minutes, then comparing.
- Reporting the best seed, or the best fold.
- Using sklearn's `MLPClassifier(early_stopping=True)`, which picks a random
  stopping slice and so trains on future minutes relative to the slice. M1c
  requires a chronological slice.
- Reading high train AUC as skill.
- Comparing models by overlapping confidence intervals instead of a paired
  test.

## Connection to team guides

| Team | Use |
|---|---|
| [Momentum](../../teams/momentum.md) | R1 versions C and D (random forest vs. MLP selectors) are this comparison on ORB candidates |
| [Reversals](../../teams/reversals.md) | R2 reversal meta-labeling: the same bake-off on fade candidates; the fixture's planted effect is a reversal |
| [Stat arb](../../teams/stat-arb.md) | Less direct; the paired-bootstrap comparison applies to any two spread signals |

## Builder notes (W2)

| Item | Detail |
|---|---|
| Folder | `workshop/02_intraday_model_comparison/` with `notebook.py`, `exercises.py`, `configs/base.yaml`, `configs/{logistic,rf,lgbm,mlp}.yaml`, `README.md` |
| Required tasks | M1a, M1b, M1c merged; F1, F2, F3, F5 merged or dropped consistently; C1f or V1 for part 2. Without them, part 2 falls back to three chronological splits made in the notebook |
| Optional tasks | L1 (horizons stretch), M2 (tracking), E2 (`block_bootstrap`; the notebook has a fallback), E1 |
| Fixture effect | Existing reversal for parts 1–2. Part 3's volatility-conditional reversal is generated in the notebook. If C1o adds a `conditional` effect option, use it instead |
| Extras | `uv sync --extra ml` (LightGBM). The LSTM challenge needs `--extra nn` (torch) |
| CI check | `tests/workshop/test_02_intraday_model_comparison.py`: (1) every model's run has identical `n_rows`; (2) logistic validation AUC within its band; (3) on `signal_strength: 0`, every model's validation AUC is within a noise band around 0.5; (4) in part 3, LightGBM's mean validation AUC exceeds logistic's. Bands are measured from several seeds when W2 is built |
| Runtime | 12 runs in part 1 plus folds; keep LightGBM `n_estimators` moderate and the MLP small so the notebook runs in under 5 minutes on a laptop |
| Solutions | `workshop-solutions` branch |

## Further reading

- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. Ch. 7 (cross-validation), ch. 8 (feature importance).
- Repo: [session 3](../sessions/03-models-under-the-contract.md),
  [session 4](../sessions/04-honest-evaluation.md),
  `src/shpe_research/models/` docstrings for M1a–c,
  [research-protocol.md](../../research/research-protocol.md).
