# Session 3: Models under the contract

| | |
|---|---|
| Level | Beg–Int |
| Duration | 75 min |
| Prerequisites | Session 2 |
| Data | Fixture |
| Code used | `contracts/model.py`, `contracts/dataset.py`, `models/sklearn_base.py`, `models/logistic.py`, `experiments/runner.py` |
| Ends with a PR for | One of M1a (random forest), M1b (LightGBM), M1c (small neural network) |

## Why this session

When every model takes the same dataset and writes the same report, comparing
them is a config change, not a new notebook. The contract also carries the
rules that keep models honest: scaling fitted on training rows only, early
stopping on a slice of training, seeds recorded.

## Learning objectives

1. Describe the `Dataset` contract: `X`, `y` and `meta` with `symbol`,
   `decision_ts`, `label_end_ts`, `weight` and `split`.
2. Add a model by subclassing `SklearnModel` and registering it.
3. Run an experiment from YAML and read every file it writes.
4. Explain why scaling and early stopping must only see training rows.

## Background

### From config to report

```
config.yaml ─▶ load bars ─▶ features ─▶ labels ─▶ join on decision_ts
           ─▶ sample every N minutes ─▶ drop NaN ─▶ split by session
           ─▶ model.fit(train) ─▶ predict all rows ─▶ report per split
```

`run_experiment` in `experiments/runner.py` writes a run folder:

| File | Contents |
|---|---|
| `config.yaml` | The exact config, so the run can be repeated |
| `model/` | Saved model state and its manifest (type, params, seeds, feature names, label, train dates) |
| `predictions.parquet` | One row per sample: meta, score, label, split |
| `report.json`, `report.md` | Metrics for train, validation and test |

### The model contract

`Model` in `contracts/model.py` has `fit(dataset)`, `predict(X)`, `save` and
`load`. `fit` uses only rows with `split == "train"`. Models register by
`model_type`:

```python
@register_model
class LogisticModel(SklearnModel):
    model_type = "logistic"
    task = "classification"
    def make_estimator(self):
        return LogisticRegression(C=float(self.params.get("C", 1.0)), ...)
```

`SklearnModel` wraps the estimator in a `Pipeline` whose `StandardScaler` is
fitted inside `fit()`, so it never sees validation or test rows. Tree models
set `scale = False`.

### Splits

`split_by_day` assigns whole sessions, in time order: first 60% train, next
20% validation, last 20% test by default. `gap_days` leaves sessions between
splits as `unassigned`. Labels never cross the session close, so a day-level
split has no overlap between train and validation rows. Multi-day labels need
purging and an embargo (V1, session 4).

## Walkthrough

1. **Run the hello model and open every output:**

   ```bash
   uv run shpe run configs/experiments/hello_model.yaml
   ```

   In `report.md`, the validation AUC should be slightly above 0.5. The fixture
   plants a reversal: each minute's return leans against the previous five
   minutes' sum, with strength `k = 0.3`. A 30-minute label only partly
   reflects a 5-minute effect, so the edge is small. Check the logistic
   coefficient on `past_return_5m` in the saved model: it should be negative.

2. **Turn the signal off.** Copy the config, add `signal_strength: 0` under
   `data:` (the fixture source in `experiments/sources.py` passes `seed`
   and `signal_strength` through to `synthetic_bars`), and rerun. The AUC
   should now be 0.5 within noise. This is the simplest negative control.

3. **Add a model live.** Show the M1a stub in `models/random_forest.py`, write
   `make_estimator` with `RandomForestClassifier(n_estimators=..., max_depth=...,
   random_state=self.seed)`, set `scale = False`, register it, and run:

   ```bash
   uv run pytest tests/todo/test_models_todo.py -k random_forest
   ```

   Then change one line in the config: `model: {type: random_forest, params:
   {n_estimators: 300, max_depth: 4}}`.

4. **Break scaling on purpose.** In a notebook, fit a `StandardScaler` on all
   rows, then train the logistic model on the scaled train rows. On this
   fixture the effect is tiny, because the feature distribution barely
   changes over time. Discuss why it is still a rule: on real data with
   trends and volatility regimes, full-sample moments carry information
   about the future.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Change `sample_every_minutes` from 15 to 1 and rerun. What happens to the row count, the AUC and the IC t-statistic? | 15× more rows, but neighbouring rows' 30-minute labels overlap heavily, so they are not independent; the t-statistic overstates confidence |
| Core | Change the horizon from 30 to 5 minutes. Does the AUC go up? | Hypothesis: yes, because the planted effect acts over five minutes; confirm on the fixture |
| Stretch | Add `gap_days: 1` and explain why it changes nothing much for these labels | Labels end within the session, so there's no overlap to remove |
| Stretch | Implement M1b and set early stopping to use the last sessions of train. Show the difference when early stopping uses validation instead | Validation AUC rises slightly, but it's no longer an honest estimate |
| Challenge | Implement M1c with a chronological stopping slice (sklearn's `early_stopping=True` picks a random slice; explain why that is wrong here) | Random slices put future minutes in the stopping set |

## Discussion

- Two models have validation AUCs of 0.523 and 0.527. Which is better?
  (Session 4 answers this properly.)
- Why does the repo save the model manifest with seeds and training dates?
- What's wrong with picking the best of 20 random seeds?

## Pitfalls

- Calling `fit` on the whole `Dataset` frame outside the contract, which
  trains on all splits.
- Early stopping on validation.
- Comparing models trained on different rows (different features cause
  different NaN drops). Check `n_rows` in each report.

## Connection to team strategies

R1 (ORB selectors, `studies/r1-orb-selectors/`) compares four selectors
with the same model contract: A (all), B (rule), C (random forest), D (MLP).
Every team study uses `shpe run` configs this way. See
[research-protocol.md](../../research/research-protocol.md).

## Wrap-up PR

Claim M1a, M1b or M1c. Move the acceptance test parameter out of
`tests/todo/` and make it pass.
