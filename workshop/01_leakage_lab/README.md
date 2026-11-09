# Example 1: Leakage lab

**Lesson.** A leak makes a result look better than it is and raises no error. Plant each one, measure how much it
inflates the result, then catch it with the repo's checks, or learn that nothing in code catches it.

**Misconception.** "If the code runs and the backtest number is good, the pipeline is fine."

Spec and facilitator notes: [docs/workshop/examples/01-leakage-lab.md](../../docs/workshop/examples/01-leakage-lab.md).
Task: W1. Runs offline on the synthetic fixture (planted effect: weak short-horizon reversion), about 2 minutes.

## Run

```bash
uv run python workshop/01_leakage_lab/notebook.py        # whole walkthrough as a script
uv run --group workshop jupyter lab                      # or open notebook.py as a notebook
uv run pytest tests/workshop/test_01_leakage_lab.py      # CI check (about 5 minutes)
SHPE_WORKSHOP_REAL_DATA=1 uv run python workshop/01_leakage_lab/notebook.py   # adds the snapshot-universe cell
```

## Files

| File | Purpose |
|---|---|
| `notebook.py` | Walkthrough (jupytext percent). Defines the leaky helpers `CenteredMean`, `DayZScore`, `NextReturn` (not in `src/`) |
| `exercises.py` | Starter: three functions marked `# EXERCISE` that raise `NotImplementedError`. The notebook prints "EXERCISE pending" and carries on |
| `configs/clean.yaml` | Baseline hello model on fixture seed 7 |
| `configs/leaky_feature.yaml` | Same plus `{type: centered_mean, window: 11}` (loads after the notebook registers the helper) |

Solutions are on the `workshop-solutions` branch (`workshop/01_leakage_lab/solutions.py`), never on `main`.

## Parts

1. Feature look-ahead: three leaky helpers, `check_feature` catches each, the fixed version passes.
2. Preprocessing fit on all rows: the scaler leak, on stationary and on trending data, logistic and k-NN.
3. Snooping: 40 configurations, pick by test (A) or validation (B), rerun on seeds 101-110. Trials counted: 40.
4. Survivorship: 60 symbols, drop the worst 20% of window return, buy yesterday's losers, gross and net of costs.

## Exercises

| Tier | Exercise | Where |
|---|---|---|
| Core | `one_bar_lookahead`: how much does AUC rise, does the prefix check catch it? | `exercises.py` |
| Core | `day_zscore_expanding`: fix `DayZScore` with an expanding window | `exercises.py` |
| Core | `selection_gain`: snooping gain for 10, 40, 100 configurations | `exercises.py` |
| Stretch | Part 4 with "drop names whose price ends below a threshold", or random removal | take-home |
| Stretch | A check that fails when a feature depends on other symbols' rows | take-home |
| Challenge | Purged splits on a 1-day label (V1/C1f; daily bars need C1e) | take-home |
| Challenge | `deflated_sharpe` (E2) on part 3's best configuration, `n_trials = 40` | take-home |

## What to claim next

E2 (deflated Sharpe with the trial count from part 3), V1/C1f (purged walk-forward), C1m (point-in-time security
master), M2 (trial counting), C1k (add `check_feature` to your own feature PR). Board: [docs/tasks.md](../../docs/tasks.md).
