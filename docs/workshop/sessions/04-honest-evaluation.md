# Session 4: Honest evaluation

| | |
|---|---|
| Level | Int |
| Duration | 90 min |
| Prerequisites | Session 3; at least the logistic model and one of M1a–M1c merged |
| Data | Fixture, several seeds |
| Code used | `evaluation/report.py`; stubs `evaluation/trading.py` (E1), `evaluation/stats.py` (E2), `validation/purged.py` (V1), `labels/multi_horizon.py` (L1) |
| Ends with a PR for | L1, V1 or E1 (E2 for Adv members) |

## Why this session

On minute data, a real edge usually shows up as an AUC only a little above
0.5 (the original workshop plan used 0.53 as its example). An edge that small
is easy to fake by accident and easy to lose to noise. This session answers:
"How sure are we, and is it worth trading?"

## Learning objectives

1. Read every metric in the report: AUC vs. base rate, daily IC and its
   t-statistic, deciles.
2. Explain why the test split is opened once, and what "trial counting"
   means.
3. Estimate uncertainty with seeds, folds and a block bootstrap, not with a
   single number.
4. Turn a score into a cost-adjusted P&L and find the cost at which the
   edge disappears.

## Background

### The report's three layers

| Layer | Metric | What it answers |
|---|---|---|
| Prediction quality | AUC, accuracy vs. base rate | Does the score order outcomes at all? |
| Ranking quality | Daily IC: Spearman correlation of score and realized return across each day's rows; mean, std and t-stat across days | Is the ordering consistent day after day? |
| Economic shape | Mean forward return by score decile | Is the edge in the tails, where you would trade? |

The daily IC is computed per session because rows within a day share
market conditions. Days are closer to independent than rows are. The report
skips days with fewer than 5 rows (`MIN_ROWS_PER_DAY`).

### Validation vs. test

| Split | Use it to | Never |
|---|---|---|
| Train | Fit parameters | Report it as performance |
| Validation | Choose models, features, thresholds; compare everything | — |
| Test | One final check of the chosen configuration | Choose anything with it |

The report says so in a note: treat the test column as opened once. A test
result that is looked at and then acted on becomes a second validation set.

### Trial counting

Every configuration you try is a trial, including the ones you deleted. With
enough trials, the best one looks good by luck. The deflated Sharpe ratio
(Bailey and López de Prado, 2014) adjusts an observed Sharpe ratio for the
number of trials, and for skew and kurtosis of returns. It needs an honest
`n_trials`, which is why M2 records every run in MLflow. The probability of
backtest overfitting (Bailey, Borwein, López de Prado and Zhu, 2017) estimates
how often the configuration that was best in-sample ranks below the median
out-of-sample.

### Uncertainty

| Source | How to measure |
|---|---|
| Model randomness | Retrain with 3+ seeds; report mean and spread (M2) |
| Which days you tested on | Walk-forward folds (C1f, V1); report per fold |
| Day-to-day noise | Block bootstrap of daily IC or daily P&L; keeps consecutive days together (E2) |
| Comparing two models | Bootstrap the **daily difference** on the same days, not each model separately |

### Purging and embargo

Today's labels end inside their session, so a split by whole sessions has no
overlap. Labels that run past the close, such as overnight holds or daily
horizons in [example 3](../examples/03-cross-sectional-ranker.md), overlap
the next split. V1 removes ("purges") training rows whose
`[decision_ts, label_end_ts]` interval overlaps the test block, and drops an
embargo after it (López de Prado, *Advances in Financial Machine Learning*,
2018, ch. 7).

### Costs

A round trip in a liquid large cap costs a few basis points at least: half
the spread on each side, slippage, and sometimes commission. A 30-minute
forward return on the fixture is a few basis points in size. E1 reports
P&L at 0, 2, 5 and 10 bps per side and the break-even cost.

## Walkthrough

1. **Read a report line by line.** Use the hello-model report. For each number,
   ask what would make it misleading.

2. **Seed variance.** Run the random forest with seeds 1, 2 and 3. Plot the
   three validation AUCs. Then run the logistic model on fixture seeds 7, 8
   and 9 (`data: {source: fixture, seed: ...}`), which changes the data,
   not the model. Compare the two spreads: changing the data usually
   moves the result more than changing the model seed.

3. **Peek at test.** Run 20 configurations (feature subsets × model
   parameters), pick the one with the best **test** AUC, and then evaluate it
   on a new fixture seed. Pick the best **validation** AUC instead and
   repeat. The test-picked one's advantage should mostly disappear on the
   new seed. This is example 6 in miniature.

4. **Costs by hand.** From `predictions.parquet`, take the validation rows,
   go long when `score > 0.5` and short otherwise, and compute the mean
   `outcome` (the raw forward return) with and without 2 bps per side. This is
   what E1 automates.

5. **Decide.** Write one paragraph: does the hello model have a tradeable
   edge on the fixture, and at what cost per side does it disappear? The
   expected answer is "a real but small ranking edge", since that is what the
   fixture plants. The break-even cost is what the pair measures, not
   something known in advance. Remember that it says nothing about real
   markets.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | For logistic vs. random forest on the same config, report validation AUC as mean ± std over 3 seeds and say whether the difference is meaningful | Usually not meaningful; overlapping ranges |
| Core | Compute the daily IC t-statistic by hand from `predictions.parquet` and match the report | `mean / std * sqrt(n_days)` over days with ≥ 5 rows |
| Stretch | Implement a simple block bootstrap of daily IC (block of 5 sessions) in the notebook and compare its interval with the naive t-interval | Wider when IC is autocorrelated |
| Stretch | Count how many configurations your pair tried today, including discarded ones. Write it in your notes | This is `n_trials` for the deflated Sharpe ratio |
| Challenge | Implement E2's `deflated_sharpe` and its acceptance test: on pure-noise strategies, the best of 100 is not significant | `tests/todo/test_eval_todo.py::test_stats` |

## Discussion

- A study reports validation AUC 0.54 and test AUC 0.51. What happened, and
  what can still be concluded?
- Your best model beat the baseline on 14 of 20 folds. Is that convincing?
- Why compare to a simple baseline at all, instead of to 0.5?

## Pitfalls

- Reporting the best seed.
- Treating rows as independent: `sample_every_minutes: 1` with 30-minute
  labels makes rows highly correlated.
- Bootstrapping rows instead of days.
- Looking at test "just to check" and then changing something.

## Connection to team strategies

Every team study follows [research-protocol.md](../../research/research-protocol.md):
pre-register the configurations, compare on validation with seeds and
bootstrap, open test once, report the number of trials. The
`research-analyst` and `leakage-auditor` agents review against the same
list.

## Wrap-up PR

Claim L1 (multi-horizon labels, Beg), V1 (purged walk-forward, Int), E1
(trading metrics, Int) or E2 (statistics, Adv).
