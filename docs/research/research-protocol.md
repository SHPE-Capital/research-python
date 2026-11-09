# Research protocol

How every study in this repo is designed, run and reported. The rules exist
because a backtest is easy to fool. Most of what goes wrong is not a bug but a
person trying one more thing after seeing results.

Related: [strategy-lifecycle.md](strategy-lifecycle.md), [study-template.md](study-template.md),
[trade-management.md](trade-management.md), the
[`research-analyst`](../../.claude/agents/research-analyst.md) and
[`leakage-auditor`](../../.claude/agents/leakage-auditor.md) agents.

## 1. Pre-registration

Write these fields in the study README and commit them **before** any
validation result exists. The commit date is the evidence.

| Field | Example |
|---|---|
| Question | Does meta-labeling improve the net return per trade of R1 ORB breakouts? |
| Decision it informs | Take selector C to paper trading, or not |
| Hypothesis and null | H1: mean net bps per kept trade > take-every-breakout. H0: no difference |
| Data | Snapshot ID (D5), bar frequency, sessions |
| Universe | Universe config, D4 seed, dev and held-out symbol lists |
| Splits | Train, validation, test dates; purge and embargo lengths |
| Candidates | `CandidateGenerator` and its parameters |
| Label | Registered label (`@register_label`) and horizon or barriers |
| Features | Feature list or feature-set config |
| Models | Models, seeds, fit scheme (`@register_fit_scheme`) |
| Trade management | `ExitPolicy`, `SizingRule`, filters, `RiskLimits`, `CostModel` |
| Grid | Every parameter you will vary and its values |
| Selection metric | The single number that picks the winner on validation |
| Baselines | At least: take every candidate; a simple rule; a shuffled or random-retention control |
| Trial budget | Maximum configurations you will try |
| Success criterion | What result moves the strategy to the next gate |
| Kill criterion | What result stops it |

Changing a pre-registered field later is allowed. Record the change, the date
and the reason in the README's change log, and count the trials run before the change.

## 2. Universe and splits

### Dev and held-out symbols

The D4 universe builder (`shpe data universe`) sorts candidates into liquidity
× volatility cells and splits them, with a fixed seed, into **dev** and
**held-out** symbols in every cell.

- Develop, tune and select on dev symbols only.
- Held-out symbols are opened once, with test, after freezing.
- The universe is static (`candidates_v0.csv`, chosen at one date). It has
  survivorship bias for earlier periods. State this in every report until the
  point-in-time universe (C1m) exists.

### Train, validation, test

| Split | Used for | Opened |
|---|---|---|
| Train | Fitting models, scalers, hedge ratios, clusters | Freely |
| Validation | Choosing models, features, thresholds, trade management | Freely, but every look is a trial |
| Test | One final estimate | Once, after the frozen commit |

- Splits are by time, never random. The test period comes after validation.
- Purge rows whose label window crosses a split boundary, and add an embargo
  after each boundary of at least the longest label horizon (C1f; López de
  Prado 2018, ch. 7).
- For walk-forward, every fold fits on its own past only; "validation" is the
  concatenation of out-of-fold predictions.
- **When test opens:** after the configuration is frozen and committed, the trial
  count is recorded, and the leakage-auditor has reviewed the code. Not before.
- If you look at test twice, the result becomes validation-only. Say so in the report.

## 3. Trial counting

A trial is any configuration whose validation result you looked at. Count:

- model and hyperparameter configs
- feature sets
- label definitions and horizons
- `ExitPolicy` grids (stop, target, time exit)
- `SizingRule` and filter variants
- universe and session-window choices
- seeds, if you report the best one

Keep the count in the study README, updated as you go. The sweep runner (M2)
and `shpe run` record every run; the leakage-auditor rebuilds the count from
run directories and git history and compares.

## 4. Deflated Sharpe ratio (E2)

With many trials, the best Sharpe ratio is high by luck. The deflated Sharpe
ratio (Bailey and López de Prado 2014) asks whether the observed Sharpe beats
the best you would expect from N trials of a strategy with no skill.

`evaluation/stats.deflated_sharpe(sharpe, n_trials, n_obs, skew=0, kurtosis=3)`

1. Expected maximum Sharpe under the null, from N trials with variance V of the
   trial Sharpes (γ ≈ 0.5772, Euler–Mascheroni):
   `SR0 = sqrt(V) × ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)))`
2. Probability the true Sharpe exceeds SR0, given T observations and the
   return skew and kurtosis:
   `DSR = Φ( (SR − SR0) sqrt(T − 1) / sqrt(1 − skew·SR + (kurtosis − 1)/4 · SR²) )`

Use per-period Sharpe (per trade or per day, not annualised) with the matching T.
Report DSR alongside the plain Sharpe. A DSR below 0.95 means the result is not
distinguishable from the best of N lucky tries.

For a check that needs no distributional assumptions, use
`probability_of_backtest_overfitting(performance, n_splits=16)` (Bailey,
Borwein, López de Prado and Zhu 2017). It runs combinatorially symmetric
cross-validation over the trial-by-period performance matrix and returns the
probability that the in-sample winner ranks below the median out of sample.

## 5. Parameter surfaces

Show the metric over the whole grid, not just the winner (T6 draws these).

- **Plateau:** neighbours of the chosen point perform about as well. The result
  is likely to survive small changes in the market.
- **Peak:** the chosen point is much better than its neighbours. Treat it as noise.
- Choose the centre of the plateau, even if a single point nearby scores higher.
- Express parameters in units that transfer: volatility multiples, half-lives,
  z-scores, not dollars or fixed bps.

## 6. Uncertainty: block bootstrap

Trade and daily returns are autocorrelated and overlap. Plain bootstrap
understates the error.

- `evaluation/stats.block_bootstrap(values, block_length, n=10_000, seed=0)`.
- Block length: at least the label horizon or holding period; for daily P&L, 5–20 days.
- To compare against a baseline, bootstrap the **paired difference** on the same
  timestamps, not two separate intervals.
- Report the 95% interval and the share of resamples above zero.

## 7. Reporting format

Every study reports with the E1 trading metrics and the
[study template](study-template.md):

| Section | Content |
|---|---|
| Summary | Question, decision, net result with interval, trials, DSR, biggest caveat |
| Setup | Link to the pre-registration commit; any changes with reasons |
| Predictive | IC and IC decay across horizons (`trading_metrics(...).ic_decay`) |
| Trading | Net return per trade and per day at costs 0, 2, 5, 10 bps and at platform defaults; turnover; hit rate; max drawdown |
| Breakdowns | `by_group`: symbol, month, universe cell, time of day |
| Baselines | Each baseline with the same metrics; paired bootstrap differences |
| Robustness | Parameter surfaces; 2× costs; held-out symbols; sub-periods |
| Trials | Count and DSR; PBO where the grid allows |
| What failed | Every variant that did not work, briefly |
| Decision | Go, no-go, or needs more data, and the next stage |

## 8. Negative results

A study that finds nothing is finished, not failed. Write it up with the same
template, keep the configs, and add it to the team's study index. The next
person who has the same idea needs to find it. Never delete a study folder
because the result was bad.

## 9. Fooling yourself: a catalogue

| Trap | What it looks like | Guard |
|---|---|---|
| Look-ahead in features | Centered windows, `shift(-k)`, full-day high/low/VWAP intraday | `assert_no_lookahead`; `check_feature` (C1k); trailing windows only |
| `ts` vs `ts_available` | Using the 10:00 bar at 10:00 | Align on `ts_available`; enter on the next bar's open |
| Global fitting | Scaler, PCA, hedge ratio or clustering fit on all rows | Fit inside the model or fit scheme, on train rows only |
| Overlapping labels | Label windows cross the split boundary | Purge and embargo (C1f) |
| Survivorship | Today's universe used for past years | State it; use C1m when it lands |
| Corporate actions | Split gaps look like 50% returns | D2 audit; D6 adjustments |
| Forward-filled prices | Gaps filled with the last price create fake zero returns | Leave missing; mask in features |
| Selection on test | Tuning after seeing test | Freeze, then open once |
| Uncounted trials | Exit and sizing grids left out of N | Count every grid point |
| Best seed | Reporting the luckiest seed | Report the mean and spread over seeds |
| Optimistic costs | Research costs below platform fills | Use platform defaults unless R5 calibrates lower |
| Pooled averages | One symbol or one day carries the result | `by_group`; drop the top contributor and recompute |
| Story after the fact | Rationale written to fit the result | Rationale in the spec card before the study |
| Small samples | 40 trades, Sharpe 3 | Interval from block bootstrap; minimum trade count in pre-registration |
| Regime luck | Works only in one volatility regime | Sub-period and volatility-regime breakdowns |
| Same-bar ambiguity | Assuming the target fills before the stop in one bar | Platform rule: the stop wins |
| Fills at the signal price | Entering at the close that produced the signal | Next-bar open plus half spread and slippage |

## References

- Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5).
- Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4).
- Harvey, C. R., Liu, Y. and Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4).
- Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428).
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5).
