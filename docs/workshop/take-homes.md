# Take-home projects

These are optional projects for participants who finish the workshop and want a
longer piece of work. Each one runs over two to four weeks, outside the
sessions. Each needs library pieces that are not on the critical path for the
sessions, so the dependencies are listed and some projects cannot start until
those tasks are merged.

The last section explains why reinforcement learning and LLM price prediction
are not offered, and what a responsible use of language models looks like here.

| ID | Project | Level | Teams | Needs (task IDs) | Can start when |
|---|---|---|---|---|---|
| TH1 | [EDGAR fundamentals and filing text](#th1-edgar-fundamentals-and-filing-text) | Adv | All | C1g (EDGAR source), C1m (security master), C1e (daily frequency), F8, E2 | C1g and C1m are merged |
| TH2 | [Post-earnings drift event study](#th2-post-earnings-drift-event-study) | Int | Momentum | C1b (events), C1e, F8, E2 | C1b and C1e are merged |
| TH3 | [Sequence models vs. gradient boosting](#th3-sequence-models-vs-gradient-boosting) | Int–Adv | Reversals | M1b, M1c, C1f, optional `nn` extra (PyTorch) | Now, on the fixture |
| TH4 | [Extensions of examples 5–9](#th4-extensions-of-examples-59) | Int–Adv | By example | As listed per example | When the example's W task is merged |

## Rules for every take-home

These are the same rules as the sessions ([research/research-protocol.md](../research/research-protocol.md)):

1. Write the hypothesis and the economic reason **before** looking at results.
   Use [research/study-template.md](../research/study-template.md).
2. Every feature has a `ts_available`. Nothing is used before it.
3. Split by time, with a gap. Keep the D4 held-out symbols untouched until the end.
4. Count every configuration run. Report the deflated Sharpe ratio (E2), not
   the best raw Sharpe.
5. Use daily P&L or date-clustered statistics. Same-day observations are correlated.
6. Run the `leakage-auditor` agent (`.claude/agents/leakage-auditor.md`) before
   presenting.
7. A null result, written up properly, is a complete take-home.

Put the work in a study folder, `studies/<team>/TH<n>-<slug>/`, with the
config, a `README.md` from the study template, and the report. If no team owns
it, use `studies/take-homes/TH<n>-<slug>/`.

---

## TH1: EDGAR fundamentals and filing text

| | |
|---|---|
| **Level** | Adv |
| **Duration** | 3–4 weeks |
| **Data** | SEC EDGAR filings (10-K, 10-Q, 8-K) via C1g; daily bars (C1e) |
| **Library tasks** | C1g, C1m, C1e, F8, E2 |
| **Output** | Two point-in-time daily features (one fundamental, one text) and a cross-sectional evaluation through the example 3 pipeline |

### Why

Fundamentals and filing text are standard inputs at cross-sectional equity
desks. They are also the easiest place to leak: databases often store values by
fiscal period end, not by the date the market could read them, and restated
numbers overwrite the originals. The project teaches point-in-time handling on
data where the gap between "period" and "available" is weeks, not minutes.

### Background

**Point-in-time rule.** For a filing, `ts_available` is the EDGAR **acceptance
time**, not the period end and not the filing date alone. A filing accepted
after the close can only be used for the next session's decisions. Use the
values as first filed. Do not use later amendments for earlier dates.

**Survivorship.** The universe at each date must come from C1m, the
point-in-time security master. A universe built from today's listed names
leaves out delisted companies and makes any "quality" feature look better.

**Text feature: filing similarity.** Cohen, Malloy & Nguyen (2020), "Lazy
Prices", compare a company's 10-K or 10-Q with its previous filing of the same
type. They report that companies whose filings change more go on to have lower
returns, and that the market is slow to react to these changes. A simple
version:

```
sim_t = cosine(tfidf(filing_t), tfidf(filing_{t-1}))     same company, same form type
feature = sim_t, available from acceptance time of filing_t
```

The TF-IDF vocabulary and weights must be fit on training-period filings only.

**Fundamental feature.** Pick one standard ratio, e.g. accruals or a
profitability ratio, computed from the XBRL facts as first filed.

### Steps

1. Build an EDGAR loader through C1g that stores, per filing: CIK, form type,
   period, acceptance time, and the raw document or XBRL facts.
2. Map CIKs to symbols through C1m. Record unmapped filings; do not drop them silently.
3. Build both features as daily series. Forward-fill from acceptance time until
   the next filing. Cap the forward-fill at the expected filing interval plus a margin.
4. Write a test that fails if any feature value is used before its acceptance time.
5. Evaluate cross-sectionally as in [Example 3](examples/03-cross-sectional-ranker.md):
   rank IC by date, quintile spreads, turnover. Neutralize to market beta (F8)
   and, if available, sector, as in [Example 8](examples/08-portfolio-construction.md).
6. Report date-clustered t-statistics and the trial count.

### Hypothesis to test

Lower filing similarity predicts lower subsequent returns, over horizons of
weeks to months, as in the paper. Treat it as a hypothesis: the workshop
universe is small and the sample is short, so low power is the most likely
outcome. Say how many years of data would be needed to detect the paper's effect
size at your universe size.

### Pitfalls

- Using `filing_date` without the time, so an after-close filing is used the same day.
- Using restated values for earlier dates.
- Fitting the TF-IDF on all filings, including future ones.
- Treating an unmapped CIK as "no filing".
- Monthly-horizon labels with daily rebalancing: overlapping labels inflate t-statistics.
  Use non-overlapping evaluation or Newey–West errors.

---

## TH2: Post-earnings drift event study

| | |
|---|---|
| **Level** | Int |
| **Duration** | 2 weeks |
| **Data** | Daily bars (C1e), earnings announcement dates and times, SPY for the market return (F8) |
| **Library tasks** | C1b (events layer), C1e, F8, E2 |
| **Output** | Cumulative abnormal return (CAR) curves by surprise quintile, with date-clustered confidence bands |

### Why

Post-earnings-announcement drift (PEAD) is one of the oldest documented
anomalies, described in detail by Bernard & Thomas (1989): prices continue to
move in the direction of an earnings surprise for weeks after the announcement.
It is a momentum effect around a known event, so it fits the momentum team. The
project teaches event studies, which the C1b events layer was built for:
decisions at event times, not every N minutes.

### Background

**Surprise without analyst data.** Consensus forecasts are not in the club's
data. Use the **announcement-day abnormal return** as the surprise proxy:

```
AR_d = r_{i,d} − β̂_i · r_{SPY,d}          β̂ from a pre-event window, e.g. days −250 to −30
surprise = AR on the event day (day 0)
```

**Event day.** If the announcement is after the close, day 0 is the next
session. If before the open, day 0 is the same session. If the time is unknown,
use the later of the two and say so.

**Drift.** Sort events into quintiles of surprise **using only past events'
breakpoints** (or within-date ranks when there are enough events per day).
Measure

```
CAR(+2, +H) = Σ_{d=2}^{H} AR_d        H up to 60 trading days
```

Starting at day +2 keeps the announcement reaction out of the drift. Plot mean
CAR by quintile against `H`.

**Statistics.** Events cluster in earnings season, and same-day events share
market shocks. Use date-clustered standard errors, or form a calendar-time
portfolio (long top quintile, short bottom) and test its daily returns.

### Steps

1. Load the earnings calendar with announcement times through C1g or a file source.
2. Register an `earnings` event type in C1b. Each event has `ts_available` at
   the announcement time.
3. Compute `AR` with F8's market return and a pre-event β.
4. Build quintiles and CAR curves, with bootstrap bands over dates (E2).
5. Turn the top-minus-bottom spread into a calendar-time portfolio and apply T5
   default costs. Report the net daily Sharpe and its deflated value.

### Hypothesis to test

The top-quintile CAR from +2 to +60 exceeds the bottom-quintile CAR. The
published effect is reported to be weaker in recent decades, so a small or
insignificant spread is a plausible result. Report it with its confidence band.

### Pitfalls

- Using the close of day 0 as the entry for an after-close announcement.
- Quintile breakpoints computed over the whole sample.
- Ignoring delisted companies (C1m) in the event sample.
- Counting overlapping 60-day windows as independent observations.

---

## TH3: Sequence models vs. gradient boosting

| | |
|---|---|
| **Level** | Int–Adv |
| **Duration** | 2–3 weeks |
| **Data** | The fixture (`synthetic_bars`, seed 7) first, then the workshop snapshot |
| **Library tasks** | M1b (LightGBM), M1c (MLP with chronological early stopping), C1f (walk-forward), optional `nn` extra for PyTorch |
| **Output** | A model comparison on identical splits, with a paired test of the daily difference |

### Why

Sequence models (LSTM, 1-D convolutions, small transformers) are often
expected to find patterns that tabular models miss. [Example 2](examples/02-intraday-model-comparison.md)
compares models on the same lag features. This take-home asks whether a model
that reads the **raw sequence** does better than gradient boosting on
**hand-built features**, when both use the same splits, the same label and the
same evaluation.

### Background

The fixture plants a linear 5-lag reversal (`signal_strength` 0.3) with
constant volatility. On the fixture, a linear model on the last 5 returns is
the best possible predictor. Any model that beats it out of sample has a bug,
most likely leakage through normalisation or windowing. This makes the fixture a
check, not a benchmark.

On the snapshot, there is no known ceiling. The question is whether the
sequence model's out-of-sample IC is higher than LightGBM's by more than the
spread across seeds.

### Steps

1. Implement the sequence model as a model plug-in that meets the C1k
   `check_model` kit. PyTorch goes in an optional `nn` extra so the base
   install stays light.
2. Build sequence inputs from the same bars as the lag features: windows of the
   last `L` returns (e.g. `L` = 30), normalised with statistics from the
   training fold only.
3. Use C1f walk-forward splits shared with LightGBM. Use chronological early
   stopping (the last part of each training fold), as in M1c.
4. Train each model with at least 5 seeds. Report the mean and spread of IC and
   of daily net P&L.
5. Test the daily IC difference with a paired block bootstrap (E2).

### Expected results

| Data | Expected |
|---|---|
| Fixture | Linear AR(5) ≥ LightGBM ≈ sequence model. A sequence model above linear is a bug to find. |
| Snapshot | Unknown. A difference smaller than the seed spread means "no difference". |

### Pitfalls

- Normalising each window with statistics that include the target bar.
- Windows that cross session boundaries without a flag.
- Random shuffling for early stopping.
- Comparing one lucky seed of one model with the mean of the other.
- Tuning the sequence model more than LightGBM. Give both the same trial
  budget and count both.

---

## TH4: Extensions of examples 5–9

Each example's Challenge exercises are a starting point. These are longer
versions.

| Example | Extension | Extra needs |
|---|---|---|
| [5 Volatility forecasting](examples/05-volatility-forecast.md) | Intraday HAR with time-of-day seasonality removed first, used for inverse-vol sizing in example 9's sizing ladder | F3, T2 |
| [6 Overfitting casino](examples/06-overfitting-casino.md) | Run CSCV PBO on one team's real study grid (RM3, RR5 or RS5) and add the result to its spec card | M2 registry for that study |
| [7 Unsupervised pairs](examples/07-unsupervised-pairs.md) | Rolling formation windows: how often do selected pairs survive into the next window? Compare cluster-based selection with distance-based on survival rate | SS1, SS2, D5 snapshot with a longer history |
| [8 Portfolio construction](examples/08-portfolio-construction.md) | Add a turnover penalty to the score-to-weight step and plot net Sharpe against the penalty | The score-to-weight module (no task ID yet; see the example's builder notes) |
| [9 Trade management](examples/09-trade-management.md) | Full parity run of the team's chosen trade management over 20 sessions, with a written explanation of every mismatch | T7, the platform deliverable for the team (P7, P8 or P9) |

---

## Why reinforcement learning and LLM price prediction are excluded

These topics come up often. They are excluded on purpose, not forgotten. The
short version is also in [README.md](README.md#excluded-topics).

### Reinforcement learning for trading

| Problem | Why it matters here |
|---|---|
| No simulator | RL learns by acting and seeing the result. A historical replay does not react to the agent's orders: no market impact, no change in fills. The agent learns to exploit the replay. |
| Reward hacking | The agent maximises the backtest, including its flaws: fill assumptions, same-bar stop and target order, session-end flattening. The result looks like skill and is a simulator artefact. |
| Small data | A few years of minute bars on a few dozen symbols is far less data than RL usually needs. Rewards are noisy and many trades share the same days. |
| Hard to validate | Policies are sensitive to seeds and hyperparameters, so the trial count is large and hard to track. |

The useful idea, choosing actions under costs, is covered with fewer
assumptions by meta-labeling ([Example 4](examples/04-meta-labeling-orb.md))
and by explicit trade management ([Example 9](examples/09-trade-management.md)).

### LLMs predicting prices

| Problem | Why it matters here |
|---|---|
| Look-ahead in the weights | A model trained on text up to its cutoff has read news, filings and commentary written after most historical decision times in the backtest. That information cannot be removed, and its effect cannot be measured. |
| Not reproducible | Hosted model versions change and are retired. A result from one version may not be reproducible a few months later. |
| No `ts_available` | There is no way to say when the model "learned" a fact, so the point-in-time contract cannot be enforced. |

### What is acceptable

Language models can be used as a **feature extractor on text**, if all of
these hold:

1. The input text has its own `ts_available` (e.g. EDGAR acceptance time, as in TH1).
2. The test period is entirely **after the model's training cutoff**. Earlier
   periods can be used only for engineering, never as evidence.
3. The model version is pinned and recorded in the study config.
4. Every prompt and output is cached and stored with the study, so the feature
   can be rebuilt without calling the model again.
5. The extracted feature is evaluated like any other feature: IC, decay,
   neutralization, trial count.

The model's output is a feature, for example "tone of this filing" on a fixed
scale. It is never a forecast of the return.

## Further reading

- Bernard, V. & Thomas, J. (1989). Post-earnings-announcement drift: delayed price response or risk premium? *Journal of Accounting Research*.
- Cohen, L., Malloy, C. & Nguyen, Q. (2020). Lazy prices. *Journal of Finance*.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*. (TH4, example 5.)
- Bailey, D. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*.
- Harvey, C., Liu, Y. & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*.
