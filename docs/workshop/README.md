# ML for stock research: workshop spec

A series of sessions that teaches SHPE Capital members how to use machine
learning in stock research honestly, using this repo's own tools. It doubles
as onboarding: **every session ends with each participant opening a pull
request** for a real task.

| | |
|---|---|
| Audience | SHPE Capital members at every level of the ladder in [tasks.md](../tasks.md): **Beg** (can write basic programs), **Int** (can organise modules and tests), **Adv** (comfortable with statistics or systems design) |
| Format | Part 1: five foundation sessions. Part 2: nine in-depth examples |
| Length | 60–90 minutes per foundation session, 90–120 minutes per example |
| Data | Synthetic fixture with planted effects, then the frozen workshop snapshot. No API keys needed |
| Setup | [setup.md](setup.md) |
| Running it | [facilitator-guide.md](facilitator-guide.md) |

## Goals

By the end, a participant can:

1. Build a dataset in which no row uses information from after its decision
   time, and explain why each rule exists.
2. Train several model families under one contract and compare them fairly:
   same rows, same splits, several seeds, and uncertainty reported.
3. Tell a real signal from luck: validation vs. test, trial counting,
   positive and negative controls.
4. Turn a signal into a strategy: candidates, exit policies, sizing, costs and
   risk limits that match the trading platform.
5. Write a study that a teammate can reproduce from its config and that
   reports negative results as clearly as positive ones.

## Format

### Part 1: foundations

These replace the earlier five-session plan in `docs/workshop.md`. They
teach the repo and its rules. Every member takes Part 1.

| # | Session | Level | Duration | Ends with a PR for |
|---|---|---|---|---|
| 1 | [Data and time](sessions/01-data-and-time.md) | Beg | 60 min | D6, or an audit improvement issue |
| 2 | [Features without leakage](sessions/02-features-without-leakage.md) | Beg–Int | 75 min | One of F1–F5 |
| 3 | [Models under the contract](sessions/03-models-under-the-contract.md) | Beg–Int | 75 min | One of M1a–M1c |
| 4 | [Honest evaluation](sessions/04-honest-evaluation.md) | Int | 90 min | L1, V1 or E1 |
| 5 | [From signal to strategy](sessions/05-signal-to-strategy.md) | Int | 90 min | L2/T1, a T-task or an S-task |

### Part 2: in-depth examples

Each example is one research question, worked end to end on the fixture first
and then on the snapshot. Teams take the examples closest to their strategy.
Every member should take 1, 5 and 6, the two controls and the leakage lab.

| # | Example | Teams | Level | Duration | Build task |
|---|---|---|---|---|---|
| 1 | [Leakage, survivorship and snooping lab](examples/01-leakage-lab.md) | All | Beg–Int | 90 min | W1 |
| 2 | [Intraday model comparison](examples/02-intraday-model-comparison.md) | Reversals | Int | 120 min | W2 |
| 3 | [Cross-sectional ranker](examples/03-cross-sectional-ranker.md) | Momentum | Int–Adv | 120 min | W3 |
| 4 | [Meta-labeling ORB breakouts](examples/04-meta-labeling-orb.md) | Momentum, Reversals | Int–Adv | 120 min | W4 |
| 5 | [Volatility forecast (positive control)](examples/05-volatility-forecast.md) | All | Int | 90 min | W5 |
| 6 | [Overfitting casino (negative control)](examples/06-overfitting-casino.md) | All | Int | 90 min | W6 |
| 7 | [PCA and clustering for pair discovery](examples/07-unsupervised-pairs.md) | Stat arb | Int–Adv | 120 min | W7 |
| 8 | [Portfolio construction and neutralization](examples/08-portfolio-construction.md) | All | Adv | 120 min | W8 |
| 9 | [Trade management](examples/09-trade-management.md) | All | Int | 120 min | W9 |

Optional extensions are in [take-homes.md](take-homes.md). For examples 5–9,
the header table in each example document overrides any level or duration
listed here.

## Schedule

The default is one session a week. Part 2 runs in a teaching order, not in
numeric order:

| Week | Session | Why here |
|---|---|---|
| 1–5 | Foundation sessions 1–5 | The rules come before the research |
| 6 | Example 1: leakage lab | Learn how results get faked before producing any |
| 7 | Example 2: intraday model comparison | The first fair model bake-off |
| 8 | Example 5: volatility forecast | Positive control: what a real, forecastable effect looks like |
| 9 | Example 6: overfitting casino | Negative control: what pure noise looks like after a search |
| 10 | Example 3: cross-sectional ranker | Signals across stocks; breadth and turnover |
| 11 | Example 4: meta-labeling ORB | From signal to trade selection |
| 12 | Example 7: unsupervised pairs | Structure without labels |
| 13 | Example 9: trade management | Holding period, stops, targets and sizing per team |
| 14 | Example 8: portfolio construction | Combining strategies under risk limits |
| 15 | Team presentations | Each team presents a strategy spec card ([template](../research/spec-card-template.md)) |

Teams that meet more often can run their own examples (see [teams](#teams))
in parallel after week 9.

## Prerequisites and build dependencies

An example can only run once the library tasks it uses have landed. Each
example has a build task (W1–W9); its packet is `docs/handoff/tasks/W<n>.md`.
W0 provides the shared infrastructure and must land first.

| Build task | Example | Library tasks required | Useful but optional | Fixture effect needed (C1o) |
|---|---|---|---|---|
| W0 | Infrastructure | C1o | C1n | — |
| W1 | 01 Leakage lab | C0 (done), D4/D5 (done), M1a | V1, E2, C1m | Intraday reversal (exists today) |
| W2 | 02 Model comparison | M1a, M1b, M1c, L1, F1, F2, F3, F5, C1f or V1 | M2, E1, E2 | Intraday reversal; optionally a regime-conditional variant |
| W3 | 03 Cross-sectional ranker | C1a, C1b, C1d, C1e, L1, V1, E1, M1b, SM3 | C1i, C1m, E2 | Daily cross-sectional momentum and short-term reversal |
| W4 | 04 Meta-labeling ORB | SM1, F1, F2, F3, F4, F8, T1 (absorbs L2), C1b, C1c, M1a, M1c, E1, E2 | T2, M2 | Breakout continuation that depends on a feature |
| W5 | 05 Volatility forecast | F3, C1c, C1e, C1f, M1b | E2 | Volatility clustering |
| W6 | 06 Overfitting casino | E2, M2, C1f | V1 | Pure noise (signal_strength = 0) |
| W7 | 07 Unsupervised pairs | SS1, SS2, SS3, F7, C1d | SS4, T1 | Cointegrated pairs inside clusters |
| W8 | 08 Portfolio construction | C1d, T2, T4, T5, E1 | C1i, C1m | Several planted effects with known correlations |
| W9 | 09 Trade management | T1–T6, SM1, SR2, SS4 | T7 | Momentum, reversal and pair effects |

The packets in `docs/handoff/tasks/` give the exact files and acceptance
tests.

## Data policy

1. **Fixture first.** Every example runs on synthetic bars
   (`shpe_research.data.fixtures.synthetic_bars`) with **known, planted
   effects**. The answer is known in advance, so a participant who gets a much
   stronger result has a bug. Today's fixture plants one effect: each minute's
   return leans against the previous five minutes
   (`r_t = e_t − (k/5)·(r_{t−1} + … + r_{t−5})`, `k = signal_strength = 0.3`).
   C1o adds momentum, cointegrated pairs and volatility clustering.
2. **Fresh seeds are free out-of-sample data.** A new fixture seed is a new,
   independent history with the same planted effects. Use one to show what
   happens to a model chosen by peeking at test data. Both `seed` and
   `signal_strength` can be set in a config's `data:` section
   (`{source: fixture, seed: 11, signal_strength: 0}`); `signal_strength: 0`
   is pure noise, the simplest negative control.
3. **Then the snapshot.** The frozen workshop snapshot (D5, `workshop-v0`:
   minute and daily bars, 2026-07-01 to 2026-09-30, the D4 universe plus SPY)
   gives everyone identical real data. Participants load `role: dev` only. The
   facilitator opens `role: held_out` once, at the end of an example, and only
   for the final pre-registered model.
4. **Never draw conclusions about a strategy from fixture results.** The
   fixture tests the pipeline, not the market.
5. **Snapshots are members-only.** They contain Alpaca data, so never commit
   them or post them publicly. Never put keys in notebooks.

## Notebooks, solutions and CI

- Notebooks are **jupytext percent-format `.py` files** under
  `workshop/NN_slug/`, e.g. `workshop/01_leakage_lab/notebook.py`. Plain
  `.py` files diff cleanly, review like code and run without Jupyter
  (`uv run python workshop/01_leakage_lab/notebook.py`). Open one as a
  notebook with jupytext. `.ipynb` files are git-ignored.
- Each example folder holds `notebook.py` (the walkthrough), `exercises.py`
  (exercise stubs with `TODO` cells), `configs/` (experiment YAMLs) and a
  short `README.md` linking back to the example document.
- **Solutions live on the `workshop-solutions` branch**, which is kept up to
  date by merging `main` into it. Facilitators share it after the session.
- **CI runs every notebook on the fixture** (`tests/workshop/`). Each notebook
  ends with check cells that assert the planted effect was recovered within a
  tolerance, and that the deliberately broken versions are caught. A library
  change that breaks a workshop notebook fails CI on `main`.
- Reusable code that a notebook needs goes into `src/shpe_research/`
  through a normal task, not into the notebook.

## Every session ends with a PR

The rule from the original plan stays. Each session's last 10 minutes are
spent claiming a task (an issue titled `<task id>: <name>`, following
[CONTRIBUTING.md](../../CONTRIBUTING.md)) and opening a draft PR. A failing
acceptance test is an acceptable first commit. In Part 2, an acceptable PR is:

- a library task the example depends on or extends,
- a study config and README section in the team's study folder, or
- a solved challenge exercise, written up as a short note in the team's study.

## Teams

| Team | Guide | Core examples | Also useful |
|---|---|---|---|
| Momentum | [teams/momentum.md](../teams/momentum.md) | 3, 4, 9 | 2, 8 |
| Reversals | [teams/reversals.md](../teams/reversals.md) | 2, 4, 9 | 5, 7 |
| Stat arb | [teams/stat-arb.md](../teams/stat-arb.md) | 7, 9 | 3, 8 |

Every example document ends with a "Connection to team strategies" section.

## Assessment rubric

The workshop has no grade. The rubric is how reviewers judge a participant's
PRs and the final team presentation, and how a member knows they are ready for
the next level. Each criterion is scored 1–4.

| Criterion | 1: not yet | 2: developing | 3: solid | 4: strong |
|---|---|---|---|---|
| **Point-in-time correctness** | Uses future data | Passes `assert_no_lookahead` but labels or splits leak | No leaks; can explain `ts` vs. `ts_available`, label overlap and purging | Finds a leak in someone else's code |
| **Honest evaluation** | Tunes on test | Uses validation but ignores variance | Reports seeds, folds and trial counts; test opened once | Uses paired bootstrap or deflated Sharpe to support a claim |
| **Reproducibility** | Results come from a notebook state | A config exists but the run does not match | `shpe run <config>` reproduces the result | The study README pre-registers its setup before results |
| **Economic sense** | Reports AUC only | Adds IC or deciles | Adds turnover, costs and a cost break-even | Connects the result to an exit policy, sizing and the platform config |
| **Communication** | No write-up | Write-up of positive results only | Negative and null results reported plainly | A teammate can act on the write-up without asking questions |
| **Code quality** | Fails lint or tests | Passes, but logic lives in notebooks | Library code in `src/` with tests | Reviewed someone else's PR with useful findings |

Ready for the next level: a 3 in every criterion on two consecutive PRs.

## Why these examples

Each example teaches one lesson that a real quant research process depends on,
and each mirrors a piece of that process:

| Example | Lesson | Industry analogue |
|---|---|---|
| 1 | Most impressive backtests are leaks | Research review and data-integrity checks before any result is trusted |
| 2 | Model choice matters less than people think; variance matters more | Model selection with fixed data and validation |
| 3 | Signals are ranked across stocks; breadth and turnover decide value | Cross-sectional equity alpha research (factor signals and their combination) |
| 4 | A model can decide which trades to take instead of predicting prices | Meta-labeling and trade selection on top of a discretionary or rule-based strategy |
| 5 | What success looks like when the target really is predictable | Risk models and volatility forecasting |
| 6 | Searching hard enough finds "signals" in noise | Multiple-testing control, deflated Sharpe ratio |
| 7 | Structure without labels: clusters, factors, pairs | Statistical arbitrage pair and basket selection |
| 8 | A signal is not a portfolio | Portfolio construction, neutralization and risk limits |
| 9 | Exits and sizing change results as much as entries do | Trade management and execution research |

Examples 5 and 6 are controls. One shows that the pipeline finds a real effect
when there is one; the other shows that it finds a fake one when misused. Every
later result is judged against both.

### Excluded topics

| Topic | Why it is out |
|---|---|
| Reinforcement learning for trading | It needs a market simulator. A historical replay cannot react to the agent's own orders, so the agent learns the replay, not the market. Rewards are noisy and sparse, and the club's data sizes are far too small. The useful part, choosing actions under costs, is taught more honestly by examples 4 and 9. |
| LLMs predicting prices | A language model trained on text up to a recent date has seen what happened after most historical decision times. That is look-ahead that cannot be removed or audited. Results also depend on a model version that changes and cannot be reproduced. Text **as features** with point-in-time timestamps (EDGAR filings) is a take-home instead. |
| Fundamentals and text | Valuable, but needs point-in-time fundamentals (C1g sources, C1m security master) that are not built yet. Covered in [take-homes.md](take-homes.md). |
