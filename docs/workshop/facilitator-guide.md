# Facilitator guide

How to run a session. Read the session or example document first. This page
covers what the documents don't: preparation, timing, mixed levels, and the
places where participants get stuck.

## Before the series starts

| When | Do | Check |
|---|---|---|
| 3 weeks before | Confirm W0 has landed: `workshop/` exists, `tests/workshop/` runs in CI, the `workshop-solutions` branch exists | CI on `main` is green, including the workshop notebooks |
| 2 weeks before | Build or locate the current snapshot (`shpe data snapshot build`, see the [README](../../README.md#real-market-data)); put the zip on the club's private drive | `uv run shpe data snapshot verify <zip>` passes on a second machine |
| 2 weeks before | Send [setup.md](setup.md) to participants | — |
| 1 week before | Run the setup check-in: 15 minutes online, everyone runs the quickstart | Everyone has a hello-model report with validation AUC slightly above 0.5 |
| Each week | Check that the library tasks the next example needs have landed (see the dependency table in the [README](README.md#prerequisites-and-build-dependencies)) | The example's notebook passes in CI on `main` |

If a library task an example needs has not landed, run the example on the
`workshop-solutions` branch, where the solution may stub the missing
piece, or swap in the next example in the schedule. Don't live-code the missing
library task during the session.

## Session shape

Every session and example follows the same 4-part shape. The minutes are for a
90-minute slot. Scale them for 60 or 120 minutes.

| Part | Minutes | Who | What |
|---|---|---|---|
| Hook | 10 | Facilitator | The question, and a result that looks too good or too bad to be true |
| Live code | 30 | Facilitator, projected | Walk through `notebook.py`. Participants follow along and run every cell |
| Exercises | 40 | Pairs | `exercises.py`, core tier first. Facilitators circulate |
| Wrap-up and PR | 10 | Everyone | Two or three pairs report results. Everyone claims a task and opens a draft PR |

Rules that keep the session on time:

- **Live code from a finished notebook**, revealing cells one at a time. Do not
  type code from scratch; typos eat the session.
- **Run every cell before the session**, on the machine you will project
  from, with the fixture seed the document gives. Results should match the
  "Expected on the fixture" column in the example to the stated tolerance.
- **Stop the exercise block on time** even if pairs aren't finished. Core
  exercises can be finished at home; the solution branch has them.

## Mixed levels

Pair people across levels: one Beg or Int with one Int or Adv. The less
experienced partner types. The more experienced partner explains and reviews.

Each exercise set has three tiers:

| Tier | Who it's for | Expectation |
|---|---|---|
| Core | Everyone | Finished in the session by most pairs. Reproduces the lesson |
| Stretch | Int | Changes one design choice and explains the effect |
| Challenge | Adv | Open-ended; usually becomes a library task or study note |

Advanced members who finish early review a Beg member's PR from the previous
week instead of starting the challenge tier. Reviewing is part of the
rubric.

## Verifying planted effects

The fixture's planted effects are the workshop's answer key. Before each
example, confirm the notebook recovers them:

```bash
uv run pytest tests/workshop -k <example slug>
```

The CI check cells assert, for each example, that:

1. The honest pipeline recovers the planted effect within a tolerance. The
   tolerance band is set when the notebook is written, from runs over several
   fixture seeds, and is written in the notebook's check cell. For the hello
   model the band sits slightly above an AUC of 0.5.
2. Each deliberately broken pipeline shows the expected failure (a much higher
   score, or a check that raises).
3. With `signal_strength=0`, the honest pipeline finds nothing beyond chance.

If (1) fails after a library change, the change probably broke something.
If (2) fails, a check got weaker. Don't run the session until it passes.

Today's fixture plants only the 5-minute reversal. The other effects
(momentum, cointegrated pairs, volatility clustering, the conditional ORB
effect) come with C1o. Each example lists the effect it needs.

## Common failure points

| Where | Symptom | Cause | Fix |
|---|---|---|---|
| Setup | `uv sync` resolves but `import lightgbm` fails on macOS | Missing OpenMP | `brew install libomp` |
| Setup | `uv sync --extra nn` takes very long | PyTorch is large | Do it before the session; only example 2's LSTM needs it |
| Any | Times differ by 4–5 hours from a chart | Stored times are UTC | Convert with `tz_convert("America/New_York")` for display only |
| Any | A merge raises or returns too few rows | Joined on `ts` instead of `ts_available` | Features and labels both key on `decision_ts = ts_available` |
| Session 2 | `assert_no_lookahead` fails on a feature that "obviously" only uses past data | `rolling(..., center=True)`, a groupby-transform over the whole day, or `shift(-1)` | Find the operation that reads later rows |
| Session 3 | Validation AUC around 0.7+ on the fixture | Leakage, almost always | Run the leakage checklist in [example 1](examples/01-leakage-lab.md) |
| Session 3 | Results differ between two participants | Different seed, or different `uv.lock` | Compare `config.yaml` and the run manifest in the two run folders |
| Session 4 | Daily IC is NaN for many days | Fewer than 5 rows per day (`MIN_ROWS_PER_DAY`) | More symbols, or a denser `sample_every_minutes` |
| Example 2 | MLP or LSTM scores vary a lot between runs | Seed variance on a weak signal | That is the lesson: report mean and spread over seeds |
| Example 3 | Momentum signal is all NaN | Not enough daily history for a 12-month lookback | Use the long-history fixture or C1e daily store, not the 3-month snapshot |
| Example 4 | Almost no candidates on the fixture | ORB effect not planted, or the entry window is too short | Check the C1o ORB parameters; use the solution config |
| Snapshot | `load_snapshot` raises a checksum error | Corrupted or edited zip | Download again; never edit snapshot files |

## Using the held-out names

Each example that uses the snapshot ends with a single held-out evaluation:

1. Pairs pick their final configuration using `role: dev` and validation
   only, and write it into their study README before the reveal.
2. The facilitator runs those configs with `role: held_out` and shows all
   results together, including the ones that got worse.
3. Nobody changes a configuration after seeing held-out results. If they
   want to, the change goes into the next study as a new pre-registered trial.

Showing every team's held-out drop at once is the most effective lesson in the
series. Make time for it.

## After each session

- Merge `main` into `workshop-solutions` if it has fallen behind, and push
  the solution for the session.
- Triage the draft PRs: label each with its task ID, assign a reviewer from a
  different level.
- Write two lines in the workshop channel: what went wrong, what to change.
  Feed fixes back to the example document through a PR.
