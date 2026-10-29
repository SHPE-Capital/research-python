# ML workshop

Five sessions of 60–90 minutes. **Every session ends with each participant
opening a pull request** for a real task from the [task board](tasks.md), so the
workshop doubles as onboarding.

**Before session 1:** install uv, clone this repo, and run the three
quickstart commands in the README. No Alpaca keys are needed. Sessions use the
synthetic fixture or the frozen workshop snapshot (D5).

## 1. Data and time

- Bars, quotes and trades: what each can and cannot tell you.
- `ts` vs. `ts_available`: why the 10:00 bar is only usable at 10:01.
- Sessions, early closes, missing minutes (never filled in), raw vs. split-adjusted prices.
- Walk through `docs/contracts.md` and run `shpe data audit` on the snapshot.
- **Exercise:** claim a data task (D6) or review a D-track module and file one
  improvement issue.

## 2. Features without leakage

- How a feature can cheat: centered windows, full-day statistics, scaling with future data.
- Read `features/returns.py` and `assert_no_lookahead`; break it on purpose and watch the test fail.
- **Exercise:** claim one of F1–F5 and get its acceptance test running (still red is fine).

## 3. Models under the contract

- The Model contract: fit, predict, save, manifest. Scaling lives inside the model.
- Run the hello model; read its manifest and report.
- **Exercise:** claim M1a, M1b or M1c and run it through `shpe run` with a new config.

## 4. Honest evaluation

- Train, validation and test periods; why test is opened once.
- IC, its daily spread and t-statistic; decile tables; why AUC 0.53 can still matter and still lose money.
- Costs and multiple testing: the more configurations you try, the higher the bar (E2's deflated Sharpe).
- **Exercise:** compare everyone's session 3 models on the same validation
  period, and record every attempt.

## 5. From signal to strategy

- Meta-labeling: a strategy proposes candidates, a model decides which to take (Christopher's ORB plan as the worked example).
- Exit-policy labels (L2) and how they match the platform's fills.
- **Exercise:** explain one complete trade end to end, using the Replay viewer
  (P4) if it has landed, or the R1 study's trade ledger.

## Where participants go next

| If you can… | Next step |
|---|---|
| Follow a tutorial but struggle to modify it | Take a Beg feature task (F2, F3) and rebuild it from the spec |
| Write basic programs but struggle to organize them | Take a data or label task (D6, L1): modules, tests, debugging |
| Build a working notebook independently | Turn it into a module plus an experiment config |
| Train models but struggle to evaluate them | Take E1 or V1: baselines, metrics, leakage, validation |
| Build reliable Python projects | Take M2, M3 or a platform task (P1–P3) |
