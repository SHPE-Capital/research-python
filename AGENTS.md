# AGENTS.md

Rules for every coding agent (and every person) working in this repo. Agent
definitions for specific jobs are in [.claude/agents/](.claude/agents/). The
documentation map, with all task IDs and paths, is [docs/README.md](docs/README.md).

## What this repo is

`shpe-research` is the SHPE Capital research library. It turns market data into
features, labels, datasets, models, trade simulations and reports. It also holds
the team studies that decide what the trading platform
([SHPE-Capital/trading-platform](https://github.com/SHPE-Capital/trading-platform))
runs. Python 3.12, managed with uv, src layout, package `shpe_research`, CLI `shpe`.

## Commands

| Command | What it does |
|---|---|
| `uv sync` | Install the core dependencies and dev tools |
| `uv sync --extra ml` | Add LightGBM, ONNX, MLflow (model tasks M1–M3) |
| `uv sync --extra nn` | Add PyTorch |
| `uv run pytest` | Run the tests (`tests/todo/` acceptance tests are skipped until claimed) |
| `uv run ruff check` | Lint (line length 120; rules E, F, W, I, B, UP) |
| `uv run shpe fixtures` | Write deterministic synthetic bars for offline work |
| `uv run shpe run configs/experiments/hello_model.yaml` | Run one experiment end to end into `runs/<stamp>-<name>/` |
| `uv run shpe data bars \| verify \| audit \| ticks \| tick-summary \| universe \| snapshot build \| snapshot verify` | Data track (needs Alpaca keys in `.env`, except audit, verify and snapshot verify) |

`shpe new <feature|model|strategy|study>` arrives with C1j. Until then, copy the
closest existing module.

## Where code goes

The full design is [docs/design/architecture.md](docs/design/architecture.md);
the decisions behind it are in [docs/adr/](docs/adr/). In short, lower layers
never import higher ones (C1l enforces this with import-linter):

```
contracts → data → features / transforms → events / strategies → labels → trade
         → validation → models → portfolio → evaluation / analysis → experiments → deploy → cli
```

| You are adding… | Put it in | Register with |
|---|---|---|
| A per-symbol feature | `features/<theme>.py` | `@register_feature` |
| A cross-sectional transform (rank, z-score, neutralize) | `transforms/` | (C1d) |
| A decision sampler | `events/` | `@register_sampler` |
| An entry rule for a strategy family | `strategies/<family>/` as a `CandidateGenerator` returning a `CandidateFrame` | `@register_strategy` |
| A label | `labels/` | `@register_label` |
| Exits, sizing, filters, risk, costs | `trade/` (`ExitPolicy`, `SizingRule`, `RiskLimits`, `CostModel`, `simulate_trades`) | — |
| A splitter / fit scheme | `validation/` | `@register_fit_scheme` |
| A model | `models/` (Model contract: fit, predict, save, manifest) | `@register_model` |
| Trade diagnostics (horizon curve, MFE/MAE, surfaces) | `analysis/` | — |
| Platform config export or parity check | `deploy/` (`export_platform_config`, `ParityReport`) | — |
| A research question and its result | `studies/<team>/<id>-<slug>/` | — |

Library code that another team could reuse belongs in `src/shpe_research/`, not
in a study folder or notebook.

## Non-negotiables

These come from [CONTRIBUTING.md](CONTRIBUTING.md). A PR that breaks one is
rejected whatever its results.

1. **No future data.** A value computed for time `t` uses only information
   available at `t`. A bar is usable at `ts_available`, not `ts` (the 10:00 bar
   is known at 10:01). Every feature passes `assert_no_lookahead`
   (`features/base.py`). Labels end at `label_end_ts`, and validation purges any
   training row whose label overlaps the evaluation period.
2. **Scale inside the model.** Means, standard deviations, scalers, PCA, and
   neutralization betas are fit on training rows only, inside the model or a
   fit-on-train processor. Never during ingestion.
3. **Record every attempt.** Every configuration you ran counts toward the trial
   count used by the deflated Sharpe ratio (E2), including the ones you dropped.
   Exit-policy and sizing grids count too.
4. **Never commit data, keys or run outputs.** `data/`, `runs/`, `snapshots/`
   and `.env` stay out of git, as do notebook outputs. Keys never appear in
   code, logs, notebooks or screenshots.
5. **Test data is opened once.** Parameters are chosen on validation. The test
   period and the held-out universe symbols are evaluated once, after the study
   is frozen.
6. **Match the platform's fill rules.** Any simulated trade uses the bar fill
   rules in trading-platform `docs/architecture/backtest-execution.md`: next-bar
   open plus or minus half spread and slippage. A stop fills at its price, or at
   the open if the bar gapped through it. When stop and target are both touched
   in one bar, the stop wins.

## Working from a task packet

Tasks are split into packets so several agents or worktrees can run in
parallel. See [docs/handoff/README.md](docs/handoff/README.md).

1. Read your packet in `docs/handoff/tasks/<ID>.md` in full before writing code.
2. Check its **Depends on** list. Do not start if a dependency has not merged,
   unless the packet names a stub you may code against.
3. Edit **only** the files under its **Owns** list. If you need a change outside
   it (a shared contract, another package, `pyproject.toml`), stop and record
   the request in your report instead of making it.
4. Branch as `<agent-or-name>/<ID>`, one worktree per packet.
5. Un-skip the packet's acceptance tests in `tests/todo/` and make them pass.

## Definition of done

- The packet's acceptance tests pass; `uv run pytest` and `uv run ruff check` are clean.
- New public functions have a docstring stating their inputs, outputs and
  timing (what is known at decision time).
- New features pass `assert_no_lookahead`; new models pass `check_model`; new
  labels and strategies pass `check_label` / `check_strategy` once C1k lands.
- No `TODO(<ID>)` remains for your ID.
- `docs/tasks.md` status updated in the same PR.

## Reporting results honestly

- State what was run, on which split, with how many trials.
- Report negative and null results with the same care as positive ones. A
  study that finds no edge is a finished study.
- Never describe a result on validation data as out-of-sample performance.
- Give net-of-cost numbers next to gross ones, using the cost model in
  `trade/costs.py` (T5) or the platform fill defaults.
- If a test fails or a step was skipped, say so with the output.
- Use the study template ([docs/research/study-template.md](docs/research/study-template.md))
  and the spec card ([docs/research/spec-card-template.md](docs/research/spec-card-template.md)).

## Specialised agents

| Agent | Use it when |
|---|---|
| [repo-architect](.claude/agents/repo-architect.md) | Deciding where code goes, reviewing layering, writing or checking an ADR |
| [feature-engineer](.claude/agents/feature-engineer.md) | Implementing F-tasks or any new feature or transform |
| [strategy-builder](.claude/agents/strategy-builder.md) | Implementing SM/SR/SS tasks, or turning an idea into a `StrategySpec` |
| [research-analyst](.claude/agents/research-analyst.md) | Running a pre-registered study and writing it up |
| [trade-management-analyst](.claude/agents/trade-management-analyst.md) | Holding period, stops, targets, sizing, filters, cost hurdles |
| [leakage-auditor](.claude/agents/leakage-auditor.md) | Reviewing any PR or result for look-ahead, survivorship or snooping (read-only) |
| [platform-parity](.claude/agents/platform-parity.md) | Exporting a platform config and reconciling research with platform backtests |
| [workshop-author](.claude/agents/workshop-author.md) | Writing workshop notebooks, exercises and solutions |

The research method all of them follow is in [docs/research/](docs/research/).
