# Pipeline

This page describes how one experiment config becomes one recorded run: the
stages, the registries that plug into them, how decision points are chosen,
how labels and fit schemes work, and what a run writes to disk.

Today this logic lives in `experiments/runner.py` (`build_dataset` and
`run_experiment`, task C0). C1a splits it into stages in `pipeline/`, and the
runner becomes a thin wrapper. Decisions:
[ADR 0003](../adr/0003-pipeline-stages-and-registries.md),
[0004](../adr/0004-events-candidates-unit-of-decision.md),
[0006](../adr/0006-fit-schemes-purged-walk-forward.md),
[0007](../adr/0007-context-symbols-cross-sectional-transforms.md),
[0008](../adr/0008-typed-configs-pydantic.md). Exact signatures:
[C1a](../handoff/tasks/C1a.md), [C1b](../handoff/tasks/C1b.md),
[C1c](../handoff/tasks/C1c.md), [C1d](../handoff/tasks/C1d.md),
[C1f](../handoff/tasks/C1f.md), [C1h](../handoff/tasks/C1h.md).

## What the C0 runner does today

```
load_bars(cfg.data)                       experiments/sources.py: fixture | parquet | store | snapshot
  -> session filter                       cfg.session, default "regular"
  -> compute_features                     ts_available renamed to decision_ts
  -> ForwardReturn label only             any other label type raises
  -> keep rows where minutes_since_open % sample_every_minutes == 0
  -> drop rows with missing features or label
  -> split_by_day(train, validation, test, gap_days)
  -> Dataset(X, y, meta, label_name)
  -> fit once on train; score every split
  -> runs/<stamp>-<name>/{config.yaml, model/, predictions.parquet, report.json, report.md}
```

It is correct for one model, one label, one split and intraday labels that end
inside their session. Everything below generalises it without changing the
contracts it already honours: `Dataset` validation, test metrics kept out of the
model manifest, and features joined at `ts_available`.

## Stages

| # | Stage | Input | Output | Registry |
|---|---|---|---|---|
| 1 | Load | `data:` config | bars (+ context bars) | data source registry (C1g) |
| 2 | Universe | bars, security master | bars restricted to the point-in-time universe | — (C1m) |
| 3 | Session filter | bars | bars | — |
| 4 | Features | bars, context bars | per-symbol feature columns at `decision_ts` | `@register_feature` |
| 5 | Transforms | feature frame | cross-sectional columns | transform registry (C1d) |
| 6 | Decisions | bars, features | `CandidateFrame` | `@register_sampler`, `@register_strategy` |
| 7 | Labels | bars, candidates | label value(s), `label_end_ts` | `@register_label` |
| 8 | Join | features, candidates, labels | `Dataset` with `era` and `split` | — |
| 9 | Fit scheme | `Dataset` | folds of (train rows, score rows) | `@register_fit_scheme` |
| 10 | Model | each fold | out-of-sample scores per fold | `@register_model` |
| 11 | Evaluate | predictions, ledger | report sections | `@register_report_section` |
| 12 | Record | everything above | run directory | — |

Each stage is a function from frames to frames. Stages never read config files
or touch the file system, except Load and Record. That keeps every stage
testable on the fixture with no I/O.

## Registries

All registries follow the existing `FEATURE_REGISTRY` pattern in
`features/base.py`: a module-level dict, a decorator that raises on a duplicate
key, and a `build_*` function that takes `{type: <key>, ...params}` from config.

| Decorator | Registers | Lives in | Example keys |
|---|---|---|---|
| `@register_feature` | `Feature` subclasses | `features/base.py` (exists) | `past_return`, `minute_of_day`, `realized_vol` |
| `@register_model` | `Model` subclasses | `contracts/model.py` (exists) | `logistic`, `random_forest`, `lightgbm`, `mlp` |
| `@register_label` | labels | `pipeline/registry.py` (C1a); labels in `labels/` (C1c) | `forward_return`, `multi_horizon`, `triple_barrier` |
| `@register_sampler` | decision-point samplers | `pipeline/registry.py` (C1a) | `every_n_minutes`, `candidates`, `session_close` |
| `@register_strategy` | `CandidateGenerator` families | `strategies/registry.py` (C1a); families in `strategies/<team>/` (S*) | `orb`, `bar_fade`, `pairs` |
| `@register_fit_scheme` | fit schemes | `pipeline/registry.py` (C1a); schemes over `validation/` (C1f) |
| `@register_report_section` | report sections | `pipeline/registry.py` (C1a); sections in `evaluation/` (C1i) | `single`, `walk_forward`, `purged_walk_forward` |

A registered class only becomes visible when its module is imported. Each
package's `__init__.py` imports its modules, as `features/__init__.py` does
today. `check_feature` and the other kits in `testing/` verify that a new class
is registered and buildable from config (see [testing.md](testing.md)).

## Decision points: samplers, events and candidates

A **decision point** is a (`symbol`, `decision_ts`) at which the strategy could
act. Every row of a `Dataset` is one decision point. There are three ways to
produce them:

| Source | Package | Example | Use when |
|---|---|---|---|
| Sampler over bars | `pipeline/` samplers | `every_n_minutes` (today's `sample_every_minutes`), `session_close` for daily | Studying a signal everywhere, e.g. IC of a feature |
| Generic event | `events/` | CUSUM filter on returns, threshold crossing of a feature, session open | Decisions should cluster where something happened |
| Strategy candidate | `strategies/` `CandidateGenerator` | ORB breakout, bar fade, pairs z-score entry | Meta-labeling or backtesting a real strategy |

All three return a `CandidateFrame` (`symbol`, `decision_ts`, `side`,
`strategy_id`, plus optional columns). The `candidates` sampler simply passes a
strategy's or event's frame through. A sampler without a natural side sets
`side = long` and the label is read as a signed forward return.

Rules every generator follows:

- `decision_ts` is a bar's `ts_available`, never its `ts`.
- A generator only reads bars and features with `ts_available <= decision_ts`.
- Candidates are emitted before filters. Filters (`trade/filters.py`) mark rows
  as skipped with a reason; they do not delete them, so a study can count what
  it filtered.

## Labels

The label registry (C1c) replaces the hard-coded `ForwardReturn` branch. A label
takes bars and a `CandidateFrame` and returns, per candidate, one or more value
columns and a `label_end_ts`.

| Key | Task | Value | `label_end_ts` |
|---|---|---|---|
| `forward_return` | C0 (exists) | close-to-close return over `horizon_minutes` | end of the horizon bar |
| `multi_horizon` | L1 | returns at several horizons, raw, cost-adjusted and volatility-adjusted | end of the longest horizon |
| `triple_barrier` | C1c over T1 | `net_return` and `exit_reason` from `simulate_trades` with an `ExitPolicy` | the trade's `exit_ts` |

The label config also chooses the **target** the model fits: `sign` (today's
default), `raw`, or a label-specific choice such as `exit_reason == take_profit`.
For `triple_barrier`, the `ExitPolicy`, `FillModel` and `CostModel` parameters
are part of the label and go into the run manifest. See
[trade-layer.md](trade-layer.md#why-labels-reuse-the-simulator).

`Dataset` already requires `label_end_ts > decision_ts`. Fit schemes use that
interval for purging, so a label that misreports its end leaks silently.
`check_label` tests it.

## Context symbols

Some features need other symbols: the market return (F8) needs SPY, a pair
spread (F7) needs the second leg, a residual reversal (SR3) needs SPY for beta.
These are **context symbols**:

- The `data:` config lists them separately, for example
  `context_symbols: [SPY]`. In a snapshot they load from the `context` role,
  which `experiments/sources.py` already supports.
- Context bars are passed to features that declare they need them. Context
  symbols never become `Dataset` rows and never enter the trade universe unless
  they are also listed as trade symbols.
- Context features obey the same `ts_available` rule: SPY's 10:00 bar is usable
  at 10:01, like the stock's.

F7 and F8 share this contract extension, which is why tasks.md lists them as
depending on each other. The C1d packet fixes the interface; F7 and F8 build on
it.

## Cross-sectional transforms

`transforms/` (C1d) operates across symbols at one `decision_ts`. It runs after
per-symbol features and before the join.

| Transform | What it does |
|---|---|
| `rank` | Percentile rank across the universe at each timestamp |
| `zscore` | Subtract the cross-sectional mean, divide by the standard deviation |
| `winsorize` | Clip at cross-sectional quantiles |
| `demean` | Subtract the cross-sectional mean (or a group mean) |
| `neutralize` | Residualise against exposures (market beta, sector) by cross-sectional regression |

Because a transform only sees rows with the same `decision_ts`, it cannot look
ahead in time. It can still leak through the universe: ranking against symbols
that were not listed yet, or that later get delisted, uses future knowledge.
Transforms therefore run on the point-in-time universe from the security master
(C1m), not on whatever symbols the snapshot happens to contain.

Sparse timestamps are a practical issue at minute frequency: if only 3 of 60
symbols have a bar at 10:07, a rank is noisy. Each transform takes a
`min_count` parameter and returns NaN below it.

## Frequencies

C1e adds daily bars next to minute bars. The contracts already carry
`timeframe` in bar metadata. For daily bars `ts` is the session date at the
open, and `ts_available` is the session close. Features declare the frequencies
they support, and a config uses one frequency per run. Daily momentum (SM3) and
the volatility forecast (workshop example 5) are the first users.

## Fit schemes

A fit scheme decides which rows train each model and which rows it scores.
C1f wraps the splitters in `validation/` as registered fit schemes.

| Key | Splitter | Behaviour |
|---|---|---|
| `single` | `validation/splits.py:split_by_day` (exists) | One chronological train / validation / test split by session, with optional `gap_days`. Today's behaviour. |
| `walk_forward` | `validation/` (C1f) | Train on a window, score the next block, roll forward. Expanding or rolling window. Refit per fold. |
| `purged_walk_forward` | `validation/purged.py:purged_walk_forward` (V1) | Walk forward, plus: drop training rows whose `[decision_ts, label_end_ts]` overlaps the scored block (purge), and drop training rows that start within `embargo` after it (embargo). |

Rules shared by all schemes:

- **The test block is held out first.** The config reserves the final block as
  `test`. Fit schemes run only on the earlier sessions. Test is scored once, by
  a model fitted under the same scheme, when the study's pre-registration says
  so (see [research-protocol.md](../research/research-protocol.md)).
- **Every prediction is out of sample.** Predictions carry a `fold` column. A
  row is scored only by a model that did not train on it.
- **Default embargo.** At least the longest label horizon in the config. For
  intraday labels that end inside their session, one session is enough. For
  multi-day holds, use the maximum holding period.
- **Era-aware folds.** Fold boundaries fall on era boundaries, never inside a
  session.

The method follows López de Prado, *Advances in Financial Machine Learning*
(2018), chapter 7.

## Processors: stateless vs. fit-on-train

A processor changes feature values before the model sees them.

| Kind | Examples | Where it runs |
|---|---|---|
| Stateless, per row | log, sign, clip at fixed bounds | In the feature itself, or as a transform |
| Stateless, per timestamp | cross-sectional rank, z-score, neutralize | `transforms/`, before the join |
| Fit on train | standard scaler, quantile bounds learned from data, PCA, target encoding | Inside the `Model`, fitted on each fold's training rows only |

Fit-on-train processors live inside the model, as `SklearnModel` already does
with its scaler. Two reasons: they are then refitted per fold automatically, so
validation statistics never leak into training, and the ONNX export (M3)
carries them, so the platform applies exactly the same scaling.

## The era column

`Dataset.meta` gains an `era` column: the session date for intraday data, the
week for daily data unless the config says otherwise. The idea and the name
come from Numerai, where an era is the unit of cross-validation and reporting.

Eras are used for:

- fold boundaries in every fit scheme;
- the by-era report section (C1i): IC and P&L per era, the share of positive
  eras, the worst era;
- era-based bootstrap blocks in E2.

`era` is an additive contract change. The C1a packet bumps the contract version
and updates [contracts.md](../contracts.md).

## Run directory and manifest

Target layout for `runs/<stamp>-<name>/`:

```
config.yaml              the resolved, validated config (defaults filled in)
manifest.json            run manifest (below)
candidates.parquet       the CandidateFrame, with skip reasons
predictions.parquet      meta + y + score + fold + era + split
ledger.parquet           TradeLedger, when the run simulates trades
model/                   final model; fold_<k>/ subfolders for walk-forward
report.json              all report sections, machine-readable
report.md                the same, for people
```

The run manifest adds to the existing `ModelManifest` (feature names, label,
params, seeds, metrics, package versions, contract version, git commit):

- the data source, snapshot id and checksum (D5 snapshots are already
  checksummed);
- the registry key and parameters of every feature, transform, sampler, label,
  fit scheme and model;
- the fold boundaries and the embargo;
- the trial number within the study, for the deflated Sharpe ratio;
- whether the test block was scored.

Test metrics stay out of the model manifest, as today.

## Config validation

C1h replaces the `ExperimentConfig` dataclass with Pydantic models.

- Unknown keys are errors (`extra="forbid"`), so a typo such as `horizon_minute`
  fails at load instead of being ignored.
- Each registered class can declare a Pydantic params model. Building from
  config validates params against it and lists the registered keys on a miss,
  as `build_feature` does today.
- Validation runs before any data loads, so a bad config fails in milliseconds.
- `StrategySpec` (`strategies/spec.py`) uses the same machinery, which is what
  lets `export_platform_config` trust its input.

An illustrative target config (the C1h packet has the schema):

```yaml
name: orb_metalabel_v1
seed: 0
data:
  source: snapshot
  snapshot: workshop_v0
  role: dev
  context_symbols: [SPY]
  frequency: 1min
session: regular
features:
  - {type: past_return, window: 5}
  - {type: or_breakout_dist}
  - {type: market_return, window: 15}
transforms:
  - {type: rank, columns: [past_return_5m]}
decisions:
  type: candidates
  strategy: configs/strategies/orb_default.yaml
label:
  type: triple_barrier
  target: sign
fit_scheme:
  type: purged_walk_forward
  n_folds: 5
  embargo: 1 session
  test_block: 20%
model:
  type: lightgbm
  params: {n_estimators: 200}
```
