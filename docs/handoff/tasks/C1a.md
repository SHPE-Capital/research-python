# C1a: Pipeline stages, registries and package skeleton

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 0 | Adv | ADR sign-off ([0002](../../adr/0002-layered-package-structure.md)–[0005](../../adr/0005-strategies-and-shared-trade-toolkit.md)), the Wave 0 migration PR | every other C1, T, S and W packet | `repo-architect` (review: lead) |

## Goal

Split `experiments/runner.py` into the twelve stages of
[design/pipeline.md](../../design/pipeline.md), add the registries every later
packet plugs into, and create the new packages with their shared types as
stubs that carry final names, fields and docstrings. After this packet merges,
no other packet needs to edit the runner.

## Owns (exclusive)

- `src/shpe_research/pipeline/__init__.py`, `pipeline/stages.py`, `pipeline/registry.py`
- `src/shpe_research/experiments/runner.py` (rewritten as a thin wrapper over `pipeline/`)
- `src/shpe_research/contracts/candidates.py` (stub; C1b fills the validator)
- `src/shpe_research/contracts/dataset.py` and `contracts/version.py` (the `era` column only)
- `src/shpe_research/events/__init__.py`, `events/types.py`
- `src/shpe_research/trade/__init__.py`, `trade/types.py`, `trade/simulate.py` (stub)
- `src/shpe_research/strategies/__init__.py`, `strategies/registry.py`,
  `strategies/momentum/__init__.py`, `strategies/reversal/__init__.py`, `strategies/statarb/__init__.py`
- `src/shpe_research/portfolio/__init__.py`, `analysis/__init__.py`, `transforms/__init__.py`
- `src/shpe_research/deploy/__init__.py`, `deploy/platform_config.py` (stub), `deploy/parity.py` (stub)
- `src/shpe_research/testing/__init__.py` (empty; C1k fills it)
- `tests/unit/pipeline/test_registry.py`, `tests/unit/pipeline/test_stages.py`,
  `tests/unit/contracts/test_candidates_schema.py`
- `tests/integration/golden/report_hello_model.json`

## Reads

- [design/architecture.md](../../design/architecture.md) (layer ranks, import rules)
- [design/pipeline.md](../../design/pipeline.md), [design/trade-layer.md](../../design/trade-layer.md)
- `src/shpe_research/features/base.py` (the registry pattern to copy)
- `src/shpe_research/contracts/model.py` (`register_model`, `build_model`)
- `src/shpe_research/experiments/runner.py` (current behaviour to preserve)
- `src/shpe_research/contracts/dataset.py` (`SPLITS`, `Dataset.validate`)

## Interface

### Registries (`pipeline/registry.py`)

Copy the `features/base.py` pattern exactly: a module-level
`dict[str, type]`, a decorator keyed on a `ClassVar` that raises on a duplicate
key, and a `build_*` function that pops `"type"`, imports the owning package so
built-ins register, and raises `ValueError` listing the registered keys when a
key is unknown.

| Decorator | Registry | Key ClassVar | Builder | Owning package imported by the builder |
|---|---|---|---|---|
| `register_sampler` | `SAMPLER_REGISTRY` | `sampler_type` | `build_sampler(spec)` | `shpe_research.events` |
| `register_label` | `LABEL_REGISTRY` | `label_type` | `build_label(spec)` | `shpe_research.labels` |
| `register_fit_scheme` | `FIT_SCHEME_REGISTRY` | `scheme_type` | `build_fit_scheme(spec)` | `shpe_research.validation` |
| `register_report_section` | `REPORT_SECTION_REGISTRY` | `section_type` | `build_report_section(spec)` | `shpe_research.evaluation` |

`strategies/registry.py` holds `register_strategy` / `STRATEGY_REGISTRY` /
`build_strategy(spec)` keyed on `strategy_type`, importing
`shpe_research.strategies`.

C1a registers only what keeps the hello model working:
`every_n_minutes` (sampler), `forward_return` (label, wrapping
`labels/forward_return.py`), `single` (fit scheme, wrapping
`validation/splits.split_by_day`), and the existing report as section
`signal` (wrapping `evaluation/report.py`). `candidates` and `session_close`
belong to C1b; the other labels to C1c; the other schemes to C1f.

### Stages (`pipeline/stages.py`)

One function per stage, each taking and returning plain DataFrames or the
`Dataset` contract, in this order: `load`, `universe`, `session_filter`,
`features`, `transforms`, `decisions`, `labels`, `join`, `fit_scheme`,
`model`, `evaluate`, `record`. `transforms` is the identity until C1d.
`run_pipeline(cfg, out_root) -> RunResult` calls them in order.
`experiments/runner.py` keeps `EVAL_SPLITS`, `ExperimentConfig`, `RunResult`,
`build_dataset` and `run_experiment` as thin wrappers so existing imports keep
working until C1h replaces `ExperimentConfig`.

### Shared stubs

| Name | Module | Content in C1a | Filled by |
|---|---|---|---|
| `CANDIDATE_COLUMNS`, `OPTIONAL_CANDIDATE_COLUMNS`, `validate_candidates` | `contracts/candidates.py` | Column names and dtypes; validator raises `NotImplementedError` | C1b |
| `CandidateFrame` | `contracts/candidates.py` | Type alias `pd.DataFrame` with the schema in its docstring | C1b |
| `CandidateGenerator` | `events/types.py` | `typing.Protocol` with `strategy_type: ClassVar[str]` and `generate(bars, context=None) -> CandidateFrame`; re-exports `CandidateFrame` | C1b |
| `ExitPolicy`, `FillModel`, `TRADE_LEDGER_COLUMNS`, `TradeLedger` | `trade/types.py` | Frozen dataclasses with the fields in [T1](T1.md); ledger column list | T1 |
| `SizingRule` | `trade/types.py` | Frozen dataclass, fields in [T2](T2.md) | T2 |
| `RiskLimits` | `trade/types.py` | Frozen dataclass, fields in [T4](T4.md) | T4 |
| `CostModel` | `trade/types.py` | Frozen dataclass, fields in [T5](T5.md) | T5 |
| `simulate_trades(bars, candidates, policy, fill=FillModel(), costs=None, sizing=None, risk=None) -> TradeLedger` | `trade/simulate.py` | Raises `NotImplementedError("TODO(T1)")` | T1 |
| `export_platform_config(spec)`, `ParityReport` | `deploy/platform_config.py`, `deploy/parity.py` | Signatures and docstrings only | T7 |

`CandidateFrame` schema (`contracts/candidates.py`):

| Column | Dtype | Required | Meaning |
|---|---|---|---|
| `symbol` | string | yes | Traded symbol (leg 1 for pairs) |
| `decision_ts` | `datetime64[ns, UTC]` | yes | Equal to the `ts_available` of the bar the decision used |
| `side` | `long` \| `short` | yes | For pairs: `long` = long spread (buy leg 1, sell leg 2) |
| `strategy_id` | string | yes | `<strategy_type>:<config hash>` |
| `action` | `enter` \| `exit` | no, default `enter` | `exit` rows close an open position by signal (SR1, SS4) |
| `exit_reason` | string | on `exit` rows | From the T1 vocabulary |
| `trigger_value`, `zscore`, `stretch` | float | no | What the generator knew at `decision_ts` |
| `counterpart_symbol`, `hedge_ratio` | string, float | pairs only | Leg 2 and β at `decision_ts` |
| `stop_price`, `target_price` | float | no | Price levels for order exits |
| `entry_price` | float | no | Trigger level for a stop entry (T1 `entry_order: stop`, e.g. the opening-range high) |
| `qty` | int | no | Strategy-native quantity (SR1 `qtyPerTrade`, SS4 leg 1 qty) |

The `action` / `exit_reason` columns are a packet-level addition to
[design/trade-layer.md](../../design/trade-layer.md); see the open question in
[../README.md](../README.md).

### Era column

Add `era` to the dataset meta columns: the New York session date for intraday
rows, the ISO week (`YYYY-Www`) for daily rows. This is additive: bump
`CONTRACT_VERSION` from `"0.1.0"` to `"0.2.0"` and add a line to
[contracts.md](../../contracts.md) (lead review).

### Run directory

`record` writes `config.yaml`, `manifest.json`, `predictions.parquet`,
`report.json`, `report.md`, and `model/fold_<k>/`; it writes
`candidates.parquet` and `ledger.parquet` when those stages produced them. The
manifest adds `snapshot_checksum` (null for fixture data), `registry_keys`
with params per stage, `folds` (boundaries and embargo), `trial_number`
(null unless set by M2) and `test_scored` (bool).

## Behaviour

- The hello model run (`configs/hello_model.yaml`) produces a `report.json`
  equal to the golden file except for run-specific fields (`run_id`, timestamps,
  git commit, package versions). Capture the golden from `main` before the
  refactor.
- Unknown registry keys fail before any data loads, listing the valid keys.
- Library code never imports `testing/`. Every new package respects the rank
  table in [design/architecture.md](../../design/architecture.md); in
  particular `trade/` imports `contracts/candidates.py`, never `events/`.
- No stage reads data from a later stage. `labels` runs after `decisions`, and
  `join` drops rows with a missing label rather than filling them.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/pipeline/test_registry.py` | `test_duplicate_key_raises` | Registering two classes with one key raises for each of the four registries and `register_strategy` |
| | `test_unknown_key_lists_registered` | `build_label({"type": "nope"})` raises `ValueError` whose message contains `forward_return` |
| | `test_builtins_registered_on_import` | `every_n_minutes`, `forward_return`, `single`, `signal` resolve without importing their modules explicitly |
| `tests/unit/pipeline/test_stages.py` | `test_stage_order` | `run_pipeline` calls the twelve stages in the documented order (monkeypatched spies) |
| | `test_runner_wrapper_compat` | `experiments.runner.run_experiment` and `build_dataset` still import and return the old types |
| `tests/unit/contracts/test_candidates_schema.py` | `test_schema_names` | Required and optional column names equal the table above |
| | `test_trade_does_not_import_events` | Importing `shpe_research.trade` does not put `shpe_research.events` in `sys.modules` |
| `tests/test_pipeline.py` (existing) | `test_hello_model_golden` (new) | Hello report matches `tests/integration/golden/report_hello_model.json` within 1e-9 on numeric fields |
| `tests/test_contracts.py` (existing) | `test_era_column` (new) | `era` is present, equals the NY date for minute rows; `CONTRACT_VERSION == "0.2.0"` |

## Out of scope

- Any label other than `forward_return`, any sampler other than
  `every_n_minutes`, any fit scheme other than `single` (C1b, C1c, C1f).
- Pydantic configs (C1h). C1a keeps the dataclass config.
- Filling any trade-layer behaviour (T1–T9).
- Moving files: the Wave 0 migration PR does that before this packet.

## Done when

The hello model golden test passes, the registry tests pass, every stub named
above imports, and `uv run pytest` is green on the rebased branch.
