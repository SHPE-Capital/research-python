# C1h: Typed config schema

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 0 | Int | C1a | every task that adds a config key; T8 (`StrategySpec`) | `repo-architect` (review: lead) |

## Goal

Validate every experiment, strategy and sweep config with Pydantic models that
reject unknown keys, so a typo fails at load time instead of silently using a
default.

## Owns (exclusive)

- `src/shpe_research/config/__init__.py`, `config/experiment.py`, `config/strategy.py`, `config/sweep.py`
- `tests/unit/config/test_schema.py`

## Reads

- [adr/0008-typed-configs-pydantic.md](../../adr/0008-typed-configs-pydantic.md)
- `src/shpe_research/experiments/runner.py`, `configs/*.yaml`
- [design/trade-layer.md](../../design/trade-layer.md) (spec fields)

## Interface

- All models use `model_config = ConfigDict(extra="forbid")`.
- `ExperimentConfig`: `name`, `seed`, `data{source, symbols, start, end,
  frequency, context_symbols}`, `universe`, `session_filter`, `features[]`,
  `transforms[]`, `decisions`, `labels`, `fit_scheme`, `model`, `evaluate`,
  `report[]`, `portfolio`.
- Component blocks (`features[]`, `labels`, `decisions`, `model`, ...) are typed
  as `{type: str, **params}`; the params are validated by the registered
  class's own `Params` model when it defines one, else passed through.
- `portfolio`: `scheme` (`top_k`\|`rank`\|`zscore`\|`vol_scaled`), `top_k`,
  `neutralize` (list), `neutralize_proportion`, `gross`, `turnover{no_trade_band,
  partial_adjustment}` (T9).
- `StrategySpec` (T8 fills the behaviour): `strategy_type`, `params`,
  `filters[]`, `exit_policy`, `sizing`, `risk_limits`, `cost_model`,
  `fill_model`, `universe`, `version`.
- `SweepConfig`: `base`, `grid` or `optuna{n_trials, sampler, seed}`, `metric`,
  `direction`.
- `load_config(path) -> ExperimentConfig | StrategySpec | SweepConfig` by a
  top-level `kind` key, defaulting to experiment.

## Behaviour

- Unknown keys raise `ValidationError` naming the dotted path.
- Defaults reproduce today's runner behaviour; the hello model golden is
  unchanged.
- `split:` is accepted for one release as an alias of `fit_scheme:` with a
  deprecation warning.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/config/test_schema.py` | `test_existing_configs_validate` | Every YAML under `configs/` validates |
| | `test_unknown_key_rejected` | `fetaures:` raises with the path in the message |
| | `test_component_params_validated` | A feature with a `Params` model rejects a bad param |
| | `test_split_alias_warns` | `split:` loads with a `DeprecationWarning` |
| | `test_portfolio_block` | The T9 keys validate; an unknown scheme raises |

## Out of scope

Behaviour behind the keys. C1h only types and validates.

## Done when

Tests pass, the runner loads through `load_config`, and the golden is unchanged.
