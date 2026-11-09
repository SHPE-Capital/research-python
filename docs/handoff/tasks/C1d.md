# C1d: Context symbols and cross-sectional transforms

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 1 | Int–Adv | C1a, C1o | F7/F8, SM3, SR3, T9, W3, W8 | `feature-engineer` (review: `leakage-auditor`, lead for the contract bump) |

## Goal

Let features read symbols that are not traded (SPY, sector ETFs, a pair's
other leg), and add the Transforms stage that turns per-symbol features into
cross-sectional ones (rank, z-score, neutralized residuals) without leaking
across time.

## Owns (exclusive)

- `src/shpe_research/features/base.py` (the `context` argument of `compute_features` only)
- `src/shpe_research/transforms/cross_sectional.py`, `transforms/neutralize.py`, `transforms/base.py`
- `docs/contracts.md` and `contracts/version.py` (the context-symbol bump)
- `tests/unit/transforms/test_cross_sectional.py`, `tests/unit/transforms/test_neutralize.py`,
  `tests/unit/features/test_context_symbols.py`

## Reads

- [adr/0007-context-symbols-cross-sectional-transforms.md](../../adr/0007-context-symbols-cross-sectional-transforms.md)
- [design/pipeline.md](../../design/pipeline.md) (Load and Transforms stages)
- `src/shpe_research/features/base.py`, `experiments/sources.py`, `data/snapshot.py` (role `context`)

## Interface

- Config: `data.context_symbols: [SPY, ...]`. Load fetches them with the traded
  symbols; Universe keeps them out of the traded set.
- `compute_features(bars, features, context=None)`. `context` is a
  `dict[str, pd.DataFrame]` of context-symbol bars. A feature declares
  `context_symbols: ClassVar[tuple[str, ...]] = ()` or reads a constructor
  param (`benchmark`, `other`).
- Context alignment helper `align_context(bars, ctx, column)`: for each row of
  `bars`, the latest `ctx` value with `ctx.ts_available <= bars.ts_available`
  (backward as-of join on `ts_available`, per session).
- Transforms (`transforms/base.py`: `Transform` ABC, `transform_type`
  ClassVar, a `register_transform` registry built like `register_feature`):

| Key | Params | Output |
|---|---|---|
| `rank` | `column`, `pct=True`, `min_count=5` | Cross-sectional rank at each `decision_ts` |
| `zscore` | `column`, `min_count=5` | (x − mean) / std across symbols at each `decision_ts` |
| `winsorize` | `column`, `lower=0.01`, `upper=0.99`, `min_count=5` | Clipped at the cross-sectional quantiles |
| `demean` | `column`, `by=None` (`sector` or a column) | x − group mean |
| `neutralize` | `column`, `exposures` (list of columns, e.g. `beta`, sector dummies), `proportion=1.0`, `min_count=10` | Residual of a per-timestamp OLS of x on the exposures, scaled back by `proportion` (0 = untouched, 1 = fully neutral) |

## Behaviour

- Every transform groups by `decision_ts` only. No transform uses a value from
  a different timestamp. Rows in groups smaller than `min_count` get NaN.
- `neutralize` adds an intercept; with `proportion=p` it returns
  `x − p · fitted_without_intercept`. T9 and W8 call it.
- Context alignment never uses a context bar whose `ts_available` is after the
  row's `ts_available`; a missing context bar yields NaN, not the previous
  session's value.
- Contract bump: `CONTRACT_VERSION` minor bump (on top of C1a's era bump) and a
  `contracts.md` note that context symbols appear in the bars frame with a
  `role` of `context` and are never in the dataset rows.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/features/test_context_symbols.py` | `test_align_backward_only` | A context bar stamped 1 ns after the row's `ts_available` is not used |
| | `test_context_not_traded` | With `context_symbols: [SPY]`, no dataset row has `symbol == "SPY"` |
| `tests/unit/transforms/test_cross_sectional.py` | `test_rank_zscore_hand` | Hand-built 2-timestamp frame gives exact ranks and z-scores |
| | `test_no_cross_time_mixing` | Shuffling rows of other timestamps leaves a timestamp's output unchanged |
| | `test_min_count` | A timestamp with fewer than `min_count` rows is all NaN |
| `tests/unit/transforms/test_neutralize.py` | `test_planted_beta_removed` | On C1o `factors` plus `xs_alpha{contaminate: {beta: 0.5}}`, corr(neutralized score, β) < 0.05 and the residual keeps IC > 0 |
| | `test_proportion` | `proportion=0` returns x; `proportion=0.5` halves the exposure coefficient |

## Out of scope

The features that use context (F7/F8, SR3), portfolio weights (T9), daily data (C1e).

## Done when

Tests pass, the contract bump is reviewed by the lead, and a config with
`context_symbols: [SPY]` and a `transforms:` list runs on the fixture.
