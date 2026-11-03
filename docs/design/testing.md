# Testing

Research code has a failure mode ordinary code does not: it can run, pass its
unit tests, and still produce a result that is wrong because it peeked at the
future. The test strategy is built around catching that. Tasks:
[C1k](../handoff/tasks/C1k.md) (check kits), [C1o](../handoff/tasks/C1o.md)
(planted-effects fixture), [W0](../handoff/tasks/W0.md) (workshop CI),
[T7](../handoff/tasks/T7.md) (parity goldens).

## Test pyramid

| Level | Folder | What it covers | Data | Speed budget |
|---|---|---|---|---|
| Unit | `tests/unit/` | One function or class: shapes, edge cases, no-lookahead | Tiny hand-built frames, fixture slices | Whole suite under 60 s |
| Integration | `tests/integration/` | A config run end to end through the pipeline; downloaders against `tests/fakes/` | Planted-effects fixture | Under 3 min |
| Statistical | `tests/integration/` (marked `statistical`) | The pipeline finds what was planted and nothing else | Planted-effects fixture | Under 3 min |
| Parity | `tests/integration/golden/` | Research fills and decisions match the platform on golden sessions | Frozen golden bars | Under 1 min |
| Workshop | `tests/workshop/` | Every workshop notebook runs top to bottom | Fixture | Under 5 min |
| Acceptance (temporary) | `tests/todo/` | Skipped tests that define "done" for open tasks | Fixture | — |

Today `tests/` is flat. The architecture migration
([architecture.md](architecture.md#migration-current-tree-to-target-tree)) moves `test_pipeline.py` and
`test_bars_download.py` to `integration/`, the rest to `unit/`, and
`fake_alpaca.py` to `tests/fakes/`. `tests/todo/` keeps its role: a task's PR
moves its acceptance test out of `todo/` into `unit/` or `integration/` and
removes the skip marker.

Nothing in CI touches the network. Downloaders are tested against
`tests/fakes/`, as `test_bars_download.py` does today with `fake_alpaca.py`.

## Check kits

`testing/` ships inside the package (`shpe_research.testing`) so studies,
notebooks and team code can use the same checks as the library's own tests. It
is rank 6 in the layer table and nothing below it may import it (a forbidden
contract in import-linter; see [architecture.md](architecture.md)).

Each kit is one call that runs a battery of checks on a registered component
and raises `AssertionError` with a message that names the failing check.

| Kit | Checks |
|---|---|
| `check_feature(feature_cls, params)` | Registered and buildable from `{type: ..., **params}`; output aligned to input rows; float dtype; no lookahead (`assert_no_lookahead` in `features/base.py`, prefix-truncation at several cutoffs); stable under row order within a symbol; NaN only in the declared warm-up; context symbols, if declared, read only up to `decision_ts` |
| `check_model(model_cls, params)` | Registered; fits and predicts on the fixture `Dataset`; same seed gives same predictions; `save` then `load_model` gives identical predictions; probabilities in [0, 1]; fitted processors are refitted per fit, not reused; for `ml` models, ONNX export matches within 1e-5 |
| `check_label(label_cls, params)` | Registered; `label_end_ts > decision_ts` on every row; `label_end_ts` is the true last bar used (recomputing after deleting bars past `label_end_ts` gives the same value); no value for a candidate whose horizon runs past the data |
| `check_strategy(generator_cls, params)` | Registered; returns a valid `CandidateFrame`; `decision_ts` is always a bar's `ts_available`; no lookahead (candidates up to time t do not change when bars after t are removed); deterministic; with an `ExitPolicy`, `simulate_trades` returns a valid `TradeLedger` whose `exit_reason` values are in the vocabulary of [trade-layer.md](trade-layer.md); for exportable families, `export_platform_config` accepts the default spec |

`shpe new feature|model|strategy` (C1j) generates a test file that calls the
matching kit, so a new component starts with its checks in place.

The prefix-truncation idea generalises: any component that maps history to a
value at time t must give the same value when the history after t is deleted.
`check_feature`, `check_label` (with the window ending at `label_end_ts`) and
`check_strategy` all apply it.

## Planted-effects fixture (C1o)

`data/fixtures.synthetic_bars` today plants one effect: short-horizon mean
reversion, where each minute's return leans against the previous five
(`signal_strength`, default 0.3). C1o extends it so each strategy family has a
known answer to recover and a known null to stay silent on. The function keeps
its current signature and defaults, so existing tests do not change; new
effects are opt-in through parameters fixed in the C1o packet.

Every effect lives on its own symbols so tests can check that the pipeline
finds the effect where it was planted and not elsewhere.

| Effect | Planted how | Symbols | Test asserts |
|---|---|---|---|
| Short-horizon reversal | Existing AR term against the last 5 minutes | reversal set | A reversal feature has negative IC on these symbols; SR1 bar fade earns a positive gross return before costs |
| Intraday momentum | Returns after the opening range continue in the breakout direction with a set drift | momentum set | ORB candidates (SM1) have a positive mean forward return; a meta-model's AUC is above 0.5 on held-out sessions |
| Cointegrated pair | Leg 2 = β × leg 1 + an OU spread with a known half-life | one or more pairs | SS2 recovers β within a tolerance; SS3 recovers the half-life within a tolerance; SS1 selects the planted pair and rejects random pairings |
| Volatility clustering | GARCH-like variance process | vol set | Realised-vol features forecast next-period realised vol (positive control for workshop example 5) |
| Null symbols | Random walk with the same volume curve and vol level, no planted dependence | null set | Every signal's IC is within noise; the deflated Sharpe ratio rejects the best of many random strategies; no pair passes the cointegration gate more often than its significance level |

Two tests matter most:

- **The null test.** On null symbols, the full pipeline with a purged
  walk-forward fit scheme must not report a significant edge. A pass on
  planted symbols with a fail on null symbols is the strongest evidence that
  there is no leakage.
- **The ceiling test.** No method should recover more than was planted. The
  existing fixture docstring already says this: a much stronger signal than
  planted means leakage. Statistical tests assert an upper bound on IC or AUC,
  not only a lower one.

Statistical tests use a fixed seed and tolerances wide enough that they do not
flake, then are checked across several seeds once when written. The tolerances
and the seed list live in the C1o packet.

## Workshop notebooks in CI

Workshop notebooks are jupytext percent-format `.py` files under
`workshop/NN_slug/` ([ADR 0010](../adr/0010-workshop-notebooks-jupytext.md),
[workshop README](../workshop/README.md)). `tests/workshop/` runs each one on
the fixture as a plain Python script in a subprocess and fails on any
exception. Each notebook ends with asserts on its key numbers (for example,
the leakage lab's leaky model must beat its clean model by a set margin), so a
library change that silently changes a workshop result fails CI. The
`workshop-solutions` branch runs the same tests on the completed exercises.

## Golden files

Golden files pin outputs that must not drift without someone deciding they
should.

| Golden | Location | Produced by | Regenerated when |
|---|---|---|---|
| Platform parity sessions | `tests/integration/golden/<family>/` | Platform backtest export (T7) | Platform `BACKTEST_ENGINE_VERSION` changes, or the family's logic changes on purpose |
| ONNX golden rows | `<run_dir>/export/golden.parquet` (in tests, a small committed copy) | `export_onnx` (M3) | The model changes |
| Report snapshot | `tests/integration/golden/report_hello_model.json` | `hello_model.yaml` on the fixture | The report format changes (C1i) |

Regenerate with an explicit command (named in the T7 and C1i packets), never as
a side effect of running tests. A PR that changes a golden says why in its
description.

## CI

`.github/workflows/ci.yml` today runs `uv sync --locked`, `ruff check` and
`pytest`. Target:

| Step | Command | Added by |
|---|---|---|
| Lint | `uv run ruff check` | exists |
| Layer contracts | `uv run lint-imports` | C1l |
| Unit and integration | `uv run pytest tests/unit tests/integration` | migration |
| Workshop | `uv run pytest tests/workshop` | W0 |
| Docs build | `uv run mkdocs build --strict` | C1n |

Optional extras (`ml`, `nn`) are installed in CI so model and ONNX checks run.
Tests that need an extra use `pytest.importorskip` so a minimal install still
passes.
