# Contracts (v0.1.0)

Every layer of this repo talks to the next through a small, versioned
contract. If your code takes or returns one of these shapes, any other
member's code can use it without coordination. The code is the source of
truth; this page explains the reasoning.

| Contract | Code | Validated by |
|---|---|---|
| Bars | `contracts/bars.py` | `validate_bars(df)` |
| Quotes, trades | `contracts/ticks.py` | `validate_quotes(df)`, `validate_trades(df)` |
| Dataset | `contracts/dataset.py` | `Dataset(...)` validates on construction |
| Model | `contracts/model.py` | the `Model` base class and `ModelManifest` |
| Feature | `features/base.py` | `assert_no_lookahead(feature, bars)` |

The version (`contracts/version.py`) is written into stored data, snapshots and
model manifests. Additive changes bump the minor version; anything that would
break existing readers bumps the major version.

## Time

All timestamps are `datetime64[ns, UTC]`. pandas 3 infers microseconds when
parsing strings, so convert with `contracts.base.to_utc_ns`, which keeps
Alpaca's nanosecond precision.

Two times matter for every observation:

- **`ts`**: when the interval *starts* (Alpaca's bar `t`). The 10:00 bar covers 10:00–10:01.
- **`ts_available`**: when it is fully known. For the 10:00 minute bar that is
  10:01. For a daily bar it is the session close, including early closes.

A decision may use a bar only at or after its `ts_available`. This one rule
prevents most look-ahead bugs.

## Bars

| Column | Type | Notes |
|---|---|---|
| `symbol` | str | |
| `ts`, `ts_available` | UTC ns | see above |
| `open`, `high`, `low`, `close` | float64 | as delivered; adjustment recorded in metadata |
| `volume` | int64 | |
| `vwap` | float64 | NaN when the source has none |
| `trade_count` | Int64 (nullable) | |
| `session` | str | `pre`, `regular`, `post` or `off` (`data/calendar.py`) |

- Rows are sorted by `(symbol, ts)` and unique on that pair.
- **Minutes without trades have no bar.** They are never filled in, and missing
  observations are never treated as zero returns.
- Feed (`sip`/`iex`), timeframe, adjustment (`raw`/`split`/`all`), source and
  contract version are stored as `BarMetadata` in the Parquet file and in the
  store manifest, not as columns.
- Default adjustment is `raw`, so prices match what was executable that day.
  Comparisons across days that span a split use `split`-adjusted bars.

## Quotes and trades

- **Quotes**: bid/ask price, bid/ask size, exchanges, tape and conditions.
  **Sizes are in round lots** (usually 100 shares) exactly as Alpaca sends
  them; convert explicitly before treating them as share counts. One-sided
  (`bid <= 0`) and crossed (`ask < bid`) quotes are dropped during
  normalization, and the drop is counted.
- **Trades**: price, size in shares, exchange, tape, conditions and trade id.
- Ticks may share a timestamp, so `ts` is not unique. A tick is available at
  its own `ts`; any receipt delay is a modelling assumption made explicitly,
  never baked into stored data.

## Dataset

`Dataset(X, y, meta, label_name)` holds three parts aligned on one index:

- `X`: numeric features in a fixed column order.
- `y`: the label, with no missing values.
- `meta`: `symbol`, `decision_ts`, `label_end_ts`, `weight` and `split`
  (`train`/`validation`/`test`/`unassigned`). Extra columns, such as the raw
  return behind a sign label, are allowed.

`label_end_ts` is when the label's outcome is fully known. Splitters use it to
drop training rows whose outcome overlaps a later period. A row whose label
ends at or before its decision is rejected.

## Model

A model subclasses `contracts.model.Model`, sets `model_type` and `task`, and
implements `_fit`, `_predict`, `_save_state` and `_load_state`. Decorate it
with `@register_model` so experiment configs and `load_model()` can find it.

- `predict(X)` reorders columns to the training order and fails on missing
  features. For classification it returns the probability of the positive class.
- **Scaling and imputation are fitted inside `_fit`** on the rows the model is
  given, never during ingestion. Otherwise statistics from validation or test
  data leak into training.
- `save(dir)` writes the model state and `manifest.json`: feature order, label,
  parameters, seeds, training window, metrics, library versions, contract
  version and git commit. The trading platform will read the same manifest when
  it loads exported models (task M3).

## Feature

A feature subclasses `features.base.Feature`, sets `feature_type` and
`requires`, gives each instance a `name` that encodes its parameters (for
example `past_return_5m`), and implements `compute(bars)` for one symbol's bars
sorted by `ts`. Register it with `@register_feature`.

The value on a row must be computable at that row's `ts_available` from that
row and earlier rows only. `assert_no_lookahead(feature, bars)` recomputes the
feature on truncated histories and fails if any value changes. Every feature
test calls it.
