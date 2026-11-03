# Session 2: Features without leakage

| | |
|---|---|
| Level | Beg–Int |
| Duration | 75 min |
| Prerequisites | Session 1 |
| Data | Fixture |
| Code used | `features/base.py`, `features/returns.py`, the F1–F5 stubs |
| Ends with a PR for | One of F1 (relative volume), F2 (VWAP distance), F3 (realized volatility), F4 (opening range), F5 (regime set) |

## Why this session

A feature is a function of the past. The repo makes that testable: compute
the feature on all the data, then on a prefix with the later bars removed,
and the overlapping values must be identical. If they differ, the feature
read something from after its own row.

## Learning objectives

1. Write a feature as a class registered with `@register_feature`.
2. Explain why `assert_no_lookahead` catches look-ahead and what it cannot
   catch.
3. Recognise the common pandas operations that look at future rows.
4. Handle session boundaries: a 30-minute return at 09:35 is NaN, not a
   return that reaches into yesterday.

## Background

### The feature contract

```python
@register_feature
class PastReturn(Feature):
    feature_type = "past_return"
    def __init__(self, window: int): ...
    @property
    def name(self) -> str:            # "past_return_5m"
        ...
    def compute(self, bars) -> pd.Series:
        """One symbol's bars sorted by ts -> one value per row, known at ts_available."""
```

`compute_features` calls `compute` once per symbol and returns one column per
feature, keyed by `symbol` and `ts_available`. A config builds features
by type:

```yaml
features:
  - {type: past_return, window: 5}
  - {type: past_return, window: 30}
  - {type: minute_of_day}
```

### The prefix check

`assert_no_lookahead(feature, bars, cutoffs)` in `features/base.py`:

1. computes the feature on the full history,
2. recomputes it on prefixes ending at several cutoffs,
3. fails if any value on a row inside the prefix changed.

A causal feature cannot change when later rows are removed. This catches the
bugs that matter most: centered windows, `shift(-k)`, whole-day statistics
and normalisation by full-sample moments.

What it cannot catch:

| Not caught | Why | Where it's caught instead |
|---|---|---|
| A scaler fitted on all rows | Happens in the model, not the feature | `SklearnModel` fits its `StandardScaler` inside `fit()` on train rows only |
| Survivorship (the universe chosen with hindsight) | The feature is causal; the set of symbols is not | D4 universe `as_of`; [example 1](../examples/01-leakage-lab.md) |
| A label that overlaps the next split | Labels, not features | Labels never cross the session close; V1 purging for multi-day labels |
| Choosing features by looking at test | A human decision | Research protocol, trial counting |

### Pandas operations that look ahead

| Operation | Problem |
|---|---|
| `rolling(n, center=True)` | Uses `n/2` future rows |
| `shift(-k)` | Reads `k` rows ahead |
| `groupby(day).transform("mean")` | Every row sees the whole day |
| `(x - x.mean()) / x.std()` | Full-sample moments |
| `bfill()` | Copies a future value backwards |
| `resample(...).last()` labelled at the interval start | The value is from the end of the interval |
| `merge_asof(..., direction="forward")` or `"nearest"` | Picks a later row |

## Walkthrough

1. **Read `PastReturn`** in `features/returns.py`. It uses `merge_asof`
   within the session, so the 30-minute return at 09:35 is NaN.

2. **Run the check on it:**

   ```python
   from shpe_research.data.fixtures import synthetic_bars
   from shpe_research.features.base import assert_no_lookahead
   from shpe_research.features.returns import PastReturn
   one = synthetic_bars(symbols=("AAA",), end="2026-03-31")
   assert_no_lookahead(PastReturn(5), one)   # passes silently
   ```

3. **Break it three ways.** Write three small features live and watch each
   fail with the row and values in the error message:

   - `CenteredMean`: `close.rolling(11, center=True).mean() / close - 1`
   - `DayZScore`: the close's z-score within its whole session
   - `NextReturn`: `close.shift(-1) / close - 1`

   Then show one that passes but is still wrong: a 5-minute return computed
   across the overnight gap. It is causal, but it mixes the overnight gap into
   an intraday feature. The check cannot catch that kind of mistake, so tests
   must.

4. **Write a real one together:** F3 realized volatility, a rolling standard
   deviation of 1-minute log returns, reset at each session open. Run
   `assert_no_lookahead`, then the acceptance test:

   ```bash
   uv run pytest tests/todo -k test_realized_vol
   ```

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Fix `DayZScore` so it uses only the session so far (expanding mean and standard deviation) | `groupby(day).expanding()` or cumulative sums; the first rows of the day are NaN |
| Core | Explain, for each row of the look-ahead table, why the prefix check fails | Each operation's value at row i depends on rows after i |
| Stretch | Write a feature that passes `assert_no_lookahead` but leaks anyway, then write a unit test that catches it | E.g. a "previous close" that is really today's close when bars are missing; test with a hand-built frame |
| Stretch | Show that `assert_no_lookahead` with a single cutoff can miss a leak that a different cutoff catches | Run `DayZScore` with a cutoff exactly at a session boundary (passes: every day in the prefix is complete) and one mid-session (fails) |
| Challenge | Implement F4 (opening range) to its spec in `features/opening_range.py`, including the worked example (H=100, L=98, close 100.10 → breakout distance 0.001) | The range is known at 09:45; earlier rows are NaN |

## Discussion

- Why does the repo compare prefix recomputations instead of reading the
  feature's code?
- A feature uses yesterday's full-day volume. Is it causal at 09:31 today?
- When is a NaN feature better than a filled one?

## Pitfalls

- Computing a feature on all symbols at once without grouping, so one
  symbol's rolling window spills into the next.
- Forgetting session resets: a 30-minute window at 09:40 that includes
  yesterday's last 20 minutes.
- Dropping NaN rows inside `compute`. That changes the row count and breaks
  the alignment; return NaN instead.

## Connection to team strategies

| Team | Features from this session |
|---|---|
| [Momentum](../../teams/momentum.md) | F1 relative volume and F4 opening range drive the ORB (SM1) |
| [Reversals](../../teams/reversals.md) | F2 VWAP distance and F3 realized volatility drive the stretch fade (SR2) |
| [Stat arb](../../teams/stat-arb.md) | F7 pair spread follows the same rules with two symbols, which needs the shared cross-asset contract (F7/F8) |

The `feature-engineer` agent (`.claude/agents/feature-engineer.md`) follows
the same checklist.

## Wrap-up PR

Claim one of F1–F5. The first commit can be the class skeleton plus the
acceptance test moved out of `tests/todo/`.
