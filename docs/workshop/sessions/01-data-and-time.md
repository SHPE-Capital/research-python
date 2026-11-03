# Session 1: Data and time

| | |
|---|---|
| Level | Beg |
| Duration | 60 min |
| Prerequisites | [Setup](../setup.md) done; quickstart ran |
| Data | Fixture; optionally the snapshot |
| Code used | `data/calendar.py`, `data/audit.py`, `data/fixtures.py`, `contracts/bars.py` |
| Ends with a PR for | D6 (corporate actions), or a data-audit issue found in the snapshot |

## Why this session

Almost every bad result in this repo's future will come from one mistake: a
row using information that was not available when the decision was made.
Before features or models, participants need a precise sense of when each
piece of data becomes known.

## Learning objectives

1. Read a bar and say exactly when it became known.
2. Explain sessions (pre, regular, post, off), early closes and why the repo
   stores UTC.
3. Run a data audit and interpret coverage, duplicates, invalid prices and
   split gaps.
4. Explain why a missing minute is left missing, never filled.

## Background

### Bars, quotes and trades

| Data | What one row is | Where in the repo |
|---|---|---|
| Trade | One print: time, price, size, exchange | `data/ticks.py` (D3) |
| Quote | Best bid and ask at a moment | `data/ticks.py` (D3) |
| Bar | Summary of the trades in an interval: open, high, low, close, volume, VWAP, trade count | `data/bars.py` (D1) |
| Minute summary | Per-minute spread, imbalance and signed volume from ticks (Lee-Ready signing) | `shpe data tick-summary` (D3) |

The workshop uses 1-minute bars. Ticks appear only in F6 and the reversals
team's microstructure work.

### `ts` vs. `ts_available`

The bars contract (`contracts/bars.py`) has two times per row:

| Column | Meaning | 10:00 bar |
|---|---|---|
| `ts` | Start of the interval | 10:00:00 |
| `ts_available` | Earliest time the bar is fully known | 10:01:00 |

The 10:00 bar contains trades from 10:00:00 to 10:00:59. Its close is not known
until 10:01. **A decision made using that bar happens at 10:01 or later.**
The contract rejects any row where `ts_available <= ts`.

The pipeline enforces this: `build_dataset` renames `ts_available` to
`decision_ts` and joins features to labels on it. Labels start from the
decision bar's close at `decision_ts`.

### Sessions and the calendar

`TradingCalendar` (XNYS, from `exchange_calendars`) knows every session,
holiday and early close. `tag_sessions` labels each timestamp `pre`,
`regular`, `post` or `off`. Experiments default to `session: regular`.

Everything is stored in UTC. New York time shifts by an hour twice a year,
so a "09:30" filter written against UTC breaks in March and November.
Convert to `America/New_York` only to display or to express rules like
"first 15 minutes".

### Missing data

An illiquid stock may have no trades in a minute, so there is no bar. The repo
never fills that minute. Forward-filling the close invents a price at which
nobody traded and creates zero returns that look like a calm market. Labels
return NaN when no bar printed during the horizon (`labels/forward_return.py`).

## Walkthrough

1. **Look at a bar.** Load the fixture and print the first five rows of
   one symbol:

   ```python
   from shpe_research.data.fixtures import synthetic_bars
   bars = synthetic_bars()
   bars[bars.symbol == "AAA"].head()[["ts", "ts_available", "open", "close", "session"]]
   ```

   Ask: at what New York time could you first act on the second row?

2. **Calendar.** Show the schedule for a week with a holiday and find an early
   close:

   ```python
   from shpe_research.data.calendar import get_calendar
   cal = get_calendar()
   cal.schedule("2026-11-23", "2026-11-28")  # Thanksgiving week: closed Thu, early close Fri
   ```

3. **Audit.** Run the audit on the fixture, then break the data and run it
   again:

   ```python
   from shpe_research.data.audit import audit_bars
   report = audit_bars(bars)
   report.symbols              # one row per symbol
   broken = bars.drop(bars.sample(frac=0.02, random_state=0).index)
   audit_bars(broken).issues   # coverage now below 100%
   ```

   Then add a duplicate row and a bar with `high < close`, and find both in
   `report.issues`.

4. **Split gaps.** Multiply one symbol's prices before a chosen date by 2 to
   fake an unadjusted 2-for-1 split. The audit flags an overnight gap larger
   than `split_threshold` (0.25 in log terms by default). The audit only
   flags it. Confirming it against real corporate actions is D6.

5. **Optional, with the snapshot:** `load_snapshot(...).bars(...)` on one dev
   name, then `audit_bars` on it. Real data has low-coverage days; find one
   and look at what happened that day.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | For a decision at 10:15 New York time, list which fixture bars may be used. Write it as a pandas filter on `ts_available` | `ts_available <= decision`; the 10:14 bar is the last one |
| Core | Count regular-session minutes on a normal day and on the day after Thanksgiving | 390 and 210 |
| Core | Plant three faults (missing minutes, duplicate, invalid OHLC) and find each in `audit_bars(...).issues` | Each fault has its own `kind` in the issues table |
| Stretch | Show that forward-filling missing minutes changes the 5-minute realized volatility of an illiquid symbol | Filled minutes add zero returns, so volatility drops |
| Challenge | Start D6: given the audit's split-gap flags, confirm them against Alpaca corporate actions (`data/corporate_actions.py`) | Acceptance test: `tests/todo/test_experiments_todo.py::test_corporate_actions` |

## Discussion

- A model that uses the 10:00 bar to decide at 10:00:30 looks fine in a
  backtest. What happens live?
- Why does the repo record trades and quotes separately instead of only bars?
- When would you want the pre-market session in a feature, and what new time
  problems does it bring?

## Pitfalls

- Filtering on `ts` instead of `ts_available` when building "data known by
  time T".
- Writing time-of-day rules in UTC.
- Treating a missing bar as "price unchanged".
- Using split-adjusted daily bars and raw minute bars together without saying
  which is which. Snapshots carry both (`1Min:raw`, `1Day:split`).

## Connection to team strategies

Every team guide ([momentum](../../teams/momentum.md),
[reversals](../../teams/reversals.md), [stat arb](../../teams/stat-arb.md))
lists decision times for its strategies. The ORB's opening range
09:30–09:44 is known at 09:45, not 09:44; a pair's spread is known only when
both legs' bars are available.

## Wrap-up PR

Claim D6, or open an issue for any audit finding on the snapshot, with the
`audit_bars` output attached as a table (not the data).
