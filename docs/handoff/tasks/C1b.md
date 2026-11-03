# C1b: Events and candidates layer

| Wave | Level | Depends on | Blocks | Agent |
|---|---|---|---|---|
| 1 | Int–Adv | C1a, C1o | SM1, SM2, SM3, SR1–SR3, SS4, W3, W4, W6 | `repo-architect` (review: `leakage-auditor`) |

## Goal

Make "when do we decide" a pluggable stage. Fill in the `CandidateFrame`
validator, implement the generic event samplers, and add the `candidates`
sampler that runs a registered strategy's `CandidateGenerator` so a study can
model only the bars where a strategy would act.

## Owns (exclusive)

- `src/shpe_research/contracts/candidates.py` (fill `validate_candidates`; names fixed by C1a)
- `src/shpe_research/events/types.py` (protocol docstrings only; signature fixed by C1a)
- `src/shpe_research/events/samplers.py`, `events/thresholds.py`, `events/breakouts.py`
- `tests/unit/events/test_samplers.py`, `tests/unit/events/test_candidates.py`

## Reads

- [design/pipeline.md](../../design/pipeline.md) (Decisions stage)
- [adr/0004-events-candidates-unit-of-decision.md](../../adr/0004-events-candidates-unit-of-decision.md)
- `src/shpe_research/pipeline/registry.py`, `strategies/registry.py` (C1a)
- `src/shpe_research/contracts/bars.py` (`ts`, `ts_available`, sessions)

## Interface

`validate_candidates(df) -> None` raises `ContractError("candidates", problems)`
on: a missing required column, `side` outside `{long, short}`, `action` outside
`{enter, exit}`, an `exit` row without `exit_reason`, a non-UTC `decision_ts`,
or duplicate `(symbol, decision_ts, strategy_id, action)` rows.

Registered samplers (`@register_sampler`), each `sample(bars, **params) -> CandidateFrame`:

| Key | Params | Emits |
|---|---|---|
| `every_n_minutes` | `n` | Moved from C1a's shim; one row per symbol every `n` regular-session minutes |
| `session_open` | `offset_minutes` | One row per symbol per session at 09:30 + offset (decision at that bar's `ts_available`) |
| `session_close` | `offset_minutes=1` | One row per symbol per session at the last regular bar's `ts_available` minus offset; for daily bars, the bar's `ts_available` (the close). Used by W3 and SM3 |
| `threshold` | `column`, `upper`, `lower` | Rows where a precomputed column crosses a level (first bar after the cross) |
| `cusum` | `column`, `h` | Symmetric CUSUM filter (López de Prado, AFML ch. 2) on a precomputed column; resets after each event |
| `range_breakout` | `range_minutes`, `entry_cutoff`, `confirm` (`close`\|`touch`), `side` | First close above (or touch of) the opening-range high after the range ends, one per symbol and session. W6 and SM1 use it; SM1 adds stop and target levels on top |
| `candidates` | `strategy` (path to a strategy config) | Calls `build_strategy(...).generate(bars, context)` |

`side` defaults to `long` for generic samplers; `range_breakout` with
`side: short` uses the range low.

## Behaviour

- Every `decision_ts` equals the `ts_available` of a bar that exists for that
  symbol. A sampler never emits a decision at a time no bar was known.
- Samplers read only bars with `ts_available <= decision_ts`. `range_breakout`
  computes H from bars whose `ts_available` is at most 09:30 + `range_minutes`;
  the breakout check starts on the next bar.
- Pre- and post-market bars are ignored unless the sampler's params say so.
- Missing minutes: the sampler skips them, it never forward-fills a bar.
- Ties: one row per `(symbol, decision_ts, strategy_id, action)`.
- Output is sorted by `decision_ts`, then `symbol`, with a fresh RangeIndex.

## Acceptance tests

| Path | Test | Asserts |
|---|---|---|
| `tests/unit/events/test_candidates.py` | `test_validator_rejects` | Each problem above raises `ContractError` naming the column |
| | `test_validator_accepts_minimal` | Four required columns pass; `action` defaults to `enter` |
| `tests/unit/events/test_samplers.py` | `test_decision_is_bar_ts_available[key]` | For each sampler on `synthetic_bars()`, every `decision_ts` is in the symbol's set of `ts_available` |
| | `test_session_close_daily_and_minute` | One row per symbol per session; daily rows equal the bar's `ts_available` |
| | `test_range_breakout_planted` | On C1o `momentum` fixture with `range_minutes=15`, events fire on the planted breakout sessions and on no session before 09:45 |
| | `test_range_breakout_no_lookahead` | Truncating bars after `decision_ts` leaves every earlier event unchanged |
| | `test_cusum_hand_case` | A hand-built series with known crossings returns exactly those rows |
| | `test_candidates_sampler_dispatch` | `{type: candidates, strategy: ...}` builds a registered dummy strategy and returns its frame validated |

Once C1k merges, each sampler also passes `check_strategy`.

## Out of scope

- Strategy logic (S packets). `range_breakout` is the generic event; SM1 owns
  ORB stops, targets and filters.
- Filters and skip reasons (T3).

## Done when

All tests above pass, the hello model still matches its golden report, and a
`decisions: {type: session_close}` config runs end to end on the fixture.
