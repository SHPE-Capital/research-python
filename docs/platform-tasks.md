# Platform tasks (trading-platform repo)

These tasks connect research output to the trading platform. They are
TypeScript work in [SHPE-Capital/trading-platform](https://github.com/SHPE-Capital/trading-platform);
paths below are relative to that repo. Claim them there with an issue that
references the ID.

## P1: Candidate and feature log

**Why:** every strategy proposes candidates; recording each one with its
features and outcome, whether taken or skipped, turns live paper trading and
every backtest into training data. It is also what the Replay viewer (P4) shows.

- Define a `StrategyCandidate` record: strategy id/type/version, symbol,
  decision time, side, feature values (name → number), decision
  (`accepted`/`rejected` plus reason), model score when a selector ran, and the
  signal id when one was emitted. The feature names must match this repo's
  feature names (see `docs/contracts.md` here).
- Publish it on the EventBus (`backend/src/core/engine/eventBus.ts`) from the
  strategy layer (`backend/src/strategies/base/strategy.ts`, `backend/src/types/strategy.ts`).
- Persist it in a new table via the next migration in `supabase/migrations/`
  (latest today: `0019_expire_stranded_sandbox_runs.sql`), for both backtests
  and paper runs.
- **Done when:** a backtest of an existing strategy writes one row per
  candidate, and a member can export those rows to Parquet for research.

## P2: ONNX selector wrapper

**Depends on:** M3 (export format), P1.

- A wrapper that takes any `IStrategy`, intercepts its signals as candidates,
  computes the feature row, runs the exported model in-process with
  `onnxruntime-node`, and accepts, skips or resizes the signal. Strategies may
  not call external services, so no Python inference server.
- Load `model.onnx` + `manifest.json`; refuse to start if the manifest's
  feature names are not all produced by the platform.
- **Parity test:** load M3's golden rows and require predictions within the
  manifest's tolerance in CI.
- **Done when:** a backtest can run the same strategy with and without the
  wrapper, and P1 logs the score for every candidate.

## P3: Replay data path

The Replay page is a stub today (see `docs/guides/replay.md` in that repo):

- `backend/src/core/replay/replayFilter.ts`: `applyFilter` returns its input unchanged (TODO in file).
- `backend/src/app/controllers/replayController.ts`: session listing returns
  `[]`, and `/api/replay/load` returns 501 because nothing reads `event_logs`.
- Nothing writes `event_logs` (`supabase/migrations/0000_initial_schema.sql`).
- `backend/src/core/replay/attributionCollector.ts`: the report method throws.

Rebuild it as a **read-only viewer over recorded runs**, not a second
simulator. The backtest engine stays the only source of fills:

- Implement `applyFilter`.
- Build a `ReplaySession` from a run's P1 candidates, its orders and fills, and
  the cached bars for its symbols and dates, instead of from `event_logs`.
- **Done when:** `/api/replay/load` returns a session for a saved backtest.

## P4: Replay decision viewer

**Depends on:** P3. Files: `frontend/app/replay/page.tsx`, `frontend/components/controls/ReplayControls.tsx`.

- Chart with bars and, at each candidate, its features, model score, decision,
  order and fill. Step from one candidate to the next.
- Side-by-side mode: the same day from two runs (live vs. backtest, or
  selector A vs. C), highlighting where they diverge.
- **Done when:** a member can explain one trade end to end from the viewer.
  This is a workshop deliverable.

## P5: Models tab

**Depends on:** M3.

- Read-only list of exported models: name, type, training window, validation
  metrics and status (`shadow`, `approved`, `retired`), from their manifests.
- **Done when:** the tab lists every model in the registry location the team
  chooses, and links each to its research report.

## P6: Wire position sizers

**Why:** the platform has a sizer interface (`IPositionSizer.computeQty`) and a
`SizerType` enum, but no strategy calls it. Each strategy emits its own `qty`
(pairs via `tradeNotionalUsd`, reversal via `qtyPerTrade`), and
`volatilityScaledSizer` and `kellyFractionalSizer` are scaffolds that return 0.
The risk-per-trade and inverse-volatility sizing that T2 studies, and that P7,
P8 and P9 need, has nowhere to run until the orchestrator asks a sizer for the
quantity.

**Depends on:** T2 (the research sizing rules and the evidence for their
defaults). Files: `backend/src/core/sizing/`,
`backend/src/core/engine/orchestrator.ts`, the `SizerType` definition, and
`docs/architecture/position-sizing-layer.md`.

- Implement `volatilityScaledSizer` (target dollar risk divided by price times
  a volatility estimate over a configured lookback) and `kellyFractionalSizer`
  (a fraction of the Kelly size from a configured win rate and payoff, or from
  the meta-model probability when one is given). Both return a non-negative
  integer and respect `maxPositionSizeUsd`.
- Add a risk-per-trade mode: quantity = floor(equity x `riskPerTradePct` /
  |entry - stop|), for strategies that know their stop price at entry.
- In the orchestrator, when a strategy config names a `sizerType`, replace the
  strategy `qty` with the sizer result. When it does not, keep the current
  behaviour, so existing strategies and their backtests do not change.
- Unit tests in `backend/tests/` for each sizer against hand-computed
  quantities, using the same cases as T2 (`tests/unit/trade/test_sizing.py` in
  this repo), so both sides agree on rounding and caps.
- Document the sizer types in `docs/architecture/position-sizing-layer.md`.
- **Done when:** a backtest with `sizerType: "volatility_scaled"` sizes trades
  to within one share of the T2 `inverse_vol` rule on the same bars, and every
  existing strategy backtest is unchanged when no `sizerType` is set.

## P7: Reversal strategy family

**Why:** the reversals team research (SR1-SR3) generalises the current bar fade
into a stretch fade with sigma-scaled entries and exits, an optional
SPY-residual version, time windows, regime filters and volatility sizing. P7
makes that one parameterised strategy, so any configuration the team validates
runs on the platform by changing a config rather than writing a new strategy.

**Depends on:** SR1, SR2, SR3 (definitions and goldens), T7 (exported config
and parity report). P6 if the `inverseVol` and `stretchInverseVol` sizing modes
use the shared sizers. Files: `backend/src/strategies/minuteReversal/` (becomes
the family, or moves to a new `backend/src/strategies/reversal/`),
`backend/src/config/strategyCatalog.ts`.

- New fields, as listed in the reversals guide
  ([teams/reversals.md](teams/reversals.md), section 8): `anchor` (`prevClose`
  | `vwap` | `ema`), `emaBars`, `volLookbackBars`, `entryThresholdSigma`,
  `residualVsSpy` (`{ enabled, betaLookbackBars, hedge }`), `exitAtSigma`,
  `maxHoldBars`, `disasterStopSigma`, `tradeWindow` (`{ start, end, flatBy }`),
  `regimeFilter` (`{ measure, window, max }`), `sizing` (`fixedQty` |
  `fixedNotional` | `inverseVol` | `stretchInverseVol`), `targetRiskUsd`.
- Existing fields (`symbols`, `regularHoursOnly`, `maxBarGapMs`,
  `maxPositionSizeUsd`, `cooldownMs`, `qtyPerTrade`, `maxPositionQty`) keep
  their meaning and defaults.
- `minuteReversal` and `mcarrasco_reversal` become catalog presets of the
  family. `beginnerReversal` is frozen and not changed.
- A baseline preset must reproduce the current `minuteReversal` trades exactly.
  That is the first test; it proves P7 changed no existing behaviour.
- Feature definitions (VWAP, EMA, realised sigma, SPY beta and residual,
  regime measures) must match this repo's F2, F3, F5 and F8; see
  [design/platform-parity.md](design/platform-parity.md).
- Tests in `backend/tests/strategy/` for each anchor, the sigma entry and exit,
  the disaster stop, `maxHoldBars`, `flatBy` and the regime filter. Bump the
  strategy `VERSION` when any default changes. Update
  `docs/guides/adding-a-strategy.md` if the family-with-presets pattern is new.
- **Done when:** the baseline preset matches `minuteReversal` trade for trade
  on the SR1 golden sessions, and the T7 `ParityReport` passes for the
  reversals team chosen configuration on at least 5 sessions.

## P8: Momentum ORB strategy

**Why:** `backend/src/strategies/momentum/` is a stub today, with no catalog
entry. The momentum team research (SM1, R1, RM studies) produces an
opening-range breakout with a range length, entry cutoff, stop and trail rules,
a time exit, volume and regime filters, risk-per-trade sizing and an optional
meta-model. P8 is the platform implementation of that strategy.

**Depends on:** SM1 (candidate definitions and goldens), T7 (exported config
and parity report), P6 (risk-per-trade sizing; until P6 lands, P8 may size
inside the strategy and must say so in its PR). P1 and P2 for the meta-model
fields. Files: `backend/src/strategies/momentum/`,
`backend/src/config/strategyCatalog.ts`.

- Config fields, as listed in the momentum guide
  ([teams/momentum.md](teams/momentum.md), section 8): `rangeMinutes`,
  `entryCutoff`, `confirmRule` (`close` | `touch`), `stopRule` (`range_mid` |
  `range_low` | `atr` | `none`), `stopAtrMult`, `trailRule` (`none` |
  `chandelier` | `breakeven_1r` | `target_r`), `maxHoldMs`, `flatBy`,
  `riskPerTradePct`, `maxConcurrent`, `maxNetLongPct`, `minRelVolume`,
  `regimeFilter` (`none` | `spy_above_open` | `efficiency_min`),
  `metaModelPath`, `minProbability`.
- **Durations are stored in milliseconds**, as `PairsStrategy` does with
  `maxHoldingTimeMs`: the research spec's `max_hold_minutes` becomes `maxHoldMs` in the
  stored config, and T7 converts from the research spec's minutes.
  `rangeMinutes` stays a count of minutes because it names a bar window, not a
  wait. Record the choice in the spec card.
- Entries are buy-stop orders at the range high. Under the bar model a buy stop
  fills at max(open, stop), the same rule T1 uses in research.
- No entry before the range closes (09:30 + `rangeMinutes`). SPY is a context
  symbol for `regimeFilter` and is never traded.
- Tests in `backend/tests/strategy/` for each stop and trail rule, the entry
  cutoff, `flatBy` on a normal and an early-close day, and the RVOL and regime
  filters. Add the catalog entry with a `VERSION`. Update
  `docs/guides/adding-a-strategy.md` if anything in the process changed.
- **Done when:** the strategy is in the catalog, its candidates appear in the
  P1 log, and the T7 `ParityReport` passes (candidates match exactly, fills
  match, P&L within $0.01 per trade and $0.05 per session) on at least 5
  sessions, then on one month.

## P9: Pairs v7

**Why:** the stat-arb team research (SS1-SS4, RS studies) may choose a hedge
method, stop rule, holding limit or sizing mode the current `PairsStrategy`
cannot express. P9 adds those options without changing the behaviour of
existing configs.

**Depends on:** SS2, SS3, SS4 (estimator definitions and goldens), T7
(exported config and parity report). P6 for `spread_vol` sizing through the
shared sizers. Files: `backend/src/strategies/pairs/pairsTypes.ts`,
`backend/src/strategies/pairs/pairsConfig.ts`, the pairs strategy itself,
`backend/src/services/indicators/ols.ts`, `cointegration.ts`, `zscore.ts`, and
a new Kalman estimator beside them.

- New fields, as listed in the stat-arb guide
  ([teams/stat-arb.md](teams/stat-arb.md), section 8.2): `priceSpace` (`raw`
  default | `log`), `hedgeRatioMethod` adds `kalman` and `tls`, `kalmanDelta`,
  `kalmanObsVar`, `stopRule` (`z` | `spread_loss` | `coint_break` |
  `combined`), `stopSpreadSigma`, `exitOnCointBreak`, `maxHoldHalfLives`,
  `flatBy`, `earningsBlackoutSessions`, `sizingMode` (`spread_units` |
  `dollar_neutral` | `beta_neutral` | `spread_vol`), `targetRiskUsd`,
  `maxScaleIns`, `scaleInZScore`, `postStopCooldownMs`, `metaModelPath`,
  `minProbability`.
- Every new field defaults to the current behaviour. A config without them must
  produce the same trades as the current version.
- The Kalman and TLS hedges must match the SS2 estimators in
  `features/cross_asset/pair_math.py` on the same price vectors (test vectors
  exported from this repo).
- `earningsBlackoutSessions` needs an earnings calendar on the platform; if
  none is available, ship the field disabled and record that in the PR.
- Tests in `backend/tests/strategy/` and alongside the indicator tests for each
  hedge method, stop rule and sizing mode. Bump the strategy `VERSION`.
- **Done when:** existing pairs configs reproduce their trades unchanged, the
  new estimators match the research test vectors, and the T7 `ParityReport`
  passes for the stat-arb team chosen configuration on at least 5 sessions.
