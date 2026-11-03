# Reversals team guide

How the Reversals team turns the platform's "fade every minute bar" strategies
into a researched, sized, risk-limited strategy, and what the team builds in
this repo versus what it uses from the shared toolkit.

Read first: [teams/README.md](README.md), the method in
[research/research-protocol.md](../research/research-protocol.md) and
[research/trade-management.md](../research/trade-management.md), and the
design in [design/trade-layer.md](../design/trade-layer.md). IDs on this page
come from the [documentation map](../README.md).

## 1. Mission and ownership

**Mission:** find out whether short-horizon price reversal in the club's
large-cap universe can be traded after costs with the platform's fills, and if
it can, hand the platform one parameterised reversal strategy with a spec card
and a parity check behind it.

| The team owns | Where |
|---|---|
| Reversal candidate generators SR1–SR3 | `src/shpe_research/strategies/reversal/` |
| Team studies RR0–RR6, plus R2 | `studies/reversals/` |
| R4 (order flow) after the move from `studies/r4-microstructure-ofi/` | `studies/reversals/` (jointly with the Microstructure members who started it) |
| The reversal spec card and the P7 config | `studies/reversals/spec-card.md`, trading-platform P7 |
| First claim on F2, F3, F5 (the features the team depends on most) | `src/shpe_research/features/` |

The team does **not** own the trade simulator, sizing, risk or cost code. Those
live in the shared `trade/` package (T1–T5). If a study needs something
`trade/` lacks, open an issue against the T task rather than writing a private
version in a study folder.

## 2. Baseline: what the platform runs today

Three strategies in `trading-platform/backend/src/strategies/` implement the
same rule:

| Strategy | Files | Defaults |
|---|---|---|
| `minuteReversal` | `minuteReversal/minuteReversalStrategy.ts`, `minuteReversalConfig.ts` | 10 symbols (NVDA, AAPL, MSFT, MU, AMD, AMZN, META, GOOGL, SPCX, TSLA), `qtyPerTrade: 1`, `maxPositionQty: 1`, `maxBarGapMs: 180000`, `regularHoursOnly: true`, `maxPositionSizeUsd: 10000`, `cooldownMs: 0`, market orders |
| `mcarrasco_reversal` | `mcarrasco_reversal/strategy.ts`, `manifest.ts` | SPY, 1 share, `maxPositionQty: 1`, `VERSION = 4` |
| `beginnerReversal` | `beginnerReversal/beginnerReversalStrategy.ts` | Teaching version: `qtyPerTrade`, `maxPositionQty`, bar trigger only |

**The rule:** on each new regular-session 1-minute bar, if the close is above
the previous bar's close, sell `qtyPerTrade`; if below, buy; if unchanged, do
nothing. Skip the bar if the gap to the previous bar exceeds `maxBarGapMs`.
Skip the order if it would take the absolute position past `maxPositionQty`.

What that means as a trading strategy:

- **Entry:** every non-flat bar. There is no size threshold, so most signals
  respond to moves smaller than the round-trip cost.
- **Holding period:** implicit. With `maxPositionQty: 1` the position is −1, 0
  or +1. A long opened after a down bar is closed by the next up bar, which can
  be one minute later or many.
- **Stops and targets:** none.
- **Sizing:** one share regardless of price or volatility. One share of NVDA
  and one share of SPY carry very different dollar risk.
- **Filters:** regular hours and the bar-gap check only. No regime, time-of-day
  or market-move condition.
- **Fills (bar model):** the decision is made at the bar's end. The order fills
  at the next bar's open ± `halfSpreadBps` ± `slippageBps`
  ([backtest-execution.md](https://github.com/SHPE-Capital/trading-platform/blob/main/docs/architecture/backtest-execution.md)).
  Use the platform `DEFAULT_FILL_MODEL`: 2 bps half spread, 5 bps slippage and
  $0.005 per share, so a round trip costs about **14 bps plus commission**
  (about 16 bps on a $50 stock). The `labels/exit_policy.py` stub still
  defaults to 1 / 2 / 0 (about 6 bps), which understates costs; T1 switches to
  the platform values ([research/trade-management.md](../research/trade-management.md)).

The platform risk engine still applies on top: `MAX_POSITION_SIZE`,
`ORDER_COOLDOWN`, `INTRADAY_DRAWDOWN`, `CONCENTRATION_LIMIT` and
`SHORT_SELLING_DISALLOWED`. `allowShortSelling` defaults to `true` in
`backend/src/config/defaults.ts`. If an environment disables it, the sell side
of the fade can only close longs, which halves the strategy. Record which
setting each backtest used.

## 3. Hypotheses and rationale

Every hypothesis below is a claim to test. None is a result.

| ID | Hypothesis | Why it might hold | Why it might not |
|---|---|---|---|
| H0 | The bar-fade baseline has negative expected P&L per trade after the bar model's costs. | The average 1-minute move in these names is a few bp, which is about the round-trip cost. | Large names with one-cent spreads may have enough reversal to clear 6 bp. Study RR0 measures it. |
| H1 | Much of the negative autocorrelation in 1-minute **close-to-close** returns is bid-ask bounce. | Roll (1984): if trades land randomly at the bid or the ask, successive trade-price changes are negatively autocorrelated even when the midpoint follows a random walk. A bar close is a trade price. | Megacap spreads are often one cent, about 0.5–2 bp, so the bounce component may be small next to minute volatility. Measure it, don't assume it. |
| H2 | Reversal after **large** moves (in volatility units) is strong enough to clear costs, while reversal after small moves is not. | Liquidity-provision explanations of short-horizon reversal (Lehmann 1990; Jegadeesh 1990, at weekly and monthly horizons): price pressure from a large order fades once liquidity returns. | The effect may be a different size at minute horizons, or it may sit in news-driven moves, which continue rather than revert. |
| H3 | Fading the **residual** move (stock minus β·SPY) beats fading the raw move. | A move driven by the market is not a mispricing in one stock. Avellaneda & Lee (2010) trade mean reversion of residuals against factors. | At 1-minute horizons β is noisy, and residuals add estimation error. |
| H4 | Reversal pays in mean-reverting regimes and loses on trend days. | Variance ratio < 1 and a low efficiency ratio describe the conditions in which fades work. | The regime may not persist long enough to forecast from a trailing window. |
| H5 | Order-flow exhaustion (falling signed volume, an imbalance flip) separates moves that revert from moves that continue. | Price pressure ends when the aggressive flow ends. | The signal may only exist at sub-minute horizons (R4's question). |

**Note on the fixture:** the synthetic fixture (`data/fixtures.py`) has a
planted reversal: with `signal_strength`, each minute's return leans against
the previous five. Every generator and study must find that effect on the
fixture before it runs on real data. On the fixture the edge exists by
construction; on real data it may not.

## 4. Research roadmap

Run the studies in order; each one's decision rule gates the next. All of them
use the frozen workshop snapshot (D5) and the **dev** symbols of the universe
(D4). Held-out symbols are opened once, for the final spec card. Time splits
are train / validation / test by session with a gap of at least one session.
Use the purged walk-forward splitter (V1/C1f) as soon as it lands; until then,
`split_by_day` with `gap_days >= 1`.

Every configuration a study evaluates is recorded in the experiment's trial
count, which feeds the deflated Sharpe ratio (E2). Each trial budget below is
a cap, not a target.

### RR0: Edge check and baseline parity

| | |
|---|---|
| Question | What does the platform's bar fade earn per trade, gross and net, and does research reproduce the platform backtest? |
| Data | 10 `minuteReversal` symbols plus SPY, the full snapshot. Quotes (D3) for one week of 5 symbols for the bounce check. |
| Split | None for the parity check (it is a reconciliation). Gross/net numbers reported on validation sessions. |
| Metric | Per-trade gross and net return in bp; trade count; hit rate; the share of trades whose gross move is smaller than the round-trip cost. |
| Bounce check | Lag-1 autocorrelation of 1-minute returns computed (a) on bar closes and (b) on quote midpoints at the same minute ends (D3 minute summaries). The gap between (a) and (b) estimates the bounce component. |
| Decision rule | Parity must pass (see section 8) before any other study is trusted. The net edge number becomes the baseline every later study must beat. |
| Trial budget | 1 (no tuning) |
| Toolkit | SR1, T1 (`simulate_trades` with the bar `FillModel`), T5 `CostModel`, T7 `ParityReport` |
| R2/R4 relation | Produces the baseline ("take all") row of R2. The bounce check is R4's first measurement at minute scale. |

### RR1: Cost hurdle chart

| | |
|---|---|
| Question | How large must a move be, in volatility units, before the expected reversal clears the round-trip cost? |
| Data | Every bar-fade candidate from RR0, tagged with its stretch `z = r / σ` (σ from F3, adjusted for time of day). |
| Split | Train and validation only. |
| Metric | Mean gross next-k-minute return (in the fade direction) by `|z|` decile, for k in {1, 3, 5, 10, 15}, with a horizontal line at the round-trip cost and bootstrap bands (E2). |
| Decision rule | If no `|z|` bucket shows gross edge above cost plus 1 bp with its confidence band clear of the cost line, record a negative result and go straight to RR3/RR4 with residual or regime conditioning. |
| Trial budget | 1 chart. The k values are reported together, not chosen among. |
| Toolkit | T6 `analysis/holding.py` (horizon curve), T5, F3 |
| R2/R4 relation | Sets the candidate threshold R2 starts from. |

This is the most important chart the team produces. It shows in one figure
where trading is worthwhile and where it only pays the spread.

### RR2: Stretch fades

| | |
|---|---|
| Question | Which stretch definition and threshold give the best net edge on validation? |
| Grid | anchor ∈ {prev close, session VWAP (F2), EMA(20)}; `entryThresholdSigma` ∈ {1.5, 2, 2.5, 3}; `volLookbackBars` ∈ {15, 30, 60}; optional "N same-direction bars" trigger, N ∈ {3, 5}, as a separate candidate rule |
| Metric | Net return per trade, net P&L per day, trades per day, t-stat by session (clustered) |
| Decision rule | Choose a configuration on a **plateau**: neighbours on the parameter surface (T6 `analysis/surfaces.py`) are within half a standard error of it. Reject sharp peaks. |
| Trial budget | ≤ 48 |
| Toolkit | SR2, T1, T5, T6, F2, F3 |
| R2/R4 relation | The chosen rule becomes R2's primary model (the candidate generator). |

### RR3: Residual fades

| | |
|---|---|
| Question | Does fading `r_stock − β·r_SPY` beat fading `r_stock`, and should the trade hedge with SPY? |
| Variants | β from a rolling 1-minute regression over {60, 390} minutes, or a prior-day estimate; unhedged residual signal vs. hedged trade (short β·notional of SPY) |
| Metric | Net return per trade, and correlation of daily P&L with SPY |
| Decision rule | Adopt residual if it improves net edge on validation **or** cuts SPY correlation by half at equal edge. The hedge must pay for its own extra costs (two more fills). |
| Trial budget | ≤ 12 |
| Toolkit | SR3, C1d context symbols, F8 market return, T1 (two-leg trades), T5 |
| R2/R4 relation | Adds residual features to R2. |

### RR4: Regime and time-of-day gating

| | |
|---|---|
| Question | Which conditions should switch the strategy off? |
| Filters | Variance ratio and efficiency ratio (F5) thresholds; trade window (skip the first {5, 15, 30} and last {10, 30} minutes); trend-day filter (absolute SPY return since the open > k·σ_day); gap filter (absolute open gap > g %) |
| Metric | Net edge per trade and per day **with and without** each filter. A filter that raises edge per trade but cuts trades enough to lower daily P&L is reported as such. |
| Decision rule | Keep a filter only if it improves validation net P&L per day and its threshold sits on a plateau. At most 3 filters in the final rule. |
| Trial budget | ≤ 30 |
| Toolkit | T3 `trade/filters.py`, F5, F8 |
| R2/R4 relation | Regime values become R2 features. A filter dropped here may still help the meta-label as a continuous input. |

### RR5: Holding period and exits

| | |
|---|---|
| Question | How long should a fade be held, and which exits help? |
| Method | Fit an OU process to the deviation series (price minus anchor, in σ units) per symbol on train with `strategies/statarb/ou.py` (SS3). Half-life `hl = −ln 2 / ln φ` from an AR(1) on the deviation. Then compare exits on validation. |
| Exits compared | time only at {1, 2, 3}·hl; signal exit at `exitAtSigma` ∈ {0, 0.5}; disaster stop at `disasterStopSigma` ∈ {none, 2×entry, 3×entry}; combinations |
| Metric | Net return per trade, plus the tail: 5% expected shortfall per trade, worst day, max drawdown. Also MFE/MAE plots (T6 `analysis/excursions.py`). |
| Decision rule | The stop is judged on tail and drawdown, not on mean return; see section 5.4. |
| Trial budget | ≤ 24 |
| Toolkit | T1 `ExitPolicy`, T6, SS3 |
| R2/R4 relation | Fixes the exit policy that produces R2's triple-barrier labels. **R2 cannot start labelling until RR5 is decided.** |

### RR6: Sizing and portfolio limits

| | |
|---|---|
| Question | How much per trade, and how much at once? |
| Variants | fixed 1 share (baseline); fixed notional; inverse-vol (`targetRiskUsd / (σ_h · price)`); stretch-scaled inverse-vol (× min(\|z\| / z_entry, cap)); prob→size from the R2 model (later) |
| Metric | Net Sharpe, max drawdown, worst day, daily turnover, peak gross exposure |
| Decision rule | Adopt the simplest rule within one standard error of the best Sharpe. Never adopt a rule that adds to a losing position. |
| Trial budget | ≤ 10 |
| Toolkit | T2 `SizingRule`, T4 `RiskLimits` |

### R2: Reversal meta-labeling (existing study)

R2 keeps its ID and moves to `studies/reversals/r2-reversal-metalabel/`. With
RR2–RR5 decided, its setup is fixed: the RR2/RR3 rule generates candidates,
the RR5 exit policy produces labels, and the model decides which candidates to
take. See section 6.

### R4: Order-flow imbalance (existing study)

R4 asks whether order-flow imbalance predicts midpoint moves over 10 s, 60 s
and 5 min after crossing the spread. For this team, its output is a set of F6
features (`signed_volume_ratio`, `displayed_imbalance`, `spread_bps`, `ofi`)
that R2 can use, and the bounce measurement in RR0.

## 5. Trade management

Each subsection gives the method, the grid, the hypothesis and the decision
rule. The shared method for each decision is in
[research/trade-management.md](../research/trade-management.md); this section
adapts it to reversals.

### 5.1 Entry

- **Method:** candidates come from a `CandidateGenerator` that returns a
  `CandidateFrame` (symbol, decision_ts, side, stretch z, anchor, σ). SR1
  reproduces the platform rule exactly; SR2 and SR3 add thresholds.
- **Grid:** RR2 and RR3.
- **Hypothesis:** an entry threshold of about 2σ or more is needed to clear
  6 bp in these names (H2).
- **Decision rule:** RR1 sets the minimum threshold; RR2 picks the
  configuration on a plateau.
- **Note on σ:** 1-minute volatility at 09:31 is several times that at 13:00. A
  trailing σ over 15 minutes in the morning mixes regimes. Scale σ by an
  intraday volatility curve estimated on train (the same idea as F1's volume
  curve), or start the trade window late enough that the trailing window is
  homogeneous.

### 5.2 Filters and regime

- **Method:** filters are `trade/filters.py` (T3) predicates applied to the
  `CandidateFrame` before simulation, so every study logs what each filter
  removed.
- **Grid:** RR4.
- **Hypothesis:** H4. Trend days and the first 15 minutes are where fades lose most.
- **Decision rule:** a filter stays only if it improves net P&L **per day** on
  validation, not just per trade.
- **Always on (not tuned):** `regularHoursOnly`; the `maxBarGapMs` gap check;
  no new entries after the trade window closes; no trades on a symbol's
  earnings day or after a trading halt in that session, once C1m/C1g supply
  the calendar.

### 5.3 Holding period

- **Method:** the OU half-life of the deviation (RR5), cross-checked with the
  RR1 horizon curve. If the curve peaks at k minutes and the half-life says
  hl, the two should agree within a factor of about 2. If they don't, find out why
  before choosing.
- **Grid:** time exit at {1, 2, 3}·hl, with a hard cap `maxHoldBars` ≤ 30 and
  "flat by 15:55".
- **Hypothesis:** the useful holding period is a few minutes, longer than the
  baseline's implied one bar but well under an hour.
- **Decision rule:** the holding period that maximises net return per unit of
  turnover on validation, chosen from a plateau.
- **Contrast:** the baseline's holding period is "until the next opposite bar".
  Report its distribution from RR0 alongside the new one.

### 5.4 Stops

- **Method:** compare no stop, a disaster stop and a time stop (RR5). Express
  the disaster stop in σ units beyond entry, or as the deviation reaching a
  multiple of the entry stretch (`disasterStopSigma`).
- **Hypothesis:** a stop **lowers** mean return in a mean-reversion strategy:
  it closes trades exactly when the stretch is largest and the expected
  reversal is biggest. Its job is to cut the tail: trend days, news and halts.
- **Decision rule:** keep the disaster stop if it reduces 5% expected
  shortfall per trade or max drawdown by at least 20% at a cost of at most 10%
  of mean net return. Otherwise rely on the time stop plus the trend-day
  filter, and say so in the spec card.
- **Never:** a tight stop (under 1σ) on a fade. MAE analysis will show winners
  routinely going further against you than that before reverting.

### 5.5 Targets and signal exit

- **Method:** the natural target for a fade is the anchor. Exit when the
  deviation returns inside `exitAtSigma` (0 = full reversion, 0.5 = partial).
  Compare with a fixed target in σ and with a time exit only.
- **Hypothesis:** the signal exit at 0.5σ beats full reversion (0σ), because the
  last part of the reversion is slow and costs holding time.
- **Decision rule:** RR5 plateau. Look at MFE: if most winners reach
  full reversion, a 0σ exit is reasonable.

### 5.6 Sizing

- **Method:** `SizingRule` options in T2 (RR6).
- **Grid:** fixed share, fixed notional, inverse-vol with `targetRiskUsd`,
  stretch-scaled inverse-vol (capped at 2× the base size), later prob→size from R2.
- **Hypothesis:** inverse-vol sizing improves Sharpe over one share, because
  the current rule gives NVDA, TSLA and SPY the same share count.
- **Decision rule:** the simplest rule within one standard error of the best.
- **Never:** average down. A second signal in the same direction while a
  position is open does not add size. The current `maxPositionQty` cap prevents
  this by accident; the new family must prevent it on purpose.

### 5.7 Portfolio and risk limits

- **Method:** `RiskLimits` in T4, using the platform names: `maxPositionSizeUsd`
  per symbol, `maxGrossExposurePct`, `maxNetExposurePct`, concentration, the
  intraday drawdown kill switch.
- **Reversal-specific:** fades bunch on market-wide moves. A sharp SPY drop
  triggers buy signals in every name at once, which is one market bet, not ten. Cap
  concurrent same-side positions (`maxConcurrentSameSide`, proposed) or use
  the residual signal (RR3), which removes most of that clustering.
- **Decision rule:** set limits from the validation distribution of peak
  concurrent positions and daily loss. Limits are risk choices, so they need
  a reason, not an optimisation.

### 5.8 Cooldown and re-entry

- **Method:** measure the return of the second and later entries in the same
  symbol within m minutes of a stop or exit, vs. first entries.
- **Grid:** `cooldownMs` ∈ {0, 60000, 300000}; "no re-entry after a disaster
  stop for the rest of the session".
- **Decision rule:** if re-entries after a stop have negative net edge on
  validation, block them for the session. Otherwise use the shortest cooldown
  that removes the negative-edge bucket.

### 5.9 Costs and execution

- **Method:** `CostModel` (T5) with the bar `FillModel`: next-bar open ±
  half spread ± slippage. Calibrate both from R5 paper-trading fills as soon as
  R5 has data; until then use the platform defaults and report a sensitivity
  table at {0.5×, 1×, 2×} costs.
- **Hypothesis:** this strategy is cost-dominated. A 1 bp change in assumed
  slippage moves the conclusion more than any parameter in RR2–RR5.
- **Decision rule:** a configuration ships only if its validation net edge is
  positive at **2×** the calibrated cost.
- **Execution note:** market orders at the next open are the platform default.
  A marketable limit at the touch is worth testing in R5, since a fade
  provides liquidity in spirit but takes it in practice.

## 6. ML angle

**Meta-labeling (R2).** The rule chosen in RR2/RR3 proposes trades. A model
predicts whether a trade will be profitable **after costs** under the RR5 exit
policy. The label comes from `simulate_trades` (T1) with costs (T5), so the
label is exactly the trade the platform would make.

| Feature group | Source |
|---|---|
| Stretch size and anchor distances | SR2 candidate columns, F2 |
| Volatility and its time-of-day ratio | F3 |
| Relative volume | F1 |
| Regime | F5 (`variance_ratio_*`, `autocorr_*`, `efficiency_*`) |
| Market context: SPY move this minute and since the open | F8, C1d |
| Order flow and spread | F6 via R4 (`signed_volume_ratio`, `displayed_imbalance`, `spread_bps`, `ofi`) |
| Time of day, minutes since open | candidate metadata |

- **Models:** start with logistic regression (`models/logistic.py`), then
  LightGBM (M1b). An MLP (M1c) is optional and must beat LightGBM on
  validation to be considered.
- **Validation:** labels overlap in time (trades last several minutes), so use
  purged walk-forward with an embargo of at least `maxHoldBars` (V1/C1f).
- **Use of the score:** (a) a threshold to take or skip; (b) prob→size
  through T2 (`size = 2Φ(z) − 1`, López de Prado 2018, ch. 10), capped.
- **Decision rule:** the selector ships only if it improves net P&L per day on
  test vs. the rule alone, at 2× costs, with the deflated Sharpe ratio (E2)
  counting every R2 configuration tried.
- **Workshop link:** example 2 uses the same label as its regression target;
  example 4 uses the same meta-labeling pattern on ORB.

## 7. What the team implements vs. what it uses

### Implements

| ID | Task | Packet |
|---|---|---|
| SR1 | Bar-fade baseline, matching `minuteReversal` exactly (same gap and regular-hours rules) | [handoff/tasks/SR1.md](../handoff/tasks/SR1.md) |
| SR2 | Stretch fade: anchors, σ scaling, N same-direction bars | [handoff/tasks/SR2.md](../handoff/tasks/SR2.md) |
| SR3 | Residual fade vs. β·SPY, optional hedge leg | [handoff/tasks/SR3.md](../handoff/tasks/SR3.md) |
| F2 | Distance from session VWAP (first claim) | [handoff/tasks/F2.md](../handoff/tasks/F2.md) |
| F3 | Realized volatility (first claim) | [handoff/tasks/F3.md](../handoff/tasks/F3.md) |
| F5 | Regime features (first claim, shared with Momentum) | [handoff/tasks/F5.md](../handoff/tasks/F5.md) |
| RR0–RR6, R2 | Studies in `studies/reversals/` | this page |
| P7 | Combined reversal family in trading-platform (TypeScript) | [handoff/tasks/P7.md](../handoff/tasks/P7.md) |

### Uses (owned elsewhere)

| ID | What | Packet |
|---|---|---|
| T1 | `ExitPolicy`, `FillModel`, `simulate_trades` → `TradeLedger` | [handoff/tasks/T1.md](../handoff/tasks/T1.md) |
| T2 | `SizingRule` | [handoff/tasks/T2.md](../handoff/tasks/T2.md) |
| T3 | Filters | [handoff/tasks/T3.md](../handoff/tasks/T3.md) |
| T4 | `RiskLimits` | [handoff/tasks/T4.md](../handoff/tasks/T4.md) |
| T5 | `CostModel` | [handoff/tasks/T5.md](../handoff/tasks/T5.md) |
| T6 | Horizon curves, MFE/MAE, parameter surfaces | [handoff/tasks/T6.md](../handoff/tasks/T6.md) |
| T7 | `export_platform_config`, `ParityReport` | [handoff/tasks/T7.md](../handoff/tasks/T7.md) |
| T8 | `StrategySpec` card generator | [handoff/tasks/T8.md](../handoff/tasks/T8.md) |
| SS3 | OU fit and half-life (Stat arb) | [handoff/tasks/SS3.md](../handoff/tasks/SS3.md) |
| F1, F6, F8 | Relative volume, order flow, market return | [F1](../handoff/tasks/F1.md), [F6](../handoff/tasks/F6.md), [F8](../handoff/tasks/F8.md) |
| C1b, C1d, C1f | Events layer, context symbols, purged walk-forward | [C1b](../handoff/tasks/C1b.md), [C1d](../handoff/tasks/C1d.md), [C1f](../handoff/tasks/C1f.md) |
| E1, E2, V1, M1 | Trading metrics, deflated Sharpe, purged CV, models | [handoff/README.md](../handoff/README.md) |
| R5 | Cost calibration from paper fills | [handoff/README.md](../handoff/README.md) |
| P1, P6 | Candidate log; sizers connected to the orchestrator | [platform-tasks.md](../platform-tasks.md) |

If T1 is not ready, RR0 can start on `labels/exit_policy.py` once L2 lands;
T1 absorbs that simulator.

## 8. Platform deliverable: P7 config and parity

**P7** replaces the three strategies with one parameterised `reversal` family
in trading-platform. `minuteReversal` and `mcarrasco_reversal` become presets
of it. **`beginnerReversal` stays frozen** as the course example. It is not
changed, and the course keeps pointing at it.

The research side produces the values with `export_platform_config`
(`deploy/platform_config.py`, T7) from the study's `StrategySpec`.

| Field | Type | Status | Baseline-equivalent value | Decided by |
|---|---|---|---|---|
| `symbols` | string[] | existing | the 10 names / `["SPY"]` | universe (D4), RR0 |
| `regularHoursOnly` | boolean | existing | `true` | fixed |
| `maxBarGapMs` | number | existing | `180000` | fixed |
| `maxPositionSizeUsd` | number | existing | `10000` | RR6, risk |
| `cooldownMs` | number | existing | `0` | RR5/5.8 |
| `anchor` | `"prevClose" \| "vwap" \| "ema"` | new | `"prevClose"` | RR2 |
| `emaBars` | number | new (when `anchor = "ema"`) | n/a | RR2 |
| `volLookbackBars` | number | new | n/a (no σ) | RR2 |
| `entryThresholdSigma` | number | new | `0` (any move) | RR1, RR2 |
| `residualVsSpy` | `{ enabled, betaLookbackBars, hedge }` | new | disabled | RR3 |
| `exitAtSigma` | number \| null | new | null (exit on the opposite bar) | RR5 |
| `maxHoldBars` | number | new | none | RR5 |
| `disasterStopSigma` | number \| null | new | null | RR5 |
| `tradeWindow` | `{ start: "HH:MM", end: "HH:MM", flatBy: "HH:MM" }` | new | the whole regular session | RR4 |
| `regimeFilter` | `{ measure, window, max }` \| null | new | null | RR4 |
| `sizing` | `"fixedQty" \| "fixedNotional" \| "inverseVol" \| "stretchInverseVol"` | new | `"fixedQty"` | RR6 |
| `targetRiskUsd` | number | new | n/a | RR6 |
| `qtyPerTrade`, `maxPositionQty` | number | existing | `1`, `1` | kept for `fixedQty` presets |

- **Baseline preset:** the "baseline-equivalent" column must reproduce
  today's `minuteReversal` trade for trade. That is the first parity test,
  and it shows P7 changed no existing behaviour. Bump the strategy version
  when the defaults change.
- **Sizing modes:** `inverseVol` and `stretchInverseVol` need either P6
  (shared sizers connected to the orchestrator) or sizing computed inside the
  strategy's own `qty`. State which in the P7 PR.

**Parity** (method in [design/platform-parity.md](../design/platform-parity.md)):

1. Export the config with `export_platform_config`.
2. Run the platform backtest and `simulate_trades` on the same symbols and at
   least 5 sessions, including one trend day and one day with a halt or a
   large gap.
3. `ParityReport` compares the trade lists: the same entries and exits,
   timestamps equal, and fill prices within the tolerance in the parity
   design doc.
4. Any mismatch blocks paper trading until it is explained. The usual causes
   are an off-by-one bar in σ or VWAP, a different treatment of bar gaps, or
   same-minute ordering.

Then promote as set out in
[research/strategy-lifecycle.md](../research/strategy-lifecycle.md): paper
trading with the P1 candidate log on, then small live size.

## 9. Kill criteria and monitoring

Write these into the spec card before paper trading starts:

| Criterion | Action |
|---|---|
| Rolling 20-session net P&L per trade ≤ 0 with a t-stat below −1 | Pause and review |
| Realized slippage (R5) exceeds the model by more than 2 bp per side over 100+ fills | Pause; re-run RR1 with the new costs |
| Trades per day outside the 5–95% range of the backtest for 5 sessions running | Check the data feed and filters before trusting P&L |
| One session's loss exceeds the backtest's worst day | The platform's intraday drawdown kill switch should already have fired; review the limit |
| The share of candidates removed by the regime filter changes by more than 2× vs. the backtest | Regime shift; review RR4 |
| Parity breaks after a platform or research change | Stop until parity passes again |

**Monitoring source:** the P1 candidate log (every candidate, taken or
skipped, with its features and score). Compare live feature distributions
with training distributions each week.

## 10. Workshop tie-ins

| Example | How the reversal work appears |
|---|---|
| [1. Leakage lab](../workshop/examples/01-leakage-lab.md) | Bid-ask bounce as an illusion that looks like edge: compare close-to-close reversal with midpoint and next-open-fill reversal. Also a same-bar fill as a leak. |
| [2. Intraday model comparison](../workshop/examples/02-intraday-model-comparison.md) | The stretch-fade label is the target; linear, RF, LightGBM and MLP/LSTM are compared on it. |
| [4. Meta-labeling](../workshop/examples/04-meta-labeling-orb.md) | ORB is the worked example; the reversal meta-label (R2) is the exercise. |
| [6. Overfitting casino](../workshop/examples/06-overfitting-casino.md) | The RR2 grid without trial counting, then the same grid with the deflated Sharpe ratio. |
| [9. Trade management](../workshop/examples/09-trade-management.md) | The reversal half of the comparison: target exits and disaster stops vs. momentum's trailing stops. |

## 11. Milestones and suggested plan

The week counts assume a few hours a week per member and 3–5 members.

| Milestone | Weeks | Done when | Needs |
|---|---|---|---|
| MS0: baseline parity | 1–2 | SR1 matches the platform on 5 sessions; RR0 gross/net table written | T1 (or L2), T5, T7 |
| MS1: cost hurdle | 3 | RR1 chart in the study README, with the decision recorded | F3, T6 |
| MS2: stretch and residual | 4–6 | RR2 and RR3 decided, with parameter surfaces | SR2, SR3, F2, F8, C1d |
| MS3: gating and exits | 7–8 | RR4 and RR5 decided; exit policy fixed | F5, T3, SS3 |
| MS4: sizing and limits | 9 | RR6 decided; limits written down with their reasons | T2, T4 |
| MS5: meta-label | 10–12 | R2 test result recorded, positive or negative | M1b, V1/C1f, F6 |
| MS6: handover | 13–14 | Spec card, P7 preset parity passing, paper trading started | T7, T8, P7 |

**Parallel work in the first two weeks:**
- one member on SR1 + RR0;
- one on F3 (+ the time-of-day σ curve);
- one on F2;
- one on F5.

Features don't depend on the baseline, so nobody waits.

**If M1 is negative** (no bucket clears costs): that is a valid, publishable
result. Write it up and continue with RR3/RR4 to test whether residual or
regime conditioning changes the picture. If those are negative too, the
team's deliverable becomes "do not trade minute fades in this universe at these
costs", plus the evidence. That prevents a loss.

## 12. Pitfalls

- **Bid-ask bounce counted as edge.** Close-to-close reversal includes the
  bounce. Your fills don't. Always evaluate with next-open fills and costs.
- **Same-bar information.** The decision is at the bar's end, and the fill is the
  next bar's open. VWAP, σ and regime values at decision time must use bars
  with `ts_available` ≤ decision time (`assert_no_lookahead`).
- **σ without a time-of-day adjustment.** A 2σ move at 09:35 and at 13:00 are
  different events. See section 5.1.
- **Bar gaps.** A missing minute is not a 1-minute return. Follow F3's gap rule
  everywhere, and match the platform's `maxBarGapMs`.
- **Overlapping labels.** Trades last minutes and overlap. Random K-fold leaks;
  use purged walk-forward with an embargo.
- **Tight stops on fades.** They cut winners before they revert. See section 5.4.
- **Averaging down.** Never add to a fade that is going against you.
- **Bunched signals.** Ten simultaneous buy signals during an SPY drop are one
  bet. Cap concurrency or use residuals.
- **Short availability.** Half the fades are shorts. Record `allowShortSelling`
  for every run; in live trading, hard-to-borrow names can reject shorts.
- **Grid search without a count.** Every threshold you try goes into the trial
  count. The RR2 grid alone can produce an impressive-looking backtest by luck.
- **Survivorship.** The universe is a static list chosen as of 2026-10-01. That's
  fine for minute-horizon studies over months, but say so in every result.
- **Earnings, halts and news.** These produce the largest stretches, and those
  stretches continue. Exclude them when the calendar data exists (C1m/C1g);
  until then, report results with and without the top 1% of stretches.

## 13. Reading list

- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread
  in an efficient market. *Journal of Finance*. The bid-ask bounce.
- Lehmann, B. (1990). Fads, martingales, and market efficiency. *Quarterly
  Journal of Economics*. Short-term (weekly) reversal.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns.
  *Journal of Finance*. Monthly reversal.
- Lo, A. and MacKinlay, A. C. (1988). Stock market prices do not follow random
  walks: evidence from a simple specification test. *Review of Financial
  Studies*. The variance ratio used in F5.
- Avellaneda, M. and Lee, J.-H. (2010). Statistical arbitrage in the U.S.
  equities market. *Quantitative Finance*. Residual mean reversion and OU
  s-scores; shared with the Stat arb team.
- Cont, R., Kukanov, A. and Stoikov, S. (2014). The price impact of order
  book events. *Journal of Financial Econometrics*. Order-flow imbalance (R4/F6).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  Triple-barrier labels, meta-labeling, purged CV, bet sizing (ch. 10).
- Bailey, D. and López de Prado, M. (2014). The deflated Sharpe ratio.
  *Journal of Portfolio Management*. Used by E2.

Internal: [studies/reversals/r2-reversal-metalabel/README.md](../../studies/reversals/r2-reversal-metalabel/README.md),
[studies/reversals/r4-microstructure-ofi/README.md](../../studies/reversals/r4-microstructure-ofi/README.md),
[contracts.md](../contracts.md), [tasks.md](../tasks.md),
[platform-tasks.md](../platform-tasks.md).
