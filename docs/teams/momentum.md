# Momentum team guide

How the momentum team researches, builds and hands over its strategies. The
general method is in [research/strategy-lifecycle.md](../research/strategy-lifecycle.md),
[research/research-protocol.md](../research/research-protocol.md) and
[research/trade-management.md](../research/trade-management.md). This page covers
what is specific to momentum. IDs follow [docs/README.md](../README.md).

## 1. Mission and ownership

**Mission:** turn the opening-range breakout (ORB) into a platform strategy whose
entry, exits, sizing and risk limits were each chosen by a pre-registered study,
and which beats a simple trend baseline after costs on held-out stocks.

| Area | Owned by the team |
|---|---|
| Strategy family | `src/shpe_research/strategies/momentum/`: SM1 ORB generator, SM2 EMA crossover baseline, SM3 cross-sectional 12-1 momentum |
| Studies | R1 (ORB selectors A–D, existing) and RM1–RM8 below, in `studies/momentum/` |
| Features (lead owner; anyone may claim) | F1 relative volume, F4 opening range, F8 market return |
| Platform | P8 ORB momentum strategy in `trading-platform/backend/src/strategies/momentum/` |
| Workshop | Examples 3 and 4; co-presents 9 |

The shared toolkit (T1–T8) is not owned by any team. Changes to it follow the
cross-team review rule in [teams/README.md](README.md).

## 2. Baseline: what exists today

### Platform

`trading-platform/backend/src/strategies/momentum/index.ts` is a placeholder. It
exports nothing. Its comment plans an EMA crossover: fast and slow EMAs per
symbol, long on a bullish cross, exit or short on a bearish cross, with
configurable periods, sizing and a stop. There is no config interface, no
`VERSION`, and no entry in `backend/src/config/strategyCatalog.ts`. The catalog
registers only pairs, minute reversal and beginner reversal.

What the platform already provides that P8 will use:

| Platform piece | Status | Relevance |
|---|---|---|
| Bar fill model (`docs/architecture/backtest-execution.md`) | Implemented | Decisions at the bar end; market orders fill at next open ± `halfSpreadBps` ± `slippageBps`; stops fill at the stop or at the open if the bar gapped through it |
| `core/sizing/fixedNotionalSizer.ts` | Implemented, **not wired** to the orchestrator | `floor(maxPositionSizeUsd / price)` |
| `core/sizing/volatilityScaledSizer.ts`, `kellyFractionalSizer.ts` | Scaffolds that return 0, not wired | Needed for risk-per-trade sizing unless P8 sizes inside the strategy (P6) |
| `core/risk/riskEngine.ts` | Implemented | `maxPositionSizeUsd`, `maxNotionalExposureUsd`, `maxGrossExposurePct`, `maxNetExposurePct`, `maxConcentrationPct` (default 0.30), order cooldown, intraday drawdown kill switch, short-selling check |

### Research

[studies/momentum/r1-orb-selectors/README.md](../../studies/momentum/r1-orb-selectors/README.md)
defines the working ORB:

| Element | R1 definition |
|---|---|
| Opening range | Regular bars 09:30–09:44 New York time (`range_minutes = 15`); H = max high, L = min low. Known at `ts_available` 09:45 (`features/session/opening_range.py`) |
| Entry | First completed post-range bar with close > H, through 11:00. Long only |
| Fill | Next bar's open plus half spread and slippage (`labels/exit_policy.py` defaults: 1.0 and 2.0 bps) |
| Stop | Range midpoint (H + L) / 2 |
| Target | None |
| Time exit | 30 minutes after the entry fill |
| Selection | A: take all. B: RVOL ≥ 1.2 and SPY above its session open. C: random forest. D: MLP |
| Universe | D4 `configs/universe/workshop_v0.yaml`: 6 liquidity × volatility cells, 8 dev / 2 held-out per cell |

**Known weaknesses:**

1. The 30-minute exit and the midpoint stop are assumptions. No study chose them.
2. Sizing is unspecified, which in practice means fixed size.
3. No limit on concurrent positions. Breakouts bunch on strong market days.
4. No cost model for the first 90 minutes of the session, when spreads are widest.
5. No baseline other than "take every breakout". Nothing shows ORB beats a simpler trend rule.
6. No platform strategy exists to reconcile against. L2's planned 5-session
   reconciliation can only check the fill rules, using an existing strategy,
   until P8 lands.

## 3. Hypotheses and economic rationale

State each as a hypothesis to test, not a fact.

| ID | Hypothesis | Why it might be true | What would falsify it |
|---|---|---|---|
| H1 | A close above the opening-range high predicts continued drift over the next 15–120 minutes | Overnight news is absorbed slowly. Institutions work orders through the morning. Breakouts trigger stop and trend-following flows. Gao, Han, Li & Zhou (2018) document market-level intraday momentum | The horizon curve of rule-A entries is flat or negative net of costs on validation |
| H2 | Breakouts on high relative volume continue more often | Volume marks informed or institutional participation rather than noise | RVOL buckets show no monotone relation to net return |
| H3 | Breakouts aligned with SPY's direction work better | Most single-stock intraday variance is market variance. Fighting the market adds noise | No difference between SPY-aligned and non-aligned breakouts |
| H4 | Momentum payoffs are right-skewed, so fixed targets hurt and trailing exits help | Trend days produce a few large winners | Fixed 2R/3R targets beat no-target plus trailing on validation |
| H5 | ORB's edge is more than market beta on up days | Selection by breakout picks stocks with idiosyncratic flow | Edge disappears when trade returns are hedged with SPY over the same interval |
| H6 (SM3) | 12-1 month winners outperform losers over the next month | Jegadeesh & Titman (1993); underreaction and slow information diffusion | Decile spread not distinguishable from zero on the PIT universe |

## 4. Research roadmap

Run in this order. Each study is a folder `studies/momentum/<id>-<slug>/` with a
README from [research/study-template.md](../research/study-template.md).
Pre-register before opening validation results.

**Shared setup unless stated:**
- Data: minute bars from the D5 snapshot.
- Universe: D4 `workshop_v0`. Dev symbols are used for selection; the 2 held-out
  symbols per cell are touched once, at the end.
- Split: `split_by_day` train / validation / test with a gap. Switch to C1f
  walk-forward once it lands.
- Era: the session date.
- Primary metric: **daily** net P&L (all trades of a session summed). Trades on
  the same day are correlated, so trade-level Sharpe overstates significance.
- Every grid point counts toward the E2 deflated-Sharpe trial count. Record all
  of them in the study README.

| Study | Question | Grid / trial budget | Toolkit | Decision rule |
|---|---|---|---|---|
| **RM1** Baseline and controls | Does rule A have any edge net of costs, compared with SM2 (EMA crossover) and with a random-entry control (same entry minutes, random eligible symbols)? | Rule A with the R1 exit: 1. SM2: ≤ 6 period pairs. Control: 20 seeds, which don't count as trials | SM1, SM2, T1, T5, T6, E1, E2 | Continue to RM2–RM5 if rule A beats the random control on validation daily net P&L (E2 paired block bootstrap, 90% one-sided). If not, write up the negative result. R1's selectors still run, framed as "can selection rescue an edgeless rule?" |
| **RM2** Entry definition | Which range length, confirmation rule and entry cutoff? | `range_minutes` {5, 15, 30} × `confirm_rule` {close, touch} × `entry_cutoff` {10:30, 11:00, 12:00} = 18 | SM1, F4, T1 | Choose the centre of the best plateau on the validation surface (T6 `analysis/surfaces.py`), not the single best cell. Keep R1's 15 / close / 11:00 unless a neighbourhood beats it by more than the bootstrap CI half-width |
| **RM3** Exits: holding period, stops, targets | What time exit, stop rule and target/trailing rule? | Horizon curve: free, it is descriptive. Exit grid ≤ 40 combinations, see section 5 | T1, T6 (`holding.py`, `excursions.py`) | Pick from MAE/MFE evidence plus the validation surface. Pre-register the chosen `ExitPolicy` as the RM6 label |
| **RM4** Filters and regime | Do RVOL, SPY alignment, gap size and efficiency ratio improve daily net P&L? | One filter at a time, ≤ 4 thresholds each: ~16. Then one combined rule | F1, F8, F5, T3 | Keep a filter only if it improves validation daily net P&L and holds on held-out symbols. Report the trade-count cost of each filter |
| **RM5** Sizing and portfolio | Risk-per-trade vs. fixed notional. How many concurrent positions? Net-long cap? | `riskPerTradePct` {0.25, 0.5}% × `maxConcurrent` {3, 5, 10, ∞} × sizing {fixed, risk-per-trade, 1/√n scaling} = 24 | T2, T4 | Choose on validation max drawdown and daily P&L volatility for similar mean. Sizing should change risk, not create edge |
| **RM6** Meta-labeling (R1 v2) | Re-run R1 selectors A–D with the RM3 exit policy and the RM4 filters as rule B | As R1: A, B, C (RF), D (MLP), 3 seeds each, plus LightGBM (M1b) | C1c `triple_barrier`, M1a–c, M2, V1, E2 | Adopt a selector only if it beats B and the matched random-retention control on validation and on held-out symbols. Then open test once |
| **RM7** Cost sensitivity | Does the final spec survive realistic costs at 09:45–11:00? | Costs ×1, ×2, ×3, plus the R5-calibrated time-of-day curve: 4 | T5, R5 | The spec needs positive validation P&L at ×2 costs. Report the break-even cost in bps per round trip |
| **RM8** (SM3 track) 12-1 cross-sectional momentum | Can the team replicate the textbook effect on a daily PIT universe? | Formation {6-1, 12-1} × rebalance {monthly}: 2 | SM3, C1e, C1d, C1m | Teaching and feature track. Success means a correct, honest replication with turnover and costs. There is no platform deliverable |

**Relation to R1:** R1 is already pre-registered with the 30-minute midpoint
exit. Do not change R1 halfway through. Finish R1 as written, using dev data
only up to validation. Run RM1–RM5 alongside it on the dev split. If RM3 picks a
different exit policy, RM6 is a **new** pre-registration with its own trial
count. It is not an edit to R1.

## 5. Trade management for ORB

The general methods are in [research/trade-management.md](../research/trade-management.md).
Below are the momentum-specific grids. **R** = entry fill price − initial stop
price, the per-share risk.

### 5.1 Entry and confirmation

- **Method:** the SM1 `CandidateGenerator` emits one `CandidateFrame` row per
  (symbol, session) at the first qualifying bar. Each row carries the decision
  time, H, L and the stop and target levels. The `ExitPolicy` decides fills,
  not the generator.
- **Grid:** `confirm_rule`:
  - `close`: close > H, then a market order fills at the next open. This is R1.
  - `touch`: a buy-stop at H that becomes active at 09:45. It fills at H, or
    at the bar's open if that bar opens above H. This matches the platform
    stop fill rule exactly.
  Also `range_minutes` {5, 15, 30} and `entry_cutoff` {10:30, 11:00, 12:00}.
- **Hypothesis:** `touch` enters earlier at a better price but takes more false
  breakouts. `close` pays the gap from H to the next open.
- **Decision rule:** the RM2 plateau rule.
- **Short side:** a breakdown below L is a separate study with its own trials.
  It runs only after the long side is settled. The platform default is
  `allowShortSelling: true` (`backend/src/config/defaults.ts`), but confirm the
  deployed risk config keeps it on, or the RiskEngine `SHORT_SELLING_DISALLOWED`
  check rejects the entries.

### 5.2 Filters and regime

- **Candidates:**
  - RVOL at the decision bar (F1 `rvol_{n}d`, thresholds {1.0, 1.2, 1.5, 2.0}).
  - SPY above its session open (F8 `market_return > 0`).
  - Opening gap |open / prev close − 1| (exclude above {2%, 4%}).
  - Efficiency ratio (F5 `efficiency_{w}m`, quartile cut).
  - Range width / realized vol (F4 `or_width`, F3), to drop ranges too narrow
    to clear costs.
- **Method:** T3 `trade/filters.py`. Use one filter per run, then the best
  combination as a single extra trial.
- **Hypothesis:** H2, H3. Large overnight gaps, often earnings, behave
  differently. Treat earnings days as a tagged sub-population. Do not tune on
  them silently.
- **Decision rule:** the RM4 rule. Each filter must pay for the trades it
  removes.

### 5.3 Holding period

- **Method:** for every rule-A entry, mark to market the net return at +5, 10,
  15, 30, 45, 60, 90, 120 minutes and at `flat_by`. Use T6
  `analysis/holding.py`. Plot the mean and median with bootstrap bands, overall
  and by RVOL tercile. Also plot the time of the MFE peak.
- **Grid:** `max_hold_minutes` {15, 30, 60, 120, until `flat_by`}. `flat_by` is
  fixed at 15:55; it is not tuned.
- **Hypothesis:** if the curve keeps rising past 30 minutes, the R1 time exit
  cuts winners. A trailing stop or a longer hold should then win.
- **Decision rule:** choose the shortest hold within the plateau of net return
  per trade. Shorter holds mean less exposure and fewer overlapping positions.

### 5.4 Stops

- **Grid:**
  - range midpoint (R1)
  - range low L
  - entry − k·ATR, k ∈ {1, 1.5, 2, 3}, with ATR or F3 realized vol computed only
    from bars before entry
  - none (time exit only)
- **Method:** MAE analysis (T6 `analysis/excursions.py`). Plot MAE in R units
  for eventual winners and losers. Place the stop beyond most winners' MAE.
  The "none" row tells you whether the stop adds return or only cuts the tail.
- **Hypothesis:** momentum entries are invalidated by a move back into the
  range, so a stop at or near the range should help, not hurt.
- **Decision rule:** best validation daily net P&L within a plateau. Ties go to
  the rule with lower max drawdown.

### 5.5 Targets and trailing

- **Grid:**
  - none (R1)
  - fixed 1.5R, 2R, 3R
  - chandelier trailing stop (highest high since entry − k·ATR, k ∈ {2, 3})
  - move the stop to breakeven after +1R
- **Method:** MFE analysis: how much open profit is given back at the time exit.
  T1 simulates trailing stops bar by bar, using only completed bars to update
  the level.
- **Hypothesis:** H4. Fixed targets cut the right tail. Trailing should beat
  targets if returns are right-skewed.
- **Decision rule:** compare on validation daily net P&L **and** on skew. Report
  both.

The combined exit grid for RM3 is stop (7) × target/trail (7) × hold (5). That
is too many cells. Prune in stages:
1. Fix hold = 30 and choose the stop family.
2. Choose target/trail.
3. Choose hold.
Count every evaluated cell as a trial.

### 5.6 Sizing

- **Rules (T2 `SizingRule`):**
  - Fixed notional, the platform `fixedNotionalSizer` formula.
  - Risk-per-trade: `qty = equity × riskPerTradePct / R`. The range width sets R,
    so size shrinks on wide-range days automatically.
  - Optional 1/√(open positions) scaling.
- **Caps:**
  - `maxPositionSizeUsd`.
  - A participation cap: qty ≤ a fraction of the symbol's average volume over
    the entry minute.
  - The platform `maxConcentrationPct` (0.30).
- **Later:** probability-to-size from the RM6 meta-model, `size = 2Φ(z) − 1`
  (López de Prado 2018, ch. 10), capped. Only if RM6 adopts a selector.
- **Decision rule:** the RM5 rule.
- **Platform gap:** the platform sizers are not wired (P6). Until P6 lands, P8
  computes `qty` inside the strategy from `riskPerTradePct` and the stop distance.

### 5.7 Portfolio and risk limits, including signal bunching

Breakouts bunch. On a strong market morning many names break out within minutes
of each other, and their trade returns are highly correlated. Twenty positions
on such a day are close to one large SPY bet.

- **Measure:**
  - breakouts per session
  - breakouts per 5-minute window
  - average pairwise correlation of same-day trade returns
  - share of daily P&L explained by SPY over the same window (H5)
- **Controls (T4 `RiskLimits`, same names as the platform RiskEngine):**
  - `maxConcurrent`
  - `maxNetLongPct` (maps to `maxNetExposurePct`)
  - `maxNotionalExposureUsd`
  - `maxConcentrationPct`
  - optional sector cap
  - the intraday drawdown kill switch
- **Queueing rule** when more candidates arrive than slots: first-come, or
  highest meta-model score. That is one extra trial.
- **Decision rule:** the RM5 rule. Also report P&L both raw and SPY-hedged.

### 5.8 Cooldown and re-entry

- **Default:** one entry per symbol per session. R1 takes the *first* breakout.
- **Study:** allow a second entry after a stop-out, if price closes above H
  again before the cutoff. Compare second-entry net return with first-entry.
  Re-entry is adopted only if second entries are profitable on their own.
- **Platform mapping:** `orderCooldownMs` handles order throttling. The
  once-per-session rule lives in the strategy.

### 5.9 Costs and execution

- Entries cluster at 09:45–11:00. Spreads in the first minutes after the open
  are wider than the 1.0 bp half-spread default. Use the R5-calibrated
  time-of-day cost curve (T5 `CostModel`) as soon as it exists. Until then, run
  every result at costs ×1, ×2 and ×3 (RM7).
- `close` confirmation uses a market order at the next open. `touch` uses a
  buy-stop. Both follow the platform's bar fill model, so T1 must implement both
  paths exactly as `backtest-execution.md` describes.
- Report the break-even cost: the round-trip cost in bps at which validation
  net P&L reaches zero.
- Run the quote-model platform backtest (`core/execution/quoteFillModel.ts`) on
  a sample of sessions before paper trading. It shows how far the bar model's
  open-price fills are from the real touch at the open.

## 6. ML angle: meta-labeling

| Element | Momentum choice |
|---|---|
| Primary model | The rule: SM1 candidates after RM4 filters (rule B generalised) |
| Secondary model | Predicts whether a candidate's trade is worth taking |
| Label | C1c `triple_barrier` built on T1 `simulate_trades` with the pre-registered RM3 `ExitPolicy`, net of T5 costs. R1 regresses net trade return; also report a binary `net_return > 0` variant as a separate, counted trial |
| Features | F1 `rvol_{n}d`, F8 `market_return`, F4 `or_width_{m}m` and `or_breakout_dist_{m}m`, F3 `realized_vol_{w}m`, F2 VWAP distance, minute of day, F5 efficiency ratio, gap size. Optional: SM3 12-1 rank as a daily context feature |
| Models | Logistic (existing), M1a random forest, M1b LightGBM, M1c MLP; 3 seeds each via M2 |
| Validation | V1 purged walk-forward. Labels span up to `max_hold_minutes`, so purge overlapping label windows. Embargo one session. Era = session date |
| Threshold | Chosen on validation only. Express it as a target acceptance rate |
| Controls | Rule A, rule B, and **matched random retention**: drop candidates at random down to the model's acceptance rate, many seeds. Also a context ablation that removes SPY features |
| Test | Opened once, after the README records the chosen model and threshold. Held-out symbols are reported separately |
| Export | M3 ONNX with golden rows → P2 wrapper → `metaModelPath` / `minProbability` in P8 |

**Honest comparison:** a model that "improves Sharpe" by trading 30% of the
candidates must beat random 30% subsets. Most of the apparent gain from any
filter comes from trading less on bad days. The random-retention control shows
how much of the gain is selection and how much is just trading less.

## 7. Implement vs. consume

### The team implements

| ID | What | Packet |
|---|---|---|
| SM1 | ORB `CandidateGenerator` | [handoff/tasks/SM1.md](../handoff/tasks/SM1.md) |
| SM2 | EMA crossover baseline | [handoff/tasks/SM2.md](../handoff/tasks/SM2.md) |
| SM3 | Cross-sectional 12-1 momentum | [handoff/tasks/SM3.md](../handoff/tasks/SM3.md) |
| F1 | Relative volume | [handoff/tasks/F1.md](../handoff/tasks/F1.md) |
| F4 | Opening range | [handoff/tasks/F4.md](../handoff/tasks/F4.md) |
| F8 | Market return (coordinate the context-symbol extension with the F7 owner) | [handoff/tasks/F8.md](../handoff/tasks/F8.md) |
| R1, RM1–RM8 | Studies | `studies/momentum/` |
| P8 | ORB strategy in the platform (TypeScript) | [handoff/tasks/P8.md](../handoff/tasks/P8.md) |

### The team consumes

| ID | What | Needed by | Packet |
|---|---|---|---|
| T1 | Exit policies and trade simulator | RM1–RM7 | [T1](../handoff/tasks/T1.md) |
| T2 | Sizing rules | RM5 | [T2](../handoff/tasks/T2.md) |
| T3 | Filters | RM4 | [T3](../handoff/tasks/T3.md) |
| T4 | Risk limits | RM5 | [T4](../handoff/tasks/T4.md) |
| T5 | Cost model | RM7, all P&L | [T5](../handoff/tasks/T5.md) |
| T6 | Diagnostics (horizon curve, MFE/MAE, surfaces) | RM2, RM3 | [T6](../handoff/tasks/T6.md) |
| T7 | Platform config export and parity | P8 handover | [T7](../handoff/tasks/T7.md) |
| T8 | Spec card | Handover | [T8](../handoff/tasks/T8.md) |
| C1b | Events/candidates layer | SM1 | [C1b](../handoff/tasks/C1b.md) |
| C1c | Label registry, `triple_barrier` | RM6 | [C1c](../handoff/tasks/C1c.md) |
| C1f | Walk-forward fit schemes | RM6 | [C1f](../handoff/tasks/C1f.md) |
| C1d, C1e, C1m | Cross-sectional transforms, daily frequency, PIT universe | RM8 | [C1d](../handoff/tasks/C1d.md), [C1e](../handoff/tasks/C1e.md), [C1m](../handoff/tasks/C1m.md) |
| F2, F3, F5 | VWAP distance, realized vol, regime | RM4, RM6 | `docs/tasks.md` |
| V1, E1, E2 | Purged CV, trading metrics, deflated Sharpe | All | `docs/tasks.md` |
| M1a–c, M2, M3 | Models, sweeps, ONNX | RM6 | `docs/tasks.md` |
| P1, P2, P6 | Candidate log, ONNX wrapper, sizer wiring | P8 | [platform-tasks.md](../platform-tasks.md) |
| R5 | TCA calibration | RM7 | `studies/shared/` |

## 8. Platform deliverable: P8 configuration

The study output is a `StrategySpec`. T7 `export_platform_config` turns it into
the P8 config JSON. Waits are stored in milliseconds, as `PairsStrategy` does
with `maxHoldingTimeMs`; T7 converts from the research spec's minutes.
`rangeMinutes` stays in minutes because it names a bar window, not a wait (see
[P8](../platform-tasks.md)).

| Research (`StrategySpec`) | P8 field | Type / units | R1 starting value | Chosen by |
|---|---|---|---|---|
| `range_minutes` | `rangeMinutes` | int, minutes from 09:30 | 15 | RM2 |
| `entry_cutoff` | `entryCutoff` | "HH:MM" New York | "11:00" | RM2 |
| `confirm_rule` | `confirmRule` | `close` \| `touch` | `close` | RM2 |
| `stop_rule` | `stopRule` | `range_mid` \| `range_low` \| `atr` \| `none` | `range_mid` | RM3 |
| `stop_atr_mult` | `stopAtrMult` | float, used when `stopRule = atr` | n/a | RM3 |
| `trail_rule` | `trailRule` | `none` \| `chandelier` \| `breakeven_1r` \| `target_r` | `none` | RM3 |
| `max_hold_minutes` | `maxHoldMs` | int, milliseconds from the entry fill (T7 converts from minutes) | 1800000 (30 min) | RM3 |
| `flat_by` | `flatBy` | "HH:MM" New York; earlier on early-close days | "15:55" | Fixed |
| `risk_per_trade_pct` | `riskPerTradePct` | fraction of equity | n/a (fixed size) | RM5 |
| `max_concurrent` | `maxConcurrent` | int | unlimited | RM5 |
| `max_net_long_pct` | `maxNetLongPct` | fraction of equity | n/a | RM5 |
| `min_rel_volume` | `minRelVolume` | float, F1 lookback fixed in spec | 1.2 (rule B) | RM4 |
| `regime_filter` | `regimeFilter` | `none` \| `spy_above_open` \| `efficiency_min` | `spy_above_open` (rule B) | RM4 |
| `meta_model_path` | `metaModelPath` | path to M3 ONNX + manifest | none | RM6 |
| `min_probability` | `minProbability` | float | n/a | RM6 |

Existing base-config fields also apply: `symbols` (the dev and held-out symbols
from D4, plus SPY as a context symbol that is never traded) and
`maxPositionSizeUsd`.

**Parity requirements (T7 `ParityReport`):**

1. **Candidates match exactly.** On at least 5 sessions, then one month, every
   (symbol, decision time) the research generator emits appears in the P1
   candidate log, and nothing else does. Filtered candidates appear with the
   same reject reason.
2. **Fills match.** Entry and exit prices match within floating-point tolerance
   under the bar model. Exit reasons (`stop`, `target`, `time`, `session_end`)
   match per trade.
3. **P&L matches.** Per-trade net P&L within $0.01. Per-session P&L within $0.05.
4. **Meta-model parity.** If used, M3 golden rows pass in P2's CI test, and
   live feature values match the research feature values on the same bars.
5. **Versions recorded.** The report stores `BACKTEST_ENGINE_VERSION` and the
   P8 `VERSION`. Any change to P8 logic bumps `VERSION` and requires a new
   parity run.

## 9. Kill criteria and live monitoring

Put these in the spec card **before** paper trading. The numbers come from the
backtest. Do not set them after seeing live results.

| Monitor | Trigger | Action |
|---|---|---|
| Rolling 20-session net P&L | Below the 5th percentile of 20-session windows from a block bootstrap of the validation period | Pause and review |
| Slippage | Mean realised slippage over 50 trades exceeds the cost model by more than the margin the spec card sets | Re-run RM7 with live costs; pause if break-even is crossed |
| Candidate rate | Breakouts per session outside the backtest's 1st–99th percentile range for 5 sessions | Treat as a data or logic bug until shown otherwise |
| Meta-model score drift | Population stability index of scores vs. validation above the spec card's threshold | Retrain or disable the selector (fall back to rule B) |
| Exit-reason mix | Share of stop exits outside the backtest range | Review regime; check the stop logic |
| Intraday drawdown | RiskEngine intraday drawdown limit | Kill switch, platform-enforced |

Retire the strategy (catalog `status: retired`) if a paused strategy fails its
review twice.

## 10. Workshop tie-ins

| Example | How momentum uses it |
|---|---|
| [03 Cross-sectional ranker](../workshop/examples/03-cross-sectional-ranker.md) | SM3: replicate 12-1 momentum, then combine signals. RM8 is the team's version |
| [04 Meta-labeling ORB](../workshop/examples/04-meta-labeling-orb.md) | R1 / RM6 in workshop form; the team presents |
| [06 Overfitting casino](../workshop/examples/06-overfitting-casino.md) | The RM3 exit grid is the exact situation this example warns about |
| [08 Portfolio construction](../workshop/examples/08-portfolio-construction.md) | Signal bunching and SPY-neutral evaluation (section 5.7) |
| [09 Trade management](../workshop/examples/09-trade-management.md) | The team runs T6 diagnostics on ORB and presents next to reversals and stat arb |

## 11. Milestones and weekly plan

**Milestone checklist:**

- [ ] **MS0 Features.** F1, F4, F8 merged and passing `assert_no_lookahead`.
- [ ] **MS1 Generator.** SM1 emits R1-compliant candidates on the fixture and the
  snapshot. SM2 is merged. `check_strategy` (C1k) passes.
- [ ] **MS2 Controls.** RM1 written up: rule A vs. EMA vs. random entry, with
  costs ×1/×2/×3.
- [ ] **MS3 R1.** R1 finished as pre-registered, including the negative result
  if there is one.
- [ ] **MS4 Trade management.** RM2–RM5 done; `ExitPolicy`, filters and sizing
  frozen in a draft spec card.
- [ ] **MS5 Selection.** RM6 done; selector adopted or rejected.
- [ ] **MS6 Costs.** RM7 passes at costs ×2, or the spec is rejected.
- [ ] **MS7 Platform.** P8 implemented. T7 parity report passes on 5 sessions,
  then one month.
- [ ] **MS8 Paper.** Paper trading with section 9 monitors for at least 20
  sessions before any live size.
- [ ] **Side track.** RM8 replication, presented in workshop example 3.

**Suggested plan** for a team of 4–6 members at a few hours a week each. Weeks
are approximate. Shared toolkit tasks may finish earlier or later; the team
should not build its own copies while waiting.

| Weeks | Features person | Strategy person | Research person(s) | TypeScript person |
|---|---|---|---|---|
| 1–2 | F4, then F1 | SM1 on the fixture | Pre-register R1 and RM1 | Read the platform strategy guide; draft the P8 config interface |
| 3–4 | F8 with the F7 owner | SM2; help with T1 if unclaimed | RM1 | P1 candidate log, if unclaimed |
| 5–6 | F5 or F3 support | RM2 plumbing | Finish R1 | P8 skeleton: candidates only, no orders |
| 7–9 | Support | RM3 | RM4, RM5 | P8 exits and sizing |
| 10–11 | SM3 for RM8 | RM6 | RM7 | Parity runs with T7 |
| 12+ | RM8 | Spec card | Monitoring setup | Paper trading |

## 12. Pitfalls specific to ORB

1. **Range timing.** The 09:44 bar is labelled 09:44 but known at 09:45. A
   breakout check on the 09:45 bar may use H only from bars with
   `ts_available` ≤ 09:45.
2. **Filling at the signal bar's close.** The decision is made at the bar end;
   the fill is the *next* bar's open. Filling at the close that triggered the
   signal is look-ahead.
3. **Touch entries through a gap.** If a bar opens above H, a buy-stop at H
   fills at the open, not at H.
4. **Same-bar stop and target.** Assume the stop, as in L2 and the platform.
5. **Trade-level statistics.** Same-day breakouts are correlated. Use daily P&L
   and block bootstraps.
6. **RVOL with today's future volume.** F1 uses only cumulative volume through
   the decision bar, against previous sessions' same-minute averages.
7. **Long-only beta.** On a rising sample, ORB longs can look good because the
   market rose. Report SPY-hedged results (H5).
8. **Earnings and news days.** Gaps from earnings dominate the tails. Tag them
   and report them separately. Do not drop them silently after seeing results.
9. **Early-close sessions.** `flatBy` must move earlier. The range is still
   09:30–09:44.
10. **Joint tuning.** Range length, stop and hold interact. Tuning them together
    on the same validation data multiplies trials. Use the staged pruning in
    5.5 and count everything.
11. **SM3 survivorship.** `workshop_v0` is a static universe as of 2026-10-01.
    A 12-1 study looking back years on it is survivorship-biased. Use C1m, or
    label the result as biased (workshop example 1 uses this on purpose).
12. **SPY as a candidate.** SPY is a context symbol. Exclude it from candidates
    unless a study says otherwise.

## 13. Reading list

- Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling
  losers. *Journal of Finance*.
- Moskowitz, T., Ooi, Y. H. and Pedersen, L. H. (2012). Time series momentum.
  *Journal of Financial Economics*.
- Asness, C., Moskowitz, T. and Pedersen, L. H. (2013). Value and momentum
  everywhere. *Journal of Finance*.
- Daniel, K. and Moskowitz, T. (2016). Momentum crashes. *Journal of Financial
  Economics*. Read before RM8.
- Gao, L., Han, Y., Li, S. Z. and Zhou, G. (2018). Market intraday momentum.
  *Journal of Financial Economics*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  Triple barrier and meta-labeling (ch. 3), bet sizing (ch. 10), purged CV (ch. 7).
- Bailey, D. and López de Prado, M. (2014). The deflated Sharpe ratio. *Journal
  of Portfolio Management*.
- Sweeney, J. (1997). *Maximum Adverse Excursion: Analyzing Price Fluctuations
  for Trading Management*. Wiley.
- Internal: [studies/momentum/r1-orb-selectors/README.md](../../studies/momentum/r1-orb-selectors/README.md),
  [design/trade-layer.md](../design/trade-layer.md),
  [design/platform-parity.md](../design/platform-parity.md),
  trading-platform `docs/architecture/backtest-execution.md`.
