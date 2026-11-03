# Stat arb team guide

This guide tells the statistical arbitrage team what it owns, what the platform
strategy does today, which studies to run in which order, and what to hand back
to the platform. Read it with the shared method docs:

- [research/research-protocol.md](../research/research-protocol.md): pre-registration, splits, trial counting
- [research/trade-management.md](../research/trade-management.md): holding period, stops, sizing, filters, risk, costs
- [research/strategy-lifecycle.md](../research/strategy-lifecycle.md): idea → spec → study → parity → paper → live → retire
- [research/spec-card-template.md](../research/spec-card-template.md): what you hand over at the end
- [design/trade-layer.md](../design/trade-layer.md): the `strategies/` + `trade/` + `analysis/` packages you build on

Agents working for this team: [`strategy-builder`](../../.claude/agents/strategy-builder.md),
[`research-analyst`](../../.claude/agents/research-analyst.md),
[`trade-management-analyst`](../../.claude/agents/trade-management-analyst.md),
[`platform-parity`](../../.claude/agents/platform-parity.md). Run
[`leakage-auditor`](../../.claude/agents/leakage-auditor.md) on every study before
its result is recorded.

---

## 1. Mission and ownership

**Mission:** find pairs whose spread reliably reverts within a holding period
we can trade intraday or overnight. Then calibrate the platform's
`PairsStrategy` so each parameter comes from a study, not a guess.

The platform strategy already has the most complete trade management of the
three teams: entry, exit, stop, max hold, cointegration gate, cooldown. Your
work is mostly **calibration and pair selection**, not building from scratch.

| Area | Team owns | Shared with others |
|---|---|---|
| Library code | `src/shpe_research/strategies/statarb/` (SS1–SS4), `features/cross_asset.py` pair features (F7, with the F8 owner) | `trade/`, `analysis/`, `deploy/` (T1–T8): reviewed by a member of another team |
| Studies | `studies/stat-arb/` (RS1–RS9; R3 moves here as `studies/stat-arb/r3-pairs-clustering/`) | `studies/shared/` (R5 TCA) |
| Platform | Spec and review for P9 (pairs config additions) | P1, P2, P6 |
| Workshop | Example 7, with segments in 6, 8 and 9 | |

Add a CODEOWNERS entry for `strategies/statarb/` and `studies/stat-arb/` naming
two team members.

---

## 2. Baseline: the platform `PairsStrategy`

Source: `trading-platform/backend/src/strategies/pairs/` (`pairsStrategy.ts`,
`pairsConfig.ts`, `pairsTypes.ts`). Version: `PairsStrategy.VERSION = 6`.

### 2.1 Defaults (`DEFAULT_PAIRS_CONFIG`)

| Field | Default | Meaning |
|---|---|---|
| `rollingWindowMs` | `3_600_000` (1h) | Time window of spread observations for the z-score mean and std |
| `hedgeRatioMethod` | `"fixed"` | `"fixed"` or `"rolling_ols"` |
| `fixedHedgeRatio` | `1` | β used when the method is `fixed` |
| `entryZScore` | `2` | Enter long spread at z ≤ −2, short spread at z ≥ 2 |
| `exitZScore` | `0.5` | Exit when \|z\| ≤ 0.5 |
| `stopLossZScore` | `4` | Exit when \|z\| ≥ 4; also blocks entry when \|z\| is already ≥ 4 |
| `maxHoldingTimeMs` | `86_400_000` (24h) | Forced exit; checked on every bar of either leg |
| `minObservations` | `30` | Spread observations needed before scoring |
| `tradeNotionalUsd` | `5_000` | Sets leg 1 quantity: `floor(5000 / price_leg1)` |
| `maxPositionSizeUsd` | `10_000` | Strategy-level cap |
| `cooldownMs` | `60_000` | No new entry for 1 minute after an exit |
| `olsWindowMs` | `14_400_000` (4h) | Window for the OLS hedge ratio and Engle-Granger test |
| `olsRecalcIntervalBars` | `5` | Recompute β and the test every 5 bars |
| `cointSignificanceLevel` | `0.05` | MacKinnon critical value used by the gate (0.01, 0.05 or 0.10) |
| `cointHysteresis` | `0.15` | Dead band around the critical value so the gate cannot flap |
| `priceSource` | `"mid"` | Ignored since v5; the strategy prices with bar closes |
| `executionAlgo` | `"market"` | Both legs sent as market orders |

`validatePairsConfig` requires `maxHoldingTimeMs` to be a whole number between
one minute and one year.

### 2.2 What the code actually does

These details decide whether research reproduces the platform. Each one is a
parity requirement for SS4.

1. **Decision clock.** One observation per minute, only when both legs printed
   a completed bar for the same minute, priced at each leg's **close**.
2. **Spread in raw price space.** `spread = price_leg1 − β × price_leg2`, not
   in log prices. The F7 spec in `features/cross_asset.py` says log prices;
   this mismatch has to be resolved (see §12 and §8).
3. **The z-score includes the current observation.** The spread is pushed into
   the window *before* the mean and std are computed, so the current point
   counts toward its own reference mean. This makes |z| slightly smaller.
4. **Windows are measured in time, not bars.** `RollingNumericWindow(rollingWindowMs)`
   drops observations older than 1h. Overnight, the whole window expires.
   Unless RS1 shows otherwise, this means:
   - the z-score is effectively per session;
   - each morning is idle until `minObservations` (30) new minutes accumulate;
   - a position carried overnight under the 24h max hold is scored against a
     brand-new mean the next morning.
   RS1 must confirm this.
5. **Exit order:** stop-loss, then max hold, then the normal exit. Exits are
   checked before entries.
6. **Cointegration gate** (`rolling_ols` only):
   - Engle-Granger on raw prices over `olsWindowMs`.
   - Dickey-Fuller on the residuals with no lags, against MacKinnon (1991)
     critical values.
   - Recomputed every `olsRecalcIntervalBars`, with hysteresis.
   - The gate blocks **entries only**; a pair that loses cointegration while a
     position is open does not exit.
   - β is updated **only while the pair is cointegrated**, so it goes stale
     during gate-off periods.
   - In `fixed` mode there is no gate.
7. **Sizing is not dollar-neutral.**
   - `leg1Qty = floor(tradeNotionalUsd / price_leg1)` and
     `leg2Qty = floor(leg1Qty × β)`.
   - The result is one "spread unit" per leg-1 share. With the default β = 1,
     both legs hold **equal share counts**, so leg 2's dollar exposure is
     `price_leg2 / price_leg1` times leg 1's.
   - The comment in `pairsTypes.ts` ("sell $5000 × hedgeRatio of leg2") does
     not match this code.
8. **The stop only fires at evaluation time.** A comment in `_checkExitSignals`
   notes that a skipped evaluation (cooldown, too few observations) also skips
   the stop.

Platform sizing and risk context
([`position-sizing-layer.md`](../../../trading-platform/docs/architecture/position-sizing-layer.md),
[`risk-layer.md`](../../../trading-platform/docs/architecture/risk-layer.md)):

- The shared sizers are not connected to the orchestrator (P6). The pairs
  strategy sizes itself in `_computeQty`.
- `RiskEngine` enforces `MAX_POSITION_SIZE`, `MAX_NOTIONAL_EXPOSURE`,
  `SHORT_SELLING_DISALLOWED` (shorting is allowed by default),
  `CONCENTRATION_LIMIT`, `INTRADAY_DRAWDOWN`, gross and net exposure, and
  per-symbol cooldowns.

### 2.3 What is missing

- No study behind any default: `entryZScore`, `exitZScore`, `stopLossZScore`, the 24h max hold, the window lengths, $5k per leg.
- No pair-selection process. Pairs are typed in by hand; XOM/CVX is the example in the config comments.
- A default hedge ratio of 1, which suits almost no pair.
- No earnings or corporate-event awareness.
- No stop that measures money lost, only a z level (see §5.5).

---

## 3. Hypotheses and rationale

Treat each line below as a hypothesis to test, not a result.

| ID | Hypothesis | Economic rationale | Tested in |
|---|---|---|---|
| H1 | Pairs chosen from residual-return clusters stay cointegrated out of sample more often than pairs chosen from all pairs or by the distance method | Shared factor exposure (same industry, same customers) is a reason for a spread to revert; a price coincidence is not | RS2 (R3 Q1) |
| H2 | An estimated hedge ratio (OLS, Kalman or TLS) gives a more stationary out-of-sample spread than β = 1 | β = 1 assumes equal price scale and equal factor exposure | RS3 |
| H3 | Setting max hold and window lengths from each pair's half-life beats the fixed 24h / 1h / 4h defaults on net P&L per trade | A spread that hasn't reverted in a few half-lives is probably not reverting | RS4 |
| H4 | Net P&L is flat over a band of entry/exit z values, not peaked at (2, 0.5) | A peak means the result is noise | RS5 |
| H5 | A loss stop in spread units protects better than the z-stop, because the rolling mean follows the spread | See §5.5 | RS6 |
| H6 | Volatility-targeted spread sizing cuts drawdown without lowering mean P&L per unit of risk | Equal notional on unequal spread vol takes unequal risk | RS7 |
| H7 | A shallow model can predict which entries converge before stop or max hold, especially near earnings | Spreads that break usually break on news | RS8 (R3 Q2) |

---

## 4. Research roadmap

Each study gets a folder `studies/stat-arb/<id>-<slug>/` with a README from
[study-template.md](../research/study-template.md). Pre-register the README:
dates, universe, grid and decision rule, written **before** you look at any
validation result.

### Shared setup for all studies

- **Universe:** D4 `configs/universe/workshop_v0.yaml`, development symbols
  only. Held-out symbols are used once, in the final check of each study.
  Candidate pairs come from the same sector or the same cluster. Held-out
  checks use pairs with **at least one held-out leg**.
- **Data:** D5 frozen snapshot of 1-minute bars. Daily bars (C1e) for formation statistics.
- **Splits:**
  - Rolling **formation → trading** windows. Selection and hedge ratios are
    fit on formation data only.
  - Model fits use V1 purged walk-forward with an embargo at least as long as
    the max hold, because trades can last overnight.
- **Fixture first:** C1o's synthetic data must contain one planted
  cointegrated pair and one planted non-cointegrated pair. Every study runs on
  it first and must **find** the planted pair and **reject** the fake one
  before using real data.
- **Costs:** the R5-calibrated `CostModel`. Until R5 lands, use the platform
  bar-model defaults: half spread 2 bps plus slippage 5 bps per fill, plus $0.005 per share (`DEFAULT_FILL_MODEL`).
- **Metrics** (via `analysis/` and E1):
  - net P&L per trade in spread-σ units and in bps of gross notional
  - Sharpe of the daily pair-book return
  - convergence rate (share of trades exiting on `exit`)
  - exit-reason mix (`exit` / `stop_loss` / `max_hold_exit`)
  - share of `exit` exits with negative P&L (see §5.5)
  - out-of-sample cointegration survival rate
- **Trial counting:** every pair tested and every grid point counts as a trial
  for E2's deflated Sharpe. Log the counts in the study README.

### Studies

| ID | Question | Data / split | Primary metric | Decision rule | Trial budget | Toolkit | Relation to R3 |
|---|---|---|---|---|---|---|---|
| **RS1** Baseline parity | Does research reproduce `PairsStrategy` v6 trade by trade? | 3 pairs (XOM/CVX plus two of your choice), 10 sessions, defaults | Matched trades, entry/exit times, P&L | ≥ 99% of trades match on timestamp and side; P&L within the tolerance set in [platform-parity.md](../design/platform-parity.md). Otherwise fix SS4 before running RS2–RS9 | 0 (no tuning) | SS4, T1 two-leg, T7 `ParityReport` | Prerequisite for both R3 questions |
| **RS2** Pair selection | Which selection method keeps pairs cointegrated out of sample? | Dev universe. Formation 60 sessions → trading 20 sessions, rolled | Out-of-sample cointegration survival rate; trades and net P&L per surviving pair | Adopt the method with the highest survival rate if it beats the distance-method baseline across rolls. Report all three methods | Count every pair tested in every roll | SS1, SS3, C1d, E2 | **Is** R3 question 1 |
| **RS3** Hedge ratio | Which β estimator gives the most stationary out-of-sample spread and the best net P&L? | RS2's pairs; same rolls | Out-of-sample ADF statistic of the spread; net P&L per trade | Use the simplest estimator within 1 standard error of the best on both metrics | 4 estimators × 2 price spaces | SS2, SS3 | Uses R3 pairs |
| **RS4** Half-life and holding period | Does `maxHoldingTimeMs = k × hl` (k = 1, 2, 3, 4), with windows scaled to hl, beat the fixed defaults? | RS2 pairs, RS3 β | Net P&L per trade; exit-reason mix | Adopt k at the plateau's centre; drop pairs whose hl falls outside the tradeable band (§5.4) | 4 k values × 3 window multipliers | SS3, T6 `analysis/` holding-period curve | Adds hl filter to R3's selection |
| **RS5** Entry/exit z | Where is the plateau for (entryZ, exitZ)? | Validation folds | Net P&L per trade and daily Sharpe, as a surface | Pick the centre of the plateau, not the peak; reject if the surface has no plateau | 4 × 5 = 20 points | T1, T6 `analysis/` parameter surface | |
| **RS6** Stops | Which stop rule cuts tail loss for the least mean cost? | Validation folds | 5th percentile trade P&L, max drawdown, mean P&L | Adopt the rule with the best tail at ≤ 10% mean cost (a pre-registered tolerance) | 5 rules × 3 levels | T1 `ExitPolicy`, T6 `analysis/` MFE/MAE | |
| **RS7** Sizing and portfolio | Which sizing mode and portfolio limits give the best risk-adjusted book? | Validation, all selected pairs at once | Daily Sharpe and max drawdown of the book; name overlap | Adopt the mode with the best Sharpe for the same drawdown | 4 modes × 2 scale-in settings | T2 `SizingRule`, T4 `RiskLimits`, C1 `portfolio/` | |
| **RS8** Meta-label | Can a model predict convergence before stop or max hold? | Candidates from SS4 on the chosen config; V1 purged walk-forward | AUC and **net P&L of the filtered strategy** vs. unfiltered | Ship the filter only if net P&L per trade improves on held-out symbols and the deflated Sharpe stays positive | ≤ 3 model classes × small grid | T1 labels, M1, E2, P2 | **Is** R3 question 2 |
| **RS9** Residual stat arb (optional) | Does trading each stock's residual against a sector ETF or PCA factors work better than hand-picked pairs? | Dev universe + sector ETFs | Same as RS5 | Exploratory; becomes a new strategy family only if it beats the pairs book on held-out data | Count every factor model tried | SS3, C1d, SR3 (shared code with Reversals) | Extends R3's PCA step |

Order: RS1 → RS2 → RS3 → RS4 → (RS5, RS6 in parallel) → RS7 → RS8. RS9
whenever someone has time after RS2.

---

## 5. Trade management

Each subsection lists the method, the grid, the expected result (a hypothesis)
and the decision rule. The general method is in
[trade-management.md](../research/trade-management.md); this section covers
pairs specifically.

### 5.1 Pair selection (RS2, SS1)

**Method:**
1. **Formation window** (e.g. 60 sessions; `formation_sessions` is a
   parameter). Compute daily and 1-minute log returns for every development
   symbol.
2. **Candidates, by three routes:**
   - **All pairs within a sector.** 10 names give 45 pairs.
   - **Residual-return clusters.**
     - Remove the market (SPY) and optionally the first k principal components.
     - Run PCA on the residuals.
     - Cluster the loadings with OPTICS or DBSCAN.
     - Form pairs within each cluster (workshop example 7).
   - **Distance method** (Gatev, Goetzmann & Rouwenhorst 2006): normalise each
     price series to 1 at the start of formation and rank pairs by sum of
     squared differences. Keep the top N.
3. **Filters**, applied to formation data only:
   - Engle-Granger p-value below the level. Test both leg orders; see §5.2.
   - Half-life in the tradeable band (§5.4).
   - Correlation of returns above a floor.
   - Enough liquidity (D4 cells).
4. **Trading window** (e.g. the next 20 sessions): trade the selected pairs
   with **fixed** formation estimates. Measure whether each pair stays
   cointegrated (a rolling test inside the trading window) and its net P&L.
5. Roll forward and repeat.

**Multiple testing:** testing 45 pairs at the 5% level produces about 2 false
positives even if no pair is truly cointegrated. Report:
- the number of pairs tested and the number selected;
- the **out-of-sample survival rate**: selected pairs still cointegrated in
  the trading window;
- the same rate for random pairs drawn from the same sector, as a null.

A method is only useful if it clearly beats that null.

**Hypothesis:** clustering beats all-pairs on survival rate; the distance
method finds more pairs but with lower survival. Decision rule: see RS2.

### 5.2 Hedge ratio (RS3, SS2)

β = 1 means "one share of leg 2 per share of leg 1". It is only right if the
two stocks have the same price scale and the same factor exposure, which
almost never happens.

| Estimator | Notes |
|---|---|
| Fixed 1 | Platform default; the baseline to beat |
| Rolling OLS on log prices | Matches the F7 spec; β in log space is a **dollar** ratio |
| Rolling OLS on raw prices | Matches the platform's `rolling_ols` today; β is a share ratio |
| Kalman filter (β, α as a random walk) | Updates every bar, with no window edge effects. Parameters: transition variance δ and observation variance. Tune δ on validation and count it as trials |
| Total least squares (orthogonal regression) | Treats both legs symmetrically |

**Leg-order asymmetry:** OLS of leg 1 on leg 2 does **not** give 1/β of the
reverse regression, so the Engle-Granger test result can depend on which leg
is first. TLS does not have this problem. For OLS, run the test both ways and
either:
- require both orders to pass, or
- pick the order with the more negative statistic and count both as trials.

**Grid:** 4 estimators × 2 price spaces (raw, log). The OLS window comes from
RS4's half-life multiplier.

**Hypothesis:** any estimated β beats β = 1 on stationarity. Kalman adapts
fastest but adds trading noise. TLS is the most stable when two legs have
similar volatility.

**Decision rule:** the simplest estimator within one standard error of the best
on both the out-of-sample ADF statistic and net P&L. Order of simplicity: OLS,
then TLS, then Kalman.

### 5.3 Entry and exit z (RS5)

**Grid:** `entryZScore` ∈ {1.5, 2.0, 2.5, 3.0} × `exitZScore` ∈ {0, 0.25, 0.5, 0.75, 1.0}.
That is 20 points, all logged as trials.

**Output:** a surface plot of net P&L per trade and daily Sharpe, plus trade
count. Narrow entries trade more and pay more cost; wide entries trade rarely
and are noisy.

**Reference point:** the Avellaneda & Lee (2010) residual model opens at
|s| > 1.25 and closes at 0.5 to 0.75 (asymmetric for long and short). Their
s-score comes from an OU model of a residual on daily data, not from a rolling
z on minute bars. Use it as a sanity reference, not a target.

**Cost check:** the gap `entryZ − exitZ`, times the spread σ in dollars per
unit, is the gross move a winning trade captures. If the round-trip cost (§5.9)
exceeds about half of that, the pair cannot pay at that grid point. Mark such
points on the surface.

**Hypothesis:** a plateau around entry 2.0–2.5 and exit 0.25–0.5, with most of
the result driven by pair choice (RS2) rather than these thresholds.

**Decision rule:** the centre of the plateau. If the surface has no plateau
(a single sharp peak), keep the defaults and record a negative result.

### 5.4 Holding period from the half-life (RS4, SS3)

Fit an AR(1) to the formation-window spread sampled at the decision clock (one
point per minute):

```
s_t = c + φ · s_{t−1} + ε_t        hl = −ln 2 / ln φ      (in bars; × 60_000 for ms)
```

This is equivalent to regressing Δs_t on s_{t−1} and using φ = 1 + b. Do not
calculate the half-life if φ ≥ 1 or φ ≤ 0; flag the pair instead.

**Rules to test:**

| Parameter | Rule | Default today | Implied by today's default |
|---|---|---|---|
| `maxHoldingTimeMs` | k × hl, with k ∈ {1, 2, 3, 4} | 24h | — |
| `rollingWindowMs` | m × hl, with m ∈ {3, 4, 5} | 1h (60 bars) | hl ≈ 12–20 min if m = 3–5 |
| `olsWindowMs` | ≥ 2 × `rollingWindowMs`, as the platform comment recommends | 4h | — |
| `minObservations` | ≥ 1 × hl, and enough for a stable std | 30 | — |

**Tradeable band:**
- **Too short:** the half-life is shorter than about 5 minutes, so the spread
  reverts before a next-open fill can capture it and costs dominate.
- **Too long:** the half-life is longer than about half a session (195
  minutes). The spread won't revert intraday, and the time-based windows
  expire overnight (§2.2 item 4).
- Pre-register the exact bounds, and also log results for pairs outside them.

**Overnight:** decide explicitly whether positions may be held overnight. The
24h default allows it, but the z statistics reset each morning. Test the
`flatBy` rule ("flat by 15:55") against overnight carry as part of RS4.

**Hypothesis:**
- k = 2–3 beats 24h, because most `max_hold_exit` trades at 24h are losers
  that sat open too long.
- A window mismatched with the half-life (m far from 3–5) produces a z that
  either lags or reacts to every move.

**Decision rule:** k and m at the plateau centre. The exported
`maxHoldingTimeMs` is computed per pair at export time (§8).

### 5.5 Stops (RS6)

**The z-stop problem.** The z-score is measured against a rolling mean, and the
mean follows the spread.

Example:
- Enter short spread at z = 2.0, with mean m and σ.
- The spread keeps widening slowly.
- After an hour, the mean has moved most of the way toward the spread and σ
  has grown. The trade is now losing several original σ, but z may be around
  2.5. The stop at 4 never fires.

The reverse also happens: the mean catches up to the spread, |z| falls below
`exitZScore`, and the trade exits as a "reversion" **at a loss**.

**Diagnostic for every study:** the share of `exit` exits with negative P&L,
and the distribution of loss in entry-σ units at stop or max-hold time.

| Stop rule | Definition | Grid |
|---|---|---|
| z stop (today) | \|z\| ≥ `stopLossZScore` | {3, 4, 5} |
| Spread loss stop | Adverse move of the spread from the entry price ≥ L × σ at entry (σ frozen at entry, so the mean can't drift it away) | L ∈ {1.5, 2, 3} |
| Cointegration-break exit | Exit when the gate turns off (using the existing hysteresis) while a position is open | on / off |
| Time stop only | No price stop; rely on `maxHoldingTimeMs` from §5.4 | — |
| Combined | Spread loss stop + cointegration-break exit + time stop | best L |

Use MAE analysis (T6 `analysis/`): plot the worst excursion of
eventual winners vs. losers in entry-σ units. Put the loss stop where few
winners go.

**Hypothesis:** for mean reversion, a price stop lowers mean P&L, because it
fires when the spread is most stretched. Its job is tail protection: one pair
breaking on news should not wipe out a month. Expect the spread-loss stop plus
the cointegration-break exit to dominate the z stop on tail risk.

**Decision rule:** the rule with the best 5th-percentile trade P&L and max
drawdown, at a mean cost within the pre-registered tolerance (e.g. ≤ 10% of
mean P&L per trade).

### 5.6 Sizing (RS7)

Leg quantities have to answer two separate questions: how big a position to
take, and how to split it between the legs.

**Leg ratio:**

| Mode | Leg 2 quantity | Notes |
|---|---|---|
| Spread units (today) | `floor(q1 × β)` (raw-price β) | Matches the spread you z-score |
| Dollar-neutral | `q1 × p1 / p2` | Equal notional; matches log-space β = 1 |
| Hedge-ratio (log β) | `β × q1 × p1 / p2` | Notional ratio = β |
| Beta-neutral | `q1 × p1 × β_mkt1 / (p2 × β_mkt2)` | Zero net market beta; β_mkt from formation |

**Position size:**

| Mode | Size | Notes |
|---|---|---|
| Fixed notional (today) | `q1 = floor(tradeNotionalUsd / p1)` | Baseline |
| Spread-vol target | `q1 = targetRiskUsd / (k × σ_spread$)` | A 1σ spread move costs the same $ on every pair |
| Probability scaled | Spread-vol target × size from the RS8 meta-label probability | T2 prob→size; only after RS8 passes |

**Scale-in:** test adding a second unit at z = 3 (when entry is 2), with **at
most one** add. Unlimited averaging into a widening spread is a martingale: it
looks great until a pair breaks. Report the worst trade with and without
scale-in.

**Rounding:** at $5k per leg on a high-priced stock, `floor` can leave the
hedge noticeably off. Report the realised hedge error.

**Hypothesis:** spread-vol targeting lowers drawdown at a similar mean P&L. The
leg-ratio choice matters less than pair selection.

**Decision rule:** best daily Sharpe at equal or lower max drawdown.

### 5.7 Portfolio and risk limits (RS7, T4)

Use T4's `RiskLimits`. Its names and checks match `RiskEngine`, so the limits
you choose carry straight into platform config.

| Limit | Why it matters for pairs | Research rule |
|---|---|---|
| Max open pairs | Pairs in the same sector move together on sector days | Cap; test {3, 5, 10} |
| Single-name exposure across pairs | XOM/CVX and XOM/COP both carry XOM risk | Net each symbol across pairs; cap per symbol notional (`MAX_POSITION_SIZE`, `CONCENTRATION_LIMIT`) |
| Sector net | A long-spread position in one energy pair can offset a short-spread position in another, or stack with it | Report sector net; cap with `maxNetExposurePct` per sector in research |
| Gross exposure | Two legs double gross for the same risk | `maxGrossExposurePct` |
| Earnings blackout | Earnings for one leg are the most common way a spread breaks | No entry within N sessions before either leg's earnings; optionally exit before the release. N ∈ {0, 1, 2} |
| Corporate events | M&A, spin-offs and splits break spreads permanently | Drop the pair; rely on D6 split confirmation |
| Short borrow | One leg is always short | Exclude hard-to-borrow names; note that SEC Rule 201 (short sale restriction after a 10% drop) can block the short leg |
| Daily loss | `INTRADAY_DRAWDOWN` and the kill switch | Mirror the platform default |

**Data gap:** the repo has no earnings calendar. One needs to be added as a
data source (C1g, via EDGAR or a file). Until then, the blackout cannot be
tested; record it as an open item.

### 5.8 Cooldown

`cooldownMs = 60_000` blocks a new entry for one minute after an exit. The
risk: a pair stopped out at |z| = 4 can re-enter one minute later at z = 2
from the same broken regime.

**Test:** cooldown ∈ {1 min, 1 × hl, 3 × hl, rest of session after a stop}.
Compute the return of re-entries made within 3 × hl of a stop vs. first
entries.

**Hypothesis:** re-entries after a stop lose more than first entries. Decision
rule: if so, use the longer cooldown after `stop_loss` exits only.

### 5.9 Costs and execution

**Fills:** a round trip is **4 fills**: open and close, each on both legs. On
the platform bar model (`DEFAULT_FILL_MODEL` in `core/execution/fillModel.ts`),
each fill pays half spread (2 bps) plus slippage (5 bps) at the next bar's
open, plus $0.005 per share commission. The round-trip hurdle is therefore
about 14 bps of each leg's notional before commission, or about 28 bps of
one-leg notional for an equal-notional pair: **double** a single-name
strategy's hurdle.

**Execution risk between legs:** both legs are sent as market orders and fill
at the next open. They can still fill at different prices than the closes the
signal used. T1's simulator must:
- fill each leg separately under the `FillModel`;
- report the realised spread at fill vs. the signal spread.

Use the quote model (see `backtest-execution.md`) on a sample, so the slippage
assumption for the bar model is measured rather than assumed.

**Cost in σ units:** for each pair, compute `cost$ / σ_spread$`. This one
number tells you whether a pair can ever pay. Put it in the RS2 selection
table.

**R5 link:** once R5 calibrates `halfSpreadBps` and `slippageBps` from paper
fills, rerun RS5 and RS7 with the new `CostModel`.

---

## 6. ML angle

Pairs trading has few independent samples: pairs × days is small, and trades
overlap. Keep models shallow and validation strict.

| Use | How | Notes |
|---|---|---|
| **Unsupervised pair discovery** | PCA on residual returns, OPTICS/DBSCAN on the loadings (RS2) | Workshop example 7 |
| **Meta-labeling of convergence** | Candidate = an SS4 entry signal under the chosen config. Label = did the spread reach `exitZScore` before the stop or max hold, **net of costs** (triple-barrier outcome from T1's two-leg simulator). Model: logistic regression, then a small LightGBM | RS8 = R3 Q2 |
| **Probability → size** | Spread-vol target size × the T2 prob→size mapping | Only after RS8 passes |
| **Adaptive hedge ratio** | Kalman filter (§5.2) | A state-space model, not a learned one; tune δ on validation |

**Meta-label features**, all computed as of the decision time:
- entry z, and the spread's z velocity (change over the last 5 bars)
- half-life at entry; formation half-life; their ratio
- Engle-Granger statistic and distance to the critical value; gate state history
- spread σ in dollars; cost in σ units (§5.9)
- correlation of the two legs' returns over the last session vs. formation
- each leg's realised volatility (F3) and relative volume (F1)
- market return and volatility (F8, F3 on SPY)
- **days to the next earnings for either leg** (needs the C1g earnings source)
- time of day; minutes until close
- an indicator for whether the trade would carry overnight

**Validation:**
- V1 purged walk-forward, with an embargo ≥ the max hold.
- Group folds by date, not by pair, so the same day's market move never sits
  in both train and test.
- Count every model × feature set as a trial.

**Decision rule:**
- Do **not** ship on AUC alone.
- Ship only if the filtered strategy's net P&L per trade improves on held-out
  symbols and E2's deflated Sharpe stays positive.
- A negative RS8 result is a valid study outcome; record it.

---

## 7. What the team implements vs. consumes

### Implements

| ID | What | Packet |
|---|---|---|
| SS1 | Pair selection: correlation, clustering, cointegration (both leg orders), distance method, half-life filter | [handoff/tasks/SS1.md](../handoff/tasks/SS1.md) |
| SS2 | Hedge ratios: fixed, rolling OLS (raw and log), Kalman, TLS | [handoff/tasks/SS2.md](../handoff/tasks/SS2.md) |
| SS3 | OU fit, half-life, Engle-Granger matching `services/indicators/cointegration.ts` (no-lag DF, MacKinnon values) | [handoff/tasks/SS3.md](../handoff/tasks/SS3.md) |
| SS4 | `CandidateGenerator` reproducing `PairsStrategy` v6 rules exactly (§2.2), outputting a `CandidateFrame` with both legs | [handoff/tasks/SS4.md](../handoff/tasks/SS4.md) |
| F7 | Pair spread features (z, β, half-life, cointegration stability), with the F8 owner on the context-symbol extension | [handoff/tasks/F7.md](../handoff/tasks/F7.md) |
| RS1–RS9 | Studies in `studies/stat-arb/` | — |
| P9 spec | Config additions and their tests in the trading-platform repo (§8) | [handoff/tasks/P9.md](../handoff/tasks/P9.md), [platform-tasks.md](../platform-tasks.md) |

**Contribution to shared code:** T1's simulator must support **multi-leg
trades**. Today, L2's `simulate_exits` handles one symbol, and R3 already calls
for a "two-leg version". Agree the interface with the T1 owner before SS4
starts. A trade is a set of legs with signed quantities, sharing one exit
decision; P&L and costs are summed over legs.

### Consumes

| ID | What | Packet |
|---|---|---|
| T1 | `ExitPolicy`, `FillModel`, `simulate_trades` → `TradeLedger` (multi-leg) | [T1](../handoff/tasks/T1.md) |
| T2 | `SizingRule`: spread-vol target, prob→size | [T2](../handoff/tasks/T2.md) |
| T3 | Filters: time window, earnings blackout | [T3](../handoff/tasks/T3.md) |
| T4 | `RiskLimits` with platform names | [T4](../handoff/tasks/T4.md) |
| T5 | `CostModel` (R5-calibrated) | [T5](../handoff/tasks/T5.md) |
| T6 | `analysis/`: holding-period curve, OU half-life, MFE/MAE, parameter surfaces | [T6](../handoff/tasks/T6.md) |
| T7 | `export_platform_config`, `ParityReport` | [T7](../handoff/tasks/T7.md) |
| T8 | `StrategySpec` card generator, `shpe new strategy` | [T8](../handoff/tasks/T8.md) |
| C1b | Events/candidates layer (entries are events, not every N minutes) | [C1b](../handoff/tasks/C1b.md) |
| C1d | Context symbols, cross-sectional transforms (PCA residuals) | [C1d](../handoff/tasks/C1d.md) |
| C1e | Daily bars for formation | [C1e](../handoff/tasks/C1e.md) |
| C1f / V1 | Purged walk-forward with embargo | [C1f](../handoff/tasks/C1f.md), [V1](../handoff/tasks/V1.md) |
| C1g | Pluggable data sources; an earnings calendar (EDGAR or file) is added on top, see open question | [C1g](../handoff/tasks/C1g.md) |
| C1m | Point-in-time universe and security master | [C1m](../handoff/tasks/C1m.md) |
| C1o | Fixture with a planted cointegrated pair | [C1o](../handoff/tasks/C1o.md) |
| F3, F8 | Realised vol, market return | [F3](../handoff/tasks/F3.md), [F8](../handoff/tasks/F8.md) |
| E1, E2 | Trading metrics; deflated Sharpe and trial counting | [E1](../handoff/tasks/E1.md), [E2](../handoff/tasks/E2.md) |
| M1a–c | Models for RS8 | [tasks.md](../tasks.md) |
| R5 | Cost calibration | `studies/shared/r5-tca/` |
| P1, P2, P6 | Candidate log; ONNX selector wrapper; connecting sizers | [platform-tasks.md](../platform-tasks.md) |

**Start without waiting:** SS2 and SS3 depend only on pandas and numpy, and
can begin on the C1o fixture (or a hand-made cointegrated pair) today. SS1
needs SS3. SS4 needs T1's multi-leg interface to be agreed, though not
finished.

---

## 8. Platform deliverable (P9)

Research produces a pairs config per selected pair through
`export_platform_config`. It also produces a `ParityReport` showing that the
platform backtest reproduces research on the same sessions. Any change to
strategy behaviour bumps `PairsStrategy.VERSION` (6 → 7), as the platform's
sizing doc requires.

### 8.1 Existing fields: how research sets them

| `pairsConfig` field | Set from | Study |
|---|---|---|
| `leg1Symbol`, `leg2Symbol` | Selected pair; leg order from §5.2 | RS2, RS3 |
| `hedgeRatioMethod` | Chosen estimator (`fixed` only if a study chose it) | RS3 |
| `fixedHedgeRatio` | Formation β, if a frozen β is chosen | RS3 |
| `entryZScore`, `exitZScore` | Plateau centre | RS5 |
| `stopLossZScore` | Chosen level, or set high (e.g. 6) as a backstop if the spread-loss stop is adopted | RS6 |
| `maxHoldingTimeMs` | `k × hl × 60_000`, rounded to whole minutes, per pair | RS4 |
| `rollingWindowMs` | `m × hl × 60_000` | RS4 |
| `olsWindowMs` | ≥ 2 × `rollingWindowMs` | RS4 |
| `olsRecalcIntervalBars` | Keep 5 unless RS3 shows the result is sensitive to it | RS3 |
| `minObservations` | ≥ hl, ≥ 30 | RS4 |
| `cointSignificanceLevel`, `cointHysteresis` | Keep the defaults unless RS2 shows gate flapping | RS2 |
| `tradeNotionalUsd` | From spread-vol target at today's σ (until P6 adds a sizer) | RS7 |
| `maxPositionSizeUsd` | Per-symbol cap from the portfolio limits | RS7 |
| `cooldownMs` | Chosen cooldown | §5.8 |
| `executionAlgo` | `market` (the bar model assumes it) | — |

### 8.2 Proposed new fields (P9)

| Field | Values | Why |
|---|---|---|
| `priceSpace` | `"raw"` (default, current) \| `"log"` | Brings the platform in line with the F7 definition if RS3 picks log |
| `hedgeRatioMethod` | add `"kalman"`, `"tls"` | If RS3 picks them |
| `kalmanDelta`, `kalmanObsVar` | numbers | Kalman settings |
| `stopRule` | `"z"` \| `"spread_loss"` \| `"coint_break"` \| `"combined"` | RS6 |
| `stopSpreadSigma` | number | Spread-loss stop, with σ frozen at entry |
| `exitOnCointBreak` | boolean | Today the gate blocks entries only |
| `maxHoldHalfLives` | number (optional) | Documents how `maxHoldingTimeMs` was derived; export still sends ms |
| `flatBy` | `"15:55"` \| null | If RS4 rejects overnight carry |
| `earningsBlackoutSessions` | integer | §5.7; needs an earnings feed on the platform too |
| `sizingMode` | `"spread_units"` \| `"dollar_neutral"` \| `"beta_neutral"` \| `"spread_vol"` | RS7 |
| `targetRiskUsd` | number | For `spread_vol` |
| `maxScaleIns`, `scaleInZScore` | 0–1, number | §5.6 |
| `postStopCooldownMs` | number | §5.8 |
| `metaModelPath`, `minProbability` | path, number | RS8, via the P2 wrapper |

---

## 9. Kill criteria and monitoring

Write these into the spec card **before** paper trading starts. Monitoring
reads the P1 candidate log and the broker ledger.

**Per pair, disable when any of these holds:**
- The cointegration gate stays off for more than N consecutive recalculations
  (pre-register N; e.g. a full session).
- The realised half-life over the last 20 sessions exceeds 2 × the formation
  half-life.
- `max_hold_exit` plus `stop_loss` make up more than X% of the last 20 trades,
  where X is the backtest rate plus a tolerance.
- A corporate event is announced for either leg (M&A, spin-off).

**Book level, pause and review when any of these holds:**
- The rolling 20-day net Sharpe falls below the lower bound of the backtest's
  bootstrap interval (E2).
- Realised cost per round trip exceeds the `CostModel` by more than Y bps (R5).
- Paper trades diverge from a same-day platform backtest beyond the parity
  tolerance.
- `INTRADAY_DRAWDOWN` triggers the kill switch.

**Dashboard, per pair, weekly:**
- trade count
- exit-reason mix
- share of `exit` exits at a loss
- realised vs. formation half-life
- realised hedge error
- cost per round trip

**Re-selection:** pairs are re-selected each formation roll (e.g. monthly)
using the frozen RS2 method. A pair that is retired and later re-selected
counts as a new trial.

---

## 10. Workshop tie-ins

| Example | Stat arb role |
|---|---|
| [07 Unsupervised pairs](../workshop/examples/07-unsupervised-pairs.md) | **Owned by this team.** PCA and clustering on residual returns, cointegration within clusters, out-of-sample survival vs. all pairs. This is RS2 in miniature on the snapshot. |
| [08 Portfolio construction](../workshop/examples/08-portfolio-construction.md) | The pair book: dollar- vs. beta-neutral, name overlap across pairs, sector net, spread-vol sizing (RS7). |
| [06 Overfitting casino](../workshop/examples/06-overfitting-casino.md) | Test every pair in a sector and keep the best backtest; watch it fail out of sample. Use the "45 pairs → about 2 false positives" calculation and E2's deflated Sharpe. |
| [09 Trade management](../workshop/examples/09-trade-management.md) | Half-life holding period, the z-stop problem (with a plot of the mean following the spread), cooldown after stops. Compare with momentum: mean reversion wants targets and disaster stops only. |

---

## 11. Milestones and suggested plan

Suggested roles for a team of four:
- **selection lead:** SS1, RS2
- **models lead:** SS2, SS3, RS3, RS4
- **trade-management lead:** RS5–RS7
- **platform liaison:** SS4, RS1, P9

| Milestone | Contents | Depends on | Done when |
|---|---|---|---|
| **MS0: Read and set up** | Read this guide, the R3 README, the platform pairs code, the research protocol. Each member runs the hello model. | — | Everyone can explain §2.2 items 1–8 |
| **MS1: Statistics on fixture** | SS3 (OU, half-life, Engle-Granger matching the platform) and SS2 (four estimators) | C1o or a hand-made pair | On the fixture, the planted pair is found and the fake one rejected; Engle-Granger statistics match `cointegration.ts` on 3 real pairs to 1e-6 |
| **MS2: Baseline parity** | SS4, RS1 | T1 multi-leg, T7 | RS1 decision rule met |
| **MS3: Selection** | SS1, RS2 | C1d, C1e, E2 | RS2 README records survival rates for 3 methods + null |
| **MS4: Calibration** | RS3, RS4, RS5, RS6 | T6 | Each README has a pre-registered grid, a surface or table, and a decision |
| **MS5: Book** | RS7; earnings blackout once C1g lands | T2, T4, C1g | Book-level Sharpe and drawdown on held-out symbols |
| **MS6: Hand-over** | Spec card, P9 config diff, `ParityReport`, kill criteria | T7, T8 | A platform backtest of the exported config matches research; the spec card is reviewed by another team |
| **MS7: Meta-label (optional)** | RS8; RS9 | M1, E2, P2 | Decision recorded (ship or negative result) |
| **Paper** | Paper-trade the exported config with monitoring (§9) | P1 | 20 sessions of paper trading within the kill-criteria bands |

---

## 12. Pitfalls

- **Selecting pairs on the whole sample.** Any statistic used to choose a pair
  must come from the formation window only. Choosing on the full period is
  look-ahead and the most common pairs-trading mistake.
- **Survivorship.** The D4 universe is static as of 2026-10-01. Pairs that broke
  because a company was acquired or delisted are missing. Note this in every
  multi-month study until C1m lands.
- **Raw vs. log price space.** The platform uses raw prices and the F7 spec uses
  log prices. SS2/SS4 must support both, RS1 must use raw, and the choice must
  be written into the export (`priceSpace`).
- **The z-score includes the current point** (§2.2 item 3). Copy this behaviour
  for parity, or RS1 will not match.
- **Time-based windows and overnight carry** (§2.2 item 4). Morning sessions are
  idle until 30 observations exist, and overnight positions are scored against
  a new mean.
- **Stale β.** The platform updates β only while the pair is cointegrated. A
  pair that drops out of cointegration and comes back resumes with an old β.
- **The gate is off in `fixed` mode.** A fixed-β pair trades whether or not it
  is cointegrated.
- **Repeated testing.** An Engle-Granger test every 5 bars is many tests a day.
  Its false rejection rate is higher than the nominal level, and residual
  autocorrelation (no lags in the DF step) distorts it further. Hysteresis
  reduces the flapping but does not fix the statistics.
- **"Reverted" is not "made money"** (§5.5). Always report P&L by exit reason.
- **Unequal notional.** With β = 1 in raw prices, a $500 stock paired with a
  $50 stock holds equal share counts, so leg 1 carries 10× the dollar exposure
  of leg 2.
- **Few independent samples.** Twenty pairs over three months is not 20 × 60
  independent observations; trades overlap within a day. Keep models shallow
  and use block bootstrap (E2) for intervals.
- **Corporate actions.** An unadjusted split looks like a 50% spread move. Use
  D6 confirmation before trusting any extreme z.
- **Execution risk between legs.** Research fills both legs at the next open.
  Live, they fill at slightly different moments. Measure the gap on the quote
  model.

---

## 13. Reading list

- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica* 55(2).
- MacKinnon, J. G. (1991). Critical values for cointegration tests. In *Long-Run Economic Relationships*, Oxford University Press. (The source of the platform's critical values.)
- Gatev, E., Goetzmann, W. N. & Rouwenhorst, K. G. (2006). Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies* 19(3). (Distance method.)
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley. (Cointegration framing; pair selection.)
- Avellaneda, M. & Lee, J.-H. (2010). Statistical arbitrage in the U.S. equities market. *Quantitative Finance* 10(7). (Residuals against PCA and ETFs, OU s-scores; RS9.)
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley. (Mean reversion, half-life, Kalman hedge ratios.)
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Triple barrier and meta-labeling, purged cross-validation, bet sizing.)
- Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management* 40(5).
