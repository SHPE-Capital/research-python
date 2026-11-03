# Example 8: Portfolio construction and neutralization

| | |
|---|---|
| **Teams** | All. Momentum (SM3) and reversals (cross-sectional fades) use it directly; stat arb uses it to combine many pairs into one book. |
| **Level** | Int (the feature-neutralization algebra is Adv) |
| **Duration** | 90 minutes |
| **Prerequisites** | [Example 3](03-cross-sectional-ranker.md) (cross-sectional scores), [Example 5](05-volatility-forecast.md) (volatility forecast) |
| **Data** | Synthetic fixture with planted market and sector factors plus a planted cross-sectional alpha (C1o); then example 3's scores on the workshop snapshot (D5) |
| **Library tasks used** | C1d cross-sectional transforms (rank, z-score, neutralize), C1e daily frequency, F3 realized volatility (daily HAR), F8 market return, T4 risk limits, T5 costs, E1 turnover and cost-adjusted P&L, E2 bootstrap |
| **Build task** | W8 |

## Why this example

A model produces **scores**: one number per stock per decision time. A trading
system needs **positions**. The step between them decides a lot of the outcome:

- how much of the result is the signal, and how much is market or sector exposure
- how much the strategy trades, and therefore what it pays in costs
- whether the book fits inside the platform's risk limits

Two teams with the same scores can end up with very different Sharpe ratios.
The difference is in portfolio construction.

**Industry analogue:** in a typical quant equity process, alpha models and
portfolio construction are separate steps, often owned by different people.
Portfolio construction turns alphas into weights subject to a risk model, a cost model and constraints.
Grinold & Kahn's *Active Portfolio Management* is the standard reference for
the alpha-to-weight step.

## Learning objectives

1. Convert scores to weights four ways and compare them: top-k, rank-weighted,
   z-scored and volatility-scaled.
2. Remove market, sector and feature exposure from scores or weights, and check
   that the alpha survives.
3. Measure turnover and cut it with no-trade bands and partial adjustment,
   while keeping net P&L.
4. Apply risk limits that mirror the platform's RiskEngine settings.
5. Attribute P&L to the market, sectors and the residual.

## Background

### From scores to weights

Let `s_i` be the score of stock `i` at one decision time, across `N` stocks.

| Scheme | Weight | Notes |
|---|---|---|
| Top-k / bottom-k | `+1/k` for the top k, `−1/k` for the bottom k, 0 otherwise | Simple and robust; ignores score size; turnover jumps when names cross the cut-off |
| Rank-weighted | `w_i ∝ rank(s_i) − (N+1)/2`, scaled to gross 1 | Uses the whole cross-section; robust to outliers |
| Z-scored | `w_i ∝ (s_i − mean(s)) / std(s)`, clipped at ±3 | Uses score size; sensitive to outliers without clipping |
| Volatility-scaled | `w_i ∝ z_i / σ̂_i` | Equalises risk contribution, not dollars; needs [Example 5](05-volatility-forecast.md)'s forecast |

All schemes are then scaled to a target gross exposure, e.g. `Σ|w_i| = 1`.

**Grinold & Kahn's alpha rule.** Turn a score into an expected-return forecast:

```
α_i = IC · σ_i · z_i
```

Here `IC` is the information coefficient (the correlation between score and
subsequent return), `σ_i` is the stock's residual volatility and `z_i` the
standardised score. Without a full risk model, a reasonable weight is
`w_i ∝ α_i / σ_i² = IC · z_i / σ_i`, which is the volatility-scaled scheme up
to a constant.

### Neutralization

The aim is to keep the alpha and drop exposures you are not paid for.

| Exposure | Method |
|---|---|
| Market (net) | Make `Σ w_i = 0` (dollar neutral), or `Σ w_i β_i = 0` (beta neutral), with `β_i` from a rolling regression on SPY returns (F8), estimated on past data only |
| Sector | Demean scores within each sector: `s_i − mean(s_sector(i))` |
| Arbitrary factors | Feature neutralization: project scores off a factor matrix `F` (N × k) |

Feature neutralization:

```
s_neutral = s − p · F (Fᵀ F)⁺ Fᵀ s           0 ≤ p ≤ 1
```

`(·)⁺` is the pseudo-inverse and `p` is the proportion removed (`p = 1` is full
removal). C1d's `neutralize` transform implements this per timestamp.
Afterwards, check that the correlation of `s_neutral` with each column of
`F` is close to 0.

### Turnover and costs

```
turnover_t = ½ Σ_i |w_{i,t} − w_{i,t−}|        w_{t−} = weights drifted by returns since the last rebalance
cost_t     = Σ_i |w_{i,t} − w_{i,t−}| · c_i      c_i = one-way cost (half spread + slippage), T5
```

Ways to trade less:

- **No-trade band:** only trade stock `i` if `|w_target − w_current| > b`.
- **Partial adjustment:** `w_new = w_current + λ (w_target − w_current)`, 0 < λ ≤ 1.
- **Score smoothing:** an EMA of scores over several periods. This adds lag, so
  check the IC decay curve (E1) first.
- **Cost-aware ranking:** require a bigger score change to enter than to stay (hysteresis).

Report P&L before and after costs, and the **break-even cost**: the
one-way cost in bps at which net P&L reaches zero (E1 reports it).

### Risk limits mirroring the platform

T4 uses the RiskEngine's names, so a research book that passes T4 passes the
platform's checks. Default values are in the platform's `config/defaults.ts`.

| Platform field | Meaning | Research check (T4) |
|---|---|---|
| `maxPositionSizeUsd` | Max notional in one symbol | Clip `|w_i| · equity` |
| `maxNotionalExposureUsd` | Max total notional | Scale gross down |
| `maxGrossExposurePct` | Gross exposure as % of equity | Scale gross down |
| `maxNetExposurePct` | Net exposure as % of equity | Shift weights towards dollar-neutral |
| `maxConcentrationPct` (0.30) | Max share of the book in one symbol | Clip and redistribute |
| `cashReservePct` (0.05) | Equity kept as cash | Gross ≤ 1 − reserve, for a cash-only account |
| `allowShortSelling` | Shorts allowed | If false, long-only book: shorts become 0 |
| `maxIntradayDrawdownPct` (0.05) | Kill switch | Flatten for the rest of the day in the backtest |
| Strategy budget `maxCapitalPct` | Share of equity for this strategy | Scale the strategy's gross |

Clipping changes the weights. Apply limits **after** neutralization, then check
how much neutrality was lost. If it is a lot, iterate, or solve the
constrained problem directly (challenge exercise).

## Setup

**Fixture (C1o).** The current `synthetic_bars` has no factor structure and no
cross-sectional alpha. C1o adds, at daily frequency:

| Planted effect | Expected behaviour |
|---|---|
| Market factor with β spread across stocks (0.5–1.5) | Long-short top-k books have a non-zero β unless neutralised |
| 3 sector factors | A raw score correlated with sector membership loads on sector returns |
| A cross-sectional alpha: next-day residual return `∝ a_i,t` plus noise, where `a` is visible as a feature | Neutralised books keep the alpha |
| A "contaminated" score `s = a + 0.5·β + 0.5·sector_dummy` | The trap: raw IC looks better than neutralised IC in an up-market sample |

Use 60 symbols, about two years, daily rebalancing.

Config sketch (syntax pending C1a/C1d/C1h; a `portfolio:` section is proposed, see builder notes):

```yaml
name: ex08_portfolio
seed: 7
data:
  source: fixture
  n_symbols: 60
  start: "2024-09-02"
  end: "2026-08-28"
  factors: {market: 1, sectors: 3}
  xs_alpha: {ic: 0.05, contaminate: {beta: 0.5, sector: 0.5}}   # C1o
frequency: daily                                                 # C1e
score:
  source: feature              # or: model predictions from example 3
  column: contaminated_score
portfolio:
  scheme: vol_scaled           # top_k | rank | zscore | vol_scaled
  top_k: 10
  neutralize: [market_beta, sector]                              # C1d
  neutralize_proportion: 1.0
  gross: 1.0
  turnover: {no_trade_band: 0.002, partial_adjustment: 0.5}
risk:                                                            # T4, platform names
  maxConcentrationPct: 0.30
  maxNetExposurePct: 0.10
  maxGrossExposurePct: 1.0
  cashReservePct: 0.05
  allowShortSelling: true
costs: {half_spread_bps: 2.0, slippage_bps: 5.0, commission_per_share: 0.005}  # T5 = platform defaults
report: [gross_pnl, net_pnl, turnover, break_even_cost, beta, sector_exposure, attribution]
```

## Walkthrough

1. **Score diagnostics (10 min).** Before building weights, compute the daily
   rank IC of the contaminated score, and its correlation with β and the sector
   dummies. *Expected:* a high correlation with β, and a raw IC that moves with
   the market's daily direction.

2. **Four schemes, no neutralization (15 min).** Build top-k, rank, z-score and
   vol-scaled books. Report for each: gross Sharpe, realised β to the market, and
   turnover. *Expected:* every book has a positive β; vol-scaled has the most
   stable realised volatility.

3. **Neutralize (20 min).** Apply market-beta and sector neutralization to the
   scores, then rebuild.
   - *Expected:* realised β ≈ 0 and sector exposures ≈ 0.
   - Gross Sharpe falls in the up-market part of the sample and rises in the
     down-market part. Over the full sample, the planted alpha remains.
   - Compare with the "clean" alpha `a` as an upper bound.

4. **Attribution (10 min).** Regress daily book returns on the market and sector
   returns. Report the intercept (residual alpha) and its block-bootstrap
   interval. *Expected:* after neutralization, nearly all P&L is in the
   intercept.

5. **Turnover and costs (20 min).** Sweep the no-trade band and partial
   adjustment λ. Plot net Sharpe against turnover.
   - *Expected:* a hump. Small bands barely change costs. Large bands miss the
     signal. The best setting is somewhere in between.
   - Report the break-even cost before and after.
   - The sweep is a grid: count its cells as trials ([Example 6](06-overfitting-casino.md)).

6. **Risk limits (15 min).** Apply the T4 limits at a small account size, e.g.
   $25,000 with `maxPositionSizeUsd` = $2,500. Show how many weights get
   clipped, and how much β and sector exposure come back after clipping.
   Discuss: a 60-name long-short book may not fit a small account, so either
   use fewer names or accept residual exposure.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Implement the four score-to-weight schemes with gross scaling. | Pure pandas per date: `groupby(ts).rank()`, z-score, clip ±3, divide by `σ̂` from F3, then divide by `Σ|w|`. Handle dates with fewer than `2k` valid names: skip them and count how often that happens. |
| Core | Sector-demean and beta-neutralise. Show that the book's β and sector exposures are close to 0. | Rolling 60-day β on past returns only (shift by one day). After demeaning, `Σ w_i β_i` is not exactly 0: regress weights on β and subtract the fit. |
| Stretch | Feature neutralization with `p ∈ {0, 0.5, 1}` against a matrix of 5 style features (size, momentum, volatility, liquidity, reversal). Plot IC vs. `p`. | Use `numpy.linalg.pinv`. If IC collapses at `p = 1`, the "alpha" was a known style factor. That is useful to know, not a failure. |
| Stretch | No-trade band and partial-adjustment sweep with net Sharpe and break-even cost. | Simulate drifted weights `w_{t−}` explicitly; turnover measured against target weights alone is wrong. |
| Challenge | Solve the constrained problem directly: maximise `αᵀw − (γ/2) wᵀΣw − cᵀ|w − w₀|` subject to `Σw = 0`, `|w_i| ≤ cap` and `wᵀβ = 0`. Compare with the step-by-step approach. | Use `cvxpy` (an optional extra, not in base dependencies). `Σ` from a factor model: `B F Bᵀ + D`. Shrink `Σ`; a sample covariance with 60 names and short history is badly conditioned. |
| Challenge | Run on example 3's real momentum scores. Is the 12-1 signal's P&L mostly market and sector exposure, or residual? | Six months of data is too short for 12-1. Use the snapshot's longest daily history and state the sample length clearly. |

Full solutions live on the `workshop-solutions` branch under `workshop/08_portfolio_construction/`.

## Discussion questions

1. Dollar-neutral and beta-neutral are different. When does the difference matter?
2. If neutralization removes most of a signal's IC, what was the signal?
3. When should you neutralise scores, and when weights? Can you do both?
4. A no-trade band reduces costs but also reduces responsiveness. How do you
   choose its size without overfitting?
5. Should risk limits be part of the research backtest, or only enforced live?

## Pitfalls

- **Look-ahead in β or sector data.** β fitted on the whole sample, or sector
  labels as of today, leak future information. Use rolling β on past data and
  point-in-time classifications (C1m).
- **Turnover computed against the previous target, not the drifted weights.**
  This undercounts trading.
- **Ignoring dates with few names.** Top-10/bottom-10 on a date with 15 valid
  names is a different strategy.
- **Clipping after neutralization and assuming neutrality holds.** Recheck exposures.
- **Annualising a daily long-short Sharpe without the cost of shorting.** Borrow
  fees and short availability are real costs on small names.
- **Reporting gross Sharpe only.** Always report net, turnover and break-even cost.

## Connection to team strategies

- **Momentum** ([teams/momentum.md](../../teams/momentum.md)): SM3 cross-sectional
  12-1 is a portfolio strategy by design. Raw momentum books carry large market
  and sector bets; residual momentum is the neutralised version.
- **Reversals** ([teams/reversals.md](../../teams/reversals.md)): a
  cross-sectional short-term reversal book (buy the day's losers, sell its
  winners) is very sensitive to turnover and costs. Sector-demeaned reversal is
  a common variant.
- **Stat arb** ([teams/stat-arb.md](../../teams/stat-arb.md)): many pairs make a
  book. Each pair is roughly market neutral, but the pairs together can still
  load on one sector. Apply the concentration and gross limits across pairs.
- Feeds [Example 9](09-trade-management.md): sizing rules there work per trade;
  this example works per book.

## Builder notes (W8)

| Item | Detail |
|---|---|
| Folder | `workshop/08_portfolio_construction/`: `README.md`, `ex08_portfolio.py` (jupytext), `configs/ex08_fixture.yaml`, `configs/ex08_snapshot.yaml` |
| Library tasks required | C1d (`rank`, `zscore`, `neutralize`), C1e daily, C1o factors and cross-sectional alpha, F3 daily vol forecast, F8 market return, T4 risk limits, T5 costs, E1 turnover and break-even cost, E2 bootstrap |
| Gap | No task ID owns score-to-weight conversion or turnover control. Options: (a) put them in `trade/sizing.py` under T2 as "portfolio sizers"; (b) add a `portfolio/` package with its own task ID. Until decided, W8 implements them in the workshop folder as plain functions and the library takes them over later. |
| Fixture effect needed | Daily market + 3 sector factors with spread-out β; cross-sectional alpha with known IC; a contaminated score column |
| CI check | On the C1o fixture with fixed seed: neutralised vol-scaled book has `|β| < 0.05` and sector exposures `< 0.05`; residual alpha t-stat > 2 at zero cost; contaminated raw score has `corr(s, β) > 0.3` |
| Participant scaffold | Steps 2 and 3 are `# TODO`; steps 4–6 are given with questions |

## Further reading

- Grinold, R. & Kahn, R. (1999). *Active Portfolio Management*, 2nd ed. McGraw-Hill. (Alpha forecasting and the alpha-to-portfolio step.)
- Avellaneda, M. & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*. (Factor-neutral residual books.)
- [research/trade-management.md](../../research/trade-management.md): sizing and risk sections.
- [design/trade-layer.md](../../design/trade-layer.md): where T2 and T4 sit.
