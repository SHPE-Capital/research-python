# Example 7: PCA and clustering for pair discovery

| | |
|---|---|
| **Teams** | Stat arb (the main audience); momentum and reversals benefit from the residual-return idea |
| **Level** | Int–Adv (PCA and the Marchenko–Pastur bound are Adv) |
| **Duration** | 120 minutes |
| **Prerequisites** | [Example 6](06-overfitting-casino.md) (trial counting), the foundation session on features; comfort with OLS and eigenvectors |
| **Data** | Synthetic fixture with planted factor structure and cointegrated pairs (C1o); then the workshop snapshot (D5), dev symbols, minute bars resampled to 30 minutes or daily |
| **Library tasks used** | C1o fixture, C1e daily frequency, C1d cross-sectional transforms, F7 pair spread z-score, SS1 selection, SS2 hedge ratios, SS3 OU half-life, SS4 pairs candidate generator, T1 simulator, T5 costs, E2 bootstrap |
| **Build task** | W7 |

## Why this example

Pairs trading starts with a search problem. With `n` symbols there are
`n(n−1)/2` candidate pairs: 435 for 30 symbols, 4,950 for 100. Test them all
for cointegration at 5% and about 5% of unrelated pairs will pass by chance:
roughly 22 false pairs out of 435. Most of those break down out of sample.

Unsupervised learning shrinks the search. PCA finds the common factors driving
returns. Clustering on each stock's factor loadings groups stocks that move for
the same reasons. Testing only within clusters means fewer tests, fewer false
discoveries and pairs that have an economic reason to stay together.

This example is the workshop version of study R3, question Q1
([studies/r3-pairs-clustering/README.md](../../../studies/r3-pairs-clustering/README.md)):
*does clustering on residual returns, with cointegration tested only inside
clusters, beat all-pairs testing on out-of-sample survival?*

**Industry analogue:** statistical-arbitrage desks model returns with
statistical factors (PCA) and trade the residuals. Avellaneda & Lee (2010) is
the standard public description: residuals after removing PCA or sector-ETF
factors, modelled as mean-reverting OU processes, traded on an s-score. The
distance method of Gatev et al. (2006) is the classic baseline.

## Learning objectives

1. Separate formation and trading windows and say why clustering on the whole
   sample is look-ahead.
2. Use PCA on standardised returns to find common factors, and choose the
   number of components.
3. Cluster stocks on factor loadings with OPTICS or DBSCAN, and read the result.
4. Run Engle-Granger tests within clusters, filter by half-life, and count
   trials correctly.
5. Compare candidate pair lists on **out-of-sample** survival and net P&L, and
   export the survivors as a pair-list config for SS4 and the platform.

## Background

### Formation and trading windows

All selection happens in a **formation window** (e.g. 60 sessions). The
selected pairs are traded in the following **trading window** (e.g. 20
sessions), with no reselection. Then roll forward. Selection that uses any data
from the trading window is look-ahead and makes every pair look stable.

### PCA on returns

Let `R` be a `T × n` matrix of returns in the formation window. Standardise each
column (z-score on formation data only), then compute the correlation matrix `C`
and its eigendecomposition `C = V Λ Vᵀ`.

- The first eigenvector is usually the market: every loading has the same sign.
- The next few pick up sectors, size and style.
- The loadings matrix `B = V[:, :k]` (n × k) places each stock in factor space.

**Choosing k.** Options, in order of rigour:
1. Fixed `k` (e.g. 5), or enough components for a fixed share (e.g. 50%) of variance.
2. Marchenko–Pastur cut-off (stretch exercise): for pure noise with `T`
   observations and `n` series, eigenvalues of `C` fall below
   `λ₊ = (1 + sqrt(n/T))²`. Keep components above `λ₊`.

### Clustering on loadings

Cluster the rows of `B` (one point per stock), usually after normalising each row.

- **DBSCAN**: points with at least `min_samples` neighbours within distance `eps` form a cluster. The rest are noise.
- **OPTICS**: like DBSCAN, but it does not need a single `eps`, which suits clusters of different density.
- Points labelled noise (`-1`) are dropped. Many stocks have no good partner, and the method says so.
- K-means is a weaker alternative: it forces every stock into a cluster.

### Cointegration within clusters

For each pair `(y, x)` inside a cluster, on formation-window **log prices**:

```
log y_t = α + β·log x_t + s_t              (OLS hedge regression)
ADF test on s_t                              (Engle-Granger, step 2)
```

Keep pairs with an EG p-value below the threshold. The platform's
`cointSignificanceLevel` is 0.05. Then fit an AR(1) to the spread:

```
s_t = c + φ·s_{t−1} + ε_t
half-life = −ln 2 / ln φ                    (bars; requires 0 < φ < 1)
```

Keep pairs whose half-life is long enough to cover costs, and short enough to
revert several times inside the trading window. Example rule: between 5 bars and
a quarter of the trading window.

Engle-Granger is not symmetric: regressing `y` on `x` and `x` on `y` give
different p-values. Pick one convention (e.g. the more liquid leg as `x`), or
require both directions to pass. Count both directions as trials if you test both.

### Baselines

| Method | Selection rule |
|---|---|
| All-pairs EG | Every pair in the universe, EG p < 0.05, half-life filter |
| Distance (Gatev et al. 2006) | Normalise prices to 1 at formation start; pick the pairs with the smallest sum of squared differences; trade at 2σ of the formation-window spread |
| Correlation | Top pairs by return correlation in formation, then EG |
| PCA + clustering | Within-cluster EG plus half-life filter (this example) |

### Multiple testing

The number of tested pairs is the trial count. With `m` tests at level `α`, expect
`m·α` false passes if no pair is truly cointegrated. Options:
- Bonferroni, `α/m`: strict.
- Benjamini–Hochberg on the EG p-values: controls the false discovery rate.
- Count `m` into the experiment's trials (M2) so DSR sees it.

Clustering helps mainly because `m` drops from `n(n−1)/2` to the sum of
`c(c−1)/2` over clusters of size `c`.

## Setup

**Fixture (C1o).** The current `synthetic_bars` has independent symbols, so
nothing should cluster and nothing should cointegrate. That is a useful
negative check. C1o adds:

| Planted effect | Purpose |
|---|---|
| 1 market factor plus 3 "sector" factors with known loadings | PCA should recover 4 components; clustering should recover the 3 sectors |
| 2 cointegrated pairs inside sectors: `log y = β log x + OU spread`, known β and half-life | Within-cluster EG should find them; SS3 should recover the half-life |
| 1 pair in the same sector that is correlated but **not** cointegrated | The trap: a high correlation without a stationary spread |
| Idiosyncratic symbols with no sector | Should be labelled noise by DBSCAN/OPTICS |

Use 24 symbols and about one year of history.

Config sketch (syntax pending C1a/C1h and SS1):

```yaml
name: ex07_pca_cluster_pairs
seed: 7
data:
  source: fixture
  n_symbols: 24
  start: "2025-09-01"
  end: "2026-08-28"
  factors: {market: 1, sectors: 3}                     # C1o
  cointegrated_pairs: [{beta: 1.2, half_life_bars: 12}, {beta: 0.8, half_life_bars: 30}]
bar_minutes: 30                                         # resample for formation
formation:
  sessions: 60
  trading_sessions: 20
  roll: true
selection:                                              # SS1
  method: pca_cluster                                   # or all_pairs | distance | correlation
  pca: {standardize: true, n_components: auto_mp}       # Marchenko–Pastur
  cluster: {algorithm: optics, min_samples: 3}
  coint: {test: engle_granger, significance: 0.05, both_directions: false}
  half_life_bars: {min: 5, max: 65}
  multiple_testing: bh                                  # none | bonferroni | bh
hedge: {method: rolling_ols}                            # SS2; fixed | kalman | tls
trading:                                                # SS4, platform-aligned names
  entryZScore: 2.0
  exitZScore: 0.5
  stopLossZScore: 4.0
  tradeNotionalUsd: 5000
costs: {half_spread_bps: 2.0, slippage_bps: 5.0, commission_per_share: 0.005}  # T5 = platform defaults; per leg
report: [pairs_selected, oos_survival, net_pnl, trials]
```

## Walkthrough

1. **Negative check on the plain fixture (10 min).** Run all-pairs EG on the
   current independent fixture with 24 symbols (276 pairs). *Expected:* about
   14 pairs pass at 5% (`276 × 0.05`), none survive out of sample, and BH leaves
   approximately none. Write down this false-pass count. It is the baseline for
   every result that follows.

2. **PCA (20 min).** Switch to the C1o fixture. Compute the formation-window
   correlation matrix and plot the eigenvalues (scree) with the
   Marchenko–Pastur `λ₊` line. *Expected:* 4 eigenvalues above the line, matching
   the planted market + 3 sectors. Plot the first two loadings for each stock:
   sector members sit together.

3. **Cluster (15 min).** Run OPTICS on the normalised loadings, excluding the
   first (market) component. *Expected:* 3 clusters matching the planted
   sectors. The idiosyncratic symbols are labelled noise. Report the adjusted
   Rand index against the planted labels: this is possible only because it is a fixture.

4. **Within-cluster cointegration (20 min).** Test pairs inside each cluster.
   Count the tests. *Expected:*
   - Both planted pairs pass, with fitted β and half-life near the planted values.
   - The correlated-but-not-cointegrated trap pair fails EG.
   - The test count is a small fraction of 276.

5. **Trade out of sample (25 min).** For every method in the baselines table,
   take the selected pairs into the next trading window and run SS4's z-score
   rule through the T1 simulator, with costs.
   - **Survival** = the share of selected pairs whose spread is still stationary
     in the trading window (ADF on trading-window data, as a diagnostic only)
     **and** whose half-life is within a factor of 2 of the formation estimate.
   - **Net P&L** per pair and in total, with block-bootstrap intervals over sessions.

   *Hypothesis to test:* PCA + clustering has higher survival than all-pairs
   EG at a similar or lower number of selected pairs. On the fixture this
   should hold by construction. On real data it is the open question R3 asks.

6. **Real data (20 min).** Repeat steps 2–5 on the snapshot dev symbols.
   - Expect much weaker structure: more components above `λ₊`, fuzzier clusters,
     few or no pairs passing BH.
   - Report what you find, including "nothing survived". A null result here is a
     valid outcome, and common for small universes.

7. **Export (10 min).** Write the surviving pairs to a pair-list YAML that SS4
   reads and T7 maps to the platform's pairs config.

```yaml
# configs/pairs/ex07_selected.yaml  (shape only; T7 owns the final schema)
formation_end: "2026-06-30"
method: pca_cluster
trials: 31                      # number of EG tests run, for DSR
pairs:
  - {y: SYM_A, x: SYM_B, beta: 1.18, half_life_bars: 13, coint_p: 0.004, cluster: 1}
platform:
  hedgeRatioMethod: rolling_ols
  cointSignificanceLevel: 0.05
  cointHysteresis: 0.15
  entryZScore: 2.0
  exitZScore: 0.5
  stopLossZScore: 4.0
  maxHoldingTimeMs: 86400000    # 24h; P9 proposes deriving this from half-life
```

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Run all-pairs EG on the independent fixture and count false passes. Then apply BH and Bonferroni. | Use `statsmodels.tsa.stattools.coint` on log prices. Expect about `m·0.05` passes, near zero after correction. |
| Core | PCA + OPTICS on the C1o fixture. Recover the 3 sectors and report the adjusted Rand index. | Use `sklearn.decomposition.PCA` on z-scored formation returns, then `sklearn.cluster.OPTICS(min_samples=3)` on the L2-normalised loadings of components 2..k. Drop the market component, or every stock clusters together. |
| Stretch | Add the Marchenko–Pastur cut-off and compare with "50% of variance". | `λ₊ = (1 + sqrt(n/T))²` for a correlation matrix. With intraday bars, `T` is large relative to `n`, so `λ₊` is close to 1. |
| Stretch | Implement the Gatev distance method and add it to the comparison table. | Normalise cumulative returns to 1 at formation start; SSD for every pair; top 5–20 pairs; open at 2 formation-window σ, close at the crossing. This is an all-pairs method too, so it carries the same trial count. |
| Challenge | Avellaneda–Lee variant: instead of pairs, regress each stock's returns on the top `k` PCA factors and trade the OU residual s-score. Compare with the pairs approach. | Rolling regression on formation data; the cumulative residual `X_t`; fit OU; `s = (X_t − m)/σ_eq`. Entry and exit thresholds as in the paper's illustrative rules. This links directly to SR3 (residual fade) in the reversals team. |
| Challenge | Rolling formation: repeat selection every 20 sessions over a year. Plot pair turnover between formations and survival per window. | High turnover means the "pairs" are noise. Report the median pair lifetime. |

Full solutions live on the `workshop-solutions` branch under `workshop/07_unsupervised_pairs/`.

## Discussion questions

1. Why does dropping the market component change the clusters so much?
2. Two stocks have return correlation 0.9 but their spread is not stationary.
   How can that be? Should you trade them?
3. Is a cluster-based pair "better" because it has an economic story, or only
   because fewer tests were run?
4. Engle-Granger, Johansen, or a half-life filter alone: which matters most for
   trading, and why?
5. What would you need to see before trusting a pair for six months instead of
   one trading window?

## Pitfalls

- **Clustering on the full sample.** The clusters then know the future. Fit
  scalers, PCA and clusters on formation data only.
- **Cointegration on returns.** EG runs on (log) price levels. Returns are
  stationary by construction.
- **Survivorship in the universe.** Using today's constituents to form pairs in
  the past drops delisted names and inflates stability. Use C1m when real data
  is used.
- **Not counting tests.** 276 EG tests plus 3 methods plus 2 directions is a
  large trial count. Record it (M2).
- **Ignoring costs and borrow.** A spread trade has two legs, so it pays four
  half-spreads per round trip. Short-leg borrow can be unavailable on small names.
- **Hedge-ratio look-ahead.** A static β fitted on formation data is fine. A β
  re-estimated on trading-window data, including the current bar, is not.
- **Calling a planted-fixture result evidence.** The fixture proves the code
  works; it says nothing about real markets.

## Connection to team strategies

- **Stat arb** ([teams/stat-arb.md](../../teams/stat-arb.md)): this is the
  pair-selection front end for SS1. Its output feeds SS2 (hedge), SS3 (OU and
  half-life) and SS4 (candidates). Half-life exits come up again in
  [Example 9](09-trade-management.md).
- **Reversals** ([teams/reversals.md](../../teams/reversals.md)): the PCA
  residual is a better "stretch" than raw price. SR3 subtracts β·SPY; the
  challenge exercise subtracts `k` factors.
- **Momentum** ([teams/momentum.md](../../teams/momentum.md)): residual momentum
  (momentum in factor-neutral returns) uses the same PCA machinery.

## Builder notes (W7)

| Item | Detail |
|---|---|
| Folder | `workshop/07_unsupervised_pairs/`: `README.md`, `ex07_pairs.py` (jupytext), `configs/ex07_independent.yaml`, `configs/ex07_fixture.yaml`, `configs/ex07_snapshot.yaml` |
| Library tasks required | C1o (factors and cointegrated pairs), SS1 (`all_pairs`, `pca_cluster`, `distance`, `correlation` selectors), SS2 (`fixed`, `rolling_ols`), SS3 (AR(1) half-life), SS4, F7, T1, T5, E2, M2 trial counts |
| Fixture effect needed | Market + 3 sector factors; 2 cointegrated OU pairs with known β and half-life; 1 correlated non-cointegrated trap pair; idiosyncratic noise names |
| Dependencies | `statsmodels` (EG/ADF) and `scikit-learn` (PCA, OPTICS) are base dependencies |
| CI check | On the C1o fixture with fixed seed: OPTICS adjusted Rand index ≥ 0.8; both planted pairs selected with half-life within 50% of the planted value; trap pair rejected; on the independent fixture, BH-corrected selections ≤ 1 |
| Participant scaffold | Steps 2–4 are `# TODO` cells with asserts against the planted values; step 5 is given |

## Further reading

- Gatev, E., Goetzmann, W. & Rouwenhorst, K. G. (2006). Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies*.
- Avellaneda, M. & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*.
- [studies/r3-pairs-clustering/README.md](../../../studies/r3-pairs-clustering/README.md)
- [design/platform-parity.md](../../design/platform-parity.md): how a pair list becomes a platform config.
