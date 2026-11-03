# R3: Pair selection by clustering, then spread entry meta-labeling

**Owner:** Statistical arbitrage team. Platform strategy: `pairs`.

**Questions:**
1. Does clustering stocks on residual returns (PCA, then OPTICS or DBSCAN),
   and testing cointegration only within clusters, find pairs that stay
   cointegrated out of sample more often than testing every pair?
2. Given a z-score entry, can a model predict which entries converge before the stop?

**Pieces:** D4 universe and daily bars, F7 pair spread features (z-score,
half-life, rolling cointegration stability), F3 realized volatility, F8 market
return, L2 exit-policy labels (two-leg version), V1 purged splits (pair trades
can run long).

**Caution:** the number of independent pairs × days is small. Keep models
shallow, and treat "pairs tested" as trials for E2's deflated Sharpe.

**TODO(R3):**
- [ ] Residual-return clustering on the dev universe, formation period only.
- [ ] Out-of-sample cointegration survival: clustered vs. all-pairs baseline.
- [ ] Entry meta-labeling once F7 and L2 land.
