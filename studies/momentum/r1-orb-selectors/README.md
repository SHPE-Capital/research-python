# R1: ORB trade selectors A–D

**Owner:** Momentum team (Christopher Wilches's proposal, presentation Nov 30 2026).

**Question:** does learned trade selection improve net performance of a
first-breakout opening-range strategy beyond simple rules, does it transfer to
held-out stocks, and how sensitive is it to costs?

| Version | Selection |
|---|---|
| A | Accept every eligible candidate |
| B | Accept when RVOL >= 1.2 and SPY is above its session open |
| C | Random forest predicting net trade return, threshold chosen on validation |
| D | Small neural network, same inputs and target |

**Pieces from the task board:**
- Universe: D4 with `configs/universe/workshop_v0.yaml` (6 cells, 8/2 dev/held-out split).
- Features: F1 relative volume, F8 market return, F4 opening range, F3 realized
  volatility, F2 VWAP distance, time of day (`minute_of_day`).
- Label: L2 exit policy (stop at the range midpoint, 30-minute time exit, next-open fills).
- Models: M1a random forest, M1c MLP, three seeds each (M2).
- Evaluation: E1, plus E2's paired block bootstrap against A and B.

**TODO(R1):**
- [ ] Candidate generator: first completed post-range bar with close > H, through 11:00.
- [ ] Pre-register dates, thresholds and selection constraints here before opening validation.
- [ ] Reconcile L2 labels with the platform backtest on 5 sessions.
- [ ] Results, including the matched random-retention comparison and context ablation.
