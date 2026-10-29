# R4: Order-flow imbalance and short-horizon prediction

**Owner:** Microstructure team (Team 3).

**Question:** does order-flow imbalance predict the midpoint move over 10 s,
60 s and 5 minutes, and does any of it survive crossing the spread?

**Pieces:** D3 quotes/trades and minute summaries, F6 order-flow features,
L1 forward returns (midpoint-based), E1 cost-adjusted evaluation.

**Notes:**
- This study runs offline in Python: the platform's historical engine feeds
  only bars to strategies, so quote-driven strategies cannot be backtested there.
- Start with a linear OFI regression (Cont, Kukanov & Stoikov 2014), then gradient boosting.
- Judge results on what you would earn after crossing the spread, not on
  prediction accuracy. Keep the feed (SIP vs. IEX) explicit: IEX imbalance is
  a single venue's view.
- Data volume is large (SPY: about 3M quotes plus trades per day). Start with
  a few liquid names and a few days.

**TODO(R4):**
- [ ] Recorded dataset: symbols, days, feed, from D3.
- [ ] Baseline OFI regression and signal decay by horizon.
- [ ] Cost-aware trading rule and results.
