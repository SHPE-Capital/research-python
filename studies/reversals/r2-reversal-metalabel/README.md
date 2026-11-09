# R2: Reversal meta-labeling

**Owner:** Reversals team. Platform strategies: `minuteReversal`, `beginnerReversal`, `mcarrasco_reversal`.

**Question:** among large short-horizon moves that the reversal rule would
fade, can a model pick the ones that actually revert after costs?

**Pieces:** F2 VWAP distance, F3 realized volatility (move size in volatility
units), F1 relative volume, F5 regime features (reversals should pay in
mean-reverting conditions), F8 market return; L2 exit-policy labels; M1 models.

**Note:** the same "large move" event, labeled "continued" instead of
"reverted", is a momentum question. Share candidates and features with R1
where possible.

**TODO(R2):**
- [ ] Port the reversal candidate rule from the platform strategy into a candidate generator.
- [ ] Baseline (take all) vs. rule filter vs. learned selector, as in R1.
- [ ] Pre-registered setup and results.
