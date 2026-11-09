# Studies

Team-owned research questions. A study composes library pieces (features,
labels, models, evaluation) and records a decision. Library code belongs in
`src/shpe_research/`, so other teams can reuse it. A study folder holds:

- `README.md`: the question, the pre-registered setup (dates, universe,
  thresholds, frozen *before* looking at validation results), and the result,
  including negative results.
- `configs/`: the experiment configs that produced the result.
- Notebooks or scripts for figures. Clear outputs before committing, and never
  commit data.

| ID | Study | Team |
|---|---|---|
| [R1](momentum/r1-orb-selectors/) | ORB trade selectors A–D | Momentum |
| [R2](reversals/r2-reversal-metalabel/) | Reversal meta-labeling | Reversals |
| [R3](stat-arb/r3-pairs-clustering/) | Pair selection by clustering | Stat arb |
| [R4](reversals/r4-microstructure-ofi/) | Order-flow imbalance | Microstructure |
| [R5](shared/r5-tca/) | Transaction cost analysis | Any |
