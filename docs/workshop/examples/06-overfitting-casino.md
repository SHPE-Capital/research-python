# Example 6: The overfitting casino (the negative control)

| | |
|---|---|
| **Teams** | All |
| **Level** | Int (the deflated Sharpe ratio and PBO derivations are Adv) |
| **Duration** | 90 minutes |
| **Prerequisites** | [Example 1](01-leakage-lab.md) (leakage and snooping), the foundation session on honest evaluation |
| **Data** | Synthetic fixture with `signal_strength: 0.0` (pure random walk); then the default fixture (`signal_strength: 0.3`) as a power check |
| **Library tasks used** | T1 exits and simulator, T6 parameter surfaces, E2 `deflated_sharpe` and `probability_of_backtest_overfitting`, M2 experiment registry (trial counts), E1 cost-adjusted P&L |
| **Build task** | W6 |

## Why this example

We give participants price data with **no signal at all**, and a strategy with
a handful of parameters. They search the parameters for the best backtest. In
sample, the best configuration looks good: a Sharpe above 2 is easy to find.
Out of sample it earns nothing. Then the deflated Sharpe ratio and the
probability of backtest overfitting (PBO) flag the result without seeing the
test set.

The main point for the teams: **a grid search over stops, targets and holding
periods is exactly this casino.** Every team will tune exits in
[Example 9](09-trade-management.md) and in its own studies. This example makes
them count trials and distrust the best grid cell.

**Industry analogue:** quant firms keep a record of every backtest run and set
higher bars for strategies that came out of large searches. Harvey, Liu & Zhu
(2016) argued that, given how many factors have been tested, a new factor needs
a t-statistic of about 3, not 2.

## Learning objectives

1. Show that the best of N configurations on noise has a large Sharpe ratio,
   and predict its size before running.
2. Count trials honestly: what counts as a trial and where it is recorded (M2).
3. Compute and interpret the deflated Sharpe ratio (DSR).
4. Compute and interpret PBO with combinatorially symmetric cross-validation (CSCV).
5. Apply the rules for honest tuning to exit-parameter grids: choose on validation,
   report the whole surface, prefer a plateau, test once.

## Background

### The best of N is not zero

Suppose you test N strategies, each with true Sharpe 0. Each estimated Sharpe
is noise with standard deviation about `1/sqrt(T)`, where `T` is the number of
observations. Annualise that standard deviation and call it `σ_SR`. The
expected best of N independent estimates is approximately:

```
E[max SR] ≈ σ_SR · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]      γ ≈ 0.5772 (Euler–Mascheroni)
```

This is the expected-maximum formula in Bailey & López de Prado (2014).

With six months of minute data, a daily-rebalanced strategy has `T ≈ 125` days. So
`σ_SR ≈ sqrt(252/125) ≈ 1.4` annualised. The best of 1,000 is then expected near
`1.4 × 3.2 ≈ 4.5`. Participants should write down their guess before running.

### Deflated Sharpe ratio (Bailey & López de Prado 2014)

DSR is the probability that the observed Sharpe beats the expected maximum
from N unskilled trials. It also accounts for the sample's skew `γ₃` and kurtosis `γ₄`:

```
SR₀ = E[max SR] under the null (above, from the variance of trial Sharpes)
DSR = Φ( (SR̂ − SR₀) · sqrt(T − 1) / sqrt(1 − γ₃·SR̂ + ((γ₄ − 1)/4)·SR̂²) )
```

Here `SR̂` is the per-period (non-annualised) Sharpe ratio. Signature in this
repo: `evaluation/stats.py::deflated_sharpe(sharpe, n_trials, n_obs, skew, kurtosis)` (E2).

**Read it as:** DSR > 0.95 means the result probably beats luck given how
hard you searched. The E2 acceptance test already demands that the best of 100
noise strategies is not significant.

### PBO via CSCV (Bailey et al. 2017)

1. Build a matrix `M` with T time blocks as rows and N configurations as
   columns. Each cell is that configuration's performance in that block.
2. Split the T blocks into S groups (e.g. S = 16).
3. For every way of choosing S/2 groups as in-sample (IS), the rest are
   out-of-sample (OOS):
   - Find the best IS configuration `n*`.
   - Find its relative rank `ω` among all configurations OOS.
   - Compute `λ = ln(ω / (1 − ω))`.
4. `PBO = fraction of splits with λ ≤ 0`, i.e. the IS winner ranks in the
   bottom half OOS.

On pure noise PBO is about 0.5: the IS winner is a coin flip OOS. A real effect
pushes PBO towards 0.

### What counts as a trial

Every configuration you **looked at** counts, including the ones you threw
away. The list:
- grid points
- feature sets
- filters added and removed
- different labels
- different universes
- "just one more tweak" after looking at validation results

The M2 experiment registry records every `shpe run` automatically. Trials done
by hand in a notebook must be logged by hand (CONTRIBUTING: "Record every
attempt").

When configurations are highly correlated, for example neighbouring stop
distances, the effective number of independent trials is smaller than N.
Counting raw N is conservative, which is fine for a workshop. The challenge
exercise estimates the effective N by clustering trial return series.

## Setup

**Strategy family** (a deliberately flexible one; the flexibility is the trap):

- Entry: a breakout of the `L`-minute high/low after the open, or a moving-average cross with fast/slow windows.
- Exit: the T1 `ExitPolicy` with a stop, a target and a time exit.
- Costs: T5 defaults, which reproduce the platform's `DEFAULT_FILL_MODEL`: half spread 2 bps and slippage 5 bps per side, plus $0.005 per share commission.

| Parameter | Grid | Count |
|---|---|---|
| Range length `L` | 5, 10, 15, 30, 60 | 5 |
| Side | long only, short only, both | 3 |
| Stop (σ units) | 0.5, 1, 1.5, 2, ∞ | 5 |
| Target (σ units) | 1, 2, 3, none | 4 |
| Time exit (minutes) | 15, 30, 60, 120, close | 5 |
| **Total** | | **1,500** |

Config sketch (sweep syntax pending M2/C1h):

```yaml
name: ex06_casino_noise
data:
  source: fixture
  symbols: [N01, N02, N03, N04, N05, N06, N07, N08, N09, N10]
  start: "2026-03-02"
  end: "2026-08-28"
  signal_strength: 0.0           # pure random walk: no edge exists
events:
  type: range_breakout           # C1b candidate generator
  range_minutes: {grid: [5, 10, 15, 30, 60]}
  side: {grid: [long, short, both]}
exit_policy:                     # T1
  stop_sigma: {grid: [0.5, 1.0, 1.5, 2.0, null]}
  target_sigma: {grid: [1.0, 2.0, 3.0, null]}
  max_minutes: {grid: [15, 30, 60, 120, session_end]}
costs: {half_spread_bps: 2.0, slippage_bps: 5.0, commission_per_share: 0.005}   # T5 = platform DEFAULT_FILL_MODEL
split: {train: 0.6, validation: 0.2, test: 0.2, gap_days: 1}
report:
  per_config: [sharpe, n_trades, mean_net_bps]
  stats: [deflated_sharpe, pbo]  # E2
  trials_from: registry          # M2
```

Fixture note: `synthetic_bars(signal_strength=0.0)` is a pure random walk. The
`a` filter in `fixtures.py` reduces to `[1, 0, 0, 0, 0, 0]`, so returns are i.i.d.
normal plus session gaps. **No configuration has an edge**, so the true answer
for every cell is "loses about the cost".

## Walkthrough

1. **Predict (5 min).** Everyone writes down: "the best in-sample annualised
   Sharpe out of 1,500 will be about ___." Then compute `E[max SR]` with the
   formula and compare.

2. **Run the casino (15 min).** Sweep all 1,500 configurations on the training
   period. Plot:
   - a histogram of the training Sharpes (centred slightly below 0, because of costs)
   - the best-so-far curve: max Sharpe vs. the number of configurations tried, in
     random order, with the `E[max SR]` curve overlaid

   *Expected:* the curves track each other. The best cell probably has an
   annualised Sharpe of 3 or more, and a believable story ("short side, 15-minute
   range, 2σ stop, no target").

3. **Pick the winner on validation (10 min).** Take the top 10 training
   configurations and score them on validation. *Expected:* their validation
   Sharpes scatter around 0 minus costs. Rank on train and rank on validation are
   uncorrelated (Spearman ≈ 0).

4. **DSR (15 min).** Compute DSR for the best training configuration with
   `n_trials = 1500`. *Expected:* well below 0.95. Then recompute it with
   `n_trials = 1`, as if the winner were the only thing tried. It now looks
   significant. That gap is the cost of not counting trials.

5. **PBO (15 min).** Build `M` with 16 blocks of daily P&L and run CSCV.
   *Expected:* PBO ≈ 0.5 on noise.

6. **Power check on a real effect (15 min).** Switch to the default fixture
   (`signal_strength: 0.3`, a planted short-horizon reversal). Add a
   "fade the last 5 minutes" entry family to the grid and run again.
   - *Expected:* fade configurations with short holds rise to the top, and PBO
     drops well below 0.5.
   - Whether DSR passes depends on costs. The planted reversal is a few basis
     points at a 1–5 minute horizon, and about 14 bps of round-trip cost (2 × (2 + 5)) will probably consume it at default costs.
     This is a preview of the reversals team's cost-hurdle problem.

7. **The exit-grid confession (15 min).** Reopen the grid and show that its
   stop/target/time sub-grid alone (100 cells) produces a "best stop" on noise.
   Write down the honest-tuning rules (below) as the team policy.

### Honest-tuning rules (copy into team guides)

1. Choose parameters on **validation**. Open **test once**, and test on the
   held-out symbols from D4 as well.
2. Every grid adds its size to the experiment's trial count (M2). That count
   goes into DSR (E2).
3. Report the **whole parameter surface** (T6 heatmap), not the best point.
   Reject a winner that sits on an isolated peak. Prefer a setting in the
   middle of a plateau.
4. Tune one decision at a time, in a fixed order: cost hurdle, holding period,
   stop, target, filters, sizing, risk limits
   ([research/trade-management.md](../../research/trade-management.md)). Fix the
   others while tuning one. Count every pass.
5. Check that the economics are plausible. A parameter that only works at one
   value has no story, and is probably noise.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Reproduce the best-so-far curve and overlay `E[max SR]`. | Shuffle the configuration order, take the cumulative max of training Sharpes, and repeat over 50 shuffles for a band. Use the cross-sectional standard deviation of trial Sharpes as `σ_SR`. |
| Core | Compute DSR for the winner with honest and dishonest trial counts. | Use E2's `deflated_sharpe` with per-period Sharpe, `n_obs` = number of daily observations, and the sample skew and kurtosis of the winner's daily returns. |
| Stretch | Implement CSCV PBO, or verify E2's implementation, on the 16-block matrix. | `itertools.combinations(range(16), 8)` gives 12,870 splits. Rank OOS with `scipy.stats.rankdata` and normalise to (0, 1) by dividing by N + 1. |
| Stretch | Shrink the grid to the "economically motivated" 12 cells and rerun. Does the winner change? Does DSR change? | A smaller grid lowers `SR₀`, but on noise there is still no edge. DSR rises but should stay below 0.95. The lesson: a prior narrows the search, but it does not create an edge. |
| Challenge | Estimate the effective number of independent trials: cluster the 1,500 daily return series by correlation, and use the number of clusters as N. | Hierarchical clustering on the distance `1 − ρ`, cut at a threshold, count clusters. Neighbouring stops are highly correlated, so effective N is much smaller than 1,500. Show how DSR moves. |
| Challenge | Power curve: vary `signal_strength` over 0, 0.1, 0.2, 0.3, 0.5 and plot the probability that DSR passes, at cost 0 and at the platform cost. | Fix the grid and run 20 fixture seeds per strength. The curve shows the smallest effect the protocol can detect. Reporting this is honest science, not pessimism. |

Full solutions live on the `workshop-solutions` branch under `workshop/06_overfitting_casino/`.

## Discussion questions

1. Your teammate shows a backtest with Sharpe 3. What three questions do you ask first?
2. Two people each try 50 configurations and pick their best. Is the trial
   count 50 or 100? What if they talked to each other?
3. When is it acceptable to look at the test set twice?
4. Why does a smooth parameter surface make a result more credible?
5. The platform's pairs defaults (entry z 2, exit 0.5, stop 4) came from
   somewhere. How many trials do they hide?

## Pitfalls

- **Forgetting the trials done in your head.** Ideas rejected after a quick
  look at a chart are trials.
- **Treating validation as a second training set.** Iterating on validation
  until it looks good overfits validation. Count those iterations.
- **Too few trades.** A configuration with 15 trades has a huge Sharpe standard
  error. Show `n_trades` next to every Sharpe.
- **Annualising wrongly.** Per-trade and per-day Sharpes need different
  scaling. Use daily P&L for every comparison.
- **Leaving costs out of the casino.** Without costs the noise distribution
  centres on 0, not below it.

## Connection to team strategies

- **Momentum:** R1's selectors A–D and the ORB stop/target/time grid. Count
  every selector and every exit configuration.
- **Reversals:** stretch threshold × anchor × hold × disaster stop is a large
  grid on a small edge, which is where overfitting is most likely.
- **Stat arb:** entry z × exit z × window × hedge method, multiplied by every
  pair tested. Pair selection is itself a trial count (see [Example 7](07-unsupervised-pairs.md)).
- Team guides: [teams/momentum.md](../../teams/momentum.md), [teams/reversals.md](../../teams/reversals.md), [teams/stat-arb.md](../../teams/stat-arb.md); protocol in [research/research-protocol.md](../../research/research-protocol.md).

## Builder notes (W6)

| Item | Detail |
|---|---|
| Folder | `workshop/06_overfitting_casino/`: `README.md`, `ex06_casino.py` (jupytext), `configs/ex06_noise.yaml`, `configs/ex06_reversal_fixture.yaml` |
| Library tasks required | T1 exits and simulator, C1b `range_breakout` event generator, E2 (`deflated_sharpe`, `probability_of_backtest_overfitting`), M2 sweep plus registry trial counts, T6 parameter surface plot, T5 costs |
| Fixture effect needed | None new: `signal_strength: 0.0` already gives pure noise. Symbols named `N01…N10` so nobody mistakes them for planted-effect symbols. |
| Runtime budget | 1,500 configs × 10 symbols × 6 months must run in under 5 minutes on a laptop. The simulator vectorises across configurations that share an entry set; precompute candidates once per `(L, side)`. |
| CI check | On a 2-month, 4-symbol, 60-config reduced grid with fixed seed: best training Sharpe > 1 (the trap shows up), DSR of the winner < 0.95, PBO in [0.3, 0.7] |
| Participant scaffold | Steps 1, 2, 4 and 7 are guided; step 5 is a `# TODO` |

## Further reading

- Bailey, D. & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The probability of backtest overfitting. *Journal of Computational Finance*.
- Harvey, C., Liu, Y. & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*.
- [research/research-protocol.md](../../research/research-protocol.md): pre-registration and trial counting.
