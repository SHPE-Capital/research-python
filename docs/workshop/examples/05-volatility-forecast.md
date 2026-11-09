# Example 5: Volatility forecast (the positive control)

| | |
|---|---|
| **Teams** | All. Every team sizes positions and sets stops in volatility units. |
| **Level** | Int |
| **Duration** | 90 minutes |
| **Prerequisites** | [Session: honest evaluation](../sessions/), [Example 2](02-intraday-model-comparison.md) (model contract, LightGBM) |
| **Data** | Synthetic fixture with planted volatility clustering (C1o); then the frozen workshop snapshot (D5), dev symbols only |
| **Library tasks used** | F3 realized volatility, C1e daily frequency, C1c label registry, C1f walk-forward fit scheme, M1b LightGBM, E2 block bootstrap |
| **Build task** | W5 |

## Why this example

Most of the workshop shows that returns are hard to predict. This example shows
a quantity that **is** predictable: volatility. Large moves cluster. A quiet day
tends to follow a quiet day, and a wild day follows a wild day. A simple
linear model on past realised volatility explains a real share of tomorrow's
volatility. The best return models explain almost none of tomorrow's return.

This is the **positive control**. If the pipeline cannot find volatility
predictability that is known to exist, something in the pipeline is broken.
[Example 6](06-overfitting-casino.md) is the matching negative control: there
the pipeline must *not* find an edge in pure noise.

**Industry analogue:** every risk desk forecasts volatility. Risk models,
volatility targeting, option pricing, margin and position limits all depend on
it. In a quant fund the volatility forecast is often more valuable than any
single alpha signal, because sizing and risk limits use it on every trade.

## Learning objectives

By the end, participants can:

1. Compute daily realised variance from minute bars without look-ahead.
2. Fit and explain the HAR-RV model and say why it is a strong baseline.
3. Compare forecasts with the right loss: QLIKE vs. MSE, and why it matters.
4. Run a fair comparison of HAR-RV and gradient boosting under walk-forward refitting.
5. Use a volatility forecast for position sizing and stop distances, and
   measure whether it helped.

## Background

### Realised variance

For session `d` with intraday log returns `r_{d,1}, …, r_{d,M}` (minute or
5-minute, inside the regular session):

```
RV_d = Σ_j r_{d,j}²          realised variance
RVol_d = sqrt(RV_d)          realised volatility
```

The overnight gap (previous close to today's open) is **not** part of `RV_d`.
Keep it as a separate feature, `gap_d² = (ln open_d − ln close_{d−1})²`.
Minute returns pick up microstructure noise such as bid-ask bounce. Compare
1-minute and 5-minute sampling as an exercise.

### HAR-RV (Corsi 2009)

Volatility moves on several time scales. HAR models this with three averages of
past realised variance:

```
RV_d^(w) = (1/5)  Σ_{i=0..4}  RV_{d−i}       weekly
RV_d^(m) = (1/22) Σ_{i=0..21} RV_{d−i}       monthly

RV_{d+1} = b0 + b_d·RV_d + b_w·RV_d^(w) + b_m·RV_d^(m) + ε_{d+1}
```

Fit by OLS. Fitting in logs (`log RV`) is usually more stable, because RV is
right-skewed. When you fit in logs, convert back with the log-normal correction
`exp(μ + σ²/2)`. The model has four parameters and is hard to overfit. That
makes it the baseline any machine-learning model must beat.

### Baselines that must be in the table

| Forecast | Formula |
|---|---|
| Random walk | `RV̂_{d+1} = RV_d` |
| Long-run mean | Training-set mean of `RV` per symbol |
| EWMA (RiskMetrics) | `σ̂²_{d+1} = λ σ̂²_d + (1−λ) RV_d`, λ = 0.94 |
| HAR-RV | Above, per symbol or pooled with a per-symbol intercept |
| LightGBM | HAR lags plus extra features (below), pooled over symbols |

### Losses

Realised variance is a noisy stand-in for the true variance. Only some loss
functions rank forecasts correctly when the target is that noisy (Patton 2011).
MSE and QLIKE both do:

```
MSE   = mean( (RV − RV̂)² )
QLIKE = mean( RV/RV̂ − ln(RV/RV̂) − 1 )
```

- **MSE** is dominated by a few high-volatility days.
- **QLIKE** penalises under-forecasting more than over-forecasting. That suits
  risk use: under-forecasting volatility means taking on too much size.
- Report both, plus `R²` of `log RV`.

### Why volatility is predictable when returns are not

If returns were predictable, traders would trade on them until the edge
disappeared. Volatility has no such direct trade for equity traders. Its
persistence comes from news arriving in clusters and from how traders respond to
risk. Since nothing removes it, it stays.

## Setup

**Fixture:** the current `synthetic_bars` uses a constant `sigma` for each
symbol. Its daily RV is therefore flat plus noise, so HAR should beat the
long-run mean by almost nothing. C1o adds a `vol_clustering` option: a GARCH(1,1)-style daily variance with known
persistence. Use both versions:

| Fixture | Expected result |
|---|---|
| Constant volatility (current) | HAR ≈ long-run mean. Out-of-sample `R²` of log RV ≈ 0. A large `R²` here is a bug. |
| `vol_clustering` (C1o), persistence `α+β` | HAR clearly beats the mean and the random walk. The fitted persistence is close to the planted one. |

Daily models need history. Run the fixture from `2024-01-02` (about 2.5 years),
not the default six months. HAR needs 22 sessions of warm-up.

**Real data:** workshop snapshot (D5), `role: dev`. Hold out the last 20% of
sessions as test. Open the test set once.

Config sketch (syntax pending C1a/C1c/C1f/C1h):

```yaml
name: ex05_har_vs_gbm
seed: 7
data:
  source: fixture
  symbols: [AAA, BBB, CCC, DDD, EEE, FFF]
  start: "2024-01-02"
  end: "2026-08-28"
  vol_clustering: {persistence: 0.95}    # C1o
frequency: daily                          # C1e: one row per symbol-session
features:
  - {type: realized_var, sampling_minutes: 5, lags: [1]}       # RV_d
  - {type: realized_var_mean, sampling_minutes: 5, window: 5}  # RV^(w)
  - {type: realized_var_mean, sampling_minutes: 5, window: 22} # RV^(m)
  - {type: overnight_gap_sq}
  - {type: day_of_week}
label:
  type: forward_realized_var      # C1c: RV of session d+1, label_end_ts = close of d+1
  sampling_minutes: 5
  transform: log
fit_scheme:
  type: walk_forward              # C1f: refit every 20 sessions on an expanding window
  refit_every_sessions: 20
  min_train_sessions: 250
model:
  type: har                       # thin OLS wrapper; or lightgbm
report:
  losses: [mse, qlike, r2_log]
  baselines: [random_walk, long_run_mean, ewma_094]
```

## Walkthrough

1. **Build daily RV without look-ahead (15 min).** Compute `RV_d` from 5-minute
   returns inside session `d`.
   - `ts_available` of `RV_d` is the end of session `d`, i.e. its last bar's `ts_available`.
   - A forecast for `d+1` made at the close of `d` may use `RV_d`.
   - A forecast made at 10:00 on day `d+1` may also use the first 30 minutes of `d+1`.
   - Run `assert_no_lookahead` on the feature.

   *What to look for:* the ACF of `RV_d` and of daily returns `r_d`, side by
   side. Returns show no autocorrelation. RV shows slowly decaying, positive
   autocorrelation, but only on the `vol_clustering` fixture.

2. **Naive baselines (10 min).** Score random walk, long-run mean and EWMA with
   both losses. The random walk often loses to the mean under MSE, because RV is
   noisy. EWMA usually beats both.

3. **HAR-RV (15 min).** Fit per symbol, then pooled. Print the coefficients.
   *Expected on the fixture:*
   - All three slopes are positive.
   - `b_d + b_w + b_m` is close to the planted persistence.
   - The fit is better than EWMA.

4. **LightGBM (20 min).** Use the same rows and the same walk-forward scheme.
   - Features: the HAR lags, `gap²`, day of week, `|r_d|`, and log volume vs.
     its 20-day mean.
   - Use a shallow model (`num_leaves ≤ 15`, early stopping on a chronological
     validation slice inside each training window).

   *Hypothesis:* on the fixture, LightGBM ties HAR, because the planted process
   is close to linear in the lags. On real data it may gain a little from gaps
   and volume. Test the difference with a block bootstrap on the **daily
   difference in QLIKE** (E2), not by comparing two numbers.

5. **Use the forecast (20 min).** Two uses that every team needs:
   - **Volatility-targeted sizing:** `qty = target_risk_usd / (σ̂_{d+1} · price)`.
     Run a buy-and-hold for each symbol with this sizing, then with fixed
     notional. Measure realised portfolio volatility ÷ target. A good forecast
     keeps the ratio close to 1 and stable over time.
   - **Stops in volatility units:** a stop `k·σ̂_h` below entry, where `σ̂_h` is
     the forecast scaled to the trade horizon `h` (√time scaling within the
     day, adjusted for the U-shaped intraday profile). Compare the stop-hit rate
     for a fixed-% stop and a volatility-scaled stop across symbols. The
     volatility-scaled stop should hit at a similar rate across names. The
     fixed-% stop hits far more often on volatile names.

6. **Report (10 min).** A loss table for every model and baseline, with
   bootstrap intervals, a plot of forecast vs. realised for one symbol, and
   the realised/target ratio chart.

## Exercises

| Tier | Exercise | Solution notes |
|---|---|---|
| Core | Implement daily `RV_d` from 5-minute returns and reproduce the HAR coefficients on the `vol_clustering` fixture. | Group by symbol and session. Use log returns of 5-minute closes inside the session. Rolling means use `min_periods` equal to the window and shift by one session so `RV^(m)` at `d` ends at `d`. OLS via `numpy.linalg.lstsq`. The slopes sum to roughly the planted persistence. |
| Core | Add the random walk, mean and EWMA baselines. Rank all four by MSE and QLIKE. | QLIKE needs `RV̂ > 0`. Floor forecasts at a small positive value and report how often the floor fired. Rankings can differ between the two losses; that is the point. |
| Stretch | Intraday version: forecast the realised volatility of the next 30 minutes at each decision time. Include time-of-day seasonality. | Divide each minute's squared return by its average for that minute of day (computed on training data only). Forecast the deseasonalised value, then multiply the seasonal factor back in. Without this, the model mostly learns the U-shape, which is trivial. |
| Stretch | Compare 1-minute and 5-minute RV on real data. | 1-minute RV is biased upward by bid-ask bounce on lower-priced names. The forecasts rank similarly, but 1-minute levels run higher. |
| Challenge | Volatility-targeted portfolio: show that HAR sizing gives a more stable realised/target ratio than EWMA, with a block-bootstrap interval. | The metric is the standard deviation of the rolling 20-day realised/target ratio. Bootstrap the difference between the two sizing schemes on the same dates. |
| Challenge | Add `overnight_gap_sq` and a market-wide RV (SPY's, via the F8 context-symbol extension) to LightGBM. Does market RV help single-name forecasts? | Market RV usually carries information because volatility rises across names at the same time. Make sure SPY's RV for session `d` has `ts_available` no later than the decision time. |

Full solutions live on the `workshop-solutions` branch under `workshop/05_volatility_forecast/`.

## Discussion questions

1. Why is HAR, with four parameters, so hard to beat? What would a model need
   to learn that HAR cannot?
2. Under-forecasting volatility and over-forecasting it lead to different
   mistakes. Which costs a trading team more, and which loss captures that?
3. If volatility is predictable, is there an "alpha" in it for an equity trader
   with no options? (Hint: sizing, stops and regime filters, not direction.)
4. Earnings days are huge volatility outliers. Should they be in the training
   set, in the test set, or modelled separately?

## Pitfalls

- **Including today in today's forecast.** Computing `RV_d` and then using it to
  "forecast" session `d`. Shift by one session and check with `assert_no_lookahead`.
- **Overlapping weekly/monthly windows that reach past the decision.** A
  centred rolling mean is look-ahead.
- **Scaling features on the whole sample.** Fit scalers on the training window
  inside the model (CONTRIBUTING rule "Scale inside the model").
- **Comparing models on different rows.** The warm-up period differs between
  models. Score every model on the same set of `(symbol, session)` rows.
- **Judging by MSE alone.** One crash day can decide the ranking.
- **Calling a forecast "good" from `R²` alone.** The question is whether
  sizing and stops improve (step 5).

## Connection to team strategies

- **Momentum** ([teams/momentum.md](../../teams/momentum.md)): ORB stops in
  `k·ATR` or `k·σ̂` units; risk-per-trade sizing where the stop distance comes
  from the forecast.
- **Reversals** ([teams/reversals.md](../../teams/reversals.md)): the entry
  threshold "stretch > k·σ" needs a good σ. A noisy σ turns into a noisy entry
  rule. The disaster stop is also in σ units.
- **Stat arb** ([teams/stat-arb.md](../../teams/stat-arb.md)): spread-volatility
  sizing (`1σ of spread = fixed $ risk`) needs a spread volatility forecast.
  HAR on spread RV works the same way.
- Feeds [Example 9](09-trade-management.md): the sizing ladder's "inverse vol"
  step uses this forecast.

## Builder notes (W5)

| Item | Detail |
|---|---|
| Folder | `workshop/05_volatility_forecast/`: `README.md`, `ex05_har_vs_gbm.py` (jupytext percent format, paired notebook), `configs/ex05_fixture.yaml`, `configs/ex05_snapshot.yaml` |
| Library tasks required | F3 (and its stretch: daily HAR feature), C1e daily frequency, C1c label registry with `forward_realized_var`, C1f walk-forward, M1b LightGBM, E2 `block_bootstrap` |
| Fixture effect needed | C1o: `vol_clustering` option with known persistence; constant-volatility default stays as the negative check |
| Small library additions | A `har` model registered via `@register_model` (OLS on the three lags, optional log transform); `qlike` and `r2_log` losses in `evaluation/` |
| CI check | On the fixture (about 2 years, 6 symbols), pass if: (a) HAR out-of-sample QLIKE < long-run-mean QLIKE on `vol_clustering`; (b) on constant volatility, HAR `R²` of log RV < 0.05; (c) the notebook runs headless in under 3 minutes |
| Participant scaffold | Steps 1, 3 and 5 have `# TODO` cells; steps 2 and 4 are given |

## Further reading

- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*.
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
- [research/trade-management.md](../../research/trade-management.md): volatility units for stops and sizing.
