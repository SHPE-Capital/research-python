# Strategy spec card template

The spec card is the single page that says what a strategy is. It is written
at lifecycle stage 3 (see [strategy-lifecycle.md](strategy-lifecycle.md)) and
updated at each gate. The frozen card matches the frozen `StrategySpec`, and the
platform config is exported from that spec with `export_platform_config`.

T8 adds `shpe new strategy`, which generates this card next to a new strategy.
Until then, copy the block below into `studies/<team>/<id>-<slug>/SPEC.md`.

Fields marked "to be studied" must name the study that will set them. Write
the kill criteria before any results exist.

---

~~~markdown
# <Strategy name> — spec card

| | |
|---|---|
| ID / version | <e.g. SR2-stretch v1.0> |
| Team | Momentum / Reversals / Stat arb |
| Owner(s) | <names> |
| Status | idea / spec / study / validated / tested / parity / paper / live / retired |
| Lifecycle stage | <1–11> |
| Spec commit | <hash of the frozen StrategySpec> |
| Studies | <RM/RR/RS IDs with links> |
| Last updated | <YYYY-MM-DD> |

## Rationale
<Who is on the other side of the trade and why they accept a worse price.
Prior work, including failures.>

## Hypothesis
- H1: <signed, testable claim with a horizon>
- H0: <the null>

## Universe
| Field | Value |
|---|---|
| Universe config | <path> |
| D4 seed / cells | <seed; liquidity × volatility cells used> |
| Dev / held-out symbols | <lists or file paths> |
| Survivorship note | <static universe as of …; C1m status> |
| Frequency | 1-min / daily |
| Session window | <e.g. 09:35–15:45 ET, regular hours only> |

## Entry
| Field | Value |
|---|---|
| CandidateGenerator | <registered name via @register_strategy, file path> |
| Signal | <definition, in units> |
| Entry threshold | <value or "to be studied (RR2)"> |
| Order type | market at next bar open / limit / stop |
| Side | long / short / both |
| Platform baseline it must beat | <e.g. minuteReversal, take-every-breakout> |

## Filters and meta-label
| Filter | Rule | Study |
|---|---|---|
| Time of day | | |
| Liquidity / spread | | |
| Regime | | |
| Events | | |
| Meta-label | <model, threshold, label> | |
| Cooldown | <duration; matches orderCooldownMs or strategy cooldownMs> | |

## Exits (ExitPolicy)
| Field | Value | Units | Study |
|---|---|---|---|
| Time exit | | bars / OU half-lives | |
| Stop | | ATR / σ√h / structural / none | |
| Target | | R / none | |
| Trailing | | k × ATR / none | |
| Signal exit | | | |
| Session end | flat at close? | | |

## Sizing (SizingRule)
| Field | Value |
|---|---|
| Rule | fixed / risk-per-trade / inverse vol / prob→size / fractional Kelly |
| Parameters | <risk %, target vol, Kelly fraction> |
| Bunching control | <max concurrent / 1/√n / none> |
| Max per position | <USD> |

## Risk limits (RiskLimits, platform names)
| Field | Value |
|---|---|
| maxPositionSizeUsd | |
| maxNotionalExposureUsd | |
| maxConcentrationPct | |
| maxIntradayDrawdownPct | |
| cashReservePct | |
| allowShortSelling | |
| orderCooldownMs | |
| maxGrossExposurePct / maxNetExposurePct | |
| Strategy budget / maxOrderNotionalPct | |

## Costs (CostModel / FillModel)
| Field | Value |
|---|---|
| Fill model | platform default: next_open, halfSpreadBps 2, slippageBps 5, commissionPerShare 0.005, volumeParticipationCap 0.1 |
| Calibration | <R5 / T5 calibration, or "platform default"> |
| Round trip at median price | <bps> |
| Stress case | 2× costs |

## Evidence
| Split | Trades | Net bps/trade (95% CI) | Sharpe | DSR | Max DD | Hit rate |
|---|---|---|---|---|---|---|
| Validation (dev) | | | | | | |
| Test (dev) | | | | | | |
| Test (held-out) | | | | | | |
| Paper | | | | | | |

Trials counted: <n>. PBO: <value or n/a>. Leakage-auditor review: <date, verdict>.

## Kill criteria
Written before results. Any one triggers retirement or a return to study.
- [ ] Net bps per trade below <x> over the last <n> trades
- [ ] Drawdown beyond <y>% of allocated capital
- [ ] Realised costs above the CostModel by more than <z>%
- [ ] Hit rate or trade frequency outside the test-period range for <k> weeks
- [ ] <Strategy-specific, e.g. the pair's cointegration p-value above the hysteresis band for 5 days>

## Platform config diff
Output of `export_platform_config`, as a diff against the platform's current
default config for this strategy.

| Field | Platform default | This spec | Status |
|---|---|---|---|
| <e.g. entryZScore> | 2 | 2.25 | existing |
| <e.g. maxHoldingTimeMs> | 86400000 | 5400000 | unit-converted (3 × half-life) |
| <e.g. hedgeRatioMethod> | rolling_ols | kalman | proposed new field (P9) |

```diff
- "entryZScore": 2,
+ "entryZScore": 2.25,
```

Parity: <ParityReport path, date, verdict>.

## Change log
| Date | Version | Change | Reason | Signed off by |
|---|---|---|---|---|
~~~

---

## Notes on filling it in

- Every parameter carries units. "Stop: 2" is not a spec; "stop: 2 × ATR(14) on 1-min bars" is.
- Parameters that the platform cannot express are listed in the config diff
  with status "proposed new field" and a P-task (P6–P9), or "not representable".
  A strategy with "not representable" fields cannot pass the parity gate.
- The evidence table is filled from study reports; link them rather than copy figures.
- Kill criteria are checked at each gate from stage 6 onward and daily when live.
