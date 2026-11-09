# Teams

Three strategy teams share one research library. This page explains how the
teams, the shared trade toolkit, the strategy families and the studies fit
together. IDs follow [docs/README.md](../README.md).

| Team | Guide |
|---|---|
| Momentum | [momentum.md](momentum.md) |
| Reversals | [reversals.md](reversals.md) |
| Statistical arbitrage | [stat-arb.md](stat-arb.md) |

## How the pieces fit

```
             shared, owned by nobody                    owned by one team
  ┌──────────────────────────────────────────┐   ┌──────────────────────────────┐
  │ data/ features/ labels/ validation/      │   │ strategies/<family>/         │
  │ models/ evaluation/ (D, F, L, V, M, E)   │   │   SM1–3, SR1–3, SS1–4        │
  │                                          │   │   candidate generators       │
  │ trade/ analysis/ deploy/ (T1–T8)         │──▶│                              │
  │   exits, simulate, sizing, filters,      │   │ studies/<team>/<id>-<slug>/  │
  │   risk, costs, diagnostics, export,      │   │   R1–R5, RM<n>, RR<n>, RS<n> │
  │   parity, spec card                      │   │   pre-registered READMEs     │
  └──────────────────────────────────────────┘   └──────────────┬───────────────┘
                                                                │ StrategySpec
                                                                ▼
                                       T7 export_platform_config → platform strategy
                                       T7 ParityReport           (P7, P8, P9)
```

- **Shared library** (`src/shpe_research/`): data, features, labels, validation,
  models and evaluation. Anyone can claim a task from [tasks.md](../tasks.md).
- **Shared trade toolkit** (T1–T8): the code that turns a decision point into a
  trade, a P&L and a platform config:
  - `CandidateGenerator` / `CandidateFrame`
  - `ExitPolicy`, `FillModel`, `simulate_trades` → `TradeLedger`
  - `SizingRule`, `CostModel`, `RiskLimits`
  - `StrategySpec`, `export_platform_config`, `ParityReport`

  All three teams depend on it, so no team owns it. See
  [design/trade-layer.md](../design/trade-layer.md).
- **Strategy families** (`strategies/<family>/`): team-owned code. A family
  emits candidates and may add family-specific helpers, such as OU fits for
  pairs. It does not re-implement exits, fills, sizing or costs. It uses
  `trade/`.
- **Studies** (`studies/<team>/<id>-<slug>/`): one pre-registered question per
  folder. Each folder has a README from
  [research/study-template.md](../research/study-template.md). Studies use the
  library. They do not add library code; anything reusable moves into `src/`
  through a task.
- **Platform strategies** (trading-platform repo): the deployable versions.
  Each team's final output is a spec card plus a config that passes parity.
  See [design/platform-parity.md](../design/platform-parity.md).

## Team map

| Team | Strategy family | Platform strategy | Studies | Workshop examples led |
|---|---|---|---|---|
| Momentum | SM1 ORB, SM2 EMA crossover, SM3 12-1 cross-sectional (daily, research only) | New ORB strategy (P8). The existing `momentum/index.ts` is a placeholder | R1, RM1–RM8 in `studies/momentum/` | 3, 4 |
| Reversals | SR1 bar fade, SR2 stretch fade, SR3 residual fade | `minuteReversal` and the beginner reversal, combined into one family by P7 | R2, R4, RR<n> in `studies/reversals/` | 2, 4 |
| Stat arb | SS1 selection, SS2 hedge ratios, SS3 OU / half-life, SS4 pairs generator | `PairsStrategy`, extended by P9 | R3, RS<n> in `studies/stat-arb/` | 7 |
| All teams | Shared toolkit T1–T8 | P1 candidate log, P2 ONNX wrapper, P6 sizer wiring | R5 TCA in `studies/shared/` | 1, 5, 6, 8, 9 |

Notes:

- R4 (order flow and short-horizon midpoint prediction) sits under reversals
  because its horizon and features are closest to that team.
  [tasks.md](../tasks.md) still lists it as "Microstructure"; follow
  [docs/README.md](../README.md).
- R5 (transaction cost analysis) belongs to no team. It lives in
  `studies/shared/` and calibrates T5 for everyone.
- Example 4 (meta-labeling ORB) is led by momentum, with reversals
  contributing the fade side of the same method.

## Working rules across teams

1. **Shared code needs a reviewer from another team.** Any change to `trade/`,
   `analysis/`, `deploy/`, `labels/`, `validation/` or `evaluation/` needs one
   approving review from a member of a *different* team. A change that helps
   one strategy can silently bias another. Examples: a fill rule, a cost
   default, how the same-bar stop and target are resolved.
2. **No private copies.** If a team needs a toolkit feature that does not exist
   yet, it extends the T task or opens a new one. It does not write
   `strategies/<family>/my_simulator.py`.
3. **Platform names win.** Field names in `RiskLimits`, `SizingRule` and the
   exported config use the platform's names (`maxPositionSizeUsd`,
   `maxNetExposurePct`, `maxConcentrationPct`, and so on). If research needs a
   new field, the platform task (P6–P9) adds it with the same name.
4. **Studies are pre-registered.** Grid, split, metric, decision rule and trial
   budget are written in the study README before validation results are
   opened. Every evaluated configuration counts toward E2's deflated Sharpe.
   See [research/research-protocol.md](../research/research-protocol.md).
5. **Test data is opened once per study.** Held-out universe symbols (D4) and
   the test period are reported after the choice is frozen, not used to make it.
6. **Context symbols are shared.** SPY and other context series come through
   C1d / F8 for every team. Do not load them ad hoc inside a strategy.

## The decision chain every team follows

Each team guide applies the same chain to its own strategy. The general method
for each step is in [research/trade-management.md](../research/trade-management.md).
The full path from idea to retirement is in
[research/strategy-lifecycle.md](../research/strategy-lifecycle.md).

| Step | Question | Main tools | Output |
|---|---|---|---|
| 1. Baseline and controls | Does the raw rule beat a simple baseline and a random control, net of costs? | Family generator, T1, T5, E1, E2 | Go / no-go |
| 2. Entry | Which candidate definition and confirmation rule? | Family generator, T6 surfaces | Candidate spec |
| 3. Holding period | How long does the edge last? | T6 horizon curve; OU half-life for stat arb | `max_hold` |
| 4. Stops and targets | Where is the setup wrong, and is profit given back? | T6 MFE/MAE, stop curves; T1 | `ExitPolicy` |
| 5. Filters and regime | When should the rule not trade? | T3, F1–F8 | Filter set |
| 6. Sizing | How much per trade? | T2 | `SizingRule` |
| 7. Portfolio and risk | How much at once, and how correlated? | T4 | `RiskLimits` |
| 8. Selection (ML) | Can a model pick better trades than the rule? | C1c labels, M1, V1, E2 | Optional meta-model |
| 9. Costs | Does it survive realistic and stressed costs? | T5, R5 | Break-even cost |
| 10. Handover | Does the platform reproduce research exactly? | T7, T8 | Spec card, config, `ParityReport` |
| 11. Paper, live, retire | Does it hold up live, and when do we stop? | Spec card kill criteria | Monitoring and retirement rule |

Steps 2–7 interact. Each team guide gives the order in which to fix them and how
to count trials when they are tuned in stages.

## Starting as a new team member

1. Read [research/strategy-lifecycle.md](../research/strategy-lifecycle.md) and
   your team guide.
2. Claim one feature or toolkit task from [tasks.md](../tasks.md) or the
   [handoff packets](../handoff/README.md). Feature tasks are a good first PR.
3. Join your team's first study as a second author before leading one.
4. Run workshop examples 1 and 6 before tuning anything.
