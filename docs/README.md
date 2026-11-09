# Documentation map

This page indexes every document in the repo and fixes the IDs they share
(task IDs, ADR numbers, agent names, file paths). When two documents disagree,
this page and the ADRs win; fix the other document.

## Who reads what

| You are… | Start here |
|---|---|
| Implementing a task (person or coding agent) | [handoff/README.md](handoff/README.md), then your task packet in `handoff/tasks/` |
| A team member doing strategy research | [teams/README.md](teams/README.md), then your team's guide |
| Changing how the repo is organised | [design/architecture.md](design/architecture.md) and the [ADRs](adr/) |
| Running or attending the workshop | [workshop/README.md](workshop/README.md) |
| Building or reviewing a strategy end to end | [research/strategy-lifecycle.md](research/strategy-lifecycle.md) |
| An AI coding agent | [../AGENTS.md](../AGENTS.md) and [../.claude/agents/](../.claude/agents/) |

## Directory layout of `docs/`

```
docs/
  README.md                  this page
  contracts.md               data/model contracts (existing)
  tasks.md                   task board (existing; extended with the IDs below)
  platform-tasks.md          trading-platform tasks P1–P9
  handoff/                   implementation packets, one per parallel task
    README.md                waves, worktree rules, merge order, dependency graph
    tasks/<ID>.md            one packet per task ID below
  design/                    how the library is structured, and why
    architecture.md          layers, package tree, import rules, data flow
    trade-layer.md           strategies/ + trade/ + analysis/ design
    pipeline.md              stages, registries, events, labels, fit schemes
    platform-parity.md       research → platform config export and parity checks
    testing.md               test pyramid, fixtures with planted effects, check kits
  adr/                       architecture decision records, NNNN-title.md
  research/                  general research method, used by every team
    strategy-lifecycle.md    idea → spec → study → parity → paper → live → retire
    research-protocol.md     pre-registration, splits, trial counting, reporting
    trade-management.md      holding period, stops, targets, sizing, filters, risk, costs
    spec-card-template.md    the strategy spec card every team hands over
    study-template.md        README template for a study folder
  teams/
    README.md                how teams, shared toolkit and studies fit together
    momentum.md
    reversals.md
    stat-arb.md
  workshop/
    README.md                workshop spec: audience, format, schedule, prerequisites
    facilitator-guide.md     running sessions, timing, common failure points
    setup.md                 participant setup and data
    sessions/                foundation sessions (data, features, models, evaluation)
    examples/NN-<slug>.md    the in-depth examples
    take-homes.md            optional extensions
```

## Canonical IDs

### Existing tasks (see [tasks.md](tasks.md))

D1–D6, F1–F8, L1, L2, V1, E1, E2, M1a–M1c, M2, M3, P1–P5, R1–R5.

### C1: architecture phase

| ID | Task | Package |
|---|---|---|
| C1a | Pipeline stages and registries (sampler, label, fit scheme, report section) | `pipeline/` |
| C1b | Events/candidates layer (decision points other than "every N minutes") | `events/` |
| C1c | Label registry; `forward_return`, `multi_horizon`, `triple_barrier` registered | `labels/` |
| C1d | Context symbols and cross-sectional transforms (rank, z-score, neutralize) | `transforms/` |
| C1e | Daily frequency alongside minute bars | `data/`, `features/` |
| C1f | Fit schemes: single split, walk-forward, purged walk-forward with embargo | `validation/`, `pipeline/` |
| C1g | Pluggable data sources (Alpaca, file, FRED, EDGAR) | `data/sources/` |
| C1h | Typed config validation (Pydantic) | `config.py` |
| C1i | Report upgrades: baselines, permutation importance, by-era breakdowns | `evaluation/` |
| C1j | `shpe new` scaffolding (feature, model, strategy, study) | `scaffold/` |
| C1k | Test kits: `check_feature`, `check_model`, `check_label`, `check_strategy` | `testing/` |
| C1l | Import-linter layer contracts in CI | `pyproject.toml` |
| C1m | Point-in-time universe and security master | `data/security_master.py` |
| C1n | Docs site (mkdocs-material + mkdocstrings), Diátaxis structure | `mkdocs.yml` |
| C1o | Synthetic fixture with planted effects (momentum, reversal, cointegrated pair, vol clustering) | `data/fixtures.py` |

### T: shared trade toolkit

| ID | Task | Files |
|---|---|---|
| T1 | Exit policies and trade simulator using platform fill rules (absorbs L2's simulator) | `trade/exits.py`, `trade/simulate.py` |
| T2 | Sizing: fixed, risk-per-trade, inverse vol, prob→size, fractional Kelly | `trade/sizing.py` |
| T3 | Entry filters: time windows, liquidity/spread, regime, event blackouts | `trade/filters.py` |
| T4 | Risk limits matching the platform RiskEngine names | `trade/risk.py` |
| T5 | Cost model, calibrated from R5 | `trade/costs.py` |
| T6 | Trade diagnostics: horizon curve, OU half-life, MFE/MAE, stop curves, parameter surfaces | `analysis/` |
| T7 | Platform config export and parity check | `deploy/platform_config.py`, `deploy/parity.py` |
| T8 | Spec card generator and `shpe new strategy` | `scaffold/`, `docs/research/spec-card-template.md` |
| T9 | Portfolio construction: scores to weights, neutralization, turnover-aware rebalancing | `portfolio/` |

### S: strategy families (team-owned library code)

| ID | Team | Task | Files |
|---|---|---|---|
| SM1 | Momentum | ORB candidate generator | `strategies/momentum/orb.py` |
| SM2 | Momentum | EMA crossover baseline | `strategies/momentum/ema_cross.py` |
| SM3 | Momentum | Cross-sectional 12-1 momentum (daily) | `strategies/momentum/xs_momentum.py` |
| SR1 | Reversals | Bar-fade baseline, matching the platform `minuteReversal` | `strategies/reversal/bar_fade.py` |
| SR2 | Reversals | Stretch fade (σ-scaled deviation from prev close / VWAP / EMA) | `strategies/reversal/stretch.py` |
| SR3 | Reversals | Residual fade (stock minus β·SPY) | `strategies/reversal/residual.py` |
| SS1 | Stat arb | Pair selection: correlation, clustering, cointegration, distance method | `strategies/statarb/selection.py` |
| SS2 | Stat arb | Hedge ratios: fixed, rolling OLS, Kalman, TLS | `strategies/statarb/hedge.py` |
| SS3 | Stat arb | OU fit and half-life | `strategies/statarb/ou.py` |
| SS4 | Stat arb | Pairs candidate generator matching the platform `PairsStrategy` | `strategies/statarb/pairs.py` |

### Studies

Existing R1–R5 keep their IDs. Team studies added by the team guides use
`RM<n>` (momentum), `RR<n>` (reversals), `RS<n>` (stat arb). Study folders
move to `studies/<team>/<id>-<slug>/` (R1 → `studies/momentum/`, R2 and R4 →
`studies/reversals/`, R3 → `studies/stat-arb/`, R5 → `studies/shared/`).

### W: workshop build tasks

W0 (workshop infrastructure: notebook tooling, CI on fixtures, solutions branch)
and W1–W8 (one per example below), W9 (trade-management session).

### P: platform (trading-platform repo)

P1–P5 existing. New: **P6** connect the volatility-scaled and Kelly sizers to the
orchestrator; **P7** combine the reversal strategies into one parameterised
family; **P8** ORB momentum strategy; **P9** pairs config additions (Kalman
hedge, half-life max hold, stop rule, earnings blackout, spread-vol sizing).

## Workshop examples

| # | File | Example | Teams |
|---|---|---|---|
| 1 | `examples/01-leakage-lab.md` | Leakage, survivorship and snooping lab | All |
| 2 | `examples/02-intraday-model-comparison.md` | Intraday model comparison (linear, RF, LightGBM, MLP/LSTM) | Reversals |
| 3 | `examples/03-cross-sectional-ranker.md` | Cross-sectional ranker: replicate 12-1 momentum or reversal, then combine signals | Momentum |
| 4 | `examples/04-meta-labeling-orb.md` | Meta-labeling ORB breakouts | Momentum, Reversals |
| 5 | `examples/05-volatility-forecast.md` | Volatility forecast: HAR-RV vs. gradient boosting (positive control) | All |
| 6 | `examples/06-overfitting-casino.md` | Overfitting casino (negative control) | All |
| 7 | `examples/07-unsupervised-pairs.md` | PCA and clustering for pair discovery | Stat arb |
| 8 | `examples/08-portfolio-construction.md` | Portfolio construction and neutralization | All |
| 9 | `examples/09-trade-management.md` | Trade management: holding period, stops, targets, sizing per team | All |

## Agent definitions

Root [AGENTS.md](../AGENTS.md) holds the rules every agent follows; `CLAUDE.md`
points to it. Specialised agents live in `.claude/agents/`:

| Agent | Use for |
|---|---|
| `repo-architect` | Placing new code, enforcing layers and ADRs, scaffolding |
| `feature-engineer` | Writing features that pass the no-lookahead checks |
| `strategy-builder` | Building a strategy family in `strategies/` on top of `trade/` |
| `research-analyst` | Running a pre-registered study and writing it up |
| `trade-management-analyst` | Holding period, stop, target, sizing and filter studies |
| `leakage-auditor` | Reviewing code and results for look-ahead, survivorship and snooping |
| `platform-parity` | Exporting configs and reconciling research with platform backtests |
| `workshop-author` | Writing workshop notebooks and exercises with their solutions |
