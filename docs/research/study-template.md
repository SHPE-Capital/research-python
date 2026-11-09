# Study template

A study answers one research question and records a decision. This template
extends [studies/README.md](../../studies/README.md): the folder layout stays
the same, and the README gains a fixed structure so every study can be read the
same way.

Study IDs: existing R1–R5, and team studies `RM<n>` (momentum), `RR<n>`
(reversals), `RS<n>` (stat arb). Folders live at
`studies/<team>/<id>-<slug>/` (R1 in `studies/momentum/`, R2 and R4 in
`studies/reversals/`, R3 in `studies/stat-arb/`, R5 in `studies/shared/`).
`shpe new study` (C1j) will generate this folder.

## Folder layout

```
studies/<team>/<id>-<slug>/
  README.md          this template, filled in
  SPEC.md            spec card, if the study builds or tunes a strategy
  configs/           every experiment config that produced a reported number
  notebooks/         jupytext .py notebooks or scripts for figures; outputs cleared
  PARITY.md          ParityReport summary, if the study reaches platform parity
  POSTMORTEM.md      only when a live strategy is retired
```

Never commit data, run outputs or keys. Run directories stay in the ignored
runs folder and are referenced by ID.

## How to use it

1. Copy the block below into the study's `README.md`.
2. Fill in sections 1–3 and commit **before** running anything on validation.
3. Fill in sections 4–9 as you go. Keep the trial count current.
4. Freeze, commit, ask for a `leakage-auditor` review, then open test once.
5. Fill in sections 10–12. Add the study to the team's study index.

See [research-protocol.md](research-protocol.md) for the rules behind each section.

---

~~~markdown
# <ID>: <title>

| | |
|---|---|
| Team | |
| Authors | |
| Status | pre-registered / in progress / frozen / complete / abandoned |
| Pre-registration commit | <hash, date> |
| Frozen commit | <hash, date> |
| Test opened | <date, run ID> (once) |
| Decision | go / no-go / needs more data |

## Summary
Question: …
Decision: …
Net result (test, held-out): <metric, 95% CI>
Trials: <n>; deflated Sharpe: <value>
Biggest caveat: …

## 1. Question and decision
<One question. The decision it informs. Link the spec card if any.>

## 2. Hypothesis
- H1:
- H0:
- Mechanism: <why it should work>

## 3. Pre-registration
| Field | Value |
|---|---|
| Data snapshot (D5) | |
| Frequency, sessions | |
| Universe config, D4 seed | |
| Dev symbols / held-out symbols | |
| Train | <start – end> |
| Validation | <start – end> |
| Test | <start – end> |
| Purge / embargo | |
| Fit scheme | |
| Candidates (CandidateGenerator) | |
| Label | |
| Features | |
| Models and seeds | |
| ExitPolicy / SizingRule / filters / RiskLimits | |
| CostModel | platform defaults unless stated |
| Grid | <every parameter and its values> |
| Selection metric | |
| Baselines | take every candidate; simple rule; shuffled / random-retention control |
| Trial budget | |
| Minimum trades | |
| Success criterion | |
| Kill criterion | |

### Changes to the pre-registration
| Date | Field | From | To | Reason | Trials before the change |
|---|---|---|---|---|---|

## 4. Data checks
<D2 audit summary for the universe and period, missing sessions, split handling,
survivorship note.>

## 5. Development log
| Date | Run ID / config | What changed | Validation metric | Kept? |
|---|---|---|---|---|

Running trial count: <n>

## 6. Validation results
- Predictive: IC, IC decay
- Trading: net bps per trade, per day; turnover; hit rate; max drawdown
- Costs: 0 / 2 / 5 / 10 bps, platform defaults, 2× platform
- Baselines, with paired block-bootstrap differences
- Breakdowns (`by_group`): symbol, month, universe cell, time of day

## 7. Trade management
<Horizon curve, half-life, MAE/MFE, stop and target comparisons against no stop,
sizing ladder, filters, risk-limit binding counts. Link T6 figures.>

## 8. Parameter surfaces
<Figures and a sentence per parameter: plateau or peak, and the choice.>

## 9. Freeze
Frozen config: <path, commit>. Trial count at freeze: <n>.
Leakage-auditor review: <date, verdict, findings resolved>.

## 10. Test and held-out results
| Split | Trades | Net bps/trade (95% CI) | Sharpe | DSR | Max DD |
|---|---|---|---|---|---|
| Test, dev symbols | | | | | |
| Test, held-out symbols | | | | | |

PBO (if the grid allows): <value>.

## 11. What failed
<Every variant that did not work, one line each. This section is required.>

## 12. Decision and next steps
<Go / no-go / needs more data. Next lifecycle stage. Spec card updates. Tasks to open.>

## Reproduce
```bash
uv sync --extra ml
shpe data snapshot verify <snapshot>
shpe run studies/<team>/<id>-<slug>/configs/<final>.yaml
```
~~~

---

## Negative and abandoned studies

A study that finds no effect, or is stopped early, keeps its folder and README.
Set status to `complete` (with decision no-go) or `abandoned`, fill in what
was run, the trial count, and why it stopped. The next person with the same
idea needs to find it.
