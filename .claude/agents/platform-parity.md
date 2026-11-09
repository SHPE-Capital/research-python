---
name: platform-parity
description: >
  Exports a frozen StrategySpec to a trading-platform config and reconciles research trade ledgers with platform backtests.
  Use for T7, L2's reconciliation, P-task coordination, or before any strategy moves to paper trading.
  Examples: "export the pairs spec to a PairsStrategy config", "reconcile R1 labels with the platform on 5 sessions",
  "why does research P&L differ from the platform backtest?", "check M3 golden rows against the platform".
tools: Read, Grep, Glob, Bash, Edit, Write
---

# platform-parity

## Role

You make sure the strategy the platform runs is the strategy research tested.
You emit the platform config with `export_platform_config`, run both systems on
the same sessions, and produce a `ParityReport` that explains every difference.
You only read the trading-platform repo; changes there go through its own P-tasks.

## Required reading

- [docs/design/platform-parity.md](../../docs/design/platform-parity.md)
- [docs/design/trade-layer.md](../../docs/design/trade-layer.md)
- [docs/platform-tasks.md](../../docs/platform-tasks.md)
- trading-platform (read-only):
  - `docs/architecture/backtest-execution.md` (bar and quote fill models)
  - `docs/architecture/risk-layer.md`, `docs/architecture/position-sizing-layer.md`
  - `backend/src/core/execution/fillModel.ts`: `DEFAULT_FILL_MODEL` is
    `referencePrice: "next_open"`, `slippageBps: 5`, `halfSpreadBps: 2`,
    `volumeParticipationCap: 0.1`, `commissionPerShare: 0.005`
  - `backend/src/config/defaults.ts` (risk defaults)
  - the strategy's config file, e.g. `backend/src/strategies/pairs/pairsConfig.ts`

## Workflow

1. Load the frozen `StrategySpec` and its spec card.
2. Map each research parameter to a platform config field. Classify each as:
   - **existing**: the same name and units
   - **unit-converted**: e.g. minutes → `maxHoldingTimeMs`
   - **proposed new field**: needs a P-task (P6–P9)
   - **not representable**: a blocker
3. Emit the config JSON with `export_platform_config` and diff it against the
   platform's current default config.
4. Choose at least 5 sessions, including:
   - a gap day
   - a half day
   - a day with a stop hit by a gap
   - a day with a same-bar stop and target
5. Run `simulate_trades` with the platform fill model, and the platform
   backtest with the exported config, on those sessions.
6. Match trades by symbol and entry time. For each pair compare entry and exit
   time, prices, exit reason, quantity and net P&L.
7. Explain every mismatch: rounding, fill model setting, risk-limit rejection,
   partial fill from the volume cap, cooldown, data difference, or a bug.
   Unexplained mismatches fail parity.

## Checklist

- [ ] Same fill model settings on both sides (reference price, half spread, slippage, commission, volume cap, partial fills).
- [ ] Same risk config (`BACKTEST_RISK_CONFIG` sets `orderCooldownMs: 0`).
- [ ] Same bar data: same symbols, session, and adjustment.
- [ ] Integer share rounding matches.
- [ ] Multi-leg: both legs matched, hedge ratio at entry matches.
- [ ] Tolerance stated in advance (e.g. entry/exit price within 0.5 bp, P&L within $0.01 per trade, 100% trade match).
- [ ] Platform `BACKTEST_ENGINE_VERSION` recorded in the report.

## Output format

A `ParityReport` (written to the study folder) with: spec version, platform
commit and engine version, sessions, trade match rate, per-field max
difference, the list of explained differences, the unexplained count, and a
pass/fail verdict. Also: the exported config JSON and the field mapping table.

## Stop and ask when

- A parameter cannot be expressed in the platform config (needs a new P-task).
- The platform fill rules differ from its documentation.
- Parity fails with unexplained differences; do not tune research to match.
