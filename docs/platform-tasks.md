# Platform tasks (trading-platform repo)

These tasks connect research output to the trading platform. They are
TypeScript work in [SHPE-Capital/trading-platform](https://github.com/SHPE-Capital/trading-platform);
paths below are relative to that repo. Claim them there with an issue that
references the ID.

## P1: Candidate and feature log

**Why:** every strategy proposes candidates; recording each one with its
features and outcome, whether taken or skipped, turns live paper trading and
every backtest into training data. It is also what the Replay viewer (P4) shows.

- Define a `StrategyCandidate` record: strategy id/type/version, symbol,
  decision time, side, feature values (name → number), decision
  (`accepted`/`rejected` plus reason), model score when a selector ran, and the
  signal id when one was emitted. The feature names must match this repo's
  feature names (see `docs/contracts.md` here).
- Publish it on the EventBus (`backend/src/core/engine/eventBus.ts`) from the
  strategy layer (`backend/src/strategies/base/strategy.ts`, `backend/src/types/strategy.ts`).
- Persist it in a new table via the next migration in `supabase/migrations/`
  (latest today: `0019_expire_stranded_sandbox_runs.sql`), for both backtests
  and paper runs.
- **Done when:** a backtest of an existing strategy writes one row per
  candidate, and a member can export those rows to Parquet for research.

## P2: ONNX selector wrapper

**Depends on:** M3 (export format), P1.

- A wrapper that takes any `IStrategy`, intercepts its signals as candidates,
  computes the feature row, runs the exported model in-process with
  `onnxruntime-node`, and accepts, skips or resizes the signal. Strategies may
  not call external services, so no Python inference server.
- Load `model.onnx` + `manifest.json`; refuse to start if the manifest's
  feature names are not all produced by the platform.
- **Parity test:** load M3's golden rows and require predictions within the
  manifest's tolerance in CI.
- **Done when:** a backtest can run the same strategy with and without the
  wrapper, and P1 logs the score for every candidate.

## P3: Replay data path

The Replay page is a stub today (see `docs/guides/replay.md` in that repo):

- `backend/src/core/replay/replayFilter.ts`: `applyFilter` returns its input unchanged (TODO in file).
- `backend/src/app/controllers/replayController.ts`: session listing returns
  `[]`, and `/api/replay/load` returns 501 because nothing reads `event_logs`.
- Nothing writes `event_logs` (`supabase/migrations/0000_initial_schema.sql`).
- `backend/src/core/replay/attributionCollector.ts`: the report method throws.

Rebuild it as a **read-only viewer over recorded runs**, not a second
simulator. The backtest engine stays the only source of fills:

- Implement `applyFilter`.
- Build a `ReplaySession` from a run's P1 candidates, its orders and fills, and
  the cached bars for its symbols and dates, instead of from `event_logs`.
- **Done when:** `/api/replay/load` returns a session for a saved backtest.

## P4: Replay decision viewer

**Depends on:** P3. Files: `frontend/app/replay/page.tsx`, `frontend/components/controls/ReplayControls.tsx`.

- Chart with bars and, at each candidate, its features, model score, decision,
  order and fill. Step from one candidate to the next.
- Side-by-side mode: the same day from two runs (live vs. backtest, or
  selector A vs. C), highlighting where they diverge.
- **Done when:** a member can explain one trade end to end from the viewer.
  This is a workshop deliverable.

## P5: Models tab

**Depends on:** M3.

- Read-only list of exported models: name, type, training window, validation
  metrics and status (`shadow`, `approved`, `retired`), from their manifests.
- **Done when:** the tab lists every model in the registry location the team
  chooses, and links each to its research report.
