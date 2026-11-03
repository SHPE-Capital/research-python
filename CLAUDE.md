# CLAUDE.md

Follow [AGENTS.md](AGENTS.md). It holds the commands, the layer rules, the
non-negotiables and the definition of done. [docs/README.md](docs/README.md)
maps every document and ID.

## Subagents in `.claude/agents/`

| Agent | Job |
|---|---|
| `repo-architect` | Code placement, layering, ADRs, scaffolding |
| `feature-engineer` | Features and transforms that pass the no-lookahead checks |
| `strategy-builder` | Strategy families in `strategies/` built on `trade/` |
| `research-analyst` | Pre-registered studies and their write-ups |
| `trade-management-analyst` | Holding period, stop, target, sizing and filter studies |
| `leakage-auditor` | Read-only review for look-ahead, survivorship and snooping |
| `platform-parity` | Platform config export and research/platform reconciliation |
| `workshop-author` | Workshop notebooks, exercises and solutions |

## Claude Code specifics

- **Read the packet first.** For any task ID, read `docs/handoff/tasks/<ID>.md`
  before opening source files. Stay inside its **Owns** list.
- **One worktree per packet.** Use a git worktree (or `isolation: "worktree"`
  for subagents) per packet, branch `<name>/<ID>`. Never let two packets edit
  the same file.
- **Run tests before claiming done.** `uv run pytest` and `uv run ruff check`
  must pass, and you must quote their outcome in your report. If something
  fails or you skipped a step, say so.
- **No data in the repo.** Write scratch files to the session scratchpad, not
  into `data/`, `runs/` or the source tree.
- **Commit only when asked.** Never commit to `main`.
- **Ask before** changing a contract in `contracts/`, bumping its version,
  adding a dependency, or editing a file another packet owns.
- On Windows the Bash tool is Git Bash; use forward slashes and `uv run`.
