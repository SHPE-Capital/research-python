# Agent orchestrator

Runs the handoff tasks as unattended Claude Code workers. Each task gets its
own git worktree, branch and Windows Terminal tab. When a worker finishes,
the orchestrator checks its branch and merges it into **`JC`**. Tasks that
depended on it then start without anyone stepping in.

| File | Role |
|---|---|
| `start-agents.bat` | Entry point. It passes every argument to `orchestrate.ps1`. |
| `orchestrate.ps1` | Scheduler. It polls, creates worktrees, opens tabs, merges finished work and resumes failed work. |
| `run-task.ps1` | Runs one headless worker (`claude -p`) in its tab and records how it ended. |
| `tasks.psd1` | The task graph: ID, agent, wave, dependencies and kind for each task. |

## Before the first run

1. Check out `JC` in this repository and commit everything on it. Every
   worktree starts from `JC`, and finished work is fast-forwarded into the
   `JC` checkout here.
2. While the orchestrator runs, don't edit tracked files in this checkout. A
   merge waits, and retries on the next poll, until this checkout has no
   uncommitted changes to tracked files.
3. Put `claude`, `uv` and `git` on PATH. Windows Terminal (`wt.exe`) is
   optional; without it, each task opens in its own PowerShell window.
4. The default branch prefix is your lower-cased `git config user.name`. It
   must not match any existing branch case-insensitively: on Windows, a
   branch called `JC` blocks `jc/<ID>`.

## How it runs

```
start-agents.bat -Plan          # launch order by depth, then exit
start-agents.bat -DryRun        # what would launch now
start-agents.bat                # run unattended (poll every 60 s)
start-agents.bat -Status        # state of every unmerged task
```

Each worker runs:

```
claude -p --permission-mode auto [--agent <agent>] "<prompt>"
```

The prompt points the worker at its packet, `docs/handoff/tasks/<ID>.md`, or,
for tasks with no packet, at their row in `docs/tasks.md`. It also tells the
worker:

- Never ask questions. A change that `AGENTS.md` says needs approval is
  skipped and listed under Questions in the report. That covers contract
  changes, new dependencies and edits to another packet's files.
- Commit on its own branch, `<name>/<ID>`. Never push, never merge and never
  switch branches.

On each poll the orchestrator does three things.

1. **Integrates** finished workers, one at a time:
   - The worktree must be clean, and the branch must have commits ahead of
     `JC`.
   - It runs `git merge JC` into the task branch.
   - It runs `uv sync`, `uv run ruff check` and `uv run pytest -q`. Output
     goes to `<ID>.gate.log`.
   - It runs `git merge --ff-only <name>/<ID>` in this checkout, which moves
     `JC` forward.

   If a step fails, it resumes the worker with `claude -p --continue` and a
   message saying what failed: a dirty tree, no commits, a merge conflict or
   failing checks. After `-MaxFixAttempts` resumes (default 2), the task is
   marked **failed** and its dependents stay blocked.
2. **Recovers** workers:
   - A worker that hit a usage limit is resumed once the limit resets, and
     no new launches start until then. This does not count as an attempt.
   - A worker running longer than `-TaskTimeoutMinutes` (default 240) is
     killed and resumed.
   - A worker whose tab was closed is also resumed.
3. **Launches** ready tasks, meaning every dependency is merged. They start
   in `tasks.psd1` order, up to `-MaxParallel` at once (default 6).

The orchestrator exits when one of two things happens:

- Every task is merged.
- Nothing is left that can run, because the remaining tasks are failed or
  blocked behind failed tasks.

Wave 0 runs one step after another: `MIG`, then `C1a` and `DEPS`, then
`C1h`, `C1o` and `C1l`. Every Wave 1+ task waits for all six.

## Logs and reports

These are in `..\research-python-worktrees\.orchestrator\`:

| File | Content |
|---|---|
| `<ID>.report.md` | The worker's final report for each run. Read the Questions section. |
| `<ID>.log.jsonl` | The raw event stream: every message and tool call |
| `<ID>.gate.log` | Output of the last merge check |
| `<ID>.exited` | `status=ok\|error\|ratelimit`, exit code and cost in USD |
| `<ID>.failed` | Why the task was given up on |
| `prompts\<ID>.txt` | The prompt the worker started with |

## Handling a failed task

1. Read `<ID>.report.md` and `<ID>.gate.log`.
2. Fix the problem in the task's worktree, or open the conversation with
   `start-agents.bat -Resume <ID>`, which runs interactive
   `claude --continue`.
3. Run `start-agents.bat -Retry <ID>`. It clears the failed marker and the
   attempt count, and the task runs again on the next poll in its existing
   worktree.
4. Start the orchestrator again if it has exited.

## Options

| Option | Effect |
|---|---|
| `-Name juancavallin` | Branch prefix (`<name>/<ID>`) |
| `-BaseRef JC` | The branch worktrees start from and merges go into. With `origin/<branch>`, nothing is merged: each task waits for you to merge its branch yourself. |
| `-MaxParallel 6` | Maximum workers running at once |
| `-PermissionMode auto` | Also `acceptEdits`, `bypassPermissions`, `dontAsk`, `manual`, `plan` |
| `-MaxFixAttempts 2` | Resumes per task before it is marked failed |
| `-TaskTimeoutMinutes 240` | Kill and resume a worker that runs longer than this |
| `-PollSeconds 60` | Poll interval |
| `-Only T1,C1d` | Restrict to these tasks |
| `-NoIntegrate` | Never merge; finished tasks wait for you |
| `-Status` | Print states and exit |
| `-Once` | One poll, then exit |
| `-MarkMerged MIG,C1a` | Record tasks as merged, for example after a manual squash merge |
| `-Retry C1a` | Clear a failed or finished task so it runs again |
| `-Resume T1` | Open T1's last conversation interactively |
| `-Cleanup` | Remove clean worktrees of merged tasks. Branches are kept. |
| `-WorktreeRoot <dir>` | Default: `..\research-python-worktrees` |
| `-NoWindowsTerminal` | Use plain PowerShell windows |

## States

| State | Meaning |
|---|---|
| `blocked` | A dependency is not merged yet. The missing dependencies are listed. |
| `ready` | Launches when a slot is free |
| `running` | A worker is running |
| `finished` | The worker ended; it is integrated on the next poll |
| `resume` | A fix or continue message is queued; the worker resumes when a slot is free |
| `stopped` | The tab closed or was killed before the worker ended; it gets resumed |
| `failed` | Given up after `-MaxFixAttempts`; see above |
| `merged` | In `JC` |

## Notes

- `repo-architect` has no Edit or Write tools, so its tasks (`MIG`, `C1a`,
  `C1l`, ...) start without `--agent`. Their prompt tells the worker to
  follow `.claude/agents/repo-architect.md`.
- `auto` permission mode lets a classifier approve or block each tool call.
  Nothing waits for a human. A blocked action makes the worker choose another
  approach or report the problem.
- Differences from `docs/handoff/README.md`:
  - `C1c` waits for `T1`.
  - `M3` waits for `M1a`.
  - `C1n` waits for the main Wave 2 tasks.
  - The header of `tasks.psd1` lists these.
- Platform-repo tasks (P6 to P9) and tasks already done (C0, D1 to D5) are
  not in the graph.
