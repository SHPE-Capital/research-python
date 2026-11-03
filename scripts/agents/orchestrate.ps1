<#
.SYNOPSIS
    Runs the handoff tasks as unattended Claude Code workers, one git worktree
    and one terminal tab per task, and merges each finished task into the base
    branch (JC by default) so the tasks that depend on it can start.

.DESCRIPTION
    The task graph is in tasks.psd1. Every poll:
      1. Finished workers are integrated, one at a time:
         - the worktree must be clean and the branch must have commits;
         - the base branch is merged into the task branch;
         - uv sync, uv run ruff check and uv run pytest must pass;
         - then the base branch is fast-forwarded to the task branch.
         If a step fails, the worker is resumed (claude --continue) with a
         prompt describing the failure, up to -MaxFixAttempts times; after
         that the task is marked failed and its dependents stay blocked.
      2. Workers that hit a usage limit are resumed after the limit resets,
         without counting an attempt. Workers running longer than
         -TaskTimeoutMinutes are stopped and resumed.
      3. Ready tasks (every dependency merged) start in tasks.psd1 order, up to
         -MaxParallel at once, each in worktree <root>\<ID> on branch
         <name>/<ID> created from the base branch.

    Workers run headless (claude -p) with -PermissionMode (auto by default),
    commit on their own branch, and never push or merge. The base branch must
    be checked out in this repository, with no uncommitted changes to tracked
    files, whenever a merge happens.

    Runtime state (prompts, logs, reports, markers) lives outside the repo, in
    <root>\.orchestrator.

.EXAMPLE
    scripts\agents\start-agents.bat -Plan
.EXAMPLE
    scripts\agents\start-agents.bat -DryRun
.EXAMPLE
    scripts\agents\start-agents.bat
.EXAMPLE
    scripts\agents\start-agents.bat -Retry C1a
#>
[CmdletBinding()]
param(
    [string]$Name,
    [string]$BaseRef = 'JC',
    [int]$MaxParallel = 6,
    [string]$WorktreeRoot,
    [ValidateSet('auto', 'acceptEdits', 'bypassPermissions', 'dontAsk', 'manual', 'plan')]
    [string]$PermissionMode = 'auto',
    [int]$PollSeconds = 60,
    [int]$MaxFixAttempts = 2,
    [int]$TaskTimeoutMinutes = 240,
    [string]$Only,
    [string]$MarkMerged,
    [string]$Retry,
    [string]$Resume,
    [switch]$NoIntegrate,
    [switch]$Status,
    [switch]$Plan,
    [switch]$Once,
    [switch]$DryRun,
    [switch]$Cleanup,
    [switch]$NoWindowsTerminal
)

$ErrorActionPreference = 'Continue'
Set-StrictMode -Version 2

function Fail([string]$msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

function Split-List([string]$s) {
    if (-not $s) { return @() }
    return @($s -split '[,\s]+' | Where-Object { $_ })
}

function Write-Log([string]$msg, [string]$color = 'Gray') {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg) -ForegroundColor $color
}

# ---------------------------------------------------------------- setup

$RepoRoot = & git -C $PSScriptRoot rev-parse --show-toplevel 2>$null
if (-not $RepoRoot) { Fail 'scripts\agents is not inside a git repository' }
$RepoRoot = $RepoRoot -replace '/', '\'

function Invoke-Git { & git -C $RepoRoot @args 2>$null }

if (-not $WorktreeRoot) {
    $WorktreeRoot = Join-Path (Split-Path $RepoRoot -Parent) ((Split-Path $RepoRoot -Leaf) + '-worktrees')
}
$StateDir = Join-Path $WorktreeRoot '.orchestrator'
$PromptDir = Join-Path $StateDir 'prompts'

if (-not $Name) { $Name = Invoke-Git config user.name }
$Name = ("$Name".ToLower() -replace '[^a-z0-9._-]', '')
if (-not $Name) { Fail 'no branch prefix: pass -Name <yourname>' }

$data = Import-PowerShellDataFile (Join-Path $PSScriptRoot 'tasks.psd1')
$Gate = @($data.Gate)
$Tasks = @($data.Tasks)
$ById = @{}
foreach ($t in $Tasks) {
    if (-not $t.ContainsKey('Kind')) { $t.Kind = 'packet' }
    $ById[$t.Id] = $t
}

function Get-Deps($t) {
    $d = @($t.Deps)
    if ($t.Wave -gt 0) { $d = $Gate + $d }
    return @($d | Select-Object -Unique)
}

foreach ($t in $Tasks) {
    foreach ($d in (Get-Deps $t)) {
        if (-not $ById.ContainsKey($d)) { Fail "task $($t.Id) depends on unknown task $d (tasks.psd1)" }
    }
}

# Depth = longest dependency chain below the task; also catches cycles.
$Depth = @{}
function Get-Depth([string]$id, [string[]]$stack = @()) {
    if ($Depth.ContainsKey($id)) { return $Depth[$id] }
    if ($stack -contains $id) { Fail "dependency cycle: $(($stack + $id) -join ' -> ')" }
    $max = -1
    foreach ($d in (Get-Deps $ById[$id])) {
        $x = Get-Depth $d ($stack + $id)
        if ($x -gt $max) { $max = $x }
    }
    $Depth[$id] = $max + 1
    return $Depth[$id]
}
foreach ($t in $Tasks) { [void](Get-Depth $t.Id) }

$Selected = $Tasks
$onlyIds = @(Split-List $Only)
if ($onlyIds.Count -gt 0) {
    foreach ($id in $onlyIds) { if (-not $ById.ContainsKey($id)) { Fail "unknown task in -Only: $id" } }
    $Selected = @($Tasks | Where-Object { $onlyIds -contains $_.Id })
}

if ($Plan) {
    Write-Host "Launch plan (depth = longest chain of merges before the task can start)`n"
    $Tasks | Sort-Object { $Depth[$_.Id] }, { [array]::IndexOf($Tasks, $_) } |
        Group-Object { $Depth[$_.Id] } | ForEach-Object {
            $ids = ($_.Group | ForEach-Object { $_.Id }) -join ', '
            Write-Host ("depth {0,2}: {1}" -f $_.Name, $ids)
        }
    exit 0
}

# ---------------------------------------------------------------- preflight

$BaseBranch = $BaseRef -replace '^origin/', ''

function Update-Base {
    if ($BaseRef -like 'origin/*') { Invoke-Git fetch origin --quiet | Out-Null }
}

Update-Base
$baseSha = Invoke-Git rev-parse --verify --quiet "$BaseRef^{commit}"
if (-not $baseSha) { Fail "base ref $BaseRef not found" }

foreach ($f in @('AGENTS.md', 'docs/handoff/README.md', '.claude/agents/strategy-builder.md')) {
    Invoke-Git cat-file -e "${BaseRef}:$f" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Fail "$BaseRef has no $f. Commit the handoff docs and .claude/agents to $BaseBranch first; every worktree starts from $BaseRef."
    }
}

foreach ($ref in @(Invoke-Git for-each-ref '--format=%(refname:short)' refs/heads)) {
    if ($ref -ieq $Name) {
        Fail "branch '$ref' exists, and on Windows it blocks branches named $Name/<ID>. Pass -Name with another prefix."
    }
}

$Integrate = -not $NoIntegrate -and -not ($BaseRef -like 'origin/*')
if ($Integrate -and -not $Status -and -not $DryRun) {
    $current = Invoke-Git branch --show-current
    if ($current -ne $BaseRef) {
        Fail "this repository has $current checked out; check out $BaseRef (merges fast-forward it here) or pass -NoIntegrate"
    }
}

if (-not $Status -and -not $DryRun) {
    foreach ($cmd in @('claude', 'uv')) {
        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { Fail "$cmd is not on PATH" }
    }
    New-Item -ItemType Directory -Force -Path $PromptDir | Out-Null
}

$UseWt = (-not $NoWindowsTerminal) -and [bool](Get-Command wt.exe -ErrorAction SilentlyContinue)

# Agents whose definition has no Edit/Write tool (repo-architect is review-only)
# cannot implement a task, so those tasks run as a normal session that is told
# to follow the agent file.
$AgentCanEdit = @{}
function Test-AgentCanEdit([string]$agent) {
    if (-not $AgentCanEdit.ContainsKey($agent)) {
        $file = Join-Path $RepoRoot ".claude\agents\$agent.md"
        $toolsLine = ''
        if (Test-Path $file) { $toolsLine = (Select-String -Path $file -Pattern '^tools:' | Select-Object -First 1).Line }
        $AgentCanEdit[$agent] = ((-not $toolsLine) -or ($toolsLine -match '\bEdit\b' -and $toolsLine -match '\bWrite\b'))
    }
    return $AgentCanEdit[$agent]
}

# ---------------------------------------------------------------- state

function Get-StatePath([string]$id, [string]$ext) { Join-Path $StateDir "$id.$ext" }
function Get-Branch([string]$id) { "$Name/$id" }
function Get-Worktree([string]$id) { Join-Path $WorktreeRoot $id }

function Read-KV([string]$path) {
    if (-not (Test-Path $path)) { return @{} }
    $raw = Get-Content $path -Raw
    if (-not $raw) { return @{} }
    return (ConvertFrom-StringData $raw)
}

function Get-Attempts([string]$id) {
    $p = Get-StatePath $id 'attempts'
    if (Test-Path $p) { return [int](Get-Content $p -TotalCount 1) }
    return 0
}

$MergedCache = @{}
function Test-Merged([string]$id) {
    if ($MergedCache.ContainsKey($id)) { return $MergedCache[$id] }
    $result = $false
    if (Test-Path (Get-StatePath $id 'merged')) {
        $result = $true
    } else {
        $tip = Invoke-Git rev-parse --verify --quiet "refs/heads/$(Get-Branch $id)"
        $startFile = Get-StatePath $id 'base'
        if ($tip -and (Test-Path $startFile) -and ($tip -ne (Get-Content $startFile -TotalCount 1))) {
            Invoke-Git merge-base --is-ancestor $tip $BaseRef | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $result = $true
                Set-Content -Path (Get-StatePath $id 'merged') -Value "detected $(Get-Date -Format o)" -Encoding ASCII
            }
        }
    }
    $MergedCache[$id] = $result
    return $result
}

function Get-RunnerPid([string]$id) {
    $p = Get-StatePath $id 'pid'
    if (-not (Test-Path $p)) { return 0 }
    $procId = 0
    if (-not [int]::TryParse((Get-Content $p -TotalCount 1), [ref]$procId)) { return 0 }
    if (Get-Process -Id $procId -ErrorAction SilentlyContinue) { return $procId }
    return 0
}

function Get-Waiting($t) { @(Get-Deps $t | Where-Object { -not (Test-Merged $_) }) }

function Get-TaskState($t) {
    $id = $t.Id
    if (Test-Merged $id) { return 'merged' }
    if (Test-Path (Get-StatePath $id 'failed')) { return 'failed' }
    $launched = Get-StatePath $id 'launched'
    if (Test-Path $launched) {
        if (Test-Path (Get-StatePath $id 'exited')) { return 'finished' }
        if (Get-RunnerPid $id) { return 'running' }
        $age = (Get-Date) - (Get-Item $launched).LastWriteTime
        if (-not (Test-Path (Get-StatePath $id 'pid')) -and $age.TotalSeconds -lt 120) { return 'running' }
        return 'stopped'
    }
    if (Test-Path (Get-StatePath $id 'next')) { return 'resume' }
    if (@(Get-Waiting $t).Count -eq 0) { return 'ready' }
    return 'blocked'
}

function Get-CooldownUntil {
    $p = Join-Path $StateDir 'cooldown'
    if (-not (Test-Path $p)) { return $null }
    $until = [datetime]::Parse((Get-Content $p -TotalCount 1))
    if ($until -le (Get-Date)) { Remove-Item $p -ErrorAction SilentlyContinue; return $null }
    return $until
}

# ---------------------------------------------------------------- prompts

function Test-PromptText([string]$id, [string]$text) {
    if ($text -match '["%]') { Fail "prompt for $id contains a double quote or percent sign, which breaks argument passing" }
    return $text
}

function Get-Rules([string]$branch) {
    $r = @()
    $r += 'You are running unattended: nobody will answer questions, so never wait for input. Where AGENTS.md or CLAUDE.md say to ask first (a contract change or version bump, a dependency that is not in the handoff table, editing a file another packet owns), do not make that change; list it under Questions in your final report and finish everything else.'
    $r += "You are authorised to commit. When uv run ruff check and uv run pytest both pass, commit all of your work on branch $branch with a conventional commit message. Never commit to $BaseBranch or main, never push, never merge, and do not switch branches. The orchestrator merges your branch into $BaseBranch after rerunning the checks."
    $r += 'Write scratch files outside the worktree. Leave the worktree clean: everything either committed or deleted.'
    $r += 'Finish with a report: files changed, tests added and their status, the ruff and pytest outcome, anything in the spec you could not satisfy, and questions for the owner of another task.'
    return $r
}

function New-Prompt($t) {
    $id = $t.Id
    $branch = Get-Branch $id
    $agent = $t.Agent
    $parts = @()

    if (Test-AgentCanEdit $agent) {
        $parts += "You are working in the git worktree for task $id on branch $branch, running as the $agent agent."
    } else {
        $parts += "You are working in the git worktree for task $id on branch $branch. Follow the rules and required reading in .claude/agents/$agent.md; you implement this task yourself, so its review-only default does not apply."
    }

    switch ($t.Kind) {
        'migration' {
            $parts += 'Task MIG is the Wave 0 migration PR. Read AGENTS.md, then the section Wave 0: the migration PR in docs/handoff/README.md with its checklist, and the migration table in docs/design/architecture.md.'
            $parts += 'Before changing anything, run uv run pytest and record the passed and skipped counts. Move files and fix imports only: no behaviour change, no registry key change, and keep every TODO(<ID>) docstring word for word. Afterwards uv run pytest must give the same counts and the hello-model golden output must be unchanged.'
        }
        'deps' {
            $parts += 'Task DEPS is the Wave 0 pyproject.toml dependency pass. Read AGENTS.md and the dependency table under Conflict hotspots in docs/handoff/README.md.'
            $parts += 'Add each listed dependency to the group the table names. Append only; never reorder or reformat, and do not touch [tool.importlinter], which belongs to C1l. Skip anything the table or the open questions mark as optional or undecided (for example cvxpy or torch) and list it under Questions. Then run uv lock and uv sync, and commit pyproject.toml and uv.lock.'
        }
        'stub' {
            $parts += "Task $id has no packet. Its spec is its row in docs/tasks.md and the TODO($id) docstring in the stub file that row names (search the source tree for TODO($id)); the row also names its tests."
            $parts += 'Read AGENTS.md and docs/handoff/README.md (ownership, hotspots, definition of done) first. Edit only that stub, its tests and the hotspot lines the handoff README allows. Write or un-skip the tests first and confirm they fail, then implement until they pass.'
        }
        default {
            $parts += "Read, in this order: AGENTS.md; docs/handoff/README.md (ownership, hotspots, definition of done); docs/handoff/tasks/$id.md (your packet); then every file under the packet's Reads heading."
            $parts += 'Create or edit only the files in the packet''s Owns list, plus the hotspot edits docs/handoff/README.md allows. Write the acceptance tests first, from the packet, and confirm they fail; then implement until they pass. If a public name or field listed under Interface cannot be met as written, keep the closest compatible version and explain it under Questions.'
        }
    }

    $parts += Get-Rules $branch
    return (Test-PromptText $id ($parts -join ' '))
}

function Format-ForPrompt([string[]]$lines) {
    return (($lines | ForEach-Object { $_ -replace '"', "'" -replace '%', ' pct' }) -join ' | ')
}

# Queue a resumed session (claude --continue) with $message. Counts an
# attempt unless -Free; past -MaxFixAttempts the task is marked failed.
function Request-Fix($t, [string]$reason, [string]$message, [switch]$Free) {
    $id = $t.Id
    $attempts = Get-Attempts $id
    if (-not $Free) {
        $attempts++
        if ($attempts -gt $MaxFixAttempts) {
            Set-Content -Path (Get-StatePath $id 'failed') -Value $reason -Encoding ASCII
            Write-Log "$id failed: $reason (gave up after $MaxFixAttempts fix attempts)" 'Red'
            return
        }
        Set-Content -Path (Get-StatePath $id 'attempts') -Value $attempts -Encoding ASCII
    }
    $text = ($message + ' ' + ((Get-Rules (Get-Branch $id)) -join ' '))
    Set-Content -Path (Join-Path $PromptDir "$id.next.txt") -Value (Test-PromptText $id $text) -Encoding ASCII
    Set-Content -Path (Get-StatePath $id 'next') -Value $reason -Encoding ASCII
    Remove-Item (Get-StatePath $id 'launched'), (Get-StatePath $id 'exited'), (Get-StatePath $id 'pid') -ErrorAction SilentlyContinue
    Write-Log "$id will resume: $reason" 'Yellow'
}

# ---------------------------------------------------------------- launch

function Format-Arg([string]$a) {
    if ($a -match '\s') { return '"' + $a + '"' }
    return $a
}

function Start-Task($t, [switch]$Continue, [switch]$Interactive) {
    $id = $t.Id
    $branch = Get-Branch $id
    $wt = Get-Worktree $id

    if ($DryRun) {
        Write-Host "  would launch $id ($($t.Agent)) on $branch in $wt"
        return
    }

    if (-not (Test-Path $wt)) {
        $exists = Invoke-Git rev-parse --verify --quiet "refs/heads/$branch"
        if ($exists) {
            Invoke-Git worktree add $wt $branch | Out-Null
            if (-not (Test-Path (Get-StatePath $id 'base'))) {
                Set-Content -Path (Get-StatePath $id 'base') -Value (Invoke-Git merge-base $branch $BaseRef) -Encoding ASCII
            }
        } else {
            $start = Invoke-Git rev-parse --verify --quiet "$BaseRef^{commit}"
            Invoke-Git worktree add -b $branch $wt $start | Out-Null
            Set-Content -Path (Get-StatePath $id 'base') -Value $start -Encoding ASCII
        }
        if (-not (Test-Path $wt)) {
            Write-Log "could not create worktree for $id at $wt" 'Red'
            Set-Content -Path (Get-StatePath $id 'failed') -Value 'could not create worktree' -Encoding ASCII
            return
        }
    }

    $nextMarker = Get-StatePath $id 'next'
    if (Test-Path $nextMarker) {
        $Continue = $true
        $promptFile = Join-Path $PromptDir "$id.next.txt"
        Remove-Item $nextMarker
    } else {
        $promptFile = Join-Path $PromptDir "$id.txt"
        if (-not $Continue) { Set-Content -Path $promptFile -Value (New-Prompt $t) -Encoding ASCII }
    }

    Remove-Item (Get-StatePath $id 'exited'), (Get-StatePath $id 'pid') -ErrorAction SilentlyContinue
    Set-Content -Path (Get-StatePath $id 'launched') -Value (Get-Date -Format o) -Encoding ASCII

    $agentArg = ''
    if (Test-AgentCanEdit $t.Agent) { $agentArg = $t.Agent }
    $runner = Join-Path $PSScriptRoot 'run-task.ps1'
    # Headless tabs close when the worker ends (its log and report are in
    # $StateDir); an interactive -Resume tab stays open.
    $runArgs = @()
    if ($Interactive) { $runArgs += '-NoExit' }
    $runArgs += @('-ExecutionPolicy', 'Bypass', '-File', $runner,
        '-Id', $id, '-Worktree', $wt, '-StateDir', $StateDir, '-PromptFile', $promptFile,
        '-PermissionMode', $PermissionMode, '-Label', $t.Agent)
    if ($agentArg) { $runArgs += @('-Agent', $agentArg) }
    if ($Continue) { $runArgs += '-Continue' }
    if ($Interactive) { $runArgs += '-Interactive' }

    if ($UseWt) {
        & wt.exe -w shpe-agents new-tab --title $id -d $wt powershell.exe @runArgs
    } else {
        Start-Process powershell.exe -WorkingDirectory $wt -ArgumentList (($runArgs | ForEach-Object { Format-Arg $_ }) -join ' ')
    }
    $how = 'started'
    if ($Continue) { $how = 'resumed' }
    Write-Log "$how $id ($($t.Agent)) on $branch" 'Green'
    Start-Sleep -Seconds 2
}

# ---------------------------------------------------------------- integration

function Invoke-Integrate($t) {
    $id = $t.Id
    $wt = Get-Worktree $id
    $branch = Get-Branch $id
    $exit = Read-KV (Get-StatePath $id 'exited')
    $status = 'error'
    if ($exit.ContainsKey('status')) { $status = $exit['status'] }

    if ($status -eq 'ratelimit') {
        $until = (Get-Date).AddMinutes(30)
        if ($exit.ContainsKey('resetsAt') -and [long]$exit['resetsAt'] -gt 0) {
            $until = ([datetime]'1970-01-01Z').ToLocalTime().AddSeconds([long]$exit['resetsAt'] + 120)
        }
        Set-Content -Path (Join-Path $StateDir 'cooldown') -Value $until.ToString('o') -Encoding ASCII
        Write-Log "$id hit a usage limit; no launches until $($until.ToString('HH:mm'))" 'Yellow'
        Request-Fix $t 'usage limit' 'Your previous run stopped at a usage limit. Continue the task from where you stopped, following the original instructions above.' -Free
        return
    }
    if ($status -ne 'ok') {
        Request-Fix $t "session ended with $status" 'Your previous run ended with an error before finishing. Check git status and the task, then continue from where you stopped, following the original instructions above.'
        return
    }

    $dirty = @(& git -C $wt status --porcelain 2>$null)
    if ($dirty.Count -gt 0) {
        Request-Fix $t 'uncommitted changes' ("The worktree still has uncommitted changes: " + (Format-ForPrompt ($dirty | Select-Object -First 20)) + '. Commit the files that belong to the task and delete the rest.')
        return
    }

    $commits = [int](& git -C $wt rev-list --count "$BaseRef..HEAD" 2>$null)
    if ($commits -eq 0) {
        Request-Fix $t 'no commits' 'Your branch has no commits ahead of the base branch, so there is nothing to merge. Finish the task and commit it.'
        return
    }

    & git -C $wt merge --no-edit $BaseRef 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $conflicts = @(& git -C $wt diff --name-only --diff-filter=U 2>$null)
        & git -C $wt merge --abort 2>$null | Out-Null
        Request-Fix $t 'merge conflict' ("Other tasks were merged into $BaseBranch meanwhile, and merging it into your branch conflicts in: " + (Format-ForPrompt $conflicts) + ". Run git merge $BaseBranch, resolve every conflict so both sides' changes survive, run the checks and commit the merge.")
        return
    }

    $gateLog = Get-StatePath $id 'gate.log'
    Write-Log "$id checks: uv sync, ruff, pytest" 'Cyan'
    Push-Location $wt
    & cmd.exe /c "(uv sync --quiet && uv run ruff check && uv run pytest -q) > `"$gateLog`" 2>&1"
    $ok = ($LASTEXITCODE -eq 0)
    Pop-Location
    if (-not $ok) {
        $tail = @(Get-Content $gateLog -Tail 30)
        Request-Fix $t 'checks failed' ("After merging $BaseBranch into your branch, the checks fail. Full log: $gateLog. Last lines: " + (Format-ForPrompt $tail) + ' Fix the failures and commit.')
        return
    }

    $tracked = @(Invoke-Git status --porcelain --untracked-files=no)
    if ($tracked.Count -gt 0) {
        Write-Log "cannot merge $id yet: $RepoRoot has uncommitted changes to tracked files" 'Yellow'
        return
    }
    if ((Invoke-Git branch --show-current) -ne $BaseRef) {
        Write-Log "cannot merge $id yet: $RepoRoot is not on $BaseRef" 'Yellow'
        return
    }
    Invoke-Git merge --ff-only $branch | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "fast-forward of $BaseRef to $branch failed; will retry" 'Yellow'
        return
    }
    Set-Content -Path (Get-StatePath $id 'merged') -Value "integrated $(Get-Date -Format o)" -Encoding ASCII
    $MergedCache[$id] = $true
    Write-Log "merged $id into $BaseRef ($(Invoke-Git rev-parse --short $BaseRef))" 'Green'
}

# ---------------------------------------------------------------- one-shot commands

foreach ($id in (Split-List $MarkMerged)) {
    if (-not $ById.ContainsKey($id)) { Fail "unknown task in -MarkMerged: $id" }
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
    Set-Content -Path (Get-StatePath $id 'merged') -Value "marked $(Get-Date -Format o)" -Encoding ASCII
    Write-Host "marked $id as merged"
}
if ($MarkMerged) { exit 0 }

foreach ($id in (Split-List $Retry)) {
    if (-not $ById.ContainsKey($id)) { Fail "unknown task in -Retry: $id" }
    foreach ($ext in @('failed', 'attempts', 'launched', 'exited', 'next', 'pid')) {
        Remove-Item (Get-StatePath $id $ext) -ErrorAction SilentlyContinue
    }
    Write-Host "cleared $id; it starts again on the next poll (in its existing worktree if there is one)"
}
if ($Retry) { exit 0 }

if ($Resume) {
    if (-not $ById.ContainsKey($Resume)) { Fail "unknown task in -Resume: $Resume" }
    if (-not (Test-Path (Get-Worktree $Resume))) { Fail "no worktree for $Resume yet" }
    Start-Task $ById[$Resume] -Continue -Interactive
    exit 0
}

if ($Cleanup) {
    foreach ($t in $Tasks) {
        $wt = Get-Worktree $t.Id
        if ((Test-Path $wt) -and (Test-Merged $t.Id)) {
            $dirty = & git -C $wt status --porcelain 2>$null
            if ($dirty) { Write-Host "  keep $($t.Id): worktree has uncommitted changes"; continue }
            Invoke-Git worktree remove $wt | Out-Null
            Write-Host "  removed worktree $wt (branch $(Get-Branch $t.Id) kept)"
        }
    }
    exit 0
}

# ---------------------------------------------------------------- scheduler loop

function Get-Rows {
    @(foreach ($t in $Selected) {
        [pscustomobject]@{
            Id      = $t.Id
            State   = Get-TaskState $t
            Agent   = $t.Agent
            Depth   = $Depth[$t.Id]
            Waiting = ((Get-Waiting $t) -join ',')
        }
    })
}

# Build every prompt once now, so a bad character fails here and not mid-run.
foreach ($t in $Selected) { [void](New-Prompt $t) }

if (-not $Status -and -not $DryRun) {
    $host.UI.RawUI.WindowTitle = "orchestrator ($BaseRef)"
    Write-Log "orchestrator: base $BaseRef, prefix $Name, max $MaxParallel workers, permission mode $PermissionMode, state in $StateDir" 'Cyan'
}

while ($true) {
    $MergedCache = @{}
    Update-Base
    $rows = Get-Rows

    if (-not $Status -and -not $DryRun) {
        $changed = $false
        foreach ($r in $rows) {
            $t = $ById[$r.Id]
            if ($r.State -eq 'finished' -and $Integrate) {
                Invoke-Integrate $t; $changed = $true
            } elseif ($r.State -eq 'stopped') {
                Request-Fix $t 'session interrupted' 'Your previous run was interrupted (window closed, killed or timed out). Check git status, then continue the task from where you stopped, following the original instructions above.'
                $changed = $true
            } elseif ($r.State -eq 'running') {
                $age = (Get-Date) - (Get-Item (Get-StatePath $r.Id 'launched')).LastWriteTime
                $procId = Get-RunnerPid $r.Id
                if ($procId -and $age.TotalMinutes -gt $TaskTimeoutMinutes) {
                    Write-Log "$($r.Id) ran over $TaskTimeoutMinutes minutes; stopping it" 'Yellow'
                    & taskkill.exe /T /F /PID $procId 2>$null | Out-Null
                    $changed = $true
                }
            }
        }
        if ($changed) { $MergedCache = @{}; $rows = Get-Rows }
    }

    $cooldown = $null
    if (-not $DryRun -and (Test-Path $StateDir)) { $cooldown = Get-CooldownUntil }

    if (-not $Status -and -not $cooldown) {
        $slots = $MaxParallel - @($rows | Where-Object { $_.State -eq 'running' }).Count
        foreach ($r in $rows) {
            if ($slots -le 0) { break }
            if ($r.State -notin @('ready', 'resume')) { continue }
            Start-Task $ById[$r.Id]
            if (-not $DryRun) { $r.State = 'running' }
            $slots--
        }
    }

    $counts = ($rows | Group-Object State | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join '  '
    Write-Host ''
    $line = "[{0}] {1} @ {2}  {3}" -f (Get-Date -Format 'HH:mm'), $BaseRef, (Invoke-Git rev-parse --short $BaseRef), $counts
    if ($cooldown) { $line += "  (usage limit: waiting until $($cooldown.ToString('HH:mm')))" }
    Write-Host $line -ForegroundColor Cyan
    $show = $rows | Where-Object { $_.State -in @('running', 'finished', 'stopped', 'ready', 'resume', 'failed') }
    if ($Status) { $show = $rows | Where-Object { $_.State -ne 'merged' } }
    $show | Sort-Object Depth, { [array]::IndexOf($Selected, $ById[$_.Id]) } | ForEach-Object {
        $note = $_.Waiting
        if ($_.State -eq 'failed') { $note = "$(Get-Content (Get-StatePath $_.Id 'failed') -TotalCount 1); see $StateDir\$($_.Id).report.md, then -Retry $($_.Id)" }
        if ($_.State -eq 'finished' -and -not $Integrate) { $note = 'review and merge, then -MarkMerged if squash-merged' }
        if ($_.State -eq 'running') { $a = Get-Attempts $_.Id; if ($a -gt 0) { $note = "fix attempt $a" } }
        Write-Host ("  {0,-6} {1,-9} {2,-25} {3}" -f $_.Id, $_.State, $_.Agent, $note)
    }

    $open = @($rows | Where-Object { $_.State -ne 'merged' })
    if ($open.Count -eq 0) {
        Write-Log 'All selected tasks are merged.' 'Green'
        break
    }
    $active = @($open | Where-Object { $_.State -in @('running', 'ready', 'resume', 'finished', 'stopped') })
    if ($active.Count -eq 0 -and -not $Status -and -not $DryRun) {
        $failed = @($open | Where-Object { $_.State -eq 'failed' } | ForEach-Object { $_.Id })
        Write-Log "Nothing left that can run. Failed: $($failed -join ', '). Fix them, then: start-agents.bat -Retry <ID>, and start the orchestrator again." 'Red'
        break
    }
    if ($Status -or $DryRun -or $Once) { break }
    Start-Sleep -Seconds $PollSeconds
}
