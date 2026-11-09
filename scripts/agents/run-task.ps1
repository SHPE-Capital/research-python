<#
.SYNOPSIS
    Runs one task's Claude Code worker. orchestrate.ps1 opens this in its own
    terminal tab; you do not normally call it yourself.

.DESCRIPTION
    Runs claude headless (-p) with the prompt in -PromptFile and prints a
    readable live log. The raw event stream goes to <StateDir>\<Id>.log.jsonl
    and the final report is appended to <StateDir>\<Id>.report.md.

    Writes <Id>.pid while running and <Id>.exited when done, with
    status=ok|error|ratelimit, which the orchestrator reads.

    -Continue resumes the most recent conversation in the worktree.
    -Interactive opens a normal interactive session instead (used by -Resume).
#>
param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Worktree,
    [Parameter(Mandatory = $true)][string]$StateDir,
    [string]$PromptFile,
    [string]$Agent,
    [string]$Label,
    [string]$PermissionMode = 'auto',
    [switch]$Continue,
    [switch]$Interactive
)

$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "$Id - $Label"
Set-Location $Worktree

$pidFile = Join-Path $StateDir "$Id.pid"
$exitFile = Join-Path $StateDir "$Id.exited"
$logFile = Join-Path $StateDir "$Id.log.jsonl"
$reportFile = Join-Path $StateDir "$Id.report.md"
Set-Content -Path $pidFile -Value $PID -Encoding ASCII

Write-Host "Task $Id ($Label) in $Worktree" -ForegroundColor Cyan
Write-Host "Branch: $(git branch --show-current)"
Write-Host ''

if (-not $Continue) {
    Write-Host 'uv sync ...'
    & uv sync --quiet
    if ($LASTEXITCODE -ne 0) { Write-Host 'uv sync failed; the worker can still fix it.' -ForegroundColor Yellow }
}

function Format-ToolInput($name, $inp) {
    if ($null -eq $inp) { return '' }
    foreach ($key in @('command', 'file_path', 'path', 'pattern', 'description', 'prompt')) {
        $p = $inp.PSObject.Properties[$key]
        if ($p -and $p.Value) {
            $s = "$($p.Value)" -replace '\s+', ' '
            if ($s.Length -gt 140) { $s = $s.Substring(0, 140) + '...' }
            return $s
        }
    }
    return ''
}

$status = 'error'
$resetsAt = 0
$cost = ''

if ($Interactive) {
    & claude --permission-mode $PermissionMode --continue
    $code = $LASTEXITCODE
    $status = 'ok'
} else {
    $prompt = (Get-Content -Path $PromptFile -Raw).Trim()
    $claudeArgs = @('-p', '--output-format', 'stream-json', '--verbose', '--permission-mode', $PermissionMode)
    if ($Continue) {
        $claudeArgs += '--continue'
    } elseif ($Agent) {
        $claudeArgs += @('--agent', $Agent)
    }
    $claudeArgs += $prompt

    $writer = New-Object IO.StreamWriter($logFile, $true, (New-Object Text.UTF8Encoding($false)))
    $result = $null
    $limited = $false
    try {
        & claude @claudeArgs | ForEach-Object {
            $line = $_
            $writer.WriteLine($line)
            $writer.Flush()
            try { $e = $line | ConvertFrom-Json } catch { Write-Host $line; return }
            switch ($e.type) {
                'assistant' {
                    foreach ($c in @($e.message.content)) {
                        if ($c.type -eq 'text' -and $c.text) {
                            Write-Host $c.text
                        } elseif ($c.type -eq 'tool_use') {
                            Write-Host ("  > {0} {1}" -f $c.name, (Format-ToolInput $c.name $c.input)) -ForegroundColor DarkGray
                        }
                    }
                }
                'rate_limit_event' {
                    $info = $e.rate_limit_info
                    if ($info -and $info.status -and $info.status -notlike 'allowed*') {
                        $limited = $true
                        if ($info.resetsAt) { $resetsAt = [long]$info.resetsAt }
                        Write-Host "  usage limit reached ($($info.status))" -ForegroundColor Yellow
                    }
                }
                'result' { $result = $e }
            }
        }
        $code = $LASTEXITCODE
    } finally {
        $writer.Close()
    }

    $isError = $true
    if ($result) {
        $isError = [bool]$result.is_error
        if ($result.PSObject.Properties['total_cost_usd']) { $cost = '{0:N2}' -f $result.total_cost_usd }
        $text = ''
        if ($result.PSObject.Properties['result']) { $text = "$($result.result)" }
        if ($text -match '(usage|rate) limit|limit reached|resets at') { if ($isError) { $limited = $true } }
        Add-Content -Path $reportFile -Encoding UTF8 -Value @("## $Id run ended $(Get-Date -Format o)", '', $text, '')
    }
    if ($limited -and ($isError -or $code -ne 0)) {
        $status = 'ratelimit'
    } elseif (-not $isError -and $code -eq 0) {
        $status = 'ok'
    }
}

Set-Content -Path $exitFile -Encoding ASCII -Value @("status=$status", "code=$code", "resetsAt=$resetsAt", "cost=$cost", "time=$(Get-Date -Format s)")
Remove-Item $pidFile -ErrorAction SilentlyContinue

Write-Host ''
$color = 'Green'
if ($status -ne 'ok') { $color = 'Yellow' }
Write-Host "Worker for $Id ended: $status (exit $code, cost $cost USD)." -ForegroundColor $color
Write-Host 'The orchestrator now checks and merges it, or resumes it with a fix prompt.'
Write-Host "Report: $reportFile"
