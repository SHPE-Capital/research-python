@echo off
rem Starts the agent orchestrator. All arguments go to orchestrate.ps1, e.g.
rem   start-agents.bat -Plan
rem   start-agents.bat -DryRun
rem   start-agents.bat -MaxParallel 5
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0orchestrate.ps1" %*
