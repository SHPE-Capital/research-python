@echo off
setlocal
pushd "%~dp0" || exit /b 1
where uv >nul 2>&1
if errorlevel 1 goto :missing
call uv sync --locked --python 3.12
if errorlevel 1 goto :failed
call uv run --locked python scripts\momentum\rm9.py demo --root runs\rm9-demo --workers 2
if errorlevel 1 goto :failed
call uv run --locked python scripts\momentum\rm9.py verify --root runs\rm9-demo --deep
if errorlevel 1 goto :failed
echo.
echo PASS. Open runs\rm9-demo\job\results.html for the synthetic result.
echo For a clean independent rerun see MOMENTUM_HANDOFF.md.
if not defined RM9_NO_OPEN start "" explorer.exe "%CD%\runs\rm9-demo\job"
echo.
pause
popd
exit /b 0

:missing
echo Install uv for Windows from https://docs.astral.sh/uv/getting-started/installation/
echo Close and reopen the terminal after installing, then run this file again.
pause
popd
exit /b 2

:failed
echo The demo stopped. Keep the output and check runs\rm9-demo\job\status.json.
echo See the troubleshooting table in the Juan runbook. Do not delete the run.
pause
popd
exit /b 1
