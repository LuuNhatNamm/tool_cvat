@echo off
setlocal
cd /d "%~dp0"
where pythonw >nul 2>nul
if not errorlevel 1 (
    start "" pythonw "%~dp0src\cvat_job_runner.py"
) else (
    python "%~dp0src\cvat_job_runner.py"
    if errorlevel 1 pause
)
