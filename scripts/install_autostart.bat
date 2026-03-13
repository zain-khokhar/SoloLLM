@echo off
:: ============================================================
:: SoloLLM — Install Auto-Start at Windows Logon
:: ============================================================
:: Registers a Windows Task Scheduler task that silently starts
:: the SoloLLM backend whenever the current user logs in.
:: Run this script once. To undo, run uninstall_autostart.bat.
:: ============================================================

set TASK_NAME=SoloLLM_Backend_Autostart

:: Resolve paths
set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..") do set PROJECT_ROOT=%%~fI

:: Find pythonw.exe (windowless Python)
for /f "delims=" %%P in ('where pythonw 2^>nul') do set PYTHONW=%%P
if not defined PYTHONW (
    for /f "delims=" %%P in ('where python 2^>nul') do set PYTHONW=%%P
)

if not defined PYTHONW (
    echo ERROR: Could not find python or pythonw on PATH.
    pause
    exit /b 1
)

set STARTUP_SCRIPT=%SCRIPT_DIR%start_backend.pyw

echo.
echo  SoloLLM Auto-Start Installer
echo  ─────────────────────────────
echo  Task name  : %TASK_NAME%
echo  Python     : %PYTHONW%
echo  Script     : %STARTUP_SCRIPT%
echo.

:: Create the scheduled task (runs at logon of current user, no admin needed)
schtasks /create /tn "%TASK_NAME%" ^
    /tr "\"%PYTHONW%\" \"%STARTUP_SCRIPT%\"" ^
    /sc onlogon ^
    /rl limited ^
    /f

if %errorlevel% equ 0 (
    echo.
    echo  SUCCESS: SoloLLM backend will now auto-start at logon.
    echo  To remove, run uninstall_autostart.bat
) else (
    echo.
    echo  FAILED: Could not create scheduled task.
    echo  Try running this script as Administrator.
)

echo.
pause
