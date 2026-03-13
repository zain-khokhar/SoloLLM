@echo off
:: ============================================================
:: SoloLLM — Remove Auto-Start from Windows Logon
:: ============================================================

set TASK_NAME=SoloLLM_Backend_Autostart

echo.
echo  Removing scheduled task: %TASK_NAME%
echo.

schtasks /delete /tn "%TASK_NAME%" /f

if %errorlevel% equ 0 (
    echo.
    echo  SUCCESS: Auto-start has been removed.
) else (
    echo.
    echo  Task not found or could not be removed.
)

echo.
pause
