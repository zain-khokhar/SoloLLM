"""
SoloLLM Backend — Silent Startup Script
========================================
This script is designed to be run at Windows logon via Task Scheduler.
It starts the uvicorn backend server silently (no console window) so
the backend is warmed up by the time the user opens the application.

File extension .pyw ensures Python runs without a console window.
"""

import subprocess
import sys
import os
from pathlib import Path

# Resolve project paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
LOG_FILE = PROJECT_ROOT / "data" / "backend_autostart.log"

# Ensure the data directory exists for logs
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Find the pythonw.exe next to whichever python.exe is running this script.
# pythonw.exe is the windowless Python interpreter on Windows.
python_dir = Path(sys.executable).parent
pythonw = python_dir / "pythonw.exe"
if not pythonw.exists():
    pythonw = Path(sys.executable)  # fallback to python.exe

with open(LOG_FILE, "w", encoding="utf-8") as log:
    try:
        subprocess.Popen(
            [
                str(pythonw), "-m", "uvicorn",
                "main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
            ],
            cwd=str(BACKEND_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception as e:
        log.write(f"Failed to start backend: {e}\n")
