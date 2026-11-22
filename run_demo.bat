@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment is missing. See README.md for setup instructions.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
