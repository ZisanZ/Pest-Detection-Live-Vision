@echo off
REM --- Run Pest Detector with virtual environment ---
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "cd 'G:\PestVision_1'; .\venv\Scripts\Activate.ps1; python pest_detector_live.py"
pause
