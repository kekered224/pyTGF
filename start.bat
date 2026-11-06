@echo off
chcp 65001 >nul 2>&1
title pyTGF Web UI

echo ============================================================
echo   pyTGF - Terrain Gravity Field Calculation (Python)
echo   Web Interface Launcher
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8+ and add to PATH.
    echo         Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if package is installed
python -c "import pyTGF" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] pyTGF not installed. Installing in development mode...
    pip install -e . --quiet
    if %errorlevel% neq 0 (
        echo [ERROR] Installation failed. Please run: pip install -e .
        pause
        exit /b 1
    )
    echo [OK] Installation complete.
    echo.
)

echo Starting web server on http://127.0.0.1:5000
echo.
echo The browser will open automatically.
echo Press Ctrl+C to stop the server.
echo.

REM Start browser after a short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:5000"

REM Launch the Flask server
python -m pyTGF.web

pause
