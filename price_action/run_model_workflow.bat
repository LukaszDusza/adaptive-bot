@echo off
REM ML Trading Workflow - Windows Launcher
REM This script launches the Python-based workflow tool

echo.
echo ========================================
echo   ML Trading Workflow
echo ========================================
echo.

REM Activate virtual environment
if exist "..\.venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "..\.venv\Scripts\activate.bat"
) else (
    echo WARNING: Virtual environment not found at ..\.venv
    echo Attempting to use system Python...
)

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Run the Python workflow script
python run_model_workflow.py

REM Pause if there was an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Script exited with error code: %ERRORLEVEL%
    pause
)
