@echo off
title Coal Deposit Interpolation Tool v1.0.0-rc1
color 0A

echo.
echo ================================================================
echo  🎯 COAL DEPOSIT INTERPOLATION TOOL v1.0.0-rc1
echo ================================================================
echo  🐍 Professional geological data analysis tool
echo  ✨ Ready for production use
echo ================================================================
echo.

echo 🔍 Checking Python installation...

:: Check for Python installations
set "PYTHON_FOUND=0"
set "PYTHON_CMD="

:: Try python
python --version >nul 2>&1
if %errorlevel% == 0 (
    set "PYTHON_FOUND=1"
    set "PYTHON_CMD=python"
    echo ✅ Python found: python
    python --version
    goto :run_app
)

:: Try python3
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    set "PYTHON_FOUND=1"
    set "PYTHON_CMD=python3"
    echo ✅ Python found: python3
    python3 --version
    goto :run_app
)

:: Try py launcher
py --version >nul 2>&1
if %errorlevel% == 0 (
    set "PYTHON_FOUND=1"
    set "PYTHON_CMD=py"
    echo ✅ Python found: py launcher
    py --version
    goto :run_app
)

:: No Python found
echo.
echo ❌ ERROR: Python not found!
echo.
echo 📥 Please install Python 3.8+ from:
echo    https://python.org/downloads/
echo.
echo ✅ Make sure to check "Add Python to PATH" during installation
echo.
echo 🔄 Alternative: Install from Microsoft Store
echo    Search for "Python 3.11" in Microsoft Store
echo.
goto :end

:run_app
echo.
echo 🚀 Starting Coal Interpolation Tool...
echo.

:: Change to the script's directory
cd /d "%~dp0"
echo 📁 Working directory: %CD%

:: Try to run the launcher
if exist "launch.py" (
    echo 📱 Using smart launcher...
    %PYTHON_CMD% launch.py
) else if exist "demo.py" (
    echo 🎮 Running demo version...
    %PYTHON_CMD% demo.py
) else if exist "main.py" (
    echo 🎯 Running main application...
    %PYTHON_CMD% main.py
) else (
    echo ❌ ERROR: No Python files found!
    echo    Current directory: %CD%
    echo    Looking for: launch.py, demo.py, main.py
    dir *.py
)

echo.
echo ✅ Application finished.

:end
echo.
echo Press any key to exit...
pause >nul