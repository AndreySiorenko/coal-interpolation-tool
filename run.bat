@echo off
echo ========================================
echo Coal Interpolation Tool v1.0.0-rc1
echo ========================================
echo.
echo Checking Python installation...

:: Try different Python commands
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Python found: python
    python run_demo.py
    goto :end
)

python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Python found: python3
    python3 run_demo.py
    goto :end
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    echo Python found: py
    py run_demo.py
    goto :end
)

echo.
echo ERROR: Python not found in PATH
echo.
echo Please install Python 3.8+ from https://python.org
echo Make sure to check "Add Python to PATH" during installation
echo.
echo Alternative: Use Windows Store Python
echo.

:end
echo.
echo Press any key to exit...
pause >nul