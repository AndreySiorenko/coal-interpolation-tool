@echo off
title Installing Dependencies in Virtual Environment
color 0A

echo.
echo ================================================================
echo  📦 УСТАНОВКА БИБЛИОТЕК В ВИРТУАЛЬНОЕ ОКРУЖЕНИЕ
echo  🐍 Путь: C:\Users\nirst\Desktop\Interpolation\.venv1
echo ================================================================
echo.

set VENV_PATH=C:\Users\nirst\Desktop\Interpolation\.venv1
set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe
set PIP_EXE=%VENV_PATH%\Scripts\pip.exe

cd /d "C:\Users\nirst\Desktop\Interpolation\Interpolation\Interpolation"

if not exist "%PYTHON_EXE%" (
    echo ❌ Виртуальное окружение не найдено!
    echo    Путь: %VENV_PATH%
    pause
    exit /b 1
)

echo ✅ Виртуальное окружение найдено
"%PYTHON_EXE%" --version
echo.

echo 📥 Обновление pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo.
echo 📥 Установка основных библиотек...
echo.

echo [1/6] NumPy (численные вычисления)...
"%PIP_EXE%" install numpy

echo.
echo [2/6] Pandas (работа с данными)...
"%PIP_EXE%" install pandas

echo.
echo [3/6] SciPy (научные вычисления)...
"%PIP_EXE%" install scipy

echo.
echo [4/6] Matplotlib (визуализация)...
"%PIP_EXE%" install matplotlib

echo.
echo [5/6] OpenPyXL (работа с Excel)...
"%PIP_EXE%" install openpyxl

echo.
echo [6/6] xlrd (чтение Excel)...
"%PIP_EXE%" install xlrd

echo.
echo 🔍 Проверка установки...
"%PYTHON_EXE%" -c "import numpy, pandas, scipy, matplotlib; print('✅ Все основные библиотеки установлены!')"

echo.
echo ================================================================
echo  ✅ УСТАНОВКА ЗАВЕРШЕНА!
echo ================================================================
echo.
echo Теперь запустите программу:
echo %PYTHON_EXE% main.py
echo.
echo Или используйте run_venv.bat
echo.
pause