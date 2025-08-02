@echo off
title Diagnostic and Run Tool
color 0A

echo.
echo ================================================================
echo  🔍 ДИАГНОСТИКА И ЗАПУСК ПРОГРАММЫ
echo ================================================================
echo.

echo 📊 Проверка Python установок...
echo.

:: Проверяем разные способы запуска Python
set PYTHON_FOUND=0
set PYTHON_CMD=

:: 1. Python из PATH
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ "python" найден в PATH
    python --version
    set PYTHON_FOUND=1
    set PYTHON_CMD=python
    goto check_pip
)

:: 2. Python3
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ "python3" найден
    python3 --version
    set PYTHON_FOUND=1
    set PYTHON_CMD=python3
    goto check_pip
)

:: 3. Py launcher
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ "py" launcher найден
    py --version
    set PYTHON_FOUND=1
    set PYTHON_CMD=py
    goto check_pip
)

:: 4. Python 3.13 по пути
if exist "C:\Python313\python.exe" (
    echo ✅ Python 3.13 найден в C:\Python313
    "C:\Python313\python.exe" --version 2>nul
    if %errorlevel% == 0 (
        set PYTHON_FOUND=1
        set PYTHON_CMD="C:\Python313\python.exe"
    ) else (
        echo ⚠️  Но есть проблемы с установкой
    )
)

if %PYTHON_FOUND% == 0 (
    echo ❌ Python не найден!
    echo.
    echo Установите Python с https://python.org
    echo При установке отметьте "Add Python to PATH"
    goto end
)

:check_pip
echo.
echo 📦 Проверка pip...
%PYTHON_CMD% -m pip --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ pip установлен
    %PYTHON_CMD% -m pip --version
) else (
    echo ❌ pip не установлен
    echo.
    echo Попробуйте установить pip:
    echo 1. Скачайте https://bootstrap.pypa.io/get-pip.py
    echo 2. Запустите: %PYTHON_CMD% get-pip.py
)

echo.
echo 🔍 Проверка библиотек...
echo.

%PYTHON_CMD% -c "import numpy; print('✅ NumPy установлен')" 2>nul || echo ❌ NumPy не установлен
%PYTHON_CMD% -c "import pandas; print('✅ Pandas установлен')" 2>nul || echo ❌ Pandas не установлен
%PYTHON_CMD% -c "import scipy; print('✅ SciPy установлен')" 2>nul || echo ❌ SciPy не установлен
%PYTHON_CMD% -c "import matplotlib; print('✅ Matplotlib установлен')" 2>nul || echo ❌ Matplotlib не установлен

echo.
echo ================================================================
echo  🚀 ПОПЫТКА ЗАПУСКА ПРОГРАММЫ
echo ================================================================
echo.

:: Проверяем наличие файлов
if exist "demo.py" (
    echo 🎮 Запуск ДЕМО версии (не требует библиотек)...
    echo.
    %PYTHON_CMD% demo.py
) else if exist "main.py" (
    echo 🎯 Запуск основной программы...
    echo Внимание: Если библиотеки не установлены, запустится в режиме MOCK
    echo.
    %PYTHON_CMD% main.py
) else (
    echo ❌ Файлы программы не найдены!
)

:end
echo.
echo ================================================================
echo.
pause