@echo off
title Alternative Python Installation Check
color 0A

echo.
echo ================================================================
echo  🔍 ПРОВЕРКА АЛЬТЕРНАТИВНЫХ УСТАНОВОК PYTHON
echo ================================================================
echo.

echo Проверяем различные варианты Python...
echo.

:: Проверка Python из PATH
echo [1] Проверка "python" в PATH:
python --version 2>nul
if %errorlevel% == 0 (
    echo ✅ Найден Python в PATH
    echo.
    echo Установка библиотек через этот Python:
    python -m pip install numpy pandas scipy matplotlib openpyxl xlrd
    echo.
    echo После установки запустите: python main.py
) else (
    echo ❌ Не найден
)

echo.

:: Проверка Python3
echo [2] Проверка "python3":
python3 --version 2>nul
if %errorlevel% == 0 (
    echo ✅ Найден Python3
    echo.
    echo Установка библиотек:
    python3 -m pip install numpy pandas scipy matplotlib openpyxl xlrd
) else (
    echo ❌ Не найден
)

echo.

:: Проверка py launcher
echo [3] Проверка "py" launcher:
py --version 2>nul
if %errorlevel% == 0 (
    echo ✅ Найден py launcher
    echo.
    echo Установка библиотек:
    py -m pip install numpy pandas scipy matplotlib openpyxl xlrd
    echo.
    echo После установки запустите: py main.py
) else (
    echo ❌ Не найден
)

echo.

:: Проверка стандартных путей установки
echo [4] Проверка стандартных путей установки:
echo.

if exist "C:\Python312\python.exe" (
    echo ✅ Python 3.12 найден в C:\Python312
    C:\Python312\python.exe --version
)

if exist "C:\Python311\python.exe" (
    echo ✅ Python 3.11 найден в C:\Python311
    C:\Python311\python.exe --version
)

if exist "C:\Python310\python.exe" (
    echo ✅ Python 3.10 найден в C:\Python310
    C:\Python310\python.exe --version
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    echo ✅ Python 3.12 найден в %LOCALAPPDATA%\Programs\Python\Python312
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" --version
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    echo ✅ Python 3.11 найден в %LOCALAPPDATA%\Programs\Python\Python311
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" --version
)

echo.
echo ================================================================
echo  📌 РЕКОМЕНДАЦИИ:
echo ================================================================
echo.
echo 1. Если Python найден - используйте соответствующую команду выше
echo 2. Если Python 3.13 не работает - установите Python 3.11 или 3.12
echo    с официального сайта: https://python.org
echo 3. При установке ОБЯЗАТЕЛЬНО отметьте:
echo    ✅ Add Python to PATH
echo    ✅ Install pip
echo.
echo ================================================================
echo.
pause