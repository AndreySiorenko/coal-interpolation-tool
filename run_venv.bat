@echo off
title Coal Interpolation Tool - Virtual Environment
color 0A

echo.
echo ================================================================
echo  🎯 COAL DEPOSIT INTERPOLATION TOOL v1.0.0-rc1
echo  🐍 Используется виртуальное окружение
echo ================================================================
echo.

cd /d "%~dp0"

set VENV_PATH=C:\Users\nirst\Desktop\Interpolation\.venv1
set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe

echo 📁 Рабочая директория: %CD%
echo 🐍 Python: %PYTHON_EXE%
echo.

if exist "%PYTHON_EXE%" (
    echo ✅ Виртуальное окружение найдено
    "%PYTHON_EXE%" --version
    echo.
    
    echo 🚀 Запуск программы...
    echo.
    
    "%PYTHON_EXE%" main.py
    
) else (
    echo ❌ ОШИБКА: Виртуальное окружение не найдено!
    echo    Ожидаемый путь: %VENV_PATH%
)

echo.
echo ✅ Программа завершена.
echo.
pause