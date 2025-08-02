@echo off
title Fix Circular Import and Run
color 0A

echo.
echo ================================================================
echo  🔧 ИСПРАВЛЕНИЕ ЦИКЛИЧЕСКОГО ИМПОРТА И ЗАПУСК
echo ================================================================
echo.

cd /d "%~dp0"

set VENV_PYTHON=C:\Users\nirst\Desktop\Interpolation\.venv1\Scripts\python.exe

echo 🔧 Исправление циклического импорта...
echo.

"%VENV_PYTHON%" fix_circular_import.py

echo.
echo 🚀 Запуск программы...
echo.

"%VENV_PYTHON%" main.py

echo.
pause