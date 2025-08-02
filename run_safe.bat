@echo off
title Coal Interpolation Tool - Safe Mode
color 0A

echo.
echo ================================================================
echo  COAL DEPOSIT INTERPOLATION TOOL v1.0.0-rc1
echo  Safe Mode - Problematic modules disabled
echo ================================================================
echo.

cd /d "%~dp0"

echo Запуск программы в безопасном режиме...
echo.

C:\Users\nirst\Desktop\Interpolation\.venv1\Scripts\python.exe main.py

echo.
echo Программа завершена.
echo.
pause