@echo off
title CSV Loading Test
color 0A

echo.
echo ================================================================
echo  🧪 ТЕСТИРОВАНИЕ ЗАГРУЗКИ CSV ФАЙЛА
echo ================================================================
echo.

cd /d "%~dp0"

echo 📥 Тестирование файла "Скважины устья.csv"...
echo.

C:\Users\nirst\Desktop\Interpolation\.venv1\Scripts\python.exe test_specific_csv.py

echo.
echo ================================================================
echo.
pause