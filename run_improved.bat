@echo off
title Coal Interpolation Tool - Improved Error Handling
color 0A

echo.
echo ================================================================
echo  COAL DEPOSIT INTERPOLATION TOOL v1.0.0-rc1
echo  Improved CSV Error Handling
echo ================================================================
echo.

cd /d "%~dp0"

echo Available files for testing:
echo    - wells_data_fixed.csv (UTF-8 encoded)
echo    - Скважины устья.csv (original file)
echo.

echo Запуск программы с улучшенной обработкой ошибок...
echo.

C:\Users\nirst\Desktop\Interpolation\.venv1\Scripts\python.exe main.py

echo.
echo Программа завершена.
echo.
echo Советы при проблемах с CSV:
echo    1. Попробуйте файл wells_data_fixed.csv
echo    2. В настройках попробуйте кодировку cp1251
echo    3. Проверьте разделитель (запятая/точка с запятой)
echo.
pause