@echo off
title Coal Interpolation Tool - Debug Mode
color 0A

echo.
echo ================================================================
echo  🎯 COAL DEPOSIT INTERPOLATION TOOL v1.0.0-rc1
echo  🐛 Debug Mode - Enhanced Error Reporting
echo ================================================================
echo.

cd /d "%~dp0"

echo 🚀 Запуск программы с расширенной диагностикой...
echo.

:: Run with verbose Python output
C:\Users\nirst\Desktop\Interpolation\.venv1\Scripts\python.exe -u main.py

echo.
echo ✅ Программа завершена.
echo.
echo 📝 Если были ошибки с загрузкой CSV:
echo    1. Проверьте формат файла CSV
echo    2. Убедитесь что есть заголовки столбцов  
echo    3. Проверьте кодировку файла (UTF-8 рекомендуется)
echo.
pause