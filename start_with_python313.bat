@echo off
title Coal Interpolation Tool - Python 3.13
color 0A

echo.
echo ================================================================
echo  🎯 COAL DEPOSIT INTERPOLATION TOOL v1.0.0-rc1
echo  🐍 Using Python 3.13 from C:\Python313
echo ================================================================
echo.

cd /d "%~dp0"

echo 📁 Working directory: %CD%
echo.

if exist "C:\Python313\python.exe" (
    echo ✅ Python 3.13 найден
    echo.
    
    echo 🚀 Запуск программы...
    echo.
    
    :: Пробуем запустить через launch.py для автоопределения режима
    if exist "launch.py" (
        echo 📱 Использование умного лаунчера...
        C:\Python313\python.exe launch.py
    ) else if exist "main.py" (
        echo 🎯 Запуск основного приложения...
        C:\Python313\python.exe main.py
    ) else if exist "demo.py" (
        echo 🎮 Запуск демо версии...
        C:\Python313\python.exe demo.py
    ) else (
        echo ❌ Файлы приложения не найдены!
    )
) else (
    echo ❌ ОШИБКА: Python не найден в C:\Python313
    echo.
    echo Проверьте установку Python или путь к нему.
)

echo.
echo ✅ Приложение завершено.
echo.
pause