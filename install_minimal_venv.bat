@echo off
title Quick Install - Minimal Dependencies
color 0A

echo.
echo ================================================================
echo  ⚡ БЫСТРАЯ УСТАНОВКА МИНИМАЛЬНЫХ ЗАВИСИМОСТЕЙ
echo ================================================================
echo.

cd /d "C:\Users\nirst\Desktop\Interpolation\Interpolation\Interpolation"

set VENV_PYTHON=C:\Users\nirst\Desktop\Interpolation\.venv1\Scripts\python.exe

if exist "%VENV_PYTHON%" (
    echo ✅ Используем виртуальное окружение
    echo.
    echo 📥 Установка из requirements-minimal.txt...
    "%VENV_PYTHON%" -m pip install -r requirements-minimal.txt
) else (
    echo ⚠️  Виртуальное окружение не найдено
    echo 🔍 Пробуем обычный Python...
    echo.
    python -m pip install -r requirements-minimal.txt
)

echo.
echo ✅ Готово! Запустите run_venv.bat
echo.
pause