@echo off
title Coal Interpolation - Safe Demo Mode
color 0A

echo.
echo ================================================================
echo  🎮 ЗАПУСК ДЕМО ВЕРСИИ (БЕЗ ЗАВИСИМОСТЕЙ)
echo ================================================================
echo.

cd /d "%~dp0"

echo 📁 Текущая директория: %CD%
echo.

:: Пробуем разные способы запуска Python
echo 🔍 Поиск Python...

:: 1. python
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Используем "python"
    echo.
    echo 🎮 Запуск демо интерфейса...
    python demo.py
    goto end
)

:: 2. python3
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Используем "python3"
    echo.
    echo 🎮 Запуск демо интерфейса...
    python3 demo.py
    goto end
)

:: 3. py
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Используем "py"
    echo.
    echo 🎮 Запуск демо интерфейса...
    py demo.py
    goto end
)

:: 4. C:\Python313\python.exe
if exist "C:\Python313\python.exe" (
    echo ✅ Используем Python 3.13 из C:\Python313
    echo.
    echo 🎮 Запуск демо интерфейса...
    "C:\Python313\python.exe" demo.py
    goto end
)

:: 5. Проверяем стандартные пути
for %%v in (312 311 310 39 38) do (
    if exist "C:\Python%%v\python.exe" (
        echo ✅ Найден Python в C:\Python%%v
        echo.
        echo 🎮 Запуск демо интерфейса...
        "C:\Python%%v\python.exe" demo.py
        goto end
    )
)

:: 6. Проверяем пути пользователя
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    echo ✅ Найден Python 3.12 в профиле пользователя
    echo.
    echo 🎮 Запуск демо интерфейса...
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" demo.py
    goto end
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    echo ✅ Найден Python 3.11 в профиле пользователя
    echo.
    echo 🎮 Запуск демо интерфейса...
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" demo.py
    goto end
)

:: Python не найден
echo ❌ Python не найден в системе!
echo.
echo Пожалуйста, установите Python:
echo 1. Скачайте с https://python.org
echo 2. При установке отметьте "Add Python to PATH"
echo 3. Перезапустите этот файл

:end
echo.
echo ================================================================
echo.
pause