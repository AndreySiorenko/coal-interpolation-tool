@echo off
title Installing Dependencies with Python 3.13
color 0A

echo.
echo ================================================================
echo  📦 УСТАНОВКА ЗАВИСИМОСТЕЙ ДЛЯ ПОЛНОГО РЕЖИМА
echo  🐍 Используя Python 3.13 из C:\Python313
echo ================================================================
echo.

echo 🔍 Проверка Python по пути C:\Python313...

:: Проверяем Python по конкретному пути
if exist "C:\Python313\python.exe" (
    echo ✅ Python найден в C:\Python313
    "C:\Python313\python.exe" --version
    echo.
    goto install_deps
) else (
    echo ❌ Python не найден в C:\Python313\python.exe
    echo.
    goto check_path
)

:check_path
echo 🔍 Проверка Python в PATH...
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python найден в PATH
    python --version
    set PYTHON_CMD=python
    goto install_deps
) else (
    echo ⚠️  Python не найден в PATH
    echo.
    echo 📝 Добавьте Python в PATH:
    echo    1. Win+X → Система → Дополнительные параметры системы
    echo    2. Переменные среды → Path → Изменить
    echo    3. Добавить: C:\Python313
    echo    4. Добавить: C:\Python313\Scripts
    echo    5. OK → Перезапустить командную строку
    echo.
    echo 🔄 Или используем полный путь...
    set PYTHON_CMD="C:\Python313\python.exe"
    goto install_deps
)

:install_deps
echo.
echo 📥 Установка основных зависимостей для полного режима...
echo.

:: Сначала обновим pip
echo [0/6] Обновление pip...
C:\Python313\python.exe -m pip install --upgrade pip

echo.
echo [1/6] Установка NumPy (численные вычисления)...
C:\Python313\python.exe -m pip install numpy

echo.
echo [2/6] Установка Pandas (работа с данными)...
C:\Python313\python.exe -m pip install pandas

echo.
echo [3/6] Установка SciPy (научные вычисления)...
C:\Python313\python.exe -m pip install scipy

echo.
echo [4/6] Установка Matplotlib (визуализация)...
C:\Python313\python.exe -m pip install matplotlib

echo.
echo [5/6] Установка OpenPyXL (работа с Excel)...
C:\Python313\python.exe -m pip install openpyxl

echo.
echo [6/6] Установка xlrd (чтение Excel)...
C:\Python313\python.exe -m pip install xlrd

echo.
echo 🎯 Проверка установки...
C:\Python313\python.exe -c "import numpy, pandas, scipy, matplotlib; print('✅ Все основные библиотеки установлены!')"

echo.
echo ================================================================
echo  ✅ УСТАНОВКА ЗАВЕРШЕНА!
echo ================================================================
echo.
echo Теперь вы можете запустить программу в ПОЛНОМ РЕЖИМЕ:
echo.
echo Вариант 1 (с полным путем):
echo   C:\Python313\python.exe main.py
echo.
echo Вариант 2 (если Python в PATH):
echo   python main.py
echo.
echo Или просто запустите:
echo   start_with_python313.bat
echo.
echo ================================================================
echo.
pause