@echo off
title Installing Dependencies for Full Mode
color 0A

echo.
echo ================================================================
echo  📦 УСТАНОВКА ЗАВИСИМОСТЕЙ ДЛЯ ПОЛНОГО РЕЖИМА
echo ================================================================
echo.

echo 🔍 Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден! Установите Python 3.8+
    echo    https://python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python найден
python --version
echo.

echo 📥 Установка основных зависимостей для полного режима...
echo.

echo [1/4] Установка NumPy (численные вычисления)...
pip install numpy

echo.
echo [2/4] Установка Pandas (работа с данными)...
pip install pandas

echo.
echo [3/4] Установка SciPy (научные вычисления)...
pip install scipy

echo.
echo [4/4] Установка Matplotlib (визуализация)...
pip install matplotlib

echo.
echo 🎯 Установка дополнительных компонентов...
pip install openpyxl xlrd

echo.
echo ================================================================
echo  ✅ УСТАНОВКА ЗАВЕРШЕНА!
echo ================================================================
echo.
echo Теперь вы можете запустить программу в ПОЛНОМ РЕЖИМЕ:
echo   - Запустите start.bat
echo   - Или выполните: python main.py
echo.
echo ================================================================
echo.
pause