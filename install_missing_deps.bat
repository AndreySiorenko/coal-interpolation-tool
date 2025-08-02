@echo off
title Installing Missing Dependencies
color 0A

echo.
echo ================================================================
echo  📦 УСТАНОВКА ВСЕХ НЕДОСТАЮЩИХ ЗАВИСИМОСТЕЙ
echo ================================================================
echo.

set VENV_PIP=C:\Users\nirst\Desktop\Interpolation\.venv1\Scripts\pip.exe

echo 📥 Установка недостающих библиотек...
echo.

echo [1/15] chardet (кодировки)...
"%VENV_PIP%" install chardet

echo.
echo [2/15] requests (HTTP запросы)...
"%VENV_PIP%" install requests

echo.
echo [3/15] urllib3 (HTTP клиент)...
"%VENV_PIP%" install urllib3

echo.
echo [4/15] certifi (SSL сертификаты)...
"%VENV_PIP%" install certifi

echo.
echo [5/15] idna (интернационализация доменов)...
"%VENV_PIP%" install idna

echo.
echo [6/15] pytz (временные зоны)...
"%VENV_PIP%" install pytz

echo.
echo [7/15] six (совместимость Python 2/3)...
"%VENV_PIP%" install six

echo.
echo [8/15] pillow (изображения)...
"%VENV_PIP%" install pillow

echo.
echo [9/15] packaging (пакеты)...
"%VENV_PIP%" install packaging

echo.
echo [10/15] setuptools (инструменты)...
"%VENV_PIP%" install setuptools

echo.
echo [11/15] wheel (колеса пакетов)...
"%VENV_PIP%" install wheel

echo.
echo [12/15] joblib (параллельность)...
"%VENV_PIP%" install joblib

echo.
echo [13/15] threadpoolctl (управление потоками)...
"%VENV_PIP%" install threadpoolctl

echo.
echo [14/15] kiwisolver (решатель ограничений)...
"%VENV_PIP%" install kiwisolver

echo.
echo [15/15] cycler (циклы цветов)...
"%VENV_PIP%" install cycler

echo.
echo ================================================================
echo  ✅ ВСЕ ЗАВИСИМОСТИ УСТАНОВЛЕНЫ!
echo ================================================================
echo.
echo Теперь запустите: run_fixed.bat
echo.
pause