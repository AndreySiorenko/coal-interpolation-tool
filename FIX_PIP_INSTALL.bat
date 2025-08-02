@echo off
title Fixing Python Installation - Installing pip
color 0A

echo.
echo ================================================================
echo  🔧 ИСПРАВЛЕНИЕ УСТАНОВКИ PYTHON - УСТАНОВКА PIP
echo ================================================================
echo.

echo 📥 Скачивание get-pip.py...
echo.

:: Скачиваем get-pip.py используя PowerShell
powershell -Command "Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py"

if exist get-pip.py (
    echo ✅ get-pip.py успешно скачан
    echo.
    echo 📦 Установка pip...
    echo.
    
    C:\Python313\python.exe get-pip.py
    
    echo.
    echo 🔍 Проверка установки pip...
    C:\Python313\python.exe -m pip --version
    
    echo.
    echo 🗑️ Удаление временного файла...
    del get-pip.py
    
    echo.
    echo ✅ PIP УСТАНОВЛЕН!
    echo.
    echo ================================================================
    echo  📦 ТЕПЕРЬ УСТАНОВИМ БИБЛИОТЕКИ
    echo ================================================================
    echo.
    
    echo [1/6] Установка NumPy...
    C:\Python313\python.exe -m pip install numpy
    
    echo.
    echo [2/6] Установка Pandas...
    C:\Python313\python.exe -m pip install pandas
    
    echo.
    echo [3/6] Установка SciPy...
    C:\Python313\python.exe -m pip install scipy
    
    echo.
    echo [4/6] Установка Matplotlib...
    C:\Python313\python.exe -m pip install matplotlib
    
    echo.
    echo [5/6] Установка OpenPyXL...
    C:\Python313\python.exe -m pip install openpyxl
    
    echo.
    echo [6/6] Установка xlrd...
    C:\Python313\python.exe -m pip install xlrd
    
    echo.
    echo ✅ ВСЕ БИБЛИОТЕКИ УСТАНОВЛЕНЫ!
    echo.
    echo Теперь запустите start_with_python313.bat для запуска программы
    
) else (
    echo ❌ Ошибка при скачивании get-pip.py
    echo.
    echo Попробуйте:
    echo 1. Скачать вручную: https://bootstrap.pypa.io/get-pip.py
    echo 2. Сохранить в эту папку
    echo 3. Запустить: C:\Python313\python.exe get-pip.py
)

echo.
echo ================================================================
echo.
pause