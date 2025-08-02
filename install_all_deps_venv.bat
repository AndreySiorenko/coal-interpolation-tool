@echo off
title Installing ALL Dependencies in Virtual Environment
color 0A

echo.
echo ================================================================
echo  📦 УСТАНОВКА ВСЕХ ЗАВИСИМОСТЕЙ В ВИРТУАЛЬНОЕ ОКРУЖЕНИЕ
echo ================================================================
echo.

set VENV_PATH=C:\Users\nirst\Desktop\Interpolation\.venv1
set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe
set PIP_EXE=%VENV_PATH%\Scripts\pip.exe

cd /d "C:\Users\nirst\Desktop\Interpolation\Interpolation\Interpolation"

if not exist "%PYTHON_EXE%" (
    echo ❌ Виртуальное окружение не найдено!
    pause
    exit /b 1
)

echo ✅ Виртуальное окружение найдено
"%PYTHON_EXE%" --version
echo.

echo 📥 Обновление pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo.
echo 📥 Установка ВСЕХ зависимостей из requirements.txt...
echo.

if exist "requirements.txt" (
    "%PIP_EXE%" install -r requirements.txt
    echo.
    echo ✅ Все зависимости из requirements.txt установлены!
) else (
    echo ⚠️  requirements.txt не найден, устанавливаем вручную...
    echo.
    
    echo [1/10] NumPy...
    "%PIP_EXE%" install numpy
    
    echo.
    echo [2/10] Pandas...
    "%PIP_EXE%" install pandas
    
    echo.
    echo [3/10] SciPy...
    "%PIP_EXE%" install scipy
    
    echo.
    echo [4/10] Matplotlib...
    "%PIP_EXE%" install matplotlib
    
    echo.
    echo [5/10] Scikit-learn...
    "%PIP_EXE%" install scikit-learn
    
    echo.
    echo [6/10] Plotly...
    "%PIP_EXE%" install plotly
    
    echo.
    echo [7/10] Seaborn...
    "%PIP_EXE%" install seaborn
    
    echo.
    echo [8/10] OpenPyXL...
    "%PIP_EXE%" install openpyxl
    
    echo.
    echo [9/10] xlrd...
    "%PIP_EXE%" install xlrd
    
    echo.
    echo [10/10] PyYAML...
    "%PIP_EXE%" install pyyaml
)

echo.
echo 🔍 Проверка установки основных библиотек...
echo.
"%PYTHON_EXE%" -c "import numpy; print('✅ NumPy:', numpy.__version__)"
"%PYTHON_EXE%" -c "import pandas; print('✅ Pandas:', pandas.__version__)"
"%PYTHON_EXE%" -c "import scipy; print('✅ SciPy:', scipy.__version__)"
"%PYTHON_EXE%" -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)"
"%PYTHON_EXE%" -c "import sklearn; print('✅ Scikit-learn:', sklearn.__version__)"

echo.
echo ================================================================
echo  ✅ ВСЕ ЗАВИСИМОСТИ УСТАНОВЛЕНЫ!
echo ================================================================
echo.
echo Теперь запустите программу:
echo run_venv.bat
echo.
echo Или напрямую:
echo %PYTHON_EXE% main.py
echo.
pause