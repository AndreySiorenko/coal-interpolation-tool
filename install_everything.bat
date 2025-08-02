@echo off
title Install Everything - Complete Setup
color 0A

echo.
echo ================================================================
echo  📦 ПОЛНАЯ УСТАНОВКА ВСЕХ ЗАВИСИМОСТЕЙ
echo ================================================================
echo.

set VENV_PIP=C:\Users\nirst\Desktop\Interpolation\.venv1\Scripts\pip.exe
set VENV_PYTHON=C:\Users\nirst\Desktop\Interpolation\.venv1\Scripts\python.exe

echo 📥 Установка из requirements.txt (если существует)...
if exist "requirements.txt" (
    "%VENV_PIP%" install -r requirements.txt
) else (
    echo ⚠️  requirements.txt не найден, устанавливаем вручную...
)

echo.
echo 📥 Обновление pip...
"%VENV_PYTHON%" -m pip install --upgrade pip

echo.
echo 📥 Установка основных научных библиотек...
"%VENV_PIP%" install numpy pandas scipy matplotlib scikit-learn

echo.
echo 📥 Установка визуализации...
"%VENV_PIP%" install plotly seaborn

echo.
echo 📥 Установка работы с файлами...
"%VENV_PIP%" install openpyxl xlrd pyyaml

echo.
echo 📥 Установка геопространственных библиотек...
"%VENV_PIP%" install rasterio pyproj

echo.
echo 📥 Установка утилит...
"%VENV_PIP%" install chardet requests urllib3 certifi idna pytz six

echo.
echo 📥 Установка дополнительных инструментов...
"%VENV_PIP%" install pillow packaging setuptools wheel joblib threadpoolctl kiwisolver cycler

echo.
echo 📥 Установка GUI библиотек...
"%VENV_PIP%" install PyQt5

echo.
echo 📥 Установка других зависимостей...
"%VENV_PIP%" install jsonschema numba pydantic

echo.
echo 🔍 Финальная проверка основных библиотек...
echo.
"%VENV_PYTHON%" -c "import numpy; print('✅ NumPy:', numpy.__version__)"
"%VENV_PYTHON%" -c "import pandas; print('✅ Pandas:', pandas.__version__)"
"%VENV_PYTHON%" -c "import scipy; print('✅ SciPy:', scipy.__version__)"
"%VENV_PYTHON%" -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)"
"%VENV_PYTHON%" -c "import sklearn; print('✅ Scikit-learn:', sklearn.__version__)"
"%VENV_PYTHON%" -c "import chardet; print('✅ Chardet:', chardet.__version__)"

echo.
echo ================================================================
echo  🎉 ПОЛНАЯ УСТАНОВКА ЗАВЕРШЕНА!
echo ================================================================
echo.
echo Все необходимые библиотеки установлены.
echo Теперь запустите: run_fixed.bat
echo.
pause