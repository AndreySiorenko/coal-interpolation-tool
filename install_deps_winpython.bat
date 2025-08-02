@echo off
title Installing Dependencies - WinPython 3.12.4
color 0A

echo.
echo ================================================================
echo  📦 ПРОВЕРКА И УСТАНОВКА БИБЛИОТЕК
echo  🐍 WinPython 3.12.4
echo ================================================================
echo.

set WINPYTHON_PATH=G:\PROGRAMS\WPy64-31241\python-3.12.4.amd64
set PYTHON_EXE=%WINPYTHON_PATH%\python.exe
set PIP_EXE=%WINPYTHON_PATH%\Scripts\pip.exe

if not exist "%PYTHON_EXE%" (
    echo ❌ WinPython не найден по пути: %WINPYTHON_PATH%
    pause
    exit /b 1
)

echo ✅ WinPython найден
"%PYTHON_EXE%" --version
echo.

echo 📊 Проверка установленных пакетов...
echo.

:: WinPython обычно уже содержит научные библиотеки
echo Основные библиотеки:
"%PYTHON_EXE%" -c "import numpy; print('✅ NumPy:', numpy.__version__)" 2>nul || goto install_numpy
"%PYTHON_EXE%" -c "import pandas; print('✅ Pandas:', pandas.__version__)" 2>nul || goto install_pandas
"%PYTHON_EXE%" -c "import scipy; print('✅ SciPy:', scipy.__version__)" 2>nul || goto install_scipy
"%PYTHON_EXE%" -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)" 2>nul || goto install_matplotlib

echo.
echo Дополнительные библиотеки:
"%PYTHON_EXE%" -c "import openpyxl; print('✅ OpenPyXL:', openpyxl.__version__)" 2>nul || goto install_openpyxl
"%PYTHON_EXE%" -c "import xlrd; print('✅ xlrd:', xlrd.__version__)" 2>nul || goto install_xlrd

echo.
echo ✅ Все необходимые библиотеки установлены!
echo.
echo Теперь запустите start_winpython.bat для запуска программы
goto end

:install_numpy
echo.
echo 📥 Установка NumPy...
"%PIP_EXE%" install numpy
goto check_pandas

:check_pandas
:install_pandas
echo.
echo 📥 Установка Pandas...
"%PIP_EXE%" install pandas
goto check_scipy

:check_scipy
:install_scipy
echo.
echo 📥 Установка SciPy...
"%PIP_EXE%" install scipy
goto check_matplotlib

:check_matplotlib
:install_matplotlib
echo.
echo 📥 Установка Matplotlib...
"%PIP_EXE%" install matplotlib
goto check_openpyxl

:check_openpyxl
:install_openpyxl
echo.
echo 📥 Установка OpenPyXL...
"%PIP_EXE%" install openpyxl
goto check_xlrd

:check_xlrd
:install_xlrd
echo.
echo 📥 Установка xlrd...
"%PIP_EXE%" install xlrd

echo.
echo ================================================================
echo  ✅ УСТАНОВКА ЗАВЕРШЕНА!
echo ================================================================
echo.
echo Все библиотеки установлены. Запустите start_winpython.bat

:end
echo.
pause