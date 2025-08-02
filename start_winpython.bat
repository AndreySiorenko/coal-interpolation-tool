@echo off
title Coal Interpolation Tool - WinPython 3.12.4
color 0A

echo.
echo ================================================================
echo  🎯 COAL DEPOSIT INTERPOLATION TOOL v1.0.0-rc1
echo  🐍 Using WinPython 3.12.4
echo ================================================================
echo.

cd /d "%~dp0"

set WINPYTHON_PATH=G:\PROGRAMS\WPy64-31241\python-3.12.4.amd64
set PYTHON_EXE=%WINPYTHON_PATH%\python.exe

echo 📁 Working directory: %CD%
echo 🐍 Python path: %WINPYTHON_PATH%
echo.

if exist "%PYTHON_EXE%" (
    echo ✅ WinPython 3.12.4 найден
    "%PYTHON_EXE%" --version
    echo.
    
    echo 🔍 Проверка установленных библиотек...
    echo.
    
    "%PYTHON_EXE%" -c "import numpy; print('✅ NumPy:', numpy.__version__)" 2>nul || echo ❌ NumPy не найден
    "%PYTHON_EXE%" -c "import pandas; print('✅ Pandas:', pandas.__version__)" 2>nul || echo ❌ Pandas не найден
    "%PYTHON_EXE%" -c "import scipy; print('✅ SciPy:', scipy.__version__)" 2>nul || echo ❌ SciPy не найден
    "%PYTHON_EXE%" -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)" 2>nul || echo ❌ Matplotlib не найден
    
    echo.
    echo 🚀 Запуск программы...
    echo.
    
    :: Запускаем через launch.py для автоопределения режима
    if exist "launch.py" (
        echo 📱 Использование умного лаунчера...
        "%PYTHON_EXE%" launch.py
    ) else if exist "main.py" (
        echo 🎯 Запуск основного приложения...
        "%PYTHON_EXE%" main.py
    ) else (
        echo ❌ Файлы приложения не найдены!
    )
) else (
    echo ❌ ОШИБКА: WinPython не найден по пути:
    echo    %WINPYTHON_PATH%
    echo.
    echo Проверьте путь установки WinPython
)

echo.
echo ✅ Программа завершена.
echo.
pause