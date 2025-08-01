# Руководство по установке

Это подробное руководство поможет вам установить и настроить программу для анализа и интерполяции данных скважинной геологоразведки угольных месторождений.

## Системные требования

### Минимальные требования
- **Операционная система**: 
  - Windows 10/11 (64-bit)
  - Linux (Ubuntu 20.04 LTS или выше, Debian 10+, Fedora 33+)
  - macOS 10.15 (Catalina) или выше
- **Python**: 3.8 или выше (рекомендуется 3.10+)
- **Оперативная память**: 8 ГБ
- **Свободное место на диске**: 2 ГБ
- **Процессор**: 
  - Intel Core i5 (4-го поколения) или новее
  - AMD Ryzen 3 или новее
  - Apple M1 или новее
- **Видеокарта**: Поддержка OpenGL 3.3 для 3D визуализации

### Рекомендуемые требования
- **Операционная система**: Последние версии поддерживаемых ОС
- **Python**: 3.10 или 3.11
- **Оперативная память**: 16 ГБ или более
- **Свободное место на диске**: 5 ГБ
- **Процессор**: 
  - Intel Core i7/i9 (8 ядер)
  - AMD Ryzen 7/9
  - Apple M1 Pro/Max или M2
- **Видеокарта**: Дискретная GPU с поддержкой OpenGL 4.5+

## Предварительная подготовка

### Установка Python

#### Windows

**Вариант 1: Официальный установщик Python**
1. Скачайте Python с официального сайта: https://www.python.org/downloads/
2. Запустите установщик
3. **ВАЖНО**: Обязательно отметьте галочку "Add Python to PATH"
4. Выберите "Install Now" или "Customize installation"
5. После установки откройте командную строку и проверьте:
   ```cmd
   python --version
   pip --version
   ```

**Вариант 2: Использование Windows Store**
1. Откройте Microsoft Store
2. Найдите "Python 3.11" или "Python 3.10"
3. Нажмите "Получить"

**Вариант 3: Использование Anaconda**
1. Скачайте Anaconda: https://www.anaconda.com/products/distribution
2. Установите с настройками по умолчанию
3. Используйте Anaconda Prompt для работы

#### Linux (Ubuntu/Debian)
```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Python и необходимых инструментов
sudo apt install python3.10 python3.10-venv python3-pip python3-dev -y

# Установка дополнительных системных зависимостей
sudo apt install build-essential libssl-dev libffi-dev -y
sudo apt install libgdal-dev gdal-bin -y
sudo apt install libgl1-mesa-dev libglu1-mesa-dev -y
```

#### Linux (Fedora/Red Hat)
```bash
# Установка Python
sudo dnf install python3.10 python3-devel python3-pip -y

# Дополнительные зависимости
sudo dnf install gcc gcc-c++ make -y
sudo dnf install gdal gdal-devel -y
sudo dnf install mesa-libGL-devel mesa-libGLU-devel -y
```

#### macOS
```bash
# Установка Homebrew (если еще не установлен)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установка Python
brew install python@3.10

# Дополнительные зависимости
brew install gdal
brew install vtk
```

### Установка Git

#### Windows
1. Скачайте Git: https://git-scm.com/download/win
2. Установите с настройками по умолчанию

#### Linux
```bash
# Ubuntu/Debian
sudo apt install git -y

# Fedora
sudo dnf install git -y
```

#### macOS
```bash
brew install git
```

## Установка программы

### Шаг 1: Получение исходного кода

```bash
# Клонирование репозитория
git clone https://github.com/your-organization/coal-interpolation-tool.git
cd coal-interpolation-tool

# Или загрузка архива
# Скачайте архив с GitHub и распакуйте в удобную папку
```

### Шаг 2: Создание виртуального окружения

Виртуальное окружение изолирует зависимости проекта от системных пакетов.

#### Windows
```cmd
# Создание виртуального окружения
python -m venv venv

# Или с указанием конкретной версии Python
py -3.10 -m venv venv
```

#### Linux/macOS
```bash
# Создание виртуального окружения
python3.10 -m venv venv

# Или используя python3
python3 -m venv venv
```

### Шаг 3: Активация виртуального окружения

#### Windows (Command Prompt)
```cmd
venv\Scripts\activate.bat
```

#### Windows (PowerShell)
```powershell
# Если возникает ошибка с политикой выполнения скриптов
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Активация
venv\Scripts\Activate.ps1
```

#### Linux/macOS
```bash
source venv/bin/activate
```

После активации в начале строки командной строки появится префикс `(venv)`.

### Шаг 4: Обновление pip и setuptools

```bash
# Обновление pip до последней версии
python -m pip install --upgrade pip

# Обновление setuptools и wheel
python -m pip install --upgrade setuptools wheel
```

### Шаг 5: Установка зависимостей

```bash
# Установка основных зависимостей
pip install -r requirements.txt

# Для разработчиков (включает линтеры, форматеры и инструменты тестирования)
pip install -r requirements-dev.txt
```

## Установка дополнительных компонентов

### Установка VTK для 3D визуализации

VTK может требовать дополнительных шагов на некоторых системах.

#### Windows
```bash
# Обычно устанавливается автоматически, но если возникли проблемы:
pip install vtk --no-cache-dir
```

#### Linux
```bash
# Установка системных зависимостей (уже выполнено выше)
# Установка VTK
pip install vtk

# Если возникают проблемы
pip install --upgrade --force-reinstall vtk
```

#### macOS
```bash
# VTK через pip
pip install vtk

# Альтернатива через Homebrew
brew install vtk
pip install vtk --no-binary vtk
```

### Установка GDAL для работы с геопространственными данными

#### Windows
1. Скачайте подходящий wheel файл:
   - Перейдите на https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
   - Выберите файл соответствующий вашей версии Python и архитектуре
   - Например: `GDAL‑3.7.1‑cp310‑cp310‑win_amd64.whl` для Python 3.10 64-bit

2. Установите скачанный файл:
   ```bash
   pip install путь_к_файлу/GDAL‑3.7.1‑cp310‑cp310‑win_amd64.whl
   ```

#### Linux
```bash
# Системные зависимости уже установлены выше
# Установка Python биндингов
pip install GDAL==$(gdal-config --version)
```

#### macOS
```bash
# GDAL уже установлен через Homebrew
pip install GDAL==$(gdal-config --version)
```

## Проверка установки

### Шаг 1: Проверка версий

```bash
# Проверка Python
python --version

# Проверка pip
pip --version

# Проверка установленных пакетов
pip list
```

### Шаг 2: Проверка импорта основных модулей

```python
# Создайте файл test_imports.py
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'NumPy: {numpy.__version__}')
except ImportError as e:
    print(f'NumPy import error: {e}')

try:
    import pandas
    print(f'Pandas: {pandas.__version__}')
except ImportError as e:
    print(f'Pandas import error: {e}')

try:
    import scipy
    print(f'SciPy: {scipy.__version__}')
except ImportError as e:
    print(f'SciPy import error: {e}')

try:
    import PyQt6
    print('PyQt6: Installed')
except ImportError as e:
    print(f'PyQt6 import error: {e}')

print('All imports successful!' if all([
    'numpy' in sys.modules,
    'pandas' in sys.modules,
    'scipy' in sys.modules,
    'PyQt6' in sys.modules
]) else 'Some imports failed!')
"
```

### Шаг 3: Запуск тестов

```bash
# Запуск всех unit тестов
python -m pytest tests/unit/

# Запуск с подробным выводом
python -m pytest -v

# Запуск с покрытием кода
python -m pytest --cov=src --cov-report=html

# Открыть отчет о покрытии (Windows)
start htmlcov/index.html

# Открыть отчет о покрытии (Linux/macOS)
open htmlcov/index.html
```

### Шаг 4: Первый запуск программы

```bash
# Запуск основной программы
python main.py

# Запуск с демо-данными
python main.py --demo

# Запуск в режиме отладки
python main.py --debug
```

## Решение типичных проблем

### Проблема: ModuleNotFoundError при запуске

**Причина**: Не активировано виртуальное окружение или не установлены зависимости.

**Решение**:
```bash
# Убедитесь, что виртуальное окружение активировано
# Переустановите зависимости
pip install -r requirements.txt --force-reinstall
```

### Проблема: ImportError: DLL load failed (Windows)

**Причина**: Отсутствуют системные библиотеки.

**Решение**:
1. Установите Microsoft Visual C++ Redistributable:
   https://support.microsoft.com/help/2977003/
2. Перезагрузите компьютер
3. Переустановите проблемный пакет:
   ```bash
   pip uninstall пакет
   pip install пакет --no-cache-dir
   ```

### Проблема: Permission denied при установке

**Решение**:
```bash
# Не используйте sudo с pip!
# Вместо этого используйте флаг --user
pip install --user -r requirements.txt

# Или убедитесь, что используете виртуальное окружение
```

### Проблема: SSL Certificate error

**Решение**:
```bash
# Временное решение (НЕ рекомендуется для production)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Правильное решение: обновить сертификаты
# Windows: обновить Python до последней версии
# Linux: sudo apt-get update && sudo apt-get install ca-certificates
# macOS: brew install ca-certificates
```

### Проблема: Конфликт версий пакетов

**Решение**:
```bash
# Создайте новое чистое окружение
deactivate
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows

# Создайте заново и установите
python -m venv venv
# Активируйте и установите зависимости заново
```

## Настройка для разработки

### Установка pre-commit hooks

```bash
# Установка pre-commit
pip install pre-commit

# Установка хуков
pre-commit install

# Ручной запуск проверок
pre-commit run --all-files
```

### Настройка IDE

#### VS Code
1. Установите расширение Python
2. Выберите интерпретатор из виртуального окружения (Ctrl+Shift+P → "Python: Select Interpreter")
3. Рекомендуемые расширения:
   - Python
   - Pylance
   - Python Test Explorer
   - GitLens

#### PyCharm
1. File → Settings → Project → Python Interpreter
2. Выберите интерпретатор из папки venv
3. Включите поддержку pytest в настройках

## Обновление программы

### Обновление из Git

```bash
# Сохраните локальные изменения (если есть)
git stash

# Получите последние изменения
git pull origin main

# Восстановите локальные изменения (если нужно)
git stash pop

# Обновите зависимости
pip install -r requirements.txt --upgrade
```

### Обновление зависимостей

```bash
# Посмотреть устаревшие пакеты
pip list --outdated

# Обновить конкретный пакет
pip install --upgrade numpy

# Обновить все пакеты (осторожно!)
pip install --upgrade -r requirements.txt
```

## Удаление программы

```bash
# Деактивировать виртуальное окружение
deactivate

# Удалить папку с программой
# Windows
rmdir /s coal-interpolation-tool

# Linux/macOS
rm -rf coal-interpolation-tool
```

## Docker установка (альтернативный метод)

Для упрощения развертывания можно использовать Docker:

```bash
# Сборка образа
docker build -t coal-interpolation-tool .

# Запуск контейнера
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/data:/app/data \
  coal-interpolation-tool
```

## Получение помощи

Если у вас возникли проблемы с установкой:

1. **Проверьте документацию**:
   - [README.md](README.md) - общая информация
   - [ARCHITECTURE.md](ARCHITECTURE.md) - архитектура проекта
   - [Wiki проекта](https://github.com/your-organization/coal-interpolation-tool/wiki)

2. **Поиск решения**:
   - Проверьте раздел [Issues](https://github.com/your-organization/coal-interpolation-tool/issues) на GitHub
   - Поищите в [Discussions](https://github.com/your-organization/coal-interpolation-tool/discussions)

3. **Создайте запрос о помощи**:
   - Опишите проблему детально
   - Приложите логи ошибок
   - Укажите версии ОС, Python и установленных пакетов

4. **Контакты поддержки**:
   - Email: support@coal-interpolation.com
   - Telegram: @coal_interpolation_support

---
Последнее обновление: 2025-08-01