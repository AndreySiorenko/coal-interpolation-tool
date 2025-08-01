# GitHub Setup Guide - Coal Interpolation Tool

## Шаги для создания GitHub репозитория

### 1. Создание репозитория на GitHub

1. Откройте [GitHub.com](https://github.com) и войдите в аккаунт
2. Нажмите "New repository" (зеленая кнопка)
3. Заполните данные:
   - **Repository name**: `coal-interpolation-tool`
   - **Description**: `Advanced interpolation tool for coal deposit quality analysis with multiple export formats`
   - **Visibility**: Public (рекомендуется для open-source проекта)
   - **НЕ** инициализируйте с README, .gitignore или license (у нас уже есть код)

### 2. Подключение локального репозитория

После создания репозитория на GitHub выполните команды:

```bash
# Добавить remote origin (замените YOUR_USERNAME на ваш GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/coal-interpolation-tool.git

# Проверить настройки remote
git remote -v

# Отправить код на GitHub
git push -u origin master
```

### 3. Настройка ветки по умолчанию (опционально)

Если GitHub использует `main` как ветку по умолчанию:

```bash
# Переименовать локальную ветку
git branch -M main

# Отправить на GitHub
git push -u origin main
```

### 4. Проверка

После успешного push:
- Обновите страницу репозитория на GitHub
- Убедитесь, что все файлы загружены
- README.md должен отображаться на главной странице

### 5. Настройка GitHub Actions (опционально)

Для автоматического тестирования создайте файл `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ master, main ]
  pull_request:
    branches: [ master, main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Текущий статус проекта

✅ **Завершено:**
- Локальный Git репозиторий инициализирован
- Все файлы добавлены и закоммичены
- .gitignore настроен для Python проекта
- Commit с полным описанием функций создан

🔄 **Следующие шаги:**
1. Создать репозиторий на GitHub.com  
2. Добавить remote origin
3. Push кода на GitHub
4. Настроить GitHub Actions (опционально)

## Информация о проекте

- **Название**: Coal Interpolation Tool
- **Описание**: Advanced interpolation tool for coal deposit quality analysis
- **Язык**: Python 3.8+
- **Лицензия**: MIT
- **Основные функции**:
  - Методы интерполяции: IDW, Kriging, RBF
  - Экспорт в форматы: CSV, GeoTIFF, VTK, DXF
  - PyQt6 GUI интерфейс
  - Система автоматических рекомендаций
  - Обширное тестирование (80%+ покрытие)

## Размер проекта

- **Файлов**: 82
- **Строк кода**: ~29,000
- **Тестов**: 15+ модульных тестов
- **Зависимостей**: 12 основных пакетов

---

После выполнения этих шагов ваш проект будет доступен на GitHub!