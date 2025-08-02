# Интерактивная программа для анализа и интерполяции данных скважинной геологоразведки угольных месторождений

## Описание проекта
Специализированное программное обеспечение для комплексного анализа и интерполяции данных скважинной геологоразведки угольных месторождений. Программа разработана с учетом специфики стратифицированных угольных залежей и предназначена для моделирования качественных показателей угля (зольность, содержание серы, теплотворная способность).

## Целевая аудитория
- Геологи и горные инженеры
- Специалисты по геостатистике
- Проектировщики горных работ
- Исследователи угольных месторождений

## Основные возможности

### Загрузка и управление данными
- Импорт данных из CSV и Excel (.xlsx, .xls) файлов
- Автоматическое определение структуры данных (X, Y, Z координаты, параметры качества)
- Валидация данных с детальными отчетами об ошибках
- Предварительный просмотр в табличном виде
- Композитирование скважинных данных по интервалам

### Методы интерполяции
- **IDW (Inverse Distance Weighted)** - метод обратных взвешенных расстояний
  - Настройка степени взвешивания
  - Учет анизотропии
  - Секторный поиск
- **Обычный кригинг (Ordinary Kriging)**
  - Модели вариограмм (сферическая, экспоненциальная, гауссова)
  - Трансформация данных (логнормальная, Box-Cox)
  - Co-kriging для связанных параметров
- **Радиальные базисные функции (RBF)**
  - Различные ядра (мультиквадрик, гауссово, тонкая пластина)
  - Адаптивная настройка параметров

### Визуализация и анализ
- 3D визуализация скважин с цветовой кодировкой
- Построение карт интерполяции с изолиниями
- Гистограммы и статистический анализ данных
- Вариограммный анализ с интерактивной подгонкой
- Карты плотности точек и анализ кластеризации
- Перекрестная валидация результатов

### Система анализа и рекомендаций

#### Предварительный анализ данных
- **Статистический анализ**: описательная статистика, анализ выбросов, проверка нормальности
- **Пространственный анализ**: плотность опробования, кластеризация, анизотропия
- **Вариографический анализ**: экспериментальные вариограммы, подбор теоретических моделей

#### Оптимизация параметров
- Автоматический подбор параметров методом кросс-валидации
- Сравнительный анализ методов по критериям RMSE, MAE, R²
- Учет специфики угольных месторождений

#### Формирование рекомендаций
- Выбор оптимального метода на основе характеристик данных
- Рекомендуемые параметры с обоснованием
- Оценка качества и неопределенности результатов

### Экспорт результатов
- **Множественные форматы экспорта**:
  - CSV - табличные данные с координатами и значениями
  - GeoTIFF - геореференцированные растровые данные для ГИС
  - VTK - 3D данные для научной визуализации (ParaView, VisIt)
  - DXF - векторные данные для CAD систем (AutoCAD, QCAD)
- **Настройки экспорта**:
  - Выбор координатной системы (EPSG коды)
  - Настройка сжатия и точности данных
  - Включение метаданных и атрибутов
  - Генерация контурных линий (для DXF)
- **Типы данных**:
  - Регулярная сетка (GRID) с настройкой размера ячейки
  - Исходные точки скважин с атрибутами
  - Каркасная модель (MESH) с триангуляцией
- **Интеграция с ГИС**:
  - Прямая загрузка в QGIS, ArcGIS
  - Поддержка проекций и трансформаций координат
  - Пирамиды для больших растров

## Системные требования

### Минимальные требования
- Python 3.8 или выше
- 8 ГБ оперативной памяти
- 2 ГБ свободного места на диске
- Процессор с поддержкой многопоточности

### Рекомендуемые требования
- Python 3.10 или выше
- 16 ГБ оперативной памяти
- 5 ГБ свободного места на диске
- Многоядерный процессор (4+ ядра)
- Видеокарта с поддержкой OpenGL 3.3+

### Поддерживаемые операционные системы
- Windows 10/11 (64-bit)
- Ubuntu 20.04 LTS или выше
- macOS 10.15 (Catalina) или выше

## Основные зависимости
```
numpy>=1.24.0         # Численные вычисления
pandas>=2.0.0         # Работа с табличными данными
scipy>=1.10.0         # Научные вычисления
scikit-learn>=1.3.0   # Машинное обучение
matplotlib>=3.7.0     # 2D визуализация
plotly>=5.14.0        # Интерактивная визуализация
PyQt6>=6.5.0          # Графический интерфейс
openpyxl>=3.1.0       # Работа с Excel
rasterio>=1.3.0       # Работа с растровыми данными
vtk>=9.2.0            # 3D визуализация
ezdxf>=1.0.0          # Экспорт в DXF
```

## Установка

### Быстрая установка
```bash
# Клонировать репозиторий
git clone https://github.com/your-organization/coal-interpolation-tool.git
cd coal-interpolation-tool

# Создать и активировать виртуальное окружение
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Запустить программу
python main.py
```

Подробные инструкции по установке см. в [INSTALL.md](INSTALL.md)

## Быстрый старт

1. **Запуск программы**
   ```bash
   python main.py
   ```

2. **Загрузка данных**
   - Меню "Файл" → "Загрузить данные"
   - Выберите CSV или Excel файл
   - Укажите столбцы с координатами и значениями

3. **Настройка интерполяции**
   - Выберите метод интерполяции
   - Используйте "Рекомендуемые настройки" для автоматической конфигурации
   - При необходимости скорректируйте параметры

4. **Выполнение и экспорт**
   - Нажмите "Выполнить интерполяцию"
   - Просмотрите результаты в панели визуализации
   - Экспортируйте через меню "Файл" → "Экспорт результатов"

## Примеры использования

### Программный интерфейс (API)

#### Загрузка и валидация данных
```python
from src.io.readers.csv_reader import CSVReader
from src.io.validators import DataValidator

# Загрузка данных
reader = CSVReader()
data = reader.read('data/wells.csv')

# Валидация
validator = DataValidator()
is_valid, errors = validator.validate(data)
if not is_valid:
    print(f"Ошибки валидации: {errors}")
```

#### Интерполяция с автоматическими рекомендациями
```python
from src.core.recommendations.recommendation_engine import RecommendationEngine
from src.core.interpolation.base import InterpolationTask

# Создание задачи интерполяции
task = InterpolationTask(
    data=data,
    x_col='X',
    y_col='Y',
    value_col='Ash_Content'
)

# Получение рекомендаций
engine = RecommendationEngine()
recommendations = engine.analyze_and_recommend(task)

# Применение рекомендованного метода
interpolator = recommendations['method']
interpolator.fit(data, **recommendations['parameters'])

# Генерация сетки и интерполяция
grid = generate_regular_grid(bounds, cell_size=50)
results = interpolator.predict(grid)
```

#### Пакетная обработка
```python
from src.core.batch_processor import BatchProcessor

# Настройка пакетной обработки
processor = BatchProcessor()
parameters = ['Ash_Content', 'Sulfur_Content', 'Calorific_Value']

# Выполнение интерполяции для всех параметров
results = processor.process_multiple_parameters(
    data=data,
    parameters=parameters,
    method='idw',
    grid_params={'cell_size': 50}
)

# Экспорт результатов
for param, result in results.items():
    export_to_geotiff(result, f'output/{param}.tif')
```

#### Примеры экспорта данных

##### CSV экспорт с настройками
```python
from src.io.writers.csv_writer import CSVWriter, CSVExportOptions

# Настройки экспорта CSV
csv_options = CSVExportOptions(
    delimiter=';',                # Разделитель
    precision=3,                  # Точность чисел  
    include_coordinates=True,     # Включить координаты
    include_metadata=True         # Включить метаданные как комментарии
)

writer = CSVWriter(csv_options)
writer.write_points(point_data, 'results/ash_content.csv')
writer.write_grid(grid_data, 'results/ash_content_grid.csv')
```

##### GeoTIFF экспорт для ГИС
```python
from src.io.writers.geotiff_writer import GeoTIFFWriter, GeoTIFFExportOptions

# Настройки для GeoTIFF
geotiff_options = GeoTIFFExportOptions(
    crs='EPSG:32633',            # UTM Zone 33N
    compress='lzw',              # Сжатие LZW
    tiled=True,                  # Тайловая структура
    dtype='float32',             # Тип данных
    nodata_value=-9999           # Значение NoData
)

writer = GeoTIFFWriter(geotiff_options)
writer.write_grid(grid_data, 'results/ash_content.tif')

# Автоматическое преобразование точек в растр
writer.write_points(point_data, 'results/wells.tif', cell_size=50)
```

##### DXF экспорт для CAD систем  
```python
from src.io.writers.dxf_writer import DXFWriter, DXFExportOptions

# Настройки для DXF
dxf_options = DXFExportOptions(
    units='m',                   # Единицы измерения
    layer_name='COAL_QUALITY',   # Имя слоя
    point_style='CIRCLE',        # Стиль точек: CIRCLE, POINT, CROSS
    point_size=2.0,             # Размер символов
    contour_lines=True,         # Генерировать изолинии
    contour_intervals=5.0,      # Интервал изолиний
    include_labels=True,        # Подписи значений
    color_by_value=True,        # Цветовая кодировка
    text_height=1.5             # Высота текста
)

writer = DXFWriter(dxf_options)
writer.write_points(point_data, 'results/wells.dxf')        # Скважины как точки
writer.write_grid(grid_data, 'results/ash_contours.dxf')   # Сетка с изолиниями
```

##### VTK экспорт для 3D визуализации
```python
from src.io.writers.vtk_writer import VTKWriter, VTKExportOptions

# Настройки для VTK
vtk_options = VTKExportOptions(
    file_format='xml',           # XML формат (современный)
    data_mode='binary',          # Бинарные данные (компактнее)
    compress_data=True,          # Сжатие данных
    write_scalars=True,          # Записать скалярные поля
    write_vectors=False,         # Записать векторные поля
    include_metadata=True        # Включить метаданные
)

writer = VTKWriter(vtk_options)
writer.write_points(point_data, 'results/wells.vtp')       # PolyData для точек
writer.write_grid(grid_data, 'results/ash_grid.vti')      # ImageData для сетки

# Для 3D данных
writer.write_grid(grid_3d_data, 'results/coal_seam.vti')  # 3D объемные данные
```

##### Универсальная функция экспорта
```python
def export_interpolation_results(results, output_dir, formats=['csv', 'geotiff']):
    """
    Экспорт результатов интерполяции в различные форматы.
    
    Args:
        results: Результаты интерполяции
        output_dir: Директория для сохранения
        formats: Список форматов ['csv', 'geotiff', 'vtk', 'dxf']
    """
    from pathlib import Path
    from src.io.writers import *
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    grid_data = results['grid_data']
    point_data = results['source_points']
    parameter = results['parameter']
    
    for format_name in formats:
        if format_name == 'csv':
            writer = CSVWriter()
            writer.write_grid(grid_data, output_path / f'{parameter}_grid.csv')
            writer.write_points(point_data, output_path / f'{parameter}_points.csv')
            
        elif format_name == 'geotiff':
            options = GeoTIFFExportOptions(crs='EPSG:32633')
            writer = GeoTIFFWriter(options)
            writer.write_grid(grid_data, output_path / f'{parameter}.tif')
            
        elif format_name == 'vtk':
            writer = VTKWriter()
            writer.write_grid(grid_data, output_path / f'{parameter}.vti')
            writer.write_points(point_data, output_path / f'{parameter}_points.vtp')
            
        elif format_name == 'dxf':
            options = DXFExportOptions(contour_lines=True, include_labels=True)
            writer = DXFWriter(options)
            writer.write_grid(grid_data, output_path / f'{parameter}_contours.dxf')

# Использование
export_interpolation_results(
    results=interpolation_results,
    output_dir='output/ash_content',
    formats=['csv', 'geotiff', 'vtk', 'dxf']
)
```

Больше примеров см. в папке [examples/](examples/)

## Документация
- [Архитектура приложения](ARCHITECTURE.md)
- [Руководство пользователя](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Примеры использования](examples/)

## Разработка

### Структура проекта
```
coal-interpolation-tool/
├── src/                    # Исходный код
│   ├── core/              # Ядро системы
│   │   ├── interpolation/ # Методы интерполяции
│   │   ├── variogram/     # Вариограммный анализ
│   │   ├── grid/          # Генерация сеток
│   │   └── recommendations/ # Система рекомендаций
│   ├── gui/               # Графический интерфейс
│   ├── io/                # Ввод/вывод данных
│   ├── visualization/     # Модули визуализации
│   └── utils/             # Вспомогательные утилиты
├── tests/                 # Тесты
├── docs/                  # Документация
├── examples/              # Примеры
├── resources/             # Ресурсы (иконки, стили)
└── scripts/               # Вспомогательные скрипты
```

### Запуск тестов
```bash
# Все тесты
pytest

# Только unit-тесты
pytest tests/unit/

# С покрытием кода
pytest --cov=src --cov-report=html
```

### Соглашения по коду
- Следуем PEP 8 для Python кода
- Используем type hints для всех публичных методов
- Обязательные docstrings для модулей, классов и функций
- Максимальная длина строки: 100 символов

## История изменений
См. [CHANGELOG.md](CHANGELOG.md) для полного списка изменений.

## Планы развития
См. [TODO.md](TODO.md) для списка запланированных функций и улучшений.

## Вклад в проект
Мы приветствуем вклад в развитие проекта! Пожалуйста:
1. Создайте fork репозитория
2. Создайте branch для новой функции (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## Лицензия
Распространяется под лицензией MIT. См. [LICENSE](LICENSE) для подробностей.

## Контакты и поддержка
- **Email**: support@coal-interpolation.com
- **Issues**: https://github.com/your-organization/coal-interpolation-tool/issues
- **Документация**: https://docs.coal-interpolation.com

## Авторы и благодарности
- Основная команда разработки
- Консультанты по геостатистике
- Сообщество пользователей за обратную связь и тестирование

---
**Текущая версия**: 1.0.0-rc1 - Release Candidate с полной функциональностью  
**Дата релиза**: 2025-08-01