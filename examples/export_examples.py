#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Примеры экспорта результатов интерполяции в различные форматы.

Этот модуль демонстрирует использование всех доступных экспортеров:
- CSV для табличных данных  
- GeoTIFF для геореференцированных растров
- VTK для 3D научной визуализации
- DXF для CAD систем

Автор: Coal Interpolation Tool Team
Дата: 2025-08-01
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Добавляем путь к исходному коду
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from io.writers.csv_writer import CSVWriter, CSVExportOptions
from io.writers.geotiff_writer import GeoTIFFWriter, GeoTIFFExportOptions
from io.writers.vtk_writer import VTKWriter, VTKExportOptions
from io.writers.dxf_writer import DXFWriter, DXFExportOptions
from io.writers.base import GridData, PointData


def create_sample_data():
    """
    Создание примера данных скважин угольного месторождения.
    
    Returns:
        tuple: (point_data, grid_data) - данные точек и сетки
    """
    print("Создание примера данных...")
    
    # Генерация координат скважин
    np.random.seed(42)
    n_wells = 25
    
    # UTM координаты (зона 33N, восточная Европа)
    x_coords = np.random.uniform(400000, 410000, n_wells)
    y_coords = np.random.uniform(5500000, 5510000, n_wells)
    z_coords = np.random.uniform(100, 300, n_wells)  # Высотные отметки
    
    # Качественные параметры угля
    ash_content = np.random.normal(15, 4, n_wells)      # Зольность, %
    sulfur_content = np.random.normal(2.5, 0.8, n_wells) # Содержание серы, %
    calorific_value = 30 - 0.4 * ash_content + np.random.normal(0, 1.5, n_wells)  # МДж/кг
    
    # Ограничиваем значения реалистичными диапазонами
    ash_content = np.clip(ash_content, 8, 35)
    sulfur_content = np.clip(sulfur_content, 0.5, 6.0)
    calorific_value = np.clip(calorific_value, 18, 35)
    
    # Идентификаторы скважин
    well_ids = np.array([f'BH-{i:03d}' for i in range(1, n_wells + 1)])
    
    # Создание данных точек
    coordinates = np.column_stack([x_coords, y_coords, z_coords])
    
    point_data = PointData(
        coordinates=coordinates,
        values=ash_content,
        point_ids=well_ids,
        attributes={
            'SULFUR': sulfur_content,
            'CALORIFIC': calorific_value,
            'ELEVATION': z_coords
        },
        coordinate_system='EPSG:32633',  # UTM Zone 33N
        metadata={
            'project': 'Тестовое угольное месторождение',
            'survey_date': '2023-06-15',
            'parameter': 'ash_content',
            'units': 'percent',
            'method': 'drilling_samples'
        }
    )
    
    # Создание регулярной сетки для демонстрации
    x_grid = np.linspace(x_coords.min() - 1000, x_coords.max() + 1000, 15)
    y_grid = np.linspace(y_coords.min() - 1000, y_coords.max() + 1000, 15)
    
    # Создание синтетической поверхности зольности
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Базовый тренд + случайные вариации  
    center_x, center_y = np.mean(x_coords), np.mean(y_coords)
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    base_values = 12 + 8 * np.exp(-distance / 5000)  # Экспоненциальное убывание от центра
    
    # Добавление шума и локальных аномалий
    noise = np.random.normal(0, 1.5, X.shape)
    anomaly = 5 * np.sin(X / 2000) * np.cos(Y / 3000)  # Периодические аномалии
    
    grid_values = base_values + noise + anomaly
    grid_values = np.clip(grid_values, 8, 35)  # Ограничиваем реалистичным диапазоном
    
    grid_data = GridData(
        x_coords=x_grid,
        y_coords=y_grid,
        values=grid_values,
        bounds=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
        cell_size=np.mean(np.diff(x_grid)),
        coordinate_system='EPSG:32633',
        metadata={
            'parameter': 'ash_content',
            'units': 'percent',
            'method': 'kriging_interpolation',
            'grid_resolution': f'{len(x_grid)}x{len(y_grid)}',
            'interpolation_date': '2023-06-20'
        }
    )
    
    print(f"Создано {n_wells} скважин и сетка {len(x_grid)}×{len(y_grid)}")
    return point_data, grid_data


def example_csv_export(point_data, grid_data, output_dir):
    """
    Пример экспорта в CSV формат с различными настройками.
    
    Args:
        point_data: Данные точек скважин
        grid_data: Данные интерполированной сетки  
        output_dir: Директория для сохранения файлов
    """
    print("\n=== Экспорт в CSV формат ===")
    
    # Базовый экспорт с настройками по умолчанию
    print("1. Базовый экспорт точек...")
    writer = CSVWriter()
    writer.write_points(point_data, output_dir / 'wells_basic.csv')
    
    # Экспорт с пользовательскими настройками
    print("2. Экспорт с настройками (точка с запятой, высокая точность)...")
    csv_options = CSVExportOptions(
        delimiter=';',
        precision=4,
        include_coordinates=True,
        include_metadata=True
    )
    
    writer = CSVWriter(csv_options)
    writer.write_points(point_data, output_dir / 'wells_custom.csv')
    writer.write_grid(grid_data, output_dir / 'grid_interpolated.csv')
    
    # Экспорт только значений без координат
    print("3. Экспорт только значений...")
    minimal_options = CSVExportOptions(
        include_coordinates=False,
        include_metadata=False,
        precision=2
    )
    
    writer = CSVWriter(minimal_options)
    writer.write_points(point_data, output_dir / 'wells_values_only.csv')
    
    print("CSV файлы сохранены:")
    print(f"  - wells_basic.csv: базовый экспорт точек")
    print(f"  - wells_custom.csv: настраиваемый экспорт точек")
    print(f"  - grid_interpolated.csv: интерполированная сетка")
    print(f"  - wells_values_only.csv: только значения")


def example_geotiff_export(point_data, grid_data, output_dir):
    """
    Пример экспорта в GeoTIFF формат для ГИС.
    
    Args:
        point_data: Данные точек скважин
        grid_data: Данные интерполированной сетки
        output_dir: Директория для сохранения файлов
    """
    print("\n=== Экспорт в GeoTIFF формат ===")
    
    try:
        # Базовый экспорт сетки
        print("1. Экспорт сетки в GeoTIFF...")
        geotiff_options = GeoTIFFExportOptions(
            crs='EPSG:32633',        # UTM Zone 33N
            compress='lzw',          # LZW сжатие
            tiled=True,              # Тайловая структура для производительности
            dtype='float32',         # Тип данных
            nodata_value=-9999       # Значение NoData
        )
        
        writer = GeoTIFFWriter(geotiff_options)
        writer.write_grid(grid_data, output_dir / 'ash_content_grid.tif')
        
        # Экспорт точек как растр (автоматическая растеризация)
        print("2. Экспорт точек скважин как растр...")
        writer.write_points(point_data, output_dir / 'wells_rasterized.tif', cell_size=500)
        
        # Экспорт с различными настройками сжатия
        print("3. Экспорт с JPEG сжатием...")
        jpeg_options = GeoTIFFExportOptions(
            crs='EPSG:32633',
            compress='jpeg',
            dtype='uint8',
            tiled=True
        )
        
        # Нормализуем значения для uint8 (0-255)
        normalized_grid = GridData(
            x_coords=grid_data.x_coords,
            y_coords=grid_data.y_coords,
            values=((grid_data.values - grid_data.values.min()) / 
                   (grid_data.values.max() - grid_data.values.min()) * 255).astype(np.uint8),
            bounds=grid_data.bounds,
            cell_size=grid_data.cell_size,
            coordinate_system=grid_data.coordinate_system,
            metadata=grid_data.metadata
        )
        
        writer = GeoTIFFWriter(jpeg_options)
        writer.write_grid(normalized_grid, output_dir / 'ash_content_jpeg.tif')
        
        print("GeoTIFF файлы сохранены:")
        print(f"  - ash_content_grid.tif: основная сетка (LZW)")
        print(f"  - wells_rasterized.tif: растеризованные скважины")
        print(f"  - ash_content_jpeg.tif: сжатие JPEG")
        
    except ImportError:
        print("⚠️  Библиотека rasterio не установлена - пропускаем GeoTIFF экспорт")
        print("   Установите: pip install rasterio")


def example_vtk_export(point_data, grid_data, output_dir):
    """
    Пример экспорта в VTK формат для 3D визуализации.
    
    Args:
        point_data: Данные точек скважин
        grid_data: Данные интерполированной сетки
        output_dir: Директория для сохранения файлов
    """
    print("\n=== Экспорт в VTK формат ===")
    
    try:
        # Экспорт точек скважин
        print("1. Экспорт точек скважин для 3D визуализации...")
        vtk_options = VTKExportOptions(
            file_format='xml',        # Современный XML формат
            data_mode='binary',       # Бинарные данные (компактнее)
            compress_data=True,       # Сжатие для экономии места
            write_scalars=True,       # Записать скалярные поля
            write_vectors=False,      # Векторные поля не нужны
            include_metadata=True     # Включить метаданные
        )
        
        writer = VTKWriter(vtk_options)
        writer.write_points(point_data, output_dir / 'wells_3d.vtp')
        
        # Экспорт сетки как структурированные данные
        print("2. Экспорт интерполированной сетки...")
        writer.write_grid(grid_data, output_dir / 'ash_content_grid.vti')
        
        # Экспорт в устаревшем формате для совместимости
        print("3. Экспорт в legacy формате...")
        legacy_options = VTKExportOptions(
            file_format='legacy',
            data_mode='ascii',
            write_scalars=True
        )
        
        writer = VTKWriter(legacy_options)
        writer.write_points(point_data, output_dir / 'wells_legacy.vtk')
        
        print("VTK файлы сохранены:")
        print(f"  - wells_3d.vtp: скважины в XML формате (для ParaView)")
        print(f"  - ash_content_grid.vti: сетка в XMLFormJavaScript (ImageData)")
        print(f"  - wells_legacy.vtk: скважины в legacy формате")
        print(f"💡 Откройте .vtp и .vti файлы в ParaView для 3D визуализации")
        
    except ImportError:
        print("⚠️  Библиотека vtk не установлена - пропускаем VTK экспорт")
        print("   Установите: pip install vtk")


def example_dxf_export(point_data, grid_data, output_dir):
    """
    Пример экспорта в DXF формат для CAD систем.
    
    Args:
        point_data: Данные точек скважин
        grid_data: Данные интерполированной сетки
        output_dir: Директория для сохранения файлов
    """
    print("\n=== Экспорт в DXF формат ===")
    
    try:
        # Экспорт скважин как точки с подписями
        print("1. Экспорт скважин как кружки с подписями...")
        dxf_options = DXFExportOptions(
            units='m',                      # Метры
            layer_name='WELLS',             # Слой для скважин
            point_style='CIRCLE',           # Кружки для скважин
            point_size=50.0,               # Размер символов (м)
            include_labels=True,           # Подписи значений
            color_by_value=True,           # Цветовая кодировка
            text_height=25.0               # Размер текста
        )
        
        writer = DXFWriter(dxf_options)
        writer.write_points(point_data, output_dir / 'wells_circles.dxf')
        
        # Экспорт сетки с изолиниями
        print("2. Экспорт сетки с изолиниями...")
        contour_options = DXFExportOptions(
            units='m',
            layer_name='ASH_CONTOURS',
            contour_lines=True,            # Генерировать изолинии
            contour_intervals=2.0,         # Интервал 2%
            include_labels=True,           # Подписи изолиний
            color_by_value=True,           # Разные цвета для уровней
            line_type='CONTINUOUS'         # Сплошные линии
        )
        
        writer = DXFWriter(contour_options)
        writer.write_grid(grid_data, output_dir / 'ash_contours.dxf')
        
        # Экспорт скважин как кресты без подписей
        print("3. Экспорт скважин как кресты...")
        cross_options = DXFExportOptions(
            units='m',
            layer_name='WELL_LOCATIONS',
            point_style='CROSS',
            point_size=100.0,
            include_labels=False,
            color_by_value=False
        )
        
        writer = DXFWriter(cross_options)
        writer.write_points(point_data, output_dir / 'wells_crosses.dxf')
        
        print("DXF файлы сохранены:")
        print(f"  - wells_circles.dxf: скважины как кружки с подписями")
        print(f"  - ash_contours.dxf: изолинии зольности")
        print(f"  - wells_crosses.dxf: местоположения скважин как кресты")
        print(f"💡 Откройте .dxf файлы в AutoCAD, QCAD или LibreCAD")
        
    except ImportError:
        print("⚠️  Библиотека ezdxf не установлена - пропускаем DXF экспорт")
        print("   Установите: pip install ezdxf")


def example_batch_export(point_data, grid_data, output_dir):
    """
    Пример пакетного экспорта во все форматы.
    
    Args:
        point_data: Данные точек скважин
        grid_data: Данные интерполированной сетки
        output_dir: Директория для сохранения файлов
    """
    print("\n=== Пакетный экспорт во все форматы ===")
    
    # Определение форматов и их настроек
    export_configs = {
        'csv': {
            'writer_class': CSVWriter,
            'options_class': CSVExportOptions,
            'options': {'delimiter': ',', 'precision': 3, 'include_metadata': True},
            'extensions': {'points': '.csv', 'grid': '_grid.csv'}
        },
        'vtk': {
            'writer_class': VTKWriter,
            'options_class': VTKExportOptions,
            'options': {'file_format': 'xml', 'compress_data': True},
            'extensions': {'points': '.vtp', 'grid': '.vti'}
        },
        'dxf': {
            'writer_class': DXFWriter,
            'options_class': DXFExportOptions,
            'options': {'contour_lines': True, 'include_labels': True, 'point_style': 'CIRCLE'},
            'extensions': {'points': '_points.dxf', 'grid': '_contours.dxf'}
        }
    }
    
    # Добавляем GeoTIFF если rasterio доступен
    try:
        import rasterio
        export_configs['geotiff'] = {
            'writer_class': GeoTIFFWriter,
            'options_class': GeoTIFFExportOptions,
            'options': {'crs': 'EPSG:32633', 'compress': 'lzw'},
            'extensions': {'points': '_raster.tif', 'grid': '.tif'}
        }
    except ImportError:
        pass
    
    batch_dir = output_dir / 'batch_export'
    batch_dir.mkdir(exist_ok=True)
    
    for format_name, config in export_configs.items():
        try:
            print(f"Экспорт в {format_name.upper()}...")
            
            # Создание writer с настройками
            options = config['options_class'](**config['options'])
            writer = config['writer_class'](options)
            
            # Экспорт точек
            points_file = batch_dir / f'ash_content{config["extensions"]["points"]}'
            if format_name == 'geotiff':
                writer.write_points(point_data, points_file, cell_size=500)
            else:
                writer.write_points(point_data, points_file)
            
            # Экспорт сетки  
            grid_file = batch_dir / f'ash_content{config["extensions"]["grid"]}'
            writer.write_grid(grid_data, grid_file)
            
            print(f"  ✓ {points_file.name}")
            print(f"  ✓ {grid_file.name}")
            
        except ImportError as e:
            print(f"  ⚠️ Пропуск {format_name}: библиотека не установлена")
        except Exception as e:
            print(f"  ❌ Ошибка {format_name}: {e}")
    
    print(f"\nПакетный экспорт завершен в директории: {batch_dir}")


def export_summary_report(point_data, grid_data, output_dir):
    """
    Создание отчета о возможностях экспорта.
    
    Args:
        point_data: Данные точек скважин
        grid_data: Данные интерполированной сетки
        output_dir: Директория для сохранения отчета
    """
    print("\n=== Создание отчета о экспорте ===")
    
    report_lines = [
        "# Отчет о возможностях экспорта данных",
        "# Coal Interpolation Tool Export Report",
        f"# Дата создания: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Исходные данные",
        f"Скважины: {point_data.n_points}",
        f"Координатная система: {point_data.coordinate_system}",
        f"Параметры: {list(point_data.attributes.keys())}",
        f"Сетка: {grid_data.shape}",
        f"Размер ячейки: {grid_data.cell_size:.1f} м",
        "",
        "## Доступные форматы экспорта",
        ""
    ]
    
    # Проверка доступности различных форматов
    formats_info = []
    
    # CSV - всегда доступен
    formats_info.append("✓ CSV - Табличные данные (всегда доступен)")
    formats_info.append("  - Координаты и значения")
    formats_info.append("  - Настраиваемые разделители")
    formats_info.append("  - Метаданные как комментарии")
    formats_info.append("")
    
    # GeoTIFF
    try:
        import rasterio
        formats_info.append("✓ GeoTIFF - Геореференцированные растры")
        formats_info.append("  - Поддержка проекций")
        formats_info.append("  - Различные типы сжатия")
        formats_info.append("  - Интеграция с ГИС")
    except ImportError:
        formats_info.append("❌ GeoTIFF - требует: pip install rasterio")
    formats_info.append("")
    
    # VTK
    try:
        import vtk
        formats_info.append("✓ VTK - 3D научная визуализация")
        formats_info.append("  - XML и legacy форматы") 
        formats_info.append("  - Поддержка ParaView, VisIt")
        formats_info.append("  - Скалярные и векторные поля")
    except ImportError:
        formats_info.append("❌ VTK - требует: pip install vtk")
    formats_info.append("")
    
    # DXF
    try:
        import ezdxf
        formats_info.append("✓ DXF - CAD системы")
        formats_info.append("  - AutoCAD совместимость")
        formats_info.append("  - Изолинии и контуры")
        formats_info.append("  - Цветовая кодировка")
    except ImportError:
        formats_info.append("❌ DXF - требует: pip install ezdxf")
    formats_info.append("")
    
    report_lines.extend(formats_info)
    
    # Рекомендации по использованию
    report_lines.extend([
        "## Рекомендации по использованию",
        "",
        "### Для ГИС анализа:",
        "- GeoTIFF для растровых данных",
        "- CSV для точечных данных с координатами",
        "",
        "### Для 3D визуализации:",
        "- VTK для ParaView/VisIt",
        "- Включить все доступные атрибуты",
        "",
        "### Для CAD систем:",
        "- DXF с изолиниями для планов",
        "- Настроить единицы измерения",
        "",
        "### Для обмена данными:",
        "- CSV - универсальный формат",
        "- Включить метаданные",
        "",
        "---",
        "Создано с помощью Coal Interpolation Tool"
    ])
    
    # Сохранение отчета
    report_file = output_dir / 'export_capabilities_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Отчет сохранен: {report_file}")
    
    # Также выводим на экран
    print("\n" + "\n".join(report_lines))


def main():
    """Главная функция для демонстрации всех примеров экспорта."""
    print("🚀 Примеры экспорта данных - Coal Interpolation Tool")
    print("=" * 60)
    
    # Создание директории для результатов
    output_dir = Path(__file__).parent / 'export_results'
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Результаты будут сохранены в: {output_dir.absolute()}")
    
    # Создание примера данных
    point_data, grid_data = create_sample_data()
    
    # Примеры экспорта в различные форматы
    example_csv_export(point_data, grid_data, output_dir)
    example_geotiff_export(point_data, grid_data, output_dir)
    example_vtk_export(point_data, grid_data, output_dir)
    example_dxf_export(point_data, grid_data, output_dir)
    
    # Пакетный экспорт
    example_batch_export(point_data, grid_data, output_dir)
    
    # Отчет о возможностях
    export_summary_report(point_data, grid_data, output_dir)
    
    print("\n🎉 Все примеры экспорта выполнены успешно!")
    print(f"📂 Проверьте результаты в директории: {output_dir.absolute()}")
    
    # Список созданных файлов
    if output_dir.exists():
        files = list(output_dir.rglob('*'))
        files = [f for f in files if f.is_file()]
        
        print(f"\n📋 Создано файлов: {len(files)}")
        for file in sorted(files):
            rel_path = file.relative_to(output_dir)
            size_kb = file.stat().st_size / 1024
            print(f"  - {rel_path} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()