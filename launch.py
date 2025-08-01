#!/usr/bin/env python3
"""
Универсальный лаунчер для проекта интерполяции угольных месторождений.
Автоматически определяет доступные зависимости и запускает подходящую версию.
"""

import sys
import os
import platform
from pathlib import Path

def check_dependencies():
    """Проверка доступности зависимостей."""
    deps_status = {}
    
    # Обязательные зависимости
    required = ['tkinter']
    for dep in required:
        try:
            __import__(dep)
            deps_status[dep] = True
        except ImportError:
            deps_status[dep] = False
    
    # Опциональные зависимости для полной функциональности
    optional = ['numpy', 'pandas', 'scipy', 'matplotlib', 'plotly', 'vtk', 'rasterio', 'ezdxf']
    for dep in optional:
        try:
            __import__(dep)
            deps_status[dep] = True
        except ImportError:
            deps_status[dep] = False
    
    return deps_status

def print_banner():
    """Вывод баннера приложения."""
    print("=" * 60)
    print("🎯 COAL DEPOSIT INTERPOLATION TOOL v1.0.0-rc1")
    print("=" * 60)
    print("🐍 Проект интерполяции угольных месторождений")
    print("✨ Professional geological data analysis")
    print("=" * 60)

def print_system_info():
    """Вывод информации о системе."""
    print("\n🔍 СИСТЕМНАЯ ИНФОРМАЦИЯ:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Платформа: {platform.platform()}")
    print(f"   Архитектура: {platform.architecture()[0]}")
    print(f"   Директория: {os.getcwd()}")

def analyze_project_readiness(deps):
    """Анализ готовности проекта."""
    print("\n📊 АНАЛИЗ ГОТОВНОСТИ КОМПОНЕНТОВ:")
    
    # Проверка GUI возможностей
    if deps['tkinter']:
        print("   ✅ GUI готов (Tkinter)")
    else:
        print("   ❌ GUI недоступен (Tkinter отсутствует)")
        return False
    
    # Проверка численных вычислений
    core_libs = ['numpy', 'pandas', 'scipy']
    core_ready = all(deps.get(lib, False) for lib in core_libs)
    
    if core_ready:
        print("   ✅ Численные вычисления (NumPy, Pandas, SciPy)")
    else:
        print("   ⚠️  Ограниченные численные вычисления")
    
    # Проверка визуализации
    viz_libs = ['matplotlib', 'plotly', 'vtk']
    viz_ready = any(deps.get(lib, False) for lib in viz_libs)
    
    if viz_ready:
        available_viz = [lib for lib in viz_libs if deps.get(lib, False)]
        print(f"   ✅ Визуализация ({', '.join(available_viz)})")
    else:
        print("   ⚠️  Базовая визуализация (fallback)")
    
    # Проверка экспорта
    export_libs = ['rasterio', 'ezdxf']
    export_ready = any(deps.get(lib, False) for lib in export_libs)
    
    if export_ready:
        available_export = [lib for lib in export_libs if deps.get(lib, False)]
        print(f"   ✅ Расширенный экспорт ({', '.join(available_export)})")
    else:
        print("   ⚠️  Базовый экспорт (CSV)")
    
    return True

def recommend_launch_mode(deps):
    """Рекомендация режима запуска."""
    print("\n🚀 РЕКОМЕНДАЦИИ ПО ЗАПУСКУ:")
    
    core_libs = ['numpy', 'pandas', 'scipy', 'matplotlib']
    full_mode_ready = all(deps.get(lib, False) for lib in core_libs)
    
    if full_mode_ready:
        print("   🎯 ПОЛНЫЙ РЕЖИМ - Все функции доступны")
        print("   ↳ Команда: python main.py")
        return "full"
    else:
        print("   🎮 ДЕМО РЕЖИМ - Интерфейс и базовые функции")
        print("   ↳ Команда: python demo.py")
        return "demo"

def launch_application(mode, deps):
    """Запуск приложения в выбранном режиме."""
    current_dir = Path(__file__).parent
    
    print(f"\n🎬 ЗАПУСК В {mode.upper()} РЕЖИМЕ...")
    
    if mode == "full":
        main_path = current_dir / "main.py"
        if main_path.exists():
            print("   📱 Загрузка полного GUI приложения...")
            try:
                sys.path.insert(0, str(current_dir))
                
                # Импорт и запуск главного приложения
                from src.gui.main_window import main as main_app
                print("   ✅ GUI модуль загружен")
                print("   🎯 Запуск основного приложения...")
                
                main_app()
                
            except ImportError as e:
                print(f"   ❌ Ошибка импорта GUI: {e}")
                print("   🔄 Переключение на демо режим...")
                launch_application("demo", deps)
            except Exception as e:
                print(f"   ❌ Ошибка запуска: {e}")
                return False
        else:
            print("   ❌ main.py не найден")
            print("   🔄 Переключение на демо режим...")
            launch_application("demo", deps)
    
    elif mode == "demo":
        demo_path = current_dir / "demo.py"
        if demo_path.exists():
            print("   🎮 Загрузка демо приложения...")
            try:
                sys.path.insert(0, str(current_dir))
                
                # Импорт и запуск демо
                import demo
                print("   ✅ Демо модуль загружен")
                print("   🎯 Запуск демо интерфейса...")
                
                demo.main()
                
            except Exception as e:
                print(f"   ❌ Ошибка запуска демо: {e}")
                return False
        else:
            print("   ❌ demo.py не найден")
            return False
    
    return True

def print_installation_guide(deps):
    """Вывод инструкций по установке зависимостей."""
    missing_core = []
    missing_optional = []
    
    core_libs = ['numpy', 'pandas', 'scipy', 'matplotlib']
    optional_libs = ['plotly', 'vtk', 'rasterio', 'ezdxf']
    
    for lib in core_libs:
        if not deps.get(lib, False):
            missing_core.append(lib)
    
    for lib in optional_libs:
        if not deps.get(lib, False):
            missing_optional.append(lib)
    
    if missing_core or missing_optional:
        print("\n📦 ИНСТРУКЦИИ ПО УСТАНОВКЕ:")
        print("   Для полной функциональности установите:")
        
        if missing_core:
            print(f"   🔧 Основные: pip install {' '.join(missing_core)}")
        
        if missing_optional:
            print(f"   ⭐ Опциональные: pip install {' '.join(missing_optional)}")
        
        print("\n   📋 Или используйте requirements.txt:")
        print("   pip install -r requirements.txt")

def main():
    """Главная функция лаунчера."""
    try:
        print_banner()
        print_system_info()
        
        # Проверка зависимостей
        deps = check_dependencies()
        
        # Анализ готовности
        gui_ready = analyze_project_readiness(deps)
        
        if not gui_ready:
            print("\n❌ КРИТИЧЕСКАЯ ОШИБКА:")
            print("   Tkinter не найден - GUI невозможен")
            print("   Переустановите Python с поддержкой Tkinter")
            return False
        
        # Рекомендация режима
        mode = recommend_launch_mode(deps)
        
        # Инструкции по установке
        print_installation_guide(deps)
        
        # Запуск приложения
        success = launch_application(mode, deps)
        
        if success:
            print("\n✅ ПРИЛОЖЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        else:
            print("\n❌ ОШИБКА ЗАПУСКА ПРИЛОЖЕНИЯ")
            return False
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n🛑 ПРЕРЫВАНИЕ ПОЛЬЗОВАТЕЛЕМ")
        return True
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)