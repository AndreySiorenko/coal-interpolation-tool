#!/usr/bin/env python3
"""
Диагностика Python окружения для запуска проекта интерполяции.
"""

import sys
import os
import platform
from pathlib import Path

def main():
    print("=" * 50)
    print("ДИАГНОСТИКА PYTHON ОКРУЖЕНИЯ")
    print("=" * 50)
    
    print(f"Python версия: {sys.version}")
    print(f"Python исполняемый файл: {sys.executable}")
    print(f"Платформа: {platform.platform()}")
    print(f"Архитектура: {platform.architecture()}")
    print(f"Текущая директория: {os.getcwd()}")
    
    print("\n" + "=" * 50)
    print("ПРОВЕРКА СТРУКТУРЫ ПРОЕКТА")
    print("=" * 50)
    
    current_dir = Path(__file__).parent
    key_files = [
        'main.py',
        'demo.py', 
        'run_demo.py',
        'src/gui/main_window.py',
        'src/core/interpolation/idw.py'
    ]
    
    for file_path in key_files:
        full_path = current_dir / file_path
        status = "✅" if full_path.exists() else "❌"
        print(f"{status} {file_path}")
    
    print("\n" + "=" * 50)
    print("ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    print("=" * 50)
    
    required_modules = ['tkinter', 'numpy', 'pandas', 'matplotlib']
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - НЕ УСТАНОВЛЕН")
    
    print("\n" + "=" * 50)
    print("ПОПЫТКА ЗАПУСКА ДЕМО")
    print("=" * 50)
    
    try:
        # Проверяем tkinter
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        print("✅ Tkinter GUI готов к работе")
        
        # Проверяем demo.py
        demo_path = current_dir / 'demo.py'
        if demo_path.exists():
            print("✅ demo.py найден")
            print("ЗАПУСКАЕМ ДЕМО...")
            # Импортируем и запускаем demo
            sys.path.insert(0, str(current_dir))
            try:
                import demo
                print("✅ ДЕМО ЗАПУЩЕН УСПЕШНО!")
            except Exception as e:
                print(f"❌ Ошибка запуска demo: {e}")
        else:
            print("❌ demo.py не найден")
            
    except ImportError as e:
        print(f"❌ Проблема с зависимостями: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
    
    print("\n" + "=" * 50)
    print("ИНСТРУКЦИИ ПО ЗАПУСКУ")
    print("=" * 50)
    print("1. Убедитесь что Python 3.8+ установлен")
    print("2. Установите зависимости: pip install numpy pandas matplotlib tkinter")
    print("3. Запустите demo: python demo.py")
    print("4. Или запустите GUI: python main.py")
    print("=" * 50)

if __name__ == "__main__":
    main()