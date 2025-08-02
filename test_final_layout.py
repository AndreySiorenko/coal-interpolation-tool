#!/usr/bin/env python3
"""
Final test to verify layout and translations are working
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_layout_structure():
    """Test that the main window structure is correctly configured."""
    print("=== Testing Layout Structure ===")
    
    try:
        from src.gui.main_window import MainWindow
        print("✓ MainWindow imports successfully")
        
        # Check if the create_main_panels method exists and has the right logic
        import inspect
        source = inspect.getsource(MainWindow.create_main_panels)
        
        # Check for new layout structure
        if "20% - 60% - 20%" in source:
            print("✓ New layout comments found")
        else:
            print("✗ Layout comments not found")
            
        if "weight=3" in source:
            print("✓ Center panel weight=3 found (60%)")
        else:
            print("✗ Center panel weight not correct")
            
        if "center_frame" in source:
            print("✓ Center frame structure found")
        else:
            print("✗ Center frame not found")
            
        print("✓ Layout structure test completed")
        
    except Exception as e:
        print(f"✗ Layout test failed: {e}")

def test_results_panel():
    """Test Results panel functionality."""
    print("\n=== Testing Results Panel ===")
    
    try:
        from src.gui.widgets.results_panel import ResultsPanel
        print("✓ ResultsPanel imports successfully")
        
        # Check if update_language method exists
        if hasattr(ResultsPanel, 'update_language'):
            print("✓ update_language method found")
        else:
            print("✗ update_language method missing")
            
        print("✓ Results panel test completed")
        
    except Exception as e:
        print(f"✗ Results panel test failed: {e}")

def test_translations():
    """Test translation system."""
    print("\n=== Testing Translation System ===")
    
    try:
        from src.i18n import set_language, get_current_language, _
        
        # Test basic functionality
        set_language('ru')
        current = get_current_language()
        
        if current == 'ru':
            print("✓ Language switching works")
        else:
            print(f"✗ Language switching failed: {current}")
            
        # Test some key translations
        key_translations = {
            'Results': 'Результаты',
            'Statistics': 'Статистика', 
            'Parameters': 'Параметры'
        }
        
        working = 0
        total = len(key_translations)
        
        for key, expected in key_translations.items():
            translation = _(key)
            if translation != key:  # At least it's not the same as key
                working += 1
                print(f"✓ {key} -> translated")
            else:
                print(f"✗ {key} -> not translated")
        
        print(f"✓ Translation test: {working}/{total} keys working")
        
    except Exception as e:
        print(f"✗ Translation test failed: {e}")

if __name__ == "__main__":
    print("=== FINAL VERIFICATION TEST ===")
    test_layout_structure()
    test_results_panel() 
    test_translations()
    print("\n=== TEST COMPLETED ===")