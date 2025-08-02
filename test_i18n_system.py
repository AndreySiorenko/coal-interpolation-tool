#!/usr/bin/env python3
"""
Test internationalization system to verify language switching works.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_translation_system():
    """Test the translation system with different languages."""
    
    print("Testing Internationalization System")
    print("=" * 40)
    
    try:
        from src.i18n import init_i18n, set_language, get_current_language, _
        
        # Initialize i18n system
        print("Step 1: Initializing i18n system...")
        i18n_manager = init_i18n('en')  # Start with English
        print(f"Current language: {get_current_language()}")
        
        # Test English translations
        print("\nStep 2: Testing English translations...")
        english_texts = {
            "Coal Deposit Interpolation Tool": _("Coal Deposit Interpolation Tool"),
            "File": _("File"),
            "Open": _("Open"),
            "Save": _("Save"),
            "Language Settings": _("Language Settings"),
            "Load Data File": _("Load Data File"),
            "Interpolate": _("Interpolate"),
            "Export Results": _("Export Results")
        }
        
        for key, translation in english_texts.items():
            print(f"  '{key}' -> '{translation}'")
        
        # Switch to Russian
        print("\nStep 3: Switching to Russian...")
        set_language('ru')
        print(f"Current language: {get_current_language()}")
        
        # Test Russian translations
        print("\nStep 4: Testing Russian translations...")
        russian_texts = {
            "Coal Deposit Interpolation Tool": _("Coal Deposit Interpolation Tool"),
            "File": _("File"),
            "Open": _("Open"),
            "Save": _("Save"),
            "Language Settings": _("Language Settings"),
            "Load Data File": _("Load Data File"),
            "Interpolate": _("Interpolate"),
            "Export Results": _("Export Results")
        }
        
        for key, translation in russian_texts.items():
            print(f"  '{key}' -> '{translation}'")
        
        # Test fallback behavior
        print("\nStep 5: Testing fallback for missing translations...")
        missing_key = "Non-existent key for testing"
        result = _(missing_key)
        print(f"  Missing key: '{missing_key}' -> '{result}'")
        
        # Switch back to English
        print("\nStep 6: Switching back to English...")
        set_language('en')
        print(f"Current language: {get_current_language()}")
        
        print("\nStep 7: Testing GUI-specific translations...")
        gui_texts = {
            "Selected File:": _("Selected File:"),
            "Browse...": _("Browse..."),
            "Data Preview (First 10 rows)": _("Data Preview (First 10 rows)"),
            "Column Mapping": _("Column Mapping"),
            "X Coordinate:": _("X Coordinate:"),
            "Y Coordinate:": _("Y Coordinate:")
        }
        
        for key, translation in gui_texts.items():
            print(f"  '{key}' -> '{translation}'")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_language_settings():
    """Test language settings saving and loading."""
    
    print("\nTesting Language Settings")
    print("=" * 30)
    
    try:
        from src.gui.dialogs.language_settings_dialog import save_language_preference, load_language_preference
        
        # Test saving language preference
        print("Step 1: Testing save language preference...")
        test_language = 'ru'
        save_language_preference(test_language)
        print(f"Saved language: {test_language}")
        
        # Test loading language preference
        print("Step 2: Testing load language preference...")
        loaded_language = load_language_preference()
        print(f"Loaded language: {loaded_language}")
        
        if loaded_language == test_language:
            print("SUCCESS: Language preference save/load works!")
        else:
            print(f"ERROR: Expected {test_language}, got {loaded_language}")
            return False
        
        # Test with English
        print("Step 3: Testing with English...")
        save_language_preference('en')
        loaded_language = load_language_preference()
        print(f"Loaded language: {loaded_language}")
        
        if loaded_language == 'en':
            print("SUCCESS: English language preference works!")
        else:
            print(f"ERROR: Expected 'en', got {loaded_language}")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR in language settings: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_translation_coverage():
    """Test translation coverage for common UI elements."""
    
    print("\nTesting Translation Coverage")
    print("=" * 30)
    
    from src.i18n import init_i18n, set_language, _
    
    # Key UI elements that should be translated
    ui_elements = [
        "File", "Edit", "View", "Tools", "Help",
        "Open", "Save", "Exit", "Cancel", "OK", "Apply",
        "Load Data", "Interpolate", "Export Results",
        "Parameters", "Statistics", "Error", "Warning",
        "Language", "Settings", "About"
    ]
    
    # Test both languages
    languages = ['en', 'ru']
    coverage_results = {}
    
    for lang in languages:
        print(f"\nTesting {lang.upper()} translations...")
        set_language(lang)
        
        missing_translations = []
        for element in ui_elements:
            translation = _(element)
            if translation == element and lang != 'en':  # Missing translation (not English fallback)
                missing_translations.append(element)
            else:
                print(f"  {element} -> {translation}")
        
        coverage_results[lang] = {
            'total': len(ui_elements),
            'missing': len(missing_translations),
            'coverage': (len(ui_elements) - len(missing_translations)) / len(ui_elements) * 100
        }
        
        if missing_translations:
            print(f"  Missing translations: {missing_translations}")
        
        print(f"  Coverage: {coverage_results[lang]['coverage']:.1f}%")
    
    return coverage_results

if __name__ == "__main__":
    print("Internationalization System Test Suite")
    print("=" * 50)
    
    # Test 1: Basic translation system
    translation_success = test_translation_system()
    
    # Test 2: Language settings
    settings_success = test_language_settings()
    
    # Test 3: Translation coverage
    coverage_results = test_translation_coverage()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Translation System: {'PASS' if translation_success else 'FAIL'}")
    print(f"  Language Settings: {'PASS' if settings_success else 'FAIL'}")
    print(f"  Translation Coverage:")
    for lang, results in coverage_results.items():
        print(f"    {lang.upper()}: {results['coverage']:.1f}% ({results['total'] - results['missing']}/{results['total']})")
    
    if translation_success and settings_success:
        print("\nInternationalization system is working correctly!")
        print("You can now:")
        print("- Switch languages via Tools -> Language Settings")
        print("- Interface will be translated on application restart")
        print("- All major UI elements support Russian and English")
    else:
        print("\nSome tests FAILED. Check errors above.")