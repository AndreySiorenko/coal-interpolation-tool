#!/usr/bin/env python3
"""
Debug i18n loading to find the problem.
"""

import sys
import json
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def debug_json_loading():
    """Debug JSON translation file loading."""
    print("=== JSON LOADING DEBUG ===")
    
    # Check file paths
    ru_file = Path(__file__).parent / "src" / "i18n" / "locales" / "ru" / "messages.json"
    en_file = Path(__file__).parent / "src" / "i18n" / "locales" / "en" / "messages.json"
    
    print(f"Russian file: {ru_file}")
    print(f"Exists: {ru_file.exists()}")
    
    print(f"English file: {en_file}")
    print(f"Exists: {en_file.exists()}")
    
    if ru_file.exists():
        try:
            with open(ru_file, 'r', encoding='utf-8') as f:
                ru_data = json.load(f)
            
            print(f"Russian JSON loaded successfully")
            print(f"Language: {ru_data.get('language')}")
            messages = ru_data.get('messages', {})
            print(f"Messages count: {len(messages)}")
            
            # Check specific translations
            test_keys = ['File', 'Open', 'Save', 'Language Settings']
            for key in test_keys:
                if key in messages:
                    print(f"  '{key}' -> '{messages[key]}'")
                else:
                    print(f"  '{key}' -> NOT FOUND")
                    
        except Exception as e:
            print(f"Error loading Russian JSON: {e}")
    
    print("\n" + "="*40)

def debug_translation_manager():
    """Debug TranslationManager loading."""
    print("=== TRANSLATION MANAGER DEBUG ===")
    
    try:
        from src.i18n.translation_manager import TranslationManager
        
        # Create manager
        print("Creating TranslationManager...")
        manager = TranslationManager('en')
        
        print(f"Available languages: {manager.get_available_languages()}")
        
        # Check translations
        for lang in ['en', 'ru']:
            if lang in manager.translations:
                translations = manager.translations[lang]
                print(f"{lang}: {len(translations)} translations")
                
                # Test translations
                test_keys = ['File', 'Open', 'Save']
                for key in test_keys:
                    translation = manager.translate(key, lang)
                    print(f"  {lang} '{key}' -> '{translation}'")
            else:
                print(f"{lang}: NOT LOADED")
        
        # Test language switching
        print("\nTesting language switching...")
        print(f"Current language: {manager.current_language}")
        
        manager.set_language('ru')
        print(f"After switch to ru: {manager.current_language}")
        file_ru = manager.translate('File')
        print(f"File in Russian: '{file_ru}'")
        
        manager.set_language('en')
        print(f"After switch to en: {manager.current_language}")
        file_en = manager.translate('File')
        print(f"File in English: '{file_en}'")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_i18n_module():
    """Debug the main i18n module."""
    print("\n=== I18N MODULE DEBUG ===")
    
    try:
        from src.i18n import init_i18n, set_language, get_current_language, _
        
        # Initialize
        print("Initializing i18n...")
        manager = init_i18n('en')
        print(f"Initial language: {get_current_language()}")
        
        # Test English
        print("Testing English...")
        file_en = _('File')
        open_en = _('Open')
        print(f"File: '{file_en}'")
        print(f"Open: '{open_en}'")
        
        # Switch to Russian
        print("\nSwitching to Russian...")
        set_language('ru')
        current = get_current_language()
        print(f"Current language after switch: {current}")
        
        # Test Russian
        file_ru = _('File')
        open_ru = _('Open')
        print(f"File: '{file_ru}'")
        print(f"Open: '{open_ru}'")
        
        # Check if they're different
        if file_en != file_ru:
            print("✓ Translations are working!")
        else:
            print("✗ Translations are NOT working - same text")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("I18N Loading Debug")
    print("="*50)
    
    debug_json_loading()
    debug_translation_manager()
    debug_i18n_module()
    
    print("\n" + "="*50)
    print("Debug completed")