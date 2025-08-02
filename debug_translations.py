#!/usr/bin/env python3
"""
Debug translation loading to find why Russian translations don't work.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def debug_translation_loading():
    """Debug translation loading process."""
    
    print("Translation Loading Debug")
    print("=" * 30)
    
    try:
        from src.i18n.translation_manager import TranslationManager
        
        # Create translation manager
        print("Creating TranslationManager...")
        manager = TranslationManager('en')
        
        print(f"Locale directory: {manager.locale_dir}")
        print(f"Directory exists: {manager.locale_dir.exists()}")
        
        # Check available languages
        print(f"Available languages: {list(manager.translations.keys())}")
        
        # Check Russian translations
        if 'ru' in manager.translations:
            ru_translations = manager.translations['ru']
            print(f"Russian translations count: {len(ru_translations)}")
            
            # Show some translations
            test_keys = ['File', 'Open', 'Save', 'Language Settings', 'Interpolate']
            for key in test_keys:
                translation = ru_translations.get(key, 'NOT FOUND')
                print(f"  '{key}' -> '{translation}'")
        else:
            print("Russian translations not loaded!")
        
        # Test direct translation
        print("\nTesting direct translation...")
        manager.set_language('ru')
        test_translation = manager.translate('File')
        print(f"Direct translation of 'File': '{test_translation}'")
        
        # Test with context
        manager.set_language('en')
        en_translation = manager.translate('File')
        print(f"English translation of 'File': '{en_translation}'")
        
        # Show raw translation storage
        print(f"\nRaw translations structure:")
        for lang, translations in manager.translations.items():
            print(f"  {lang}: {len(translations)} translations")
            if translations:
                first_few = dict(list(translations.items())[:3])
                print(f"    Sample: {first_few}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_json_files():
    """Check if JSON translation files exist and are valid."""
    
    print("\nJSON Files Check")
    print("=" * 20)
    
    locale_dir = Path(__file__).parent / "src" / "i18n" / "locales"
    
    for lang in ['en', 'ru']:
        json_file = locale_dir / lang / 'messages.json'
        print(f"Checking {lang}: {json_file}")
        print(f"  Exists: {json_file.exists()}")
        
        if json_file.exists():
            try:
                import json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"  Valid JSON: Yes")
                print(f"  Messages count: {len(data.get('messages', {}))}")
                
                # Check specific translations
                messages = data.get('messages', {})
                test_keys = ['File', 'Open', 'Save']
                for key in test_keys:
                    if key in messages:
                        print(f"    '{key}' -> '{messages[key]}'")
                    else:
                        print(f"    '{key}' -> NOT FOUND")
                
            except Exception as e:
                print(f"  Error loading JSON: {e}")

if __name__ == "__main__":
    print("Translation System Debug")
    print("=" * 40)
    
    # Check JSON files first
    check_json_files()
    
    # Debug translation loading
    debug_translation_loading()