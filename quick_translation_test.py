#!/usr/bin/env python3
"""
Quick test to check if translations are working.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_translations():
    """Test translation functionality."""
    print("=== TRANSLATION TEST ===")
    
    try:
        # Import the i18n system
        from src.i18n import init_i18n, set_language, get_current_language, _, get_translation_manager
        
        # Initialize
        print("1. Initializing i18n...")
        manager = init_i18n('en')
        print(f"   Initialized with language: {get_current_language()}")
        
        # Check available translations
        tm = get_translation_manager()
        print(f"   Available languages: {tm.get_available_languages()}")
        print(f"   Translations loaded:")
        for lang, translations in tm.translations.items():
            print(f"     {lang}: {len(translations)} translations")
        
        # Test English translations
        print("\n2. Testing English translations:")
        print(f"   File: '{_('File')}'")
        print(f"   Open: '{_('Open')}'")
        print(f"   Language Settings: '{_('Language Settings')}'")
        
        # Switch to Russian
        print("\n3. Switching to Russian...")
        set_language('ru')
        current = get_current_language()
        print(f"   Current language after switch: {current}")
        
        # Test Russian translations
        print("\n4. Testing Russian translations:")
        print(f"   File: '{_('File')}'")
        print(f"   Open: '{_('Open')}'")
        print(f"   Language Settings: '{_('Language Settings')}'")
        
        # Check specific Russian translations
        print("\n5. Direct translation check:")
        ru_file = tm.translate('File', 'ru')
        print(f"   Direct Russian translation of 'File': '{ru_file}'")
        
        # Check if translations exist in storage
        if 'ru' in tm.translations:
            ru_translations = tm.translations['ru']
            print(f"   'File' in Russian translations: {'File' in ru_translations}")
            if 'File' in ru_translations:
                print(f"   Russian 'File' = '{ru_translations['File']}'")
            else:
                print("   'File' key not found in Russian translations")
                print(f"   Available keys (first 10): {list(ru_translations.keys())[:10]}")
        
        print("\n6. Testing language switching back to English...")
        set_language('en')
        print(f"   File: '{_('File')}'")
        print(f"   Open: '{_('Open')}'")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_translations()
    if success:
        print("\n✓ Translation test completed")
    else:
        print("\n✗ Translation test failed")