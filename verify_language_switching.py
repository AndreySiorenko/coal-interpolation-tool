#!/usr/bin/env python3
"""
Simple verification script for language switching.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Simple test of language switching."""
    try:
        print("Testing language switching...")
        
        # Import i18n system
        from src.i18n import init_i18n, set_language, get_current_language, _
        
        # Initialize
        i18n_manager = init_i18n('en')
        
        # Test English
        print(f"English - File: {_('File')}")
        print(f"English - Open: {_('Open')}")
        print(f"English - Language Settings: {_('Language Settings')}")
        
        # Switch to Russian
        set_language('ru')
        
        # Test Russian
        print(f"Russian - File: {_('File')}")
        print(f"Russian - Open: {_('Open')}")
        print(f"Russian - Language Settings: {_('Language Settings')}")
        
        print("Language switching test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()