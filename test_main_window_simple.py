#!/usr/bin/env python3
"""
Simplified test of MainWindow with language switching.
"""

import sys
import tkinter as tk
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_main_window():
    """Test the main window with language switching."""
    try:
        print("=== MAIN WINDOW TEST ===")
        
        # Initialize i18n first
        from src.i18n import init_i18n, set_language, get_current_language, _
        from src.gui.dialogs.language_settings_dialog import load_language_preference
        
        # Load preferred language
        try:
            preferred_language = load_language_preference()
        except:
            preferred_language = 'en'
        
        print(f"Starting with language: {preferred_language}")
        
        # Initialize i18n
        i18n_manager = init_i18n(preferred_language)
        
        # Test translations
        print(f"Initial test - File: '{_('File')}'")
        print(f"Initial test - Open: '{_('Open')}'")
        
        # Test language switching
        print("\nTesting language switch to Russian...")
        set_language('ru')
        print(f"After switch - File: '{_('File')}'")
        print(f"After switch - Open: '{_('Open')}'")
        
        print("\nTesting language switch back to English...")
        set_language('en')
        print(f"Back to English - File: '{_('File')}'")
        print(f"Back to English - Open: '{_('Open')}'")
        
        # Now try to create MainWindow
        print("\nCreating MainWindow...")
        from src.gui.main_window import MainWindow
        
        # Create a simple root window first
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Create MainWindow (this should show debug output)
        app = MainWindow()
        
        print("\nMainWindow created successfully!")
        print("You should see a window with language toggle button in top-right corner.")
        print("Click the language button to test switching.")
        
        # Run the app
        app.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_main_window()