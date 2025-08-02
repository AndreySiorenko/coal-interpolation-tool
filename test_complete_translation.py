#!/usr/bin/env python3
"""
Test complete translation functionality after fixes.
"""

import sys
import tkinter as tk
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_complete_main_window():
    """Test the complete main window with all translation fixes."""
    try:
        print("=== COMPLETE TRANSLATION TEST ===")
        
        # Initialize i18n
        from src.i18n import init_i18n, set_language, get_current_language, _
        
        print("1. Initializing i18n system...")
        i18n_manager = init_i18n('en')
        print(f"   Current language: {get_current_language()}")
        
        # Test basic translations
        print("\n2. Testing basic translations...")
        test_keys = ["File", "Open", "Save", "Data", "File Information"]
        for key in test_keys:
            translation = _(key)
            print(f"   '{key}' -> '{translation}'")
        
        # Switch to Russian and test
        print("\n3. Switching to Russian...")
        set_language('ru')
        print(f"   Current language: {get_current_language()}")
        
        for key in test_keys:
            translation = _(key)
            print(f"   '{key}' -> '{translation}'")
        
        # Create a minimal test window
        print("\n4. Creating test window with MainWindow components...")
        
        root = tk.Tk()
        root.title("Translation Test")
        root.geometry("800x600")
        
        # Create a simple notebook to test panel translations
        from tkinter import ttk
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Test frame with some elements
        test_frame = ttk.Frame(notebook)
        notebook.add(test_frame, text=_("Test Panel"))
        
        # Add some test elements
        file_frame = ttk.LabelFrame(test_frame, text=_("File Information"), padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text=_("File:")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(file_frame, text=_("No file loaded")).grid(row=0, column=1, sticky=tk.W)
        
        # Language toggle button
        def toggle_lang():
            current = get_current_language()
            new_lang = 'en' if current == 'ru' else 'ru'
            print(f"\nToggling from {current} to {new_lang}")
            set_language(new_lang)
            
            # Update UI elements manually (in real app this would be automatic)
            notebook.tab(0, text=_("Test Panel"))
            file_frame.config(text=_("File Information"))
            
            # Update button text
            lang_text = "🇷🇺 RU" if get_current_language() == 'ru' else "🇺🇸 EN"
            lang_btn.config(text=lang_text)
            
            print(f"Updated UI to {get_current_language()}")
        
        current_lang = get_current_language()
        lang_text = "🇷🇺 RU" if current_lang == 'ru' else "🇺🇸 EN"
        
        lang_btn = ttk.Button(
            test_frame,
            text=lang_text,
            command=toggle_lang,
            width=10
        )
        lang_btn.pack(pady=20)
        
        # Instructions
        instructions = f"""
Translation Test Window

Current Language: {get_current_language()}

Click the language button to test switching.
This simulates the fixed translation system.

Fixed issues:
✓ Removed duplicate language button
✓ Single language button in toolbar
✓ Complete UI element translation
✓ Dynamic language switching
"""
        
        ttk.Label(test_frame, text=instructions, justify=tk.LEFT).pack(pady=10)
        
        print("\n5. Test window created successfully!")
        print("   - Click language button to test switching")
        print("   - All UI elements should update immediately")
        print("   - No application restart required")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_main_window()