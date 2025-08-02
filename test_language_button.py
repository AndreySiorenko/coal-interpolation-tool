#!/usr/bin/env python3
"""
Simple GUI test for language switching button.
"""

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def create_test_window():
    """Create a simple test window with language switching."""
    
    # Initialize i18n
    from src.i18n import init_i18n, set_language, get_current_language, _, add_language_change_listener
    
    print("Initializing i18n system...")
    manager = init_i18n('en')
    print(f"Initial language: {get_current_language()}")
    
    # Create window
    root = tk.Tk()
    root.title("Language Switch Test")
    root.geometry("400x300")
    
    # Language change handler
    def on_language_changed(old_lang, new_lang):
        print(f"Language changed: {old_lang} -> {new_lang}")
        # Update UI elements
        title_var.set(_("Language Switch Test"))
        file_var.set(_("File"))
        open_var.set(_("Open"))
        lang_var.set(f"Current: {new_lang}")
        
        # Update button
        current_lang = get_current_language()
        lang_text = "🇷🇺 RU" if current_lang == 'ru' else "🇺🇸 EN"
        lang_button.config(text=lang_text)
    
    add_language_change_listener(on_language_changed)
    
    # UI elements with StringVar for dynamic updates
    title_var = tk.StringVar(value=_("Language Switch Test"))
    file_var = tk.StringVar(value=_("File"))
    open_var = tk.StringVar(value=_("Open"))
    lang_var = tk.StringVar(value=f"Current: {get_current_language()}")
    
    # Create UI
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, textvariable=title_var, font=("Arial", 14, "bold"))
    title_label.pack(pady=(0, 20))
    
    # Current language
    lang_label = ttk.Label(main_frame, textvariable=lang_var)
    lang_label.pack(pady=(0, 20))
    
    # Test elements
    ttk.Label(main_frame, textvariable=file_var).pack(pady=5)
    ttk.Label(main_frame, textvariable=open_var).pack(pady=5)
    
    # Language toggle button
    def toggle_language():
        current_lang = get_current_language()
        new_lang = 'en' if current_lang == 'ru' else 'ru'
        print(f"Toggling from {current_lang} to {new_lang}")
        set_language(new_lang)
    
    current_lang = get_current_language()
    lang_text = "🇷🇺 RU" if current_lang == 'ru' else "🇺🇸 EN"
    
    lang_button = ttk.Button(
        main_frame,
        text=lang_text,
        command=toggle_language,
        width=10
    )
    lang_button.pack(pady=20)
    
    # Instructions
    instructions = """
Click the language button to toggle between Russian and English.
The interface should update immediately.
"""
    ttk.Label(main_frame, text=instructions, justify=tk.CENTER).pack(pady=20)
    
    print("Test window created. Click the language button to test switching.")
    return root

if __name__ == "__main__":
    try:
        root = create_test_window()
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()