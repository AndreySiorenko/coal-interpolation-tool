#!/usr/bin/env python3
"""
Test language switching in a simple GUI to verify it works.
"""

import sys
import tkinter as tk
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_language_gui():
    """Create a simple GUI to test language switching."""
    
    # Initialize i18n system
    from src.i18n import init_i18n, set_language, get_current_language, _
    from src.gui.dialogs.language_settings_dialog import show_language_settings_dialog
    
    # Start with English
    i18n_manager = init_i18n('en')
    
    # Create test window
    root = tk.Tk()
    root.title("Language Test")
    root.geometry("400x300")
    
    # Current language display
    lang_var = tk.StringVar(value=f"Current: {get_current_language()}")
    
    def update_language_display():
        lang_var.set(f"Current: {get_current_language()}")
        # Update button texts (would normally need restart)
        file_btn.config(text=_("File"))
        open_btn.config(text=_("Open"))
        save_btn.config(text=_("Save"))
        interpolate_btn.config(text=_("Interpolate"))
        settings_btn.config(text=_("Language Settings"))
    
    def switch_to_english():
        set_language('en')
        update_language_display()
    
    def switch_to_russian():
        set_language('ru')
        update_language_display()
    
    def show_language_dialog():
        result = show_language_settings_dialog(root)
        if result:
            print(f"Language changed to: {result}")
            update_language_display()
    
    # Title
    title_label = tk.Label(root, text="Language Switching Test", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)
    
    # Current language
    lang_label = tk.Label(root, textvariable=lang_var, font=("Arial", 10))
    lang_label.pack(pady=5)
    
    # Language switch buttons
    switch_frame = tk.Frame(root)
    switch_frame.pack(pady=10)
    
    tk.Button(switch_frame, text="Switch to English", command=switch_to_english).pack(side=tk.LEFT, padx=5)
    tk.Button(switch_frame, text="Switch to Russian", command=switch_to_russian).pack(side=tk.LEFT, padx=5)
    
    # Test UI elements with translations
    ui_frame = tk.Frame(root)
    ui_frame.pack(pady=20)
    
    tk.Label(ui_frame, text="Translated UI Elements:", font=("Arial", 10, "bold")).pack(pady=(0, 10))
    
    file_btn = tk.Button(ui_frame, text=_("File"), width=15)
    file_btn.pack(pady=2)
    
    open_btn = tk.Button(ui_frame, text=_("Open"), width=15)
    open_btn.pack(pady=2)
    
    save_btn = tk.Button(ui_frame, text=_("Save"), width=15)
    save_btn.pack(pady=2)
    
    interpolate_btn = tk.Button(ui_frame, text=_("Interpolate"), width=15)
    interpolate_btn.pack(pady=2)
    
    # Language settings button
    settings_btn = tk.Button(ui_frame, text=_("Language Settings"), width=15, command=show_language_dialog)
    settings_btn.pack(pady=10)
    
    # Instructions
    instructions = """
Instructions:
1. Click language switch buttons to see immediate changes
2. Click 'Language Settings' to test the full dialog
3. Note: In real app, restart would be required for full effect
"""
    
    tk.Label(root, text=instructions, justify=tk.LEFT, font=("Arial", 9)).pack(pady=10)
    
    # Start with current language display
    update_language_display()
    
    print("Language Test GUI started.")
    print("You can test language switching with the buttons.")
    print("Close the window to exit the test.")
    
    root.mainloop()

if __name__ == "__main__":
    print("Starting Language Switching Test GUI...")
    test_language_gui()