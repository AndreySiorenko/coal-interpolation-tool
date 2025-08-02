#!/usr/bin/env python3
"""
Demonstration of dynamic language switching functionality.

This script shows how the language switching works in real-time
without requiring application restart.
"""

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

class LanguageSwitchingDemo:
    """Demo application showing dynamic language switching."""
    
    def __init__(self):
        """Initialize the demo."""
        # Initialize i18n system first
        from src.i18n import init_i18n, set_language, get_current_language, _, add_language_change_listener
        
        self.i18n_manager = init_i18n('en')
        self._ = _
        self.set_language = set_language
        self.get_current_language = get_current_language
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Dynamic Language Switching Demo")
        self.root.geometry("500x400")
        
        # Store UI elements for updates
        self.ui_elements = {}
        
        # Register for language change events
        add_language_change_listener(self.on_language_changed)
        
        self.create_ui()
        
    def create_ui(self):
        """Create the demo UI."""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        self.ui_elements['title'] = ttk.Label(
            main_frame,
            text=self._("Dynamic Language Switching Demo"),
            font=("TkDefaultFont", 16, "bold")
        )
        self.ui_elements['title'].pack(pady=(0, 20))
        
        # Current language display
        self.ui_elements['current_lang'] = ttk.Label(
            main_frame,
            text=f"{self._('Current Language:')} {self.get_current_language()}"
        )
        self.ui_elements['current_lang'].pack(pady=(0, 20))
        
        # Language switching buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(0, 20))
        
        self.ui_elements['en_button'] = ttk.Button(
            button_frame,
            text=self._("Switch to English"),
            command=lambda: self.switch_language('en')
        )
        self.ui_elements['en_button'].pack(side=tk.LEFT, padx=(0, 10))
        
        self.ui_elements['ru_button'] = ttk.Button(
            button_frame,
            text=self._("Switch to Russian"),
            command=lambda: self.switch_language('ru')
        )
        self.ui_elements['ru_button'].pack(side=tk.LEFT)
        
        # Demo UI elements that will be translated
        demo_frame = ttk.LabelFrame(main_frame, text=self._("Demo UI Elements"), padding="15")
        demo_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Menu-like elements
        self.ui_elements['file_label'] = ttk.Label(demo_frame, text=f"{self._('File')}")
        self.ui_elements['file_label'].pack(anchor=tk.W, pady=2)
        
        self.ui_elements['open_label'] = ttk.Label(demo_frame, text=f"  {self._('Open')}")
        self.ui_elements['open_label'].pack(anchor=tk.W, pady=2)
        
        self.ui_elements['save_label'] = ttk.Label(demo_frame, text=f"  {self._('Save')}")
        self.ui_elements['save_label'].pack(anchor=tk.W, pady=2)
        
        self.ui_elements['export_label'] = ttk.Label(demo_frame, text=f"  {self._('Export')}")
        self.ui_elements['export_label'].pack(anchor=tk.W, pady=2)
        
        # Separator
        ttk.Separator(demo_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self.ui_elements['tools_label'] = ttk.Label(demo_frame, text=f"{self._('Tools')}")
        self.ui_elements['tools_label'].pack(anchor=tk.W, pady=2)
        
        self.ui_elements['settings_label'] = ttk.Label(demo_frame, text=f"  {self._('Language Settings')}")
        self.ui_elements['settings_label'].pack(anchor=tk.W, pady=2)
        
        self.ui_elements['stats_label'] = ttk.Label(demo_frame, text=f"  {self._('Statistics')}")
        self.ui_elements['stats_label'].pack(anchor=tk.W, pady=2)
        
        # Status
        self.ui_elements['status'] = ttk.Label(
            main_frame,
            text=self._("Ready"),
            foreground="green"
        )
        self.ui_elements['status'].pack(pady=(20, 0))
        
    def switch_language(self, language):
        """Switch to specified language."""
        try:
            print(f"Switching language to: {language}")
            self.set_language(language)
            self.ui_elements['status'].config(text=self._("Language changed successfully!"))
        except Exception as e:
            print(f"Error switching language: {e}")
            self.ui_elements['status'].config(text=f"Error: {e}")
    
    def on_language_changed(self, old_language, new_language):
        """Handle language change event and update UI."""
        print(f"Language change event: {old_language} -> {new_language}")
        
        try:
            # Update window title
            self.root.title(self._("Dynamic Language Switching Demo"))
            
            # Update all UI elements
            self.ui_elements['title'].config(text=self._("Dynamic Language Switching Demo"))
            self.ui_elements['current_lang'].config(text=f"{self._('Current Language:')} {new_language}")
            
            # Update buttons
            self.ui_elements['en_button'].config(text=self._("Switch to English"))
            self.ui_elements['ru_button'].config(text=self._("Switch to Russian"))
            
            # Update demo elements
            self.ui_elements['file_label'].config(text=self._("File"))
            self.ui_elements['open_label'].config(text=f"  {self._('Open')}")
            self.ui_elements['save_label'].config(text=f"  {self._('Save')}")
            self.ui_elements['export_label'].config(text=f"  {self._('Export')}")
            self.ui_elements['tools_label'].config(text=self._("Tools"))
            self.ui_elements['settings_label'].config(text=f"  {self._('Language Settings')}")
            self.ui_elements['stats_label'].config(text=f"  {self._('Statistics')}")
            
            # Update status
            self.ui_elements['status'].config(text=self._("Language changed successfully!"))
            
            # Refresh the display
            self.root.update_idletasks()
            
            print("UI updated successfully!")
            
        except Exception as e:
            print(f"Error updating UI: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run the demo."""
        print("Starting Dynamic Language Switching Demo...")
        print("Click the language buttons to see real-time language switching!")
        self.root.mainloop()

def main():
    """Main entry point."""
    try:
        demo = LanguageSwitchingDemo()
        demo.run()
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()