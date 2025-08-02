"""
Language settings dialog for changing application language.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Import i18n system
from ...i18n import _, get_current_language, get_available_languages, set_language


class LanguageSettingsDialog:
    """
    Dialog for changing application language.
    
    Allows users to:
    - Select preferred language
    - Preview language change
    - Save language preference
    """
    
    def __init__(self, parent):
        """
        Initialize language settings dialog.
        
        Args:
            parent: Parent window
        """
        self.parent = parent
        self.result = None
        self.current_language = get_current_language()
        self.available_languages = get_available_languages()
        
        # Create modal dialog
        self.dialog = tk.Toplevel(parent)
        self.setup_dialog()
        self.create_widgets()
        
    def setup_dialog(self):
        """Setup dialog window properties."""
        self.dialog.title(_("Language Settings"))
        self.dialog.geometry("400x300")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.cancel)
        
        # Center dialog on parent
        self.center_dialog()
        
    def center_dialog(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        # Get parent position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate center position
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
    def create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text=_("Language Settings"),
            font=("TkDefaultFont", 14, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Language selection frame
        lang_frame = ttk.LabelFrame(main_frame, text=_("Select Language:"), padding="15")
        lang_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Language selection
        self.language_var = tk.StringVar(value=self.current_language)
        
        # Create radio buttons for each available language
        self.create_language_options(lang_frame)
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text=_("Information"), padding="15")
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Current language info
        current_lang_label = ttk.Label(
            info_frame,
            text=f"{_('Current Language:')} {self.get_language_name(self.current_language)}"
        )
        current_lang_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Immediate change info
        change_info = ttk.Label(
            info_frame,
            text=_("Language will be changed immediately"),
            foreground="green",
            font=("TkDefaultFont", 9)
        )
        change_info.pack(anchor=tk.W)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Buttons
        ttk.Button(
            button_frame,
            text=_("Cancel"),
            command=self.cancel
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(
            button_frame,
            text=_("Apply"),
            command=self.apply_changes
        ).pack(side=tk.RIGHT)
        
    def create_language_options(self, parent):
        """Create language selection radio buttons."""
        
        # Language mappings
        language_names = {
            'en': _("English"),
            'ru': _("Russian")
        }
        
        for lang_code in self.available_languages:
            if lang_code in language_names:
                name = language_names[lang_code]
                
                # Create radio button
                radio = ttk.Radiobutton(
                    parent,
                    text=name,
                    variable=self.language_var,
                    value=lang_code,
                    command=self.on_language_change
                )
                radio.pack(anchor=tk.W, pady=2)
                
    def get_language_name(self, lang_code: str) -> str:
        """Get display name for language code."""
        names = {
            'en': _("English"),
            'ru': _("Russian")
        }
        return names.get(lang_code, lang_code)
        
    def on_language_change(self):
        """Handle language selection change."""
        selected_lang = self.language_var.get()
        # Could add preview functionality here
        
    def apply_changes(self):
        """Apply language changes."""
        selected_lang = self.language_var.get()
        
        if selected_lang != self.current_language:
            try:
                # Save language preference
                save_language_preference(selected_lang)
                
                # Set language for current session (immediate effect)
                set_language(selected_lang)
                
                # Show success message
                messagebox.showinfo(
                    _("Language Changed"),
                    _("Language has been changed successfully")
                )
                
                self.result = selected_lang
                self.dialog.destroy()
                
            except Exception as e:
                messagebox.showerror(
                    _("Error"),
                    f"Failed to save language settings: {str(e)}"
                )
        else:
            # No change needed
            self.dialog.destroy()
            
    def cancel(self):
        """Cancel dialog."""
        self.result = None
        self.dialog.destroy()
        
    def show(self) -> Optional[str]:
        """Show dialog and return selected language."""
        self.dialog.wait_window()
        return self.result


def show_language_settings_dialog(parent) -> Optional[str]:
    """
    Show language settings dialog.
    
    Args:
        parent: Parent window
        
    Returns:
        Selected language code or None if cancelled
    """
    dialog = LanguageSettingsDialog(parent)
    return dialog.show()


def load_language_preference() -> str:
    """
    Load saved language preference.
    
    Returns:
        Saved language code or 'en' if not found
    """
    try:
        config_file = Path.home() / '.coal_interpolation' / 'settings.json'
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('language', 'en')
                
    except Exception:
        pass
        
    return 'en'  # Default to English


def save_language_preference(language: str):
    """
    Save language preference to configuration file.
    
    Args:
        language: Language code to save
        
    Raises:
        Exception: If saving fails
    """
    try:
        # Create config directory if it doesn't exist
        config_dir = Path.home() / '.coal_interpolation'
        config_dir.mkdir(exist_ok=True)
        
        # Load existing config or create new
        config_file = config_dir / 'settings.json'
        config = {}
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # Update language setting
        config['language'] = language
        
        # Save config
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        raise Exception(f"Failed to save language preference: {e}")