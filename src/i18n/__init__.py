"""
Internationalization (i18n) package for coal deposit interpolation tool.

Provides comprehensive multi-language support including:
- Translation framework with gettext support
- Russian and English localization
- Regional number and date formatting
- Dynamic language switching
- Pluralization support
"""

from .translation_manager import TranslationManager, get_translation_manager, _
from .formatters import RegionalFormatter, get_regional_formatter
from .language_detector import LanguageDetector, detect_system_language
from .localization_utils import LocalizationUtils, format_number, format_date, format_currency

# Initialize global translation manager
_translation_manager = None
_language_change_listeners = []

def init_i18n(language: str = None, locale_dir: str = None):
    """
    Initialize internationalization system.
    
    Args:
        language: Language code (e.g., 'ru', 'en'). If None, auto-detect.
        locale_dir: Directory containing translation files. If None, use default.
    """
    global _translation_manager
    if language is None:
        language = detect_system_language()
    
    # Use the global translation manager directly
    _translation_manager = get_translation_manager()
    _translation_manager.set_language(language)
    
    return _translation_manager

def get_current_language():
    """Get current language code."""
    manager = get_translation_manager()
    return manager.current_language

def set_language(language: str):
    """Set current language and notify listeners."""
    manager = get_translation_manager()
    old_language = manager.current_language
    manager.set_language(language)
    
    # Notify all listeners about language change
    for listener in _language_change_listeners:
        try:
            listener(old_language, language)
        except Exception as e:
            import logging
            logging.error(f"Error in language change listener: {e}")

def add_language_change_listener(callback):
    """
    Add a callback to be called when language changes.
    
    Args:
        callback: Function that takes (old_language, new_language) parameters
    """
    global _language_change_listeners
    if callback not in _language_change_listeners:
        _language_change_listeners.append(callback)

def remove_language_change_listener(callback):
    """
    Remove a language change listener.
    
    Args:
        callback: Function to remove from listeners
    """
    global _language_change_listeners
    if callback in _language_change_listeners:
        _language_change_listeners.remove(callback)

def get_available_languages():
    """Get list of available languages."""
    manager = get_translation_manager()
    return manager.get_available_languages()

__all__ = [
    'TranslationManager',
    'get_translation_manager', 
    '_',
    'RegionalFormatter',
    'get_regional_formatter',
    'LanguageDetector',
    'detect_system_language',
    'LocalizationUtils',
    'format_number',
    'format_date', 
    'format_currency',
    'init_i18n',
    'get_current_language',
    'set_language',
    'get_available_languages',
    'add_language_change_listener',
    'remove_language_change_listener'
]