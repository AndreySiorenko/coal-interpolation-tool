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
    
    _translation_manager = TranslationManager(language, locale_dir)
    return _translation_manager

def get_current_language():
    """Get current language code."""
    if _translation_manager:
        return _translation_manager.current_language
    return 'en'

def set_language(language: str):
    """Set current language."""
    if _translation_manager:
        _translation_manager.set_language(language)

def get_available_languages():
    """Get list of available languages."""
    if _translation_manager:
        return _translation_manager.get_available_languages()
    return ['en', 'ru']

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
    'get_available_languages'
]