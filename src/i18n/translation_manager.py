"""
Translation manager for multi-language support.

Provides gettext-based translation system with fallback support.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import threading


class TranslationManager:
    """
    Manages translations and language switching for the application.
    
    Supports both gettext (.po/.mo files) and JSON-based translations.
    Provides thread-safe language switching and fallback mechanisms.
    """
    
    def __init__(self, default_language: str = 'en', locale_dir: Optional[str] = None):
        """
        Initialize translation manager.
        
        Args:
            default_language: Default language code
            locale_dir: Directory containing translation files
        """
        self.logger = logging.getLogger(__name__)
        self.default_language = default_language
        self.current_language = default_language
        self.fallback_language = 'en'
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Translation storage
        self.translations: Dict[str, Dict[str, str]] = {}
        self.plural_forms: Dict[str, Callable] = {}
        
        # Set up locale directory
        if locale_dir is None:
            # Default to package locale directory
            package_dir = Path(__file__).parent
            self.locale_dir = package_dir / 'locales'
        else:
            self.locale_dir = Path(locale_dir)
        
        # Initialize translations
        self._load_translations()
        
        # Try to import gettext for .po/.mo support
        self._init_gettext_support()
    
    def _init_gettext_support(self):
        """Initialize gettext support if available."""
        try:
            import gettext
            self.gettext = gettext
            self.gettext_available = True
            self.logger.info("Gettext support enabled")
        except ImportError:
            self.gettext = None
            self.gettext_available = False
            self.logger.info("Gettext not available, using JSON translations only")
    
    def _load_translations(self):
        """Load all available translations."""
        if not self.locale_dir.exists():
            self.logger.warning(f"Locale directory not found: {self.locale_dir}")
            self._create_default_translations()
            return
        
        # Load JSON translations
        for lang_dir in self.locale_dir.iterdir():
            if lang_dir.is_dir():
                lang_code = lang_dir.name
                self._load_json_translation(lang_code)
                self._load_gettext_translation(lang_code)
        
        # Ensure we have at least basic translations
        if not self.translations:
            self._create_default_translations()
        
        self.logger.info(f"Loaded translations for languages: {list(self.translations.keys())}")
    
    def _load_json_translation(self, language: str):
        """Load JSON translation file for a language."""
        json_file = self.locale_dir / language / 'messages.json'
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    translation_data = json.load(f)
                
                if language not in self.translations:
                    self.translations[language] = {}
                
                # Load messages
                if 'messages' in translation_data:
                    self.translations[language].update(translation_data['messages'])
                
                # Load plural forms
                if 'plural_forms' in translation_data:
                    self._load_plural_forms(language, translation_data['plural_forms'])
                
                self.logger.debug(f"Loaded JSON translations for {language}: {len(self.translations[language])} messages")
                
            except Exception as e:
                self.logger.error(f"Error loading JSON translation for {language}: {e}")
    
    def _load_gettext_translation(self, language: str):
        """Load gettext .mo translation file for a language."""
        if not self.gettext_available:
            return
        
        mo_file = self.locale_dir / language / 'LC_MESSAGES' / 'messages.mo'
        if mo_file.exists():
            try:
                with open(mo_file, 'rb') as f:
                    catalog = self.gettext.GNUTranslations(f)
                
                # Extract translations from catalog
                if language not in self.translations:
                    self.translations[language] = {}
                
                # Access internal catalog (implementation-specific)
                if hasattr(catalog, '_catalog'):
                    for key, value in catalog._catalog.items():
                        if isinstance(key, str):
                            self.translations[language][key] = value
                
                self.logger.debug(f"Loaded gettext translations for {language}")
                
            except Exception as e:
                self.logger.error(f"Error loading gettext translation for {language}: {e}")
    
    def _load_plural_forms(self, language: str, plural_config: Dict[str, Any]):
        """Load plural forms configuration for a language."""
        try:
            if 'rule' in plural_config:
                # Create plural function from rule
                rule = plural_config['rule']
                if language == 'ru':
                    # Russian plural rules
                    def russian_plural(n):
                        if n % 10 == 1 and n % 100 != 11:
                            return 0  # singular
                        elif n % 10 in [2, 3, 4] and n % 100 not in [12, 13, 14]:
                            return 1  # few
                        else:
                            return 2  # many
                    self.plural_forms[language] = russian_plural
                
                elif language == 'en':
                    # English plural rules
                    def english_plural(n):
                        return 0 if n == 1 else 1
                    self.plural_forms[language] = english_plural
        
        except Exception as e:
            self.logger.error(f"Error loading plural forms for {language}: {e}")
    
    def _create_default_translations(self):
        """Create default translations if none are loaded."""
        # English (base language)
        self.translations['en'] = {
            # General UI
            'Coal Deposit Interpolation Tool': 'Coal Deposit Interpolation Tool',
            'File': 'File',
            'Edit': 'Edit',
            'View': 'View',
            'Tools': 'Tools',
            'Help': 'Help',
            'Open': 'Open',
            'Save': 'Save',
            'Exit': 'Exit',
            
            # Data operations
            'Load Data': 'Load Data',
            'Import': 'Import',
            'Export': 'Export',
            'Data Summary': 'Data Summary',
            'Statistics': 'Statistics',
            'Visualization': 'Visualization',
            
            # Interpolation
            'Interpolation Method': 'Interpolation Method',
            'IDW': 'Inverse Distance Weighting',
            'Kriging': 'Kriging',
            'RBF': 'Radial Basis Functions',
            'Parameters': 'Parameters',
            'Power': 'Power',
            'Search Radius': 'Search Radius',
            'Grid Resolution': 'Grid Resolution',
            
            # Validation
            'Validation': 'Validation',
            'Cross Validation': 'Cross Validation',
            'Leave One Out': 'Leave One Out',
            'K-Fold': 'K-Fold',
            'RMSE': 'Root Mean Square Error',
            'MAE': 'Mean Absolute Error',
            'R-squared': 'R-squared',
            
            # Reports
            'Generate Report': 'Generate Report',
            'PDF Report': 'PDF Report',
            'HTML Report': 'HTML Report',
            'Excel Report': 'Excel Report',
            
            # Status messages
            'Processing...': 'Processing...',
            'Complete': 'Complete',
            'Error': 'Error',
            'Warning': 'Warning',
            'Success': 'Success',
            
            # Units and formats
            'meters': 'meters',
            'kilometers': 'kilometers',
            'tons': 'tons',
            'percentage': 'percentage',
            'points': 'points',
            'records': 'records'
        }
        
        # Russian translations
        self.translations['ru'] = {
            # General UI
            'Coal Deposit Interpolation Tool': 'Инструмент интерполяции угольных месторождений',
            'File': 'Файл',
            'Edit': 'Правка',
            'View': 'Вид',
            'Tools': 'Инструменты',
            'Help': 'Справка',
            'Open': 'Открыть',
            'Save': 'Сохранить',
            'Exit': 'Выход',
            
            # Data operations
            'Load Data': 'Загрузить данные',
            'Import': 'Импорт',
            'Export': 'Экспорт',
            'Data Summary': 'Сводка данных',
            'Statistics': 'Статистика',
            'Visualization': 'Визуализация',
            
            # Interpolation
            'Interpolation Method': 'Метод интерполяции',
            'IDW': 'Обратно взвешенное расстояние',
            'Kriging': 'Кригинг',
            'RBF': 'Радиальные базисные функции',
            'Parameters': 'Параметры',
            'Power': 'Степень',
            'Search Radius': 'Радиус поиска',
            'Grid Resolution': 'Разрешение сетки',
            
            # Validation
            'Validation': 'Валидация',
            'Cross Validation': 'Перекрестная валидация',
            'Leave One Out': 'Исключение одного',
            'K-Fold': 'K-блочная',
            'RMSE': 'Среднеквадратичная ошибка',
            'MAE': 'Средняя абсолютная ошибка',
            'R-squared': 'R-квадрат',
            
            # Reports
            'Generate Report': 'Создать отчет',
            'PDF Report': 'PDF отчет',
            'HTML Report': 'HTML отчет',
            'Excel Report': 'Excel отчет',
            
            # Status messages
            'Processing...': 'Обработка...',
            'Complete': 'Завершено',
            'Error': 'Ошибка',
            'Warning': 'Предупреждение',
            'Success': 'Успешно',
            
            # Units and formats
            'meters': 'метры',
            'kilometers': 'километры',
            'tons': 'тонны',
            'percentage': 'процент',
            'points': 'точки',
            'records': 'записи'
        }
        
        # Set up plural forms
        self.plural_forms['en'] = lambda n: 0 if n == 1 else 1
        self.plural_forms['ru'] = lambda n: (
            0 if n % 10 == 1 and n % 100 != 11 else
            1 if n % 10 in [2, 3, 4] and n % 100 not in [12, 13, 14] else
            2
        )
        
        self.logger.info("Created default translations for English and Russian")
    
    def translate(self, message: str, language: Optional[str] = None, context: Optional[str] = None) -> str:
        """
        Translate a message to the specified language.
        
        Args:
            message: Message to translate
            language: Target language (uses current if None)
            context: Translation context for disambiguation
            
        Returns:
            Translated message or original if not found
        """
        with self._lock:
            target_lang = language or self.current_language
            
            # Build key with context if provided
            key = f"{context}|{message}" if context else message
            
            # Try target language
            if target_lang in self.translations:
                if key in self.translations[target_lang]:
                    return self.translations[target_lang][key]
                elif message in self.translations[target_lang]:
                    return self.translations[target_lang][message]
            
            # Try fallback language
            if (self.fallback_language != target_lang and 
                self.fallback_language in self.translations):
                if key in self.translations[self.fallback_language]:
                    return self.translations[self.fallback_language][key]
                elif message in self.translations[self.fallback_language]:
                    return self.translations[self.fallback_language][message]
            
            # Return original message if no translation found
            return message
    
    def translate_plural(self, singular: str, plural: str, count: int, 
                        language: Optional[str] = None) -> str:
        """
        Translate a message with plural forms.
        
        Args:
            singular: Singular form
            plural: Plural form
            count: Number for plural calculation
            language: Target language (uses current if None)
            
        Returns:
            Translated message in appropriate plural form
        """
        with self._lock:
            target_lang = language or self.current_language
            
            # Get plural form index
            plural_index = 0
            if target_lang in self.plural_forms:
                plural_index = self.plural_forms[target_lang](count)
            else:
                # Default English plural rules
                plural_index = 0 if count == 1 else 1
            
            # Build message key
            if plural_index == 0:
                message = singular
            else:
                message = plural
            
            # Translate the selected form
            translated = self.translate(message, target_lang)
            
            return translated
    
    def set_language(self, language: str):
        """
        Set the current language.
        
        Args:
            language: Language code to set
        """
        with self._lock:
            if language in self.translations:
                self.current_language = language
                self.logger.info(f"Language changed to: {language}")
            else:
                self.logger.warning(f"Language not available: {language}")
                # Try to load the language
                self._load_json_translation(language)
                self._load_gettext_translation(language)
                
                if language in self.translations:
                    self.current_language = language
                    self.logger.info(f"Language loaded and set to: {language}")
    
    def get_current_language(self) -> str:
        """Get the current language code."""
        return self.current_language
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes."""
        return list(self.translations.keys())
    
    def add_translation(self, language: str, key: str, value: str, context: Optional[str] = None):
        """
        Add a translation dynamically.
        
        Args:
            language: Language code
            key: Translation key
            value: Translation value
            context: Optional context
        """
        with self._lock:
            if language not in self.translations:
                self.translations[language] = {}
            
            translation_key = f"{context}|{key}" if context else key
            self.translations[language][translation_key] = value
    
    def get_language_info(self, language: str) -> Dict[str, Any]:
        """
        Get information about a language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary with language information
        """
        language_names = {
            'en': {'name': 'English', 'native': 'English'},
            'ru': {'name': 'Russian', 'native': 'Русский'}
        }
        
        info = language_names.get(language, {'name': language, 'native': language})
        info.update({
            'code': language,
            'available': language in self.translations,
            'message_count': len(self.translations.get(language, {})),
            'has_plural_forms': language in self.plural_forms
        })
        
        return info


# Global translation manager instance
_global_translation_manager: Optional[TranslationManager] = None

def get_translation_manager() -> TranslationManager:
    """Get the global translation manager instance."""
    global _global_translation_manager
    if _global_translation_manager is None:
        _global_translation_manager = TranslationManager()
    return _global_translation_manager

def _(message: str, language: Optional[str] = None, context: Optional[str] = None) -> str:
    """
    Convenience function for translation (gettext-style).
    
    Args:
        message: Message to translate
        language: Target language (uses current if None)
        context: Translation context
        
    Returns:
        Translated message
    """
    manager = get_translation_manager()
    return manager.translate(message, language, context)

def ngettext(singular: str, plural: str, count: int, language: Optional[str] = None) -> str:
    """
    Convenience function for plural translation (gettext-style).
    
    Args:
        singular: Singular form
        plural: Plural form  
        count: Number for plural calculation
        language: Target language
        
    Returns:
        Translated message in appropriate plural form
    """
    manager = get_translation_manager()
    return manager.translate_plural(singular, plural, count, language)