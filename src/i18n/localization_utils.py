"""
Localization utilities and helper functions.

Provides high-level localization functions and integration utilities.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal

from .translation_manager import TranslationManager, get_translation_manager, _
from .formatters import RegionalFormatter, get_regional_formatter
from .language_detector import detect_system_language


class LocalizationUtils:
    """
    High-level localization utilities and integration functions.
    
    Provides a unified interface for all localization functionality
    including translations, formatting, and language management.
    """
    
    def __init__(self, language: Optional[str] = None):
        """
        Initialize localization utilities.
        
        Args:
            language: Language code (auto-detect if None)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize language
        self.current_language = language or detect_system_language()
        
        # Initialize components
        self.translation_manager = get_translation_manager()
        self.formatter = get_regional_formatter(self.current_language)
        
        # Set language
        self.set_language(self.current_language)
    
    def set_language(self, language: str):
        """
        Set the current language for all components.
        
        Args:
            language: Language code to set
        """
        self.current_language = language
        self.translation_manager.set_language(language)
        self.formatter = get_regional_formatter(language)
        self.logger.info(f"Localization language set to: {language}")
    
    def get_language(self) -> str:
        """Get current language code."""
        return self.current_language
    
    def translate(self, message: str, context: Optional[str] = None, **kwargs) -> str:
        """
        Translate a message with optional formatting.
        
        Args:
            message: Message to translate
            context: Translation context
            **kwargs: Format arguments for string interpolation
            
        Returns:
            Translated and formatted message
        """
        translated = self.translation_manager.translate(message, context=context)
        
        # Apply string formatting if kwargs provided
        if kwargs:
            try:
                translated = translated.format(**kwargs)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Error formatting translated string: {e}")
        
        return translated
    
    def translate_plural(self, singular: str, plural: str, count: int, **kwargs) -> str:
        """
        Translate a plural message with count formatting.
        
        Args:
            singular: Singular form
            plural: Plural form
            count: Count for plural determination
            **kwargs: Additional format arguments
            
        Returns:
            Translated plural message
        """
        translated = self.translation_manager.translate_plural(singular, plural, count)
        
        # Add count to format arguments
        format_args = {'count': self.format_number(count)}
        format_args.update(kwargs)
        
        try:
            translated = translated.format(**format_args)
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Error formatting plural string: {e}")
        
        return translated
    
    def format_number(self, value: Union[int, float, Decimal], **kwargs) -> str:
        """Format number according to current locale."""
        return self.formatter.format_number(value, **kwargs)
    
    def format_currency(self, value: Union[int, float, Decimal], **kwargs) -> str:
        """Format currency according to current locale."""
        return self.formatter.format_currency(value, **kwargs)
    
    def format_percentage(self, value: Union[int, float], **kwargs) -> str:
        """Format percentage according to current locale."""
        return self.formatter.format_percentage(value, **kwargs)
    
    def format_date(self, date_value: Union[datetime, date], **kwargs) -> str:
        """Format date according to current locale."""
        return self.formatter.format_date(date_value, **kwargs)
    
    def format_measurement(self, value: Union[int, float], unit: str, **kwargs) -> str:
        """Format measurement with localized unit."""
        # Translate unit if needed
        translated_unit = self.translate(unit)
        return self.formatter.format_measurement(value, translated_unit, **kwargs)
    
    def get_localized_error_message(self, error_type: str, **kwargs) -> str:
        """
        Get localized error message.
        
        Args:
            error_type: Type of error
            **kwargs: Error-specific parameters
            
        Returns:
            Localized error message
        """
        error_messages = {
            'file_not_found': _('File not found: {filename}'),
            'invalid_data': _('Invalid data format'),
            'processing_error': _('Error processing data: {details}'),
            'validation_failed': _('Validation failed: {reason}'),
            'interpolation_error': _('Interpolation error: {method} failed'),
            'export_error': _('Export failed: {format} not supported'),
            'database_error': _('Database connection error: {details}'),
            'permission_error': _('Permission denied: {operation}'),
            'memory_error': _('Insufficient memory for operation'),
            'timeout_error': _('Operation timed out after {seconds} seconds')
        }
        
        message_template = error_messages.get(error_type, _('Unknown error: {error_type}'))
        kwargs.update({'error_type': error_type})
        
        return message_template.format(**kwargs)
    
    def get_localized_status_message(self, status: str, **kwargs) -> str:
        """
        Get localized status message.
        
        Args:
            status: Status type
            **kwargs: Status-specific parameters
            
        Returns:
            Localized status message
        """
        status_messages = {
            'loading': _('Loading...'),
            'processing': _('Processing {operation}...'),
            'saving': _('Saving {filename}...'),
            'complete': _('Operation completed successfully'),
            'cancelled': _('Operation cancelled by user'),
            'ready': _('Ready'),
            'interpolating': _('Interpolating using {method}...'),
            'validating': _('Validating results...'),
            'exporting': _('Exporting to {format}...'),
            'generating_report': _('Generating {report_type} report...')
        }
        
        message_template = status_messages.get(status, _('Status: {status}'))
        kwargs.update({'status': status})
        
        return message_template.format(**kwargs)
    
    def get_localized_unit_name(self, unit: str, plural: bool = False) -> str:
        """
        Get localized unit name.
        
        Args:
            unit: Unit identifier
            plural: Whether to return plural form
            
        Returns:
            Localized unit name
        """
        unit_translations = {
            # Length units
            'm': {'singular': _('meter'), 'plural': _('meters')},
            'km': {'singular': _('kilometer'), 'plural': _('kilometers')},
            'ft': {'singular': _('foot'), 'plural': _('feet')},
            'in': {'singular': _('inch'), 'plural': _('inches')},
            
            # Weight units
            'kg': {'singular': _('kilogram'), 'plural': _('kilograms')},
            'g': {'singular': _('gram'), 'plural': _('grams')},
            't': {'singular': _('ton'), 'plural': _('tons')},
            'lb': {'singular': _('pound'), 'plural': _('pounds')},
            
            # Area units
            'm²': {'singular': _('square meter'), 'plural': _('square meters')},
            'km²': {'singular': _('square kilometer'), 'plural': _('square kilometers')},
            'ha': {'singular': _('hectare'), 'plural': _('hectares')},
            
            # Volume units
            'm³': {'singular': _('cubic meter'), 'plural': _('cubic meters')},
            'l': {'singular': _('liter'), 'plural': _('liters')},
            
            # Percentage and ratios
            '%': {'singular': _('percent'), 'plural': _('percent')},
            'ppm': {'singular': _('parts per million'), 'plural': _('parts per million')}
        }
        
        if unit in unit_translations:
            form = 'plural' if plural else 'singular'
            return unit_translations[unit][form]
        
        return unit
    
    def create_localized_menu(self, menu_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create localized menu structure.
        
        Args:
            menu_structure: Menu structure with translation keys
            
        Returns:
            Localized menu structure
        """
        def localize_item(item):
            if isinstance(item, dict):
                localized = {}
                for key, value in item.items():
                    if key == 'label':
                        localized[key] = self.translate(value)
                    elif key == 'items' and isinstance(value, list):
                        localized[key] = [localize_item(subitem) for subitem in value]
                    elif key == 'items' and isinstance(value, dict):
                        localized[key] = localize_item(value)
                    else:
                        localized[key] = value
                return localized
            elif isinstance(item, str):
                return self.translate(item)
            else:
                return item
        
        return localize_item(menu_structure)
    
    def validate_and_format_input(self, value: str, input_type: str) -> tuple[bool, Any, str]:
        """
        Validate and format user input according to locale.
        
        Args:
            value: Input value as string
            input_type: Type of input ('number', 'currency', 'date', 'percentage')
            
        Returns:
            Tuple of (is_valid, parsed_value, error_message)
        """
        try:
            if input_type == 'number':
                # Handle regional decimal separator
                normalized = value.replace(self.formatter.settings['decimal_separator'], '.')
                normalized = normalized.replace(self.formatter.settings['thousands_separator'], '')
                parsed = float(normalized)
                return True, parsed, ''
            
            elif input_type == 'currency':
                # Remove currency symbol and parse as number
                cleaned = value.replace(self.formatter.settings['currency_symbol'], '')
                return self.validate_and_format_input(cleaned.strip(), 'number')
            
            elif input_type == 'percentage':
                # Handle percentage input
                cleaned = value.replace('%', '').strip()
                is_valid, parsed, error = self.validate_and_format_input(cleaned, 'number')
                if is_valid:
                    # Convert to decimal if it looks like percentage
                    if parsed > 1:
                        parsed = parsed / 100
                return is_valid, parsed, error
            
            elif input_type == 'date':
                # Try to parse date using regional format
                try:
                    parsed = datetime.strptime(value, self.formatter.settings['date_format'])
                    return True, parsed.date(), ''
                except ValueError:
                    # Try other common formats
                    formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d.%m.%Y']
                    for fmt in formats:
                        try:
                            parsed = datetime.strptime(value, fmt)
                            return True, parsed.date(), ''
                        except ValueError:
                            continue
                    
                    return False, None, self.translate('Invalid date format')
            
            else:
                return True, value, ''
        
        except Exception as e:
            error_msg = self.get_localized_error_message('validation_failed', reason=str(e))
            return False, None, error_msg
    
    def export_translations(self, output_dir: str):
        """
        Export current translations to JSON files.
        
        Args:
            output_dir: Directory to save translation files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for language, translations in self.translation_manager.translations.items():
            lang_dir = output_path / language
            lang_dir.mkdir(exist_ok=True)
            
            # Create translation file
            translation_data = {
                'language': language,
                'messages': translations,
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'message_count': len(translations)
                }
            }
            
            # Add plural forms if available
            if language in self.translation_manager.plural_forms:
                translation_data['plural_forms'] = {
                    'rule': f'plural_{language}'
                }
            
            output_file = lang_dir / 'messages.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported {len(translations)} translations for {language} to {output_file}")


# Global localization utilities instance
_global_utils: Optional[LocalizationUtils] = None

def get_localization_utils(language: Optional[str] = None) -> LocalizationUtils:
    """Get the global localization utilities instance."""
    global _global_utils
    if _global_utils is None or (language and _global_utils.current_language != language):
        _global_utils = LocalizationUtils(language)
    return _global_utils

# Convenience functions
def format_number(value: Union[int, float, Decimal], language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for number formatting."""
    utils = get_localization_utils(language)
    return utils.format_number(value, **kwargs)

def format_date(date_value: Union[datetime, date], language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for date formatting."""
    utils = get_localization_utils(language)
    return utils.format_date(date_value, **kwargs)

def format_currency(value: Union[int, float, Decimal], language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for currency formatting."""
    utils = get_localization_utils(language)
    return utils.format_currency(value, **kwargs)

def translate_with_format(message: str, language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for translation with formatting."""
    utils = get_localization_utils(language)
    return utils.translate(message, **kwargs)

def get_error_message(error_type: str, language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for localized error messages."""
    utils = get_localization_utils(language)
    return utils.get_localized_error_message(error_type, **kwargs)

def get_status_message(status: str, language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for localized status messages."""
    utils = get_localization_utils(language)
    return utils.get_localized_status_message(status, **kwargs)