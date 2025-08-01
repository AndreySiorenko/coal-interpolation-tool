"""
Unit tests for internationalization (i18n) system.

Tests translation management, regional formatting, language detection,
and localization utilities for both Russian and English languages.
"""

import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import patch, mock_open

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from i18n.translation_manager import TranslationManager, get_translation_manager, _, ngettext
from i18n.formatters import RegionalFormatter, get_regional_formatter
from i18n.language_detector import LanguageDetector, detect_system_language
from i18n.localization_utils import LocalizationUtils, get_localization_utils


class TestTranslationManager(unittest.TestCase):
    """Test translation manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.locale_dir = Path(self.temp_dir) / 'locales'
        self.locale_dir.mkdir(parents=True)
        
        # Create test translation files
        self._create_test_translations()
        
        self.manager = TranslationManager(default_language='en', locale_dir=str(self.locale_dir))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_translations(self):
        """Create test translation files."""
        # English translations
        en_dir = self.locale_dir / 'en'
        en_dir.mkdir()
        en_translations = {
            'language': 'en',
            'messages': {
                'Hello': 'Hello',
                'Good morning': 'Good morning',
                'File': 'File',
                'Save': 'Save',
                'Error': 'Error',
                '{count} item': '{count} item',
                '{count} items': '{count} items'
            },
            'plural_forms': {'rule': 'plural_en'}
        }
        
        with open(en_dir / 'messages.json', 'w', encoding='utf-8') as f:
            json.dump(en_translations, f, indent=2)
        
        # Russian translations
        ru_dir = self.locale_dir / 'ru'
        ru_dir.mkdir()
        ru_translations = {
            'language': 'ru',
            'messages': {
                'Hello': 'Привет',
                'Good morning': 'Доброе утро',
                'File': 'Файл',
                'Save': 'Сохранить',
                'Error': 'Ошибка',
                '{count} item': '{count} элемент',
                '{count} items': '{count} элемента'
            },
            'plural_forms': {'rule': 'plural_ru'}
        }
        
        with open(ru_dir / 'messages.json', 'w', encoding='utf-8') as f:
            json.dump(ru_translations, f, indent=2, ensure_ascii=False)
    
    def test_initialization(self):
        """Test translation manager initialization."""
        self.assertEqual(self.manager.current_language, 'en')
        self.assertEqual(self.manager.default_language, 'en')
        self.assertIn('en', self.manager.translations)
        self.assertIn('ru', self.manager.translations)
    
    def test_translation_basic(self):
        """Test basic translation functionality."""
        # English (default)
        self.assertEqual(self.manager.translate('Hello'), 'Hello')
        self.assertEqual(self.manager.translate('File'), 'File')
        
        # Russian
        self.manager.set_language('ru')
        self.assertEqual(self.manager.translate('Hello'), 'Привет')
        self.assertEqual(self.manager.translate('File'), 'Файл')
    
    def test_translation_fallback(self):
        """Test translation fallback to default language."""
        self.manager.set_language('ru')
        
        # Non-existent translation should fallback to original
        result = self.manager.translate('Non-existent message')
        self.assertEqual(result, 'Non-existent message')
    
    def test_plural_translation(self):
        """Test plural form translations."""
        # English plurals
        self.manager.set_language('en')
        result = self.manager.translate_plural('{count} item', '{count} items', 1)
        self.assertEqual(result, '{count} item')
        
        result = self.manager.translate_plural('{count} item', '{count} items', 5)
        self.assertEqual(result, '{count} items')
        
        # Russian plurals
        self.manager.set_language('ru')
        result = self.manager.translate_plural('{count} item', '{count} items', 1)
        self.assertEqual(result, '{count} элемент')
        
        result = self.manager.translate_plural('{count} item', '{count} items', 5)
        self.assertEqual(result, '{count} элемента')
    
    def test_language_switching(self):
        """Test language switching functionality."""
        self.assertEqual(self.manager.current_language, 'en')
        
        self.manager.set_language('ru')
        self.assertEqual(self.manager.current_language, 'ru')
        
        self.manager.set_language('en')
        self.assertEqual(self.manager.current_language, 'en')
    
    def test_available_languages(self):
        """Test getting available languages."""
        languages = self.manager.get_available_languages()
        self.assertIn('en', languages)
        self.assertIn('ru', languages)
    
    def test_add_translation(self):
        """Test adding translations dynamically."""
        self.manager.add_translation('en', 'New Message', 'New Message')
        self.manager.add_translation('ru', 'New Message', 'Новое сообщение')
        
        self.assertEqual(self.manager.translate('New Message', 'en'), 'New Message')
        self.assertEqual(self.manager.translate('New Message', 'ru'), 'Новое сообщение')
    
    def test_language_info(self):
        """Test getting language information."""
        en_info = self.manager.get_language_info('en')
        self.assertEqual(en_info['code'], 'en')
        self.assertEqual(en_info['name'], 'English')
        self.assertTrue(en_info['available'])
        
        ru_info = self.manager.get_language_info('ru')
        self.assertEqual(ru_info['code'], 'ru')
        self.assertEqual(ru_info['name'], 'Russian')
        self.assertTrue(ru_info['available'])


class TestRegionalFormatter(unittest.TestCase):
    """Test regional formatter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.en_formatter = RegionalFormatter('en', 'US')
        self.ru_formatter = RegionalFormatter('ru', 'RU')
    
    def test_number_formatting(self):
        """Test number formatting for different regions."""
        # English formatting
        result = self.en_formatter.format_number(1234.56)
        self.assertIn('1,234', result)  # Thousands separator
        self.assertIn('.56', result)    # Decimal separator
        
        # Russian formatting
        result = self.ru_formatter.format_number(1234.56)
        self.assertIn('1 234', result)  # Space thousands separator
        self.assertIn(',56', result)    # Comma decimal separator
    
    def test_currency_formatting(self):
        """Test currency formatting for different regions."""
        # English (USD)
        result = self.en_formatter.format_currency(1234.56)
        self.assertIn('$', result)
        self.assertIn('1,234.56', result)
        
        # Russian (RUB)
        result = self.ru_formatter.format_currency(1234.56)
        self.assertIn('₽', result)
        self.assertIn('1 234,56', result)
    
    def test_percentage_formatting(self):
        """Test percentage formatting."""
        # Test decimal input (0.0-1.0)
        result = self.en_formatter.format_percentage(0.1234)
        self.assertIn('12.3%', result)
        
        # Test percentage input (already multiplied by 100)
        result = self.en_formatter.format_percentage(12.34)
        self.assertIn('12.3%', result)
    
    def test_date_formatting(self):
        """Test date formatting for different regions."""
        test_date = date(2025, 1, 15)
        
        # English (MM/DD/YYYY)
        result = self.en_formatter.format_date(test_date)
        self.assertIn('01/15/2025', result)
        
        # Russian (DD.MM.YYYY)
        result = self.ru_formatter.format_date(test_date)
        self.assertIn('15.01.2025', result)
    
    def test_large_number_formatting(self):
        """Test large number formatting with suffixes."""
        # English (short format)
        result = self.en_formatter.format_large_number(1500000)
        self.assertIn('1.5M', result)
        
        result = self.en_formatter.format_large_number(2500)
        self.assertIn('2.5K', result)
        
        # Russian (long format)
        result = self.ru_formatter.format_large_number(1500000)
        self.assertIn('1 500 000', result)
    
    def test_scientific_notation(self):
        """Test scientific notation formatting."""
        result = self.en_formatter.format_scientific(0.000123)
        self.assertIn('1.23e-04', result)
        
        # Test with Russian decimal separator
        result = self.ru_formatter.format_scientific(0.000123)
        self.assertIn('1,23e-04', result)
    
    def test_coordinate_formatting(self):
        """Test coordinate formatting."""
        # Decimal degrees
        result = self.en_formatter.format_coordinate(45.123456, 'decimal', 6)
        self.assertEqual(result, '45.123456')
        
        # DMS format
        result = self.en_formatter.format_coordinate(45.5, 'dms', 2)
        self.assertIn('45°30\'0.00', result)
    
    def test_measurement_formatting(self):
        """Test measurement formatting with units."""
        result = self.en_formatter.format_measurement(123.45, 'meters', 2)
        self.assertEqual(result, '123.45 meters')
        
        result = self.ru_formatter.format_measurement(123.45, 'метры', 2)
        self.assertIn('123,45 метры', result)
    
    def test_none_value_handling(self):
        """Test handling of None/NaN values."""
        self.assertEqual(self.en_formatter.format_number(None), 'N/A')
        self.assertEqual(self.en_formatter.format_currency(None), 'N/A')
        self.assertEqual(self.en_formatter.format_percentage(None), 'N/A')
        self.assertEqual(self.en_formatter.format_date(None), 'N/A')


class TestLanguageDetector(unittest.TestCase):
    """Test language detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIn('en', self.detector.supported_languages)
        self.assertIn('ru', self.detector.supported_languages)
        self.assertIsInstance(self.detector.language_mappings, dict)
    
    def test_language_mapping(self):
        """Test language code mapping."""
        self.assertEqual(self.detector._map_language_code('en_US'), 'en')
        self.assertEqual(self.detector._map_language_code('ru_RU'), 'ru')
        self.assertEqual(self.detector._map_language_code('русский'), 'ru')
        self.assertEqual(self.detector._map_language_code('unknown'), 'en')
    
    @patch.dict(os.environ, {'LANG': 'ru_RU.UTF-8'})
    def test_environment_detection(self):
        """Test language detection from environment variables."""
        result = self.detector._detect_from_environment()
        self.assertEqual(result, 'ru_RU')
    
    @patch('locale.getdefaultlocale')
    def test_locale_detection(self, mock_locale):
        """Test language detection from system locale."""
        mock_locale.return_value = ('en_US', 'UTF-8')
        result = self.detector._detect_from_locale()
        self.assertEqual(result, 'en_US')
        
        mock_locale.return_value = (None, None)
        result = self.detector._detect_from_locale()
        self.assertIsNone(result)
    
    def test_language_preferences(self):
        """Test getting language preferences."""
        preferences = self.detector.get_language_preferences()
        self.assertIsInstance(preferences, list)
        self.assertTrue(len(preferences) >= 1)
        self.assertIn('en', preferences)  # English should always be fallback
    
    def test_rtl_language_detection(self):
        """Test RTL language detection."""
        self.assertFalse(self.detector.is_rtl_language('en'))
        self.assertFalse(self.detector.is_rtl_language('ru'))
        self.assertTrue(self.detector.is_rtl_language('ar'))
        self.assertTrue(self.detector.is_rtl_language('he'))
    
    def test_language_info(self):
        """Test getting comprehensive language information."""
        en_info = self.detector.get_language_info('en')
        self.assertEqual(en_info['name'], 'English')
        self.assertEqual(en_info['native_name'], 'English')
        self.assertFalse(en_info['rtl'])
        
        ru_info = self.detector.get_language_info('ru')
        self.assertEqual(ru_info['name'], 'Russian')
        self.assertEqual(ru_info['native_name'], 'Русский')
        self.assertFalse(ru_info['rtl'])


class TestLocalizationUtils(unittest.TestCase):
    """Test localization utilities functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.locale_dir = Path(self.temp_dir) / 'locales'
        self.locale_dir.mkdir(parents=True)
        
        # Create test translation files
        self._create_test_translations()
        
        # Initialize utils with test locale directory
        self.utils = LocalizationUtils('en')
        self.utils.translation_manager.locale_dir = self.locale_dir
        self.utils.translation_manager._load_translations()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_translations(self):
        """Create test translation files."""
        # Create minimal test translations
        en_dir = self.locale_dir / 'en'
        en_dir.mkdir()
        en_translations = {
            'language': 'en',
            'messages': {
                'Hello {name}': 'Hello {name}',
                'Processing...': 'Processing...',
                'File not found: {filename}': 'File not found: {filename}',
                'meter': 'meter',
                'meters': 'meters'
            }
        }
        
        with open(en_dir / 'messages.json', 'w', encoding='utf-8') as f:
            json.dump(en_translations, f, indent=2)
    
    def test_initialization(self):
        """Test localization utils initialization."""
        self.assertEqual(self.utils.current_language, 'en')
        self.assertIsNotNone(self.utils.translation_manager)
        self.assertIsNotNone(self.utils.formatter)
    
    def test_language_switching(self):
        """Test language switching in utils."""
        self.assertEqual(self.utils.get_language(), 'en')
        
        self.utils.set_language('ru')
        self.assertEqual(self.utils.get_language(), 'ru')
    
    def test_translation_with_formatting(self):
        """Test translation with string formatting."""
        result = self.utils.translate('Hello {name}', name='World')
        self.assertEqual(result, 'Hello World')
    
    def test_plural_translation_with_formatting(self):
        """Test plural translation with formatting."""
        result = self.utils.translate_plural('meter', 'meters', 1)
        self.assertIn('meter', result)
        
        result = self.utils.translate_plural('meter', 'meters', 5)
        self.assertIn('meters', result)
    
    def test_number_formatting(self):
        """Test number formatting through utils."""
        result = self.utils.format_number(1234.56)
        self.assertIn('1,234', result)
        self.assertIn('.56', result)
    
    def test_currency_formatting(self):
        """Test currency formatting through utils."""
        result = self.utils.format_currency(1234.56)
        self.assertIn('$', result)
        self.assertIn('1,234.56', result)
    
    def test_percentage_formatting(self):
        """Test percentage formatting through utils."""
        result = self.utils.format_percentage(0.1234)
        self.assertIn('12.3%', result)
    
    def test_date_formatting(self):
        """Test date formatting through utils."""
        test_date = date(2025, 1, 15)
        result = self.utils.format_date(test_date)
        self.assertIn('01/15/2025', result)
    
    def test_measurement_formatting(self):
        """Test measurement formatting with translation."""
        result = self.utils.format_measurement(123.45, 'meter')
        self.assertIn('123.45', result)
        self.assertIn('meter', result)
    
    def test_error_message_localization(self):
        """Test localized error messages."""
        result = self.utils.get_localized_error_message(
            'file_not_found', 
            filename='test.txt'
        )
        self.assertIn('test.txt', result)
    
    def test_status_message_localization(self):
        """Test localized status messages."""
        result = self.utils.get_localized_status_message(
            'processing',
            operation='data'
        )
        self.assertIn('data', result)
    
    def test_unit_name_localization(self):
        """Test unit name localization."""
        result = self.utils.get_localized_unit_name('m', plural=False)
        self.assertIn('meter', result)
        
        result = self.utils.get_localized_unit_name('m', plural=True)
        self.assertIn('meters', result)
    
    def test_menu_localization(self):
        """Test menu structure localization."""
        menu_structure = {
            'label': 'File',
            'items': [
                {'label': 'Open'},
                {'label': 'Save'}
            ]
        }
        
        localized = self.utils.create_localized_menu(menu_structure)
        self.assertIn('label', localized)
        self.assertIn('items', localized)
    
    def test_input_validation(self):
        """Test input validation with locale awareness."""
        # Valid number
        is_valid, value, error = self.utils.validate_and_format_input('123.45', 'number')
        self.assertTrue(is_valid)
        self.assertEqual(value, 123.45)
        self.assertEqual(error, '')
        
        # Invalid number
        is_valid, value, error = self.utils.validate_and_format_input('not_a_number', 'number')
        self.assertFalse(is_valid)
        self.assertIsNone(value)
        self.assertNotEqual(error, '')
    
    def test_export_translations(self):
        """Test translation export functionality."""
        export_dir = Path(self.temp_dir) / 'export'
        self.utils.export_translations(str(export_dir))
        
        # Check if export files were created
        self.assertTrue(export_dir.exists())
        exported_files = list(export_dir.rglob('messages.json'))
        self.assertTrue(len(exported_files) > 0)


class TestGlobalFunctions(unittest.TestCase):
    """Test global convenience functions."""
    
    def test_global_translation_manager(self):
        """Test global translation manager function."""
        manager = get_translation_manager()
        self.assertIsInstance(manager, TranslationManager)
        
        # Should return same instance
        manager2 = get_translation_manager()
        self.assertIs(manager, manager2)
    
    def test_global_formatter(self):
        """Test global formatter function."""
        formatter = get_regional_formatter('en')
        self.assertIsInstance(formatter, RegionalFormatter)
        self.assertEqual(formatter.language, 'en')
        
        formatter_ru = get_regional_formatter('ru')
        self.assertIsInstance(formatter_ru, RegionalFormatter)
        self.assertEqual(formatter_ru.language, 'ru')
    
    def test_global_localization_utils(self):
        """Test global localization utils function."""
        utils = get_localization_utils('en')
        self.assertIsInstance(utils, LocalizationUtils)
        self.assertEqual(utils.current_language, 'en')
    
    def test_translation_function(self):
        """Test global translation function."""
        # This should work with default translations
        result = _('File')
        self.assertIsInstance(result, str)
    
    def test_plural_translation_function(self):
        """Test global plural translation function."""
        # This should work with default translations
        result = ngettext('item', 'items', 1)
        self.assertIsInstance(result, str)
        
        result = ngettext('item', 'items', 5)
        self.assertIsInstance(result, str)
    
    def test_detect_system_language_function(self):
        """Test global language detection function."""
        language = detect_system_language()
        self.assertIn(language, ['en', 'ru'])


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestTranslationManager))
    suite.addTest(unittest.makeSuite(TestRegionalFormatter))
    suite.addTest(unittest.makeSuite(TestLanguageDetector))
    suite.addTest(unittest.makeSuite(TestLocalizationUtils))
    suite.addTest(unittest.makeSuite(TestGlobalFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")