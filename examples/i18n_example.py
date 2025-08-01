"""
Internationalization (i18n) System Example

This example demonstrates the complete functionality of the i18n system
including translation management, regional formatting, language detection,
and localization utilities for the Coal Deposit Interpolation Tool.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from i18n import (
    init_i18n, 
    get_current_language, 
    set_language, 
    get_available_languages,
    _, 
    format_number, 
    format_date, 
    format_currency
)
from i18n.localization_utils import get_localization_utils
from i18n.language_detector import detect_system_language


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def demonstrate_basic_translation():
    """Demonstrate basic translation functionality."""
    print_section("Basic Translation Functionality")
    
    # Initialize i18n system
    print("Initializing i18n system...")
    init_i18n()
    
    print(f"System detected language: {detect_system_language()}")
    print(f"Available languages: {get_available_languages()}")
    print(f"Current language: {get_current_language()}")
    
    # Test basic translations in English
    print("\n--- English Translations ---")
    set_language('en')
    print(f"Application title: {_('Coal Deposit Interpolation Tool')}")
    print(f"File menu: {_('File')}")
    print(f"Open command: {_('Open')}")
    print(f"Save command: {_('Save')}")
    print(f"Interpolation: {_('Interpolation')}")
    print(f"IDW method: {_('IDW')}")
    print(f"Kriging method: {_('Kriging')}")
    
    # Test basic translations in Russian
    print("\n--- Russian Translations ---")
    set_language('ru')
    print(f"Application title: {_('Coal Deposit Interpolation Tool')}")
    print(f"File menu: {_('File')}")
    print(f"Open command: {_('Open')}")
    print(f"Save command: {_('Save')}")
    print(f"Interpolation: {_('Interpolation')}")
    print(f"IDW method: {_('IDW')}")
    print(f"Kriging method: {_('Kriging')}")


def demonstrate_regional_formatting():
    """Demonstrate regional formatting for numbers, dates, and currencies."""
    print_section("Regional Formatting")
    
    test_number = 1234567.89
    test_currency = 9876.54
    test_date = date(2025, 3, 15)
    test_percentage = 0.1234
    
    # English formatting
    print("\n--- English (US) Formatting ---")
    set_language('en')
    utils = get_localization_utils('en')
    
    print(f"Number: {utils.format_number(test_number)}")
    print(f"Currency: {utils.format_currency(test_currency)}")
    print(f"Date: {utils.format_date(test_date)}")
    print(f"Percentage: {utils.format_percentage(test_percentage)}")
    print(f"Large number: {utils.formatter.format_large_number(test_number)}")
    print(f"Scientific: {utils.formatter.format_scientific(0.000123)}")
    
    # Russian formatting
    print("\n--- Russian (RU) Formatting ---")
    set_language('ru')
    utils = get_localization_utils('ru')
    
    print(f"Number: {utils.format_number(test_number)}")
    print(f"Currency: {utils.format_currency(test_currency)}")
    print(f"Date: {utils.format_date(test_date)}")
    print(f"Percentage: {utils.format_percentage(test_percentage)}")
    print(f"Large number: {utils.formatter.format_large_number(test_number)}")
    print(f"Scientific: {utils.formatter.format_scientific(0.000123)}")


def demonstrate_geological_terminology():
    """Demonstrate geological and mining terminology translation."""
    print_section("Geological Terminology Translation")
    
    geological_terms = [
        'Coal Seam',
        'Thickness', 
        'Depth',
        'Elevation',
        'Quality',
        'Ash Content',
        'Moisture',
        'Volatile Matter',
        'Fixed Carbon',
        'Calorific Value',
        'Sulfur Content',
        'Overburden',
        'Drill Hole',
        'Core',
        'Sample',
        'Reserve',
        'Resource'
    ]
    
    # English terminology
    print("\n--- English Geological Terms ---")
    set_language('en')
    for term in geological_terms:
        print(f"{term}: {_(term)}")
    
    # Russian terminology  
    print("\n--- Russian Geological Terms ---")
    set_language('ru')
    for term in geological_terms:
        print(f"{term}: {_(term)}")


def demonstrate_error_and_status_messages():
    """Demonstrate localized error and status messages."""
    print_section("Error and Status Messages")
    
    utils = get_localization_utils()
    
    # English messages
    print("\n--- English Messages ---")
    set_language('en')
    utils.set_language('en')
    
    print("Status messages:")
    print(f"- {utils.get_localized_status_message('loading')}")
    print(f"- {utils.get_localized_status_message('processing', operation='geological data')}")
    print(f"- {utils.get_localized_status_message('saving', filename='coal_data.csv')}")
    print(f"- {utils.get_localized_status_message('interpolating', method='Kriging')}")
    print(f"- {utils.get_localized_status_message('complete')}")
    
    print("\nError messages:")
    print(f"- {utils.get_localized_error_message('file_not_found', filename='missing.las')}")
    print(f"- {utils.get_localized_error_message('invalid_data')}")
    print(f"- {utils.get_localized_error_message('interpolation_error', method='IDW')}")
    print(f"- {utils.get_localized_error_message('database_error', details='Connection timeout')}")
    
    # Russian messages
    print("\n--- Russian Messages ---")
    set_language('ru')
    utils.set_language('ru')
    
    print("Status messages:")
    print(f"- {utils.get_localized_status_message('loading')}")
    print(f"- {utils.get_localized_status_message('processing', operation='геологические данные')}")
    print(f"- {utils.get_localized_status_message('saving', filename='coal_data.csv')}")
    print(f"- {utils.get_localized_status_message('interpolating', method='Кригинг')}")
    print(f"- {utils.get_localized_status_message('complete')}")
    
    print("\nError messages:")
    print(f"- {utils.get_localized_error_message('file_not_found', filename='missing.las')}")
    print(f"- {utils.get_localized_error_message('invalid_data')}")
    print(f"- {utils.get_localized_error_message('interpolation_error', method='IDW')}")
    print(f"- {utils.get_localized_error_message('database_error', details='Таймаут соединения')}")


def demonstrate_units_and_measurements():
    """Demonstrate unit and measurement formatting."""
    print_section("Units and Measurements")
    
    utils = get_localization_utils()
    
    measurements = [
        (1250.5, 'm', 'meters'),
        (0.75, 'km', 'kilometers'), 
        (45.8, 't', 'tons'),
        (23.4, '%', 'percentage'),
        (156.789, 'kg', 'kilograms')
    ]
    
    # English measurements
    print("\n--- English Measurements ---")
    set_language('en')
    utils.set_language('en')
    
    for value, unit, unit_name in measurements:
        formatted = utils.format_measurement(value, utils.get_localized_unit_name(unit_name))
        print(f"{value} {unit} = {formatted}")
    
    # Russian measurements
    print("\n--- Russian Measurements ---")
    set_language('ru')
    utils.set_language('ru')
    
    for value, unit, unit_name in measurements:
        formatted = utils.format_measurement(value, utils.get_localized_unit_name(unit_name))
        print(f"{value} {unit} = {formatted}")


def demonstrate_menu_localization():
    """Demonstrate menu structure localization."""
    print_section("Menu Localization")
    
    # Define application menu structure
    menu_structure = {
        'label': 'File',
        'items': [
            {'label': 'New'},
            {'label': 'Open'},
            {'label': 'Save'},
            {'label': 'Save As'},
            {'label': 'Export', 'items': [
                {'label': 'PDF Report'},
                {'label': 'Excel Report'},
                {'label': 'HTML Report'}
            ]},
            {'label': 'Exit'}
        ]
    }
    
    tools_menu = {
        'label': 'Tools',
        'items': [
            {'label': 'Interpolation Method'},
            {'label': 'Validation'},
            {'label': 'Data Summary'},
            {'label': 'Visualization'}
        ]
    }
    
    utils = get_localization_utils()
    
    def print_menu(menu, indent=0):
        """Recursively print menu structure."""
        prefix = "  " * indent
        print(f"{prefix}- {menu['label']}")
        if 'items' in menu:
            for item in menu['items']:
                print_menu(item, indent + 1)
    
    # English menu
    print("\n--- English Menu ---")
    set_language('en')
    utils.set_language('en')
    
    en_file_menu = utils.create_localized_menu(menu_structure)
    en_tools_menu = utils.create_localized_menu(tools_menu)
    
    print_menu(en_file_menu)
    print_menu(en_tools_menu)
    
    # Russian menu
    print("\n--- Russian Menu ---")
    set_language('ru')
    utils.set_language('ru')
    
    ru_file_menu = utils.create_localized_menu(menu_structure)
    ru_tools_menu = utils.create_localized_menu(tools_menu)
    
    print_menu(ru_file_menu)
    print_menu(ru_tools_menu)


def demonstrate_input_validation():
    """Demonstrate locale-aware input validation."""
    print_section("Locale-Aware Input Validation")
    
    utils = get_localization_utils()
    
    test_inputs = [
        ('123.45', 'number'),
        ('1,234.56', 'number'),
        ('$1,234.56', 'currency'),
        ('12.5%', 'percentage'),
        ('15/03/2025', 'date')
    ]
    
    # English validation
    print("\n--- English Input Validation ---")
    set_language('en')
    utils.set_language('en')
    
    for input_value, input_type in test_inputs:
        is_valid, parsed_value, error = utils.validate_and_format_input(input_value, input_type)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"{input_value} ({input_type}): {status}")
        if is_valid:
            print(f"  Parsed: {parsed_value}")
        else:
            print(f"  Error: {error}")
    
    # Russian validation (with different decimal separator)
    print("\n--- Russian Input Validation ---")
    set_language('ru')
    utils.set_language('ru')
    
    russian_inputs = [
        ('123,45', 'number'),      # Russian decimal separator
        ('1 234,56', 'number'),    # Russian thousands separator
        ('1 234,56 ₽', 'currency'),
        ('12,5%', 'percentage'),
        ('15.03.2025', 'date')     # Russian date format
    ]
    
    for input_value, input_type in russian_inputs:
        is_valid, parsed_value, error = utils.validate_and_format_input(input_value, input_type)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"{input_value} ({input_type}): {status}")
        if is_valid:
            print(f"  Parsed: {parsed_value}")
        else:
            print(f"  Error: {error}")


def demonstrate_coordinate_formatting():
    """Demonstrate coordinate formatting for geological data."""
    print_section("Coordinate Formatting")
    
    utils = get_localization_utils()
    
    # Sample coordinates (latitude, longitude)
    coordinates = [
        (55.7558, 37.6176),  # Moscow
        (59.9311, 30.3609),  # St. Petersburg
        (56.8431, 60.6454),  # Yekaterinburg
        (-33.8688, 151.2093) # Sydney (for contrast)
    ]
    
    print("\n--- Coordinate Formatting Examples ---")
    
    for lat, lon in coordinates:
        print(f"\nLocation: Lat {lat}, Lon {lon}")
        
        # Decimal degrees
        print(f"  Decimal: {utils.formatter.format_coordinate(lat, 'decimal', 6)}, {utils.formatter.format_coordinate(lon, 'decimal', 6)}")
        
        # UTM-style (just formatted as large numbers)
        print(f"  UTM-style: {utils.formatter.format_coordinate(lat * 111000, 'utm')}, {utils.formatter.format_coordinate(lon * 111000, 'utm')}")


def demonstrate_data_summary():
    """Demonstrate localized data summary for geological dataset."""
    print_section("Geological Data Summary (Localized)")
    
    # Simulate geological data summary
    data_stats = {
        'sample_count': 1247,
        'coal_seams': 3,
        'total_thickness': 12.45,  # meters
        'average_ash_content': 15.2,  # percentage
        'moisture_content': 8.7,  # percentage
        'calorific_value': 6450,  # kcal/kg
        'area_coverage': 256.78,  # square kilometers
        'drill_holes': 89
    }
    
    utils = get_localization_utils()
    
    # English summary
    print("\n--- English Data Summary ---")
    set_language('en')
    utils.set_language('en')
    
    print(f"Dataset Summary:")
    print(f"  Total samples: {utils.format_number(data_stats['sample_count'], precision=0)}")
    print(f"  Coal seams identified: {data_stats['coal_seams']}")
    print(f"  Total thickness: {utils.format_measurement(data_stats['total_thickness'], 'meters')}")
    print(f"  Average ash content: {utils.format_percentage(data_stats['average_ash_content'] / 100)}")
    print(f"  Moisture content: {utils.format_percentage(data_stats['moisture_content'] / 100)}")
    print(f"  Calorific value: {utils.format_number(data_stats['calorific_value'])} kcal/kg")
    print(f"  Area coverage: {utils.format_measurement(data_stats['area_coverage'], 'square kilometers')}")
    print(f"  Drill holes: {data_stats['drill_holes']}")
    
    # Russian summary
    print("\n--- Russian Data Summary ---")
    set_language('ru')
    utils.set_language('ru')
    
    print(f"Сводка датасета:")
    print(f"  Общее количество образцов: {utils.format_number(data_stats['sample_count'], precision=0)}")
    print(f"  Выявлено угольных пластов: {data_stats['coal_seams']}")
    print(f"  Общая мощность: {utils.format_measurement(data_stats['total_thickness'], 'метры')}")
    print(f"  Средняя зольность: {utils.format_percentage(data_stats['average_ash_content'] / 100)}")
    print(f"  Влажность: {utils.format_percentage(data_stats['moisture_content'] / 100)}")
    print(f"  Теплотворная способность: {utils.format_number(data_stats['calorific_value'])} ккал/кг")
    print(f"  Площадь покрытия: {utils.format_measurement(data_stats['area_coverage'], 'квадратные километры')}")
    print(f"  Скважины: {data_stats['drill_holes']}")


def run_interactive_demo():
    """Run an interactive demonstration."""
    print_section("Interactive Language Switching Demo")
    
    utils = get_localization_utils()
    
    print("This demo allows you to see real-time language switching.")
    print("Available languages:", get_available_languages())
    
    sample_messages = [
        'Coal Deposit Interpolation Tool',
        'Load Data',
        'Interpolation Method',
        'Generate Report',
        'Processing...',
        'Complete'
    ]
    
    while True:
        print(f"\nCurrent language: {get_current_language()}")
        print("\nSample translations:")
        for msg in sample_messages:
            print(f"  {msg} → {_(msg)}")
        
        print("\nFormatting examples:")
        print(f"  Number: {utils.format_number(12345.67)}")
        print(f"  Currency: {utils.format_currency(9876.54)}")
        print(f"  Date: {utils.format_date(datetime.now().date())}")
        
        print("\nCommands:")
        print("  'en' - Switch to English")
        print("  'ru' - Switch to Russian")
        print("  'q' - Quit")
        
        choice = input("\nEnter command: ").lower().strip()
        
        if choice == 'q':
            break
        elif choice in ['en', 'ru']:
            set_language(choice)
            utils.set_language(choice)
            print(f"Language switched to: {choice}")
        else:
            print("Invalid command. Please try again.")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print(" Coal Deposit Interpolation Tool")
    print(" Internationalization (i18n) System Demonstration")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_basic_translation()
        demonstrate_regional_formatting()
        demonstrate_geological_terminology()
        demonstrate_error_and_status_messages()
        demonstrate_units_and_measurements()
        demonstrate_menu_localization()
        demonstrate_input_validation()
        demonstrate_coordinate_formatting()
        demonstrate_data_summary()
        
        # Ask if user wants interactive demo
        print_section("Interactive Demo")
        print("Would you like to run the interactive language switching demo?")
        choice = input("Enter 'y' for yes, any other key to skip: ").lower().strip()
        
        if choice == 'y':
            run_interactive_demo()
        
        print_section("Demonstration Complete")
        print("The i18n system provides comprehensive internationalization support for:")
        print("- Multi-language user interface")
        print("- Regional number and date formatting")
        print("- Geological terminology translation")
        print("- Localized error and status messages")
        print("- Unit and measurement formatting")
        print("- Menu structure localization")
        print("- Input validation with locale awareness")
        print("- Dynamic language switching")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()