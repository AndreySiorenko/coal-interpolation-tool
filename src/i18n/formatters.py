"""
Regional formatters for numbers, dates, and currencies.

Provides locale-aware formatting for different regions and languages.
"""

import locale
import logging
from typing import Optional, Union, Dict, Any
from datetime import datetime, date
from decimal import Decimal
import threading


class RegionalFormatter:
    """
    Handles regional formatting for numbers, dates, and currencies.
    
    Provides locale-aware formatting with fallback mechanisms and
    custom formatting rules for geological and mining data.
    """
    
    def __init__(self, language: str = 'en', region: Optional[str] = None):
        """
        Initialize regional formatter.
        
        Args:
            language: Language code (e.g., 'en', 'ru')
            region: Region code (e.g., 'US', 'RU')
        """
        self.logger = logging.getLogger(__name__)
        self.language = language
        self.region = region or self._get_default_region(language)
        self.locale_code = self._build_locale_code()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Formatting settings
        self.settings = self._load_regional_settings()
        
        # Try to set system locale
        self._set_system_locale()
    
    def _get_default_region(self, language: str) -> str:
        """Get default region for a language."""
        defaults = {
            'en': 'US',
            'ru': 'RU'
        }
        return defaults.get(language, 'US')
    
    def _build_locale_code(self) -> str:
        """Build locale code from language and region."""
        return f"{self.language}_{self.region}"
    
    def _load_regional_settings(self) -> Dict[str, Any]:
        """Load regional formatting settings."""
        settings = {
            'en_US': {
                'decimal_separator': '.',
                'thousands_separator': ',',
                'currency_symbol': '$',
                'currency_position': 'before',
                'date_format': '%m/%d/%Y',
                'time_format': '%I:%M:%S %p',
                'datetime_format': '%m/%d/%Y %I:%M:%S %p',
                'number_precision': 2,
                'large_number_format': 'short'  # 1.2K, 1.5M
            },
            'ru_RU': {
                'decimal_separator': ',',
                'thousands_separator': ' ',  # Non-breaking space
                'currency_symbol': '₽',
                'currency_position': 'after',
                'date_format': '%d.%m.%Y',
                'time_format': '%H:%M:%S',
                'datetime_format': '%d.%m.%Y %H:%M:%S',
                'number_precision': 2,
                'large_number_format': 'long'  # 1200, 1500000
            }
        }
        
        return settings.get(self.locale_code, settings['en_US'])
    
    def _set_system_locale(self):
        """Try to set system locale."""
        try:
            # Common locale variations to try
            locale_variants = [
                self.locale_code,
                f"{self.locale_code}.UTF-8",
                f"{self.locale_code}.utf8",
                self.language,
                f"{self.language}.UTF-8"
            ]
            
            for variant in locale_variants:
                try:
                    locale.setlocale(locale.LC_ALL, variant)
                    self.logger.debug(f"System locale set to: {variant}")
                    self.system_locale_available = True
                    return
                except locale.Error:
                    continue
            
            # Fallback to C locale
            locale.setlocale(locale.LC_ALL, 'C')
            self.system_locale_available = False
            self.logger.warning(f"Could not set locale {self.locale_code}, using C locale")
            
        except Exception as e:
            self.logger.error(f"Error setting locale: {e}")
            self.system_locale_available = False
    
    def format_number(self, value: Union[int, float, Decimal], 
                     precision: Optional[int] = None,
                     thousands_sep: bool = True,
                     prefix: str = '',
                     suffix: str = '') -> str:
        """
        Format a number according to regional settings.
        
        Args:
            value: Number to format
            precision: Decimal precision (uses regional default if None)
            thousands_sep: Whether to include thousands separator
            prefix: Prefix to add (e.g., unit symbols)
            suffix: Suffix to add (e.g., unit names)
            
        Returns:
            Formatted number string
        """
        with self._lock:
            try:
                if precision is None:
                    precision = self.settings['number_precision']
                
                # Handle None/NaN values
                if value is None or (isinstance(value, float) and str(value).lower() in ['nan', 'inf', '-inf']):
                    return 'N/A'
                
                # Convert to float for processing
                num_value = float(value)
                
                # Format the number
                if thousands_sep and self.system_locale_available:
                    # Use system locale if available
                    formatted = locale.format_string(f"%.{precision}f", num_value, grouping=True)
                else:
                    # Manual formatting
                    formatted = f"{num_value:.{precision}f}"
                    
                    if thousands_sep:
                        # Add thousands separator manually
                        parts = formatted.split('.')
                        integer_part = parts[0]
                        decimal_part = parts[1] if len(parts) > 1 else ''
                        
                        # Add thousands separators
                        if len(integer_part) > 3:
                            separated = ''
                            for i, digit in enumerate(reversed(integer_part)):
                                if i > 0 and i % 3 == 0:
                                    separated = self.settings['thousands_separator'] + separated
                                separated = digit + separated
                            integer_part = separated
                        
                        formatted = integer_part
                        if decimal_part:
                            formatted += self.settings['decimal_separator'] + decimal_part
                
                # Apply regional decimal separator
                if not self.system_locale_available:
                    formatted = formatted.replace('.', self.settings['decimal_separator'])
                
                return f"{prefix}{formatted}{suffix}"
                
            except Exception as e:
                self.logger.error(f"Error formatting number {value}: {e}")
                return str(value)
    
    def format_large_number(self, value: Union[int, float], 
                           precision: int = 1) -> str:
        """
        Format large numbers with appropriate suffixes (K, M, B).
        
        Args:
            value: Number to format
            precision: Decimal precision for scaled numbers
            
        Returns:
            Formatted number with suffix
        """
        if value is None or (isinstance(value, float) and str(value).lower() in ['nan', 'inf', '-inf']):
            return 'N/A'
        
        abs_value = abs(float(value))
        sign = '-' if value < 0 else ''
        
        if self.settings['large_number_format'] == 'short':
            # Use K, M, B suffixes
            if abs_value >= 1e9:
                scaled = abs_value / 1e9
                suffix = 'B'
            elif abs_value >= 1e6:
                scaled = abs_value / 1e6
                suffix = 'M'
            elif abs_value >= 1e3:
                scaled = abs_value / 1e3
                suffix = 'K'
            else:
                return self.format_number(value, precision=0)
            
            formatted = self.format_number(scaled, precision=precision, thousands_sep=False)
            return f"{sign}{formatted}{suffix}"
        else:
            # Use full numbers with thousands separators
            return self.format_number(value, precision=0)
    
    def format_percentage(self, value: Union[int, float], 
                         precision: int = 1) -> str:
        """
        Format a percentage value.
        
        Args:
            value: Percentage value (0.0-1.0 or 0-100)
            precision: Decimal precision
            
        Returns:
            Formatted percentage string
        """
        if value is None or (isinstance(value, float) and str(value).lower() in ['nan', 'inf', '-inf']):
            return 'N/A'
        
        # Assume values > 1 are already in percentage form
        percent_value = value if value > 1 else value * 100
        
        formatted = self.format_number(percent_value, precision=precision, thousands_sep=False)
        return f"{formatted}%"
    
    def format_currency(self, value: Union[int, float, Decimal],
                       currency_code: Optional[str] = None,
                       precision: Optional[int] = None) -> str:
        """
        Format a currency value.
        
        Args:
            value: Currency amount
            currency_code: Currency code (uses regional default if None)
            precision: Decimal precision (uses 2 if None)
            
        Returns:
            Formatted currency string
        """
        if value is None or (isinstance(value, float) and str(value).lower() in ['nan', 'inf', '-inf']):
            return 'N/A'
        
        if precision is None:
            precision = 2
        
        symbol = currency_code or self.settings['currency_symbol']
        formatted_number = self.format_number(value, precision=precision)
        
        if self.settings['currency_position'] == 'before':
            return f"{symbol}{formatted_number}"
        else:
            return f"{formatted_number} {symbol}"
    
    def format_date(self, date_value: Union[datetime, date], 
                   format_type: str = 'date') -> str:
        """
        Format a date according to regional settings.
        
        Args:
            date_value: Date to format
            format_type: Type of format ('date', 'time', 'datetime')
            
        Returns:
            Formatted date string
        """
        if date_value is None:
            return 'N/A'
        
        try:
            format_map = {
                'date': self.settings['date_format'],
                'time': self.settings['time_format'],
                'datetime': self.settings['datetime_format']
            }
            
            format_string = format_map.get(format_type, self.settings['date_format'])
            return date_value.strftime(format_string)
            
        except Exception as e:
            self.logger.error(f"Error formatting date {date_value}: {e}")
            return str(date_value)
    
    def format_scientific(self, value: Union[int, float], 
                         precision: int = 2) -> str:
        """
        Format number in scientific notation.
        
        Args:
            value: Number to format
            precision: Decimal precision
            
        Returns:
            Scientific notation string
        """
        if value is None or (isinstance(value, float) and str(value).lower() in ['nan', 'inf', '-inf']):
            return 'N/A'
        
        try:
            formatted = f"{float(value):.{precision}e}"
            # Replace decimal separator if needed
            if self.settings['decimal_separator'] != '.':
                formatted = formatted.replace('.', self.settings['decimal_separator'])
            return formatted
        except Exception as e:
            self.logger.error(f"Error formatting scientific notation for {value}: {e}")
            return str(value)
    
    def format_coordinate(self, value: Union[int, float], 
                         coord_type: str = 'decimal',
                         precision: int = 6) -> str:
        """
        Format coordinate values (lat/lon, UTM, etc.).
        
        Args:
            value: Coordinate value
            coord_type: Type of coordinate ('decimal', 'dms', 'utm')
            precision: Decimal precision
            
        Returns:
            Formatted coordinate string
        """
        if value is None or (isinstance(value, float) and str(value).lower() in ['nan', 'inf', '-inf']):
            return 'N/A'
        
        if coord_type == 'decimal':
            return self.format_number(value, precision=precision, thousands_sep=False)
        elif coord_type == 'dms':
            # Convert decimal degrees to degrees, minutes, seconds
            abs_value = abs(float(value))
            degrees = int(abs_value)
            minutes = int((abs_value - degrees) * 60)
            seconds = ((abs_value - degrees) * 60 - minutes) * 60
            
            direction = 'N' if coord_type == 'lat' and value >= 0 else 'S' if coord_type == 'lat' else 'E' if value >= 0 else 'W'
            
            return f"{degrees}°{minutes}'{seconds:.{precision}f}\"{direction}"
        else:
            return self.format_number(value, precision=0, thousands_sep=True)
    
    def format_measurement(self, value: Union[int, float], 
                          unit: str,
                          precision: Optional[int] = None) -> str:
        """
        Format measurement values with units.
        
        Args:
            value: Measurement value
            unit: Unit of measurement
            precision: Decimal precision
            
        Returns:
            Formatted measurement string
        """
        if value is None or (isinstance(value, float) and str(value).lower() in ['nan', 'inf', '-inf']):
            return f'N/A {unit}'
        
        if precision is None:
            # Default precision based on typical measurement ranges
            precision = 2 if abs(float(value)) < 100 else 1 if abs(float(value)) < 1000 else 0
        
        formatted_value = self.format_number(value, precision=precision)
        return f"{formatted_value} {unit}"
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current regional settings."""
        return self.settings.copy()
    
    def update_settings(self, **kwargs):
        """Update regional settings."""
        with self._lock:
            self.settings.update(kwargs)


# Global formatter instance
_global_formatter: Optional[RegionalFormatter] = None

def get_regional_formatter(language: Optional[str] = None) -> RegionalFormatter:
    """Get the global regional formatter instance."""
    global _global_formatter
    if _global_formatter is None or (language and _global_formatter.language != language):
        _global_formatter = RegionalFormatter(language or 'en')
    return _global_formatter

def format_number(value: Union[int, float, Decimal], 
                 language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for number formatting."""
    formatter = get_regional_formatter(language)
    return formatter.format_number(value, **kwargs)

def format_currency(value: Union[int, float, Decimal], 
                   language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for currency formatting."""
    formatter = get_regional_formatter(language)
    return formatter.format_currency(value, **kwargs)

def format_date(date_value: Union[datetime, date], 
               language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for date formatting."""
    formatter = get_regional_formatter(language)
    return formatter.format_date(date_value, **kwargs)

def format_percentage(value: Union[int, float], 
                     language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for percentage formatting."""
    formatter = get_regional_formatter(language)
    return formatter.format_percentage(value, **kwargs)