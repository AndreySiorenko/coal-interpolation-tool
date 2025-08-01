"""
Language detection utilities.

Automatically detects system language and provides fallback mechanisms.
"""

import os
import locale
import logging
from typing import Optional, List, Dict, Any
import platform


class LanguageDetector:
    """
    Detects system language and provides language preference management.
    
    Uses multiple detection methods with fallback hierarchy:
    1. Environment variables (LANG, LANGUAGE, LC_ALL)
    2. System locale settings
    3. Platform-specific detection
    4. User preferences (if stored)
    """
    
    def __init__(self):
        """Initialize language detector."""
        self.logger = logging.getLogger(__name__)
        self.supported_languages = ['en', 'ru']
        self.language_mappings = self._build_language_mappings()
    
    def _build_language_mappings(self) -> Dict[str, str]:
        """Build mappings from various language codes to supported languages."""
        return {
            # English variants
            'en': 'en',
            'en_US': 'en',
            'en_GB': 'en',
            'en_CA': 'en',
            'en_AU': 'en',
            'english': 'en',
            
            # Russian variants
            'ru': 'ru',
            'ru_RU': 'ru',
            'ru_UA': 'ru',
            'russian': 'ru',
            'русский': 'ru',
            
            # Common aliases
            'rus': 'ru',
            'eng': 'en'
        }
    
    def detect_system_language(self) -> str:
        """
        Detect system language using multiple methods.
        
        Returns:
            Language code ('en' or 'ru'), defaults to 'en'
        """
        detection_methods = [
            self._detect_from_environment,
            self._detect_from_locale,
            self._detect_from_platform,
            self._detect_from_region
        ]
        
        for method in detection_methods:
            try:
                detected = method()
                if detected:
                    mapped = self._map_language_code(detected)
                    if mapped in self.supported_languages:
                        self.logger.info(f"Detected system language: {mapped} (from {detected})")
                        return mapped
            except Exception as e:
                self.logger.debug(f"Language detection method failed: {e}")
        
        # Default fallback
        self.logger.info("Using default language: en")
        return 'en'
    
    def _detect_from_environment(self) -> Optional[str]:
        """Detect language from environment variables."""
        env_vars = ['LANG', 'LANGUAGE', 'LC_ALL', 'LC_MESSAGES']
        
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                # Extract language code (e.g., 'ru_RU.UTF-8' -> 'ru_RU')
                lang_code = value.split('.')[0].split('@')[0]
                if lang_code:
                    self.logger.debug(f"Found language in {var}: {lang_code}")
                    return lang_code
        
        return None
    
    def _detect_from_locale(self) -> Optional[str]:
        """Detect language from system locale."""
        try:
            # Get default locale
            default_locale = locale.getdefaultlocale()
            if default_locale and default_locale[0]:
                lang_code = default_locale[0]
                self.logger.debug(f"Found language from locale: {lang_code}")
                return lang_code
        except Exception as e:
            self.logger.debug(f"Error detecting locale: {e}")
        
        return None
    
    def _detect_from_platform(self) -> Optional[str]:
        """Detect language using platform-specific methods."""
        system = platform.system()
        
        if system == 'Windows':
            return self._detect_windows_language()
        elif system == 'Darwin':  # macOS
            return self._detect_macos_language()
        elif system == 'Linux':
            return self._detect_linux_language()
        
        return None
    
    def _detect_windows_language(self) -> Optional[str]:
        """Detect language on Windows systems."""
        try:
            import winreg
            
            # Try to read from Windows registry
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                               r"Control Panel\International") as key:
                locale_name = winreg.QueryValueEx(key, "LocaleName")[0]
                self.logger.debug(f"Windows locale: {locale_name}")
                return locale_name.split('-')[0]  # Extract language part
                
        except ImportError:
            # winreg not available (not Windows)
            pass
        except Exception as e:
            self.logger.debug(f"Error reading Windows registry: {e}")
        
        try:
            # Alternative: use ctypes to call Windows API
            import ctypes
            
            # Get user default locale
            lcid = ctypes.windll.kernel32.GetUserDefaultLCID()
            lang_id = lcid & 0x3FF  # Extract primary language ID
            
            # Map common language IDs
            lang_map = {
                0x09: 'en',  # English
                0x19: 'ru'   # Russian
            }
            
            if lang_id in lang_map:
                detected = lang_map[lang_id]
                self.logger.debug(f"Windows API language: {detected}")
                return detected
                
        except Exception as e:
            self.logger.debug(f"Error using Windows API: {e}")
        
        return None
    
    def _detect_macos_language(self) -> Optional[str]:
        """Detect language on macOS systems."""
        try:
            import subprocess
            
            # Use defaults command to get preferred languages
            result = subprocess.run(
                ['defaults', 'read', '-g', 'AppleLanguages'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse the output (it's an array format)
                output = result.stdout.strip()
                if output:
                    # Extract first language (remove quotes and parentheses)
                    lines = output.split('\n')
                    for line in lines:
                        line = line.strip(' "(),')
                        if line and not line.startswith('('):
                            lang_code = line.split('-')[0]
                            self.logger.debug(f"macOS preferred language: {lang_code}")
                            return lang_code
            
        except Exception as e:
            self.logger.debug(f"Error detecting macOS language: {e}")
        
        return None
    
    def _detect_linux_language(self) -> Optional[str]:
        """Detect language on Linux systems."""
        try:
            # Try reading from /etc/locale.conf
            locale_conf_paths = ['/etc/locale.conf', '/etc/default/locale']
            
            for path in locale_conf_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        for line in f:
                            if line.startswith('LANG='):
                                lang_value = line.split('=', 1)[1].strip().strip('"')
                                lang_code = lang_value.split('.')[0]
                                self.logger.debug(f"Linux locale config: {lang_code}")
                                return lang_code
        
        except Exception as e:
            self.logger.debug(f"Error reading Linux locale config: {e}")
        
        return None
    
    def _detect_from_region(self) -> Optional[str]:
        """Detect language based on geographic region (fallback)."""
        try:
            # Very basic region-based detection
            # This is a fallback and not very reliable
            
            # Try to get timezone as a proxy for region
            import time
            
            if hasattr(time, 'tzname'):
                tz_info = time.tzname
                if tz_info:
                    tz_str = str(tz_info)
                    # Very basic heuristics
                    if 'MSK' in tz_str or 'russia' in tz_str.lower():
                        return 'ru'
            
        except Exception as e:
            self.logger.debug(f"Error detecting region: {e}")
        
        return None
    
    def _map_language_code(self, code: str) -> str:
        """Map various language codes to supported languages."""
        if not code:
            return 'en'
        
        code_lower = code.lower()
        
        # Direct mapping
        if code_lower in self.language_mappings:
            return self.language_mappings[code_lower]
        
        # Try just the language part (before underscore)
        lang_part = code_lower.split('_')[0]
        if lang_part in self.language_mappings:
            return self.language_mappings[lang_part]
        
        # Default to English
        return 'en'
    
    def get_language_preferences(self) -> List[str]:
        """
        Get ordered list of language preferences.
        
        Returns:
            List of language codes in order of preference
        """
        preferences = []
        
        # Add detected system language
        system_lang = self.detect_system_language()
        if system_lang not in preferences:
            preferences.append(system_lang)
        
        # Add fallback languages
        for lang in self.supported_languages:
            if lang not in preferences:
                preferences.append(lang)
        
        return preferences
    
    def is_rtl_language(self, language: str) -> bool:
        """
        Check if language uses right-to-left text direction.
        
        Args:
            language: Language code
            
        Returns:
            True if RTL language
        """
        rtl_languages = ['ar', 'he', 'fa', 'ur']
        return language in rtl_languages
    
    def get_language_info(self, language: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary with language information
        """
        language_info = {
            'en': {
                'name': 'English',
                'native_name': 'English',
                'iso_639_1': 'en',
                'iso_639_2': 'eng',
                'rtl': False,
                'decimal_separator': '.',
                'thousands_separator': ',',
                'typical_regions': ['US', 'GB', 'CA', 'AU']
            },
            'ru': {
                'name': 'Russian',
                'native_name': 'Русский',
                'iso_639_1': 'ru',
                'iso_639_2': 'rus',
                'rtl': False,
                'decimal_separator': ',',
                'thousands_separator': ' ',
                'typical_regions': ['RU', 'UA', 'BY', 'KZ']
            }
        }
        
        return language_info.get(language, language_info['en'])


# Global detector instance
_global_detector: Optional[LanguageDetector] = None

def get_language_detector() -> LanguageDetector:
    """Get the global language detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = LanguageDetector()
    return _global_detector

def detect_system_language() -> str:
    """Convenience function to detect system language."""
    detector = get_language_detector()
    return detector.detect_system_language()

def get_language_preferences() -> List[str]:
    """Convenience function to get language preferences."""
    detector = get_language_detector()
    return detector.get_language_preferences()

def get_language_info(language: str) -> Dict[str, Any]:
    """Convenience function to get language information."""
    detector = get_language_detector()
    return detector.get_language_info(language)