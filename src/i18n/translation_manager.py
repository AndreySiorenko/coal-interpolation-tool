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
        
        # Try to import gettext for .po/.mo support first
        self._init_gettext_support()
        
        # Initialize translations
        self._load_translations()
    
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
            'records': 'records',
            
            # Results panel
            'Results': 'Results',
            'Method:': 'Method:',
            'Min Value:': 'Min Value:',
            'Max Value:': 'Max Value:',
            'Mean Value:': 'Mean Value:',
            'Grid Information': 'Grid Information',
            'Grid Points:': 'Grid Points:',
            'Cell Size:': 'Cell Size:',
            'Grid Extent:': 'Grid Extent:',
            'Sample Results': 'Sample Results',
            'X': 'X',
            'Y': 'Y',
            'Value': 'Value',
            'View All Results': 'View All Results',
            'Export Results': 'Export Results',
            'Quality Report': 'Quality Report',
            'All Interpolation Results': 'All Interpolation Results',
            'Interpolated Value': 'Interpolated Value',
            'Close': 'Close',
            'Info': 'Info',
            'Export results functionality - To be implemented': 'Export results functionality - To be implemented',
            
            # Analysis and Recommendations
            'Analysis and Recommendations': 'Analysis and Recommendations',
            'Analyze Data': 'Analyze Data',
            'Apply Recommendations': 'Apply Recommendations',
            'Summary': 'Summary',
            'Method Comparison': 'Method Comparison',
            'Optimal Parameters': 'Optimal Parameters',
            

            'Method comparison will appear here after analysis.': 'Method comparison will appear here after analysis.',
            'Optimal parameters will appear here after analysis.': 'Optimal parameters will appear here after analysis.',
            'Recommended parameters have been applied.\nYou can now run the interpolation.': 'Recommended parameters have been applied.\nYou can now run the interpolation.',
            
            # Status messages
            'Ready': 'Ready',
            'Interpolation completed successfully': 'Interpolation completed successfully',
            'Error occurred': 'Error occurred',
            'Loaded data': 'Loaded data',
            'rows': 'rows',
            'Running interpolation...': 'Running interpolation...',
            'No data to save': 'No data to save',
            'Save Project': 'Save Project',
            'Load Project': 'Load Project',
            'Project files': 'Project files',
            'All files': 'All files',
            'No results to export': 'No results to export',
            'Please load data first': 'Please load data first',
            'Layout reset to default': 'Layout reset to default',
            'Do you want to quit?': 'Do you want to quit?',
            'Quit': 'Quit',
            
            # Dialog titles and labels
            'Data Statistics': 'Data Statistics',
            'Basic Statistics': 'Basic Statistics',
            'Spatial Statistics': 'Spatial Statistics',
            'Grid Information': 'Grid Information',
            'Grid Details': 'Grid Details',
            'Memory Estimate': 'Memory Estimate',
            'Preferences dialog - To be implemented': 'Preferences dialog - To be implemented',
            
            # Menu items
            'Project': 'Project',
            'Reset': 'Reset',
            'Layout': 'Layout',
            'Show': 'Show',
            'Data': 'Data',
            'Panel': 'Panel',
            'Language Settings': 'Language Settings',
            'Preferences': 'Preferences',
            'User Guide': 'User Guide',
            'About': 'About',
            'Grid Info': 'Grid Info',
            
            # Recommendation engine messages
            'No suitable interpolation method found': 'No suitable interpolation method found',
            'Data analysis warning:': 'Data analysis warning:',
            'Quality evaluation failed:': 'Quality evaluation failed:',
            'No analysis performed yet': 'No analysis performed yet',
            'No recommendations available. Run analyze_and_recommend first.': 'No recommendations available. Run analyze_and_recommend first.',
            'Interpolator creation for': 'Interpolator creation for',
            'not implemented': 'not implemented',
            
            # Data insights and analysis
            'Very few data points - results may be unreliable': 'Very few data points - results may be unreliable',
            'Limited data points - consider collecting more data': 'Limited data points - consider collecting more data',
            'Highly clustered data - consider using sectoral search': 'Highly clustered data - consider using sectoral search',
            'High proportion of outliers': 'High proportion of outliers',
            'detected': 'detected',
            'Very high variability in values - results may be unstable': 'Very high variability in values - results may be unstable',
            
            # Data pattern descriptions
            'Data points are uniformly distributed': 'Data points are uniformly distributed',
            'Data shows moderate clustering with some regular areas': 'Data shows moderate clustering with some regular areas',
            'Data is highly clustered in specific regions': 'Data is highly clustered in specific regions',
            'Data shows irregular spatial distribution': 'Data shows irregular spatial distribution',
            'Values are approximately normally distributed': 'Values are approximately normally distributed',
            'Values are strongly right-skewed': 'Values are strongly right-skewed',
            'Values are strongly left-skewed': 'Values are strongly left-skewed',
            'with low variability': 'with low variability',
            'with moderate variability': 'with moderate variability',
            'with high variability': 'with high variability',
            
            # Quality assessments
            'Excellent - suitable for most interpolation methods': 'Excellent - suitable for most interpolation methods',
            'Good - suitable for interpolation with appropriate parameters': 'Good - suitable for interpolation with appropriate parameters',
            'Fair - interpolation possible but results may be less reliable': 'Fair - interpolation possible but results may be less reliable',
            'Poor - consider data collection improvements': 'Poor - consider data collection improvements',
            
            # Special considerations and best practices
            'Consider detrending or using universal kriging': 'Consider detrending or using universal kriging',
            'Strong anisotropy detected - use directional parameters': 'Strong anisotropy detected - use directional parameters',
            '3D interpolation - ensure sufficient vertical sampling': '3D interpolation - ensure sufficient vertical sampling',
            'Extreme aspect ratio - consider coordinate transformation': 'Extreme aspect ratio - consider coordinate transformation',
            'Always validate results with cross-validation': 'Always validate results with cross-validation',
            'Visualize interpolation results to check for artifacts': 'Visualize interpolation results to check for artifacts',
            'With limited data, use conservative parameters': 'With limited data, use conservative parameters',
            'Consider removing or down-weighting outliers': 'Consider removing or down-weighting outliers',
            'Account for trends in your interpretation': 'Account for trends in your interpretation',
            
            # Parameter guidelines
            'Increase for sparse data, decrease for dense data': 'Increase for sparse data, decrease for dense data',
            'Increase for more local influence (1-4 range typical)': 'Increase for more local influence (1-4 range typical)',
            'Balance between accuracy (more points) and speed': 'Balance between accuracy (more points) and speed',
            'Enable for clustered or directional data': 'Enable for clustered or directional data',
            'Adjust ratio and angle based on directional patterns': 'Adjust ratio and angle based on directional patterns',
            
            # Potential issues
            'Too few points may lead to unreliable interpolation': 'Too few points may lead to unreliable interpolation',
            'Many points on edges - extrapolation may be unreliable': 'Many points on edges - extrapolation may be unreliable',
            'Severe clustering may cause interpolation artifacts': 'Severe clustering may cause interpolation artifacts',
            
            # Dialog and interface strings
            'Load Data File': 'Load Data File',
            'File Selection': 'File Selection',
            'No file selected': 'No file selected',
            'Recent Files': 'Recent Files',
            'File Information': 'File Information',
            'Column Mapping': 'Column Mapping',
            'Value Columns': 'Value Columns',
            'CSV Settings': 'CSV Settings',
            'Data Validation': 'Data Validation',
            'Skip rows with invalid coordinates': 'Skip rows with invalid coordinates',
            'Fill missing values with interpolation': 'Fill missing values with interpolation',
            'Remove duplicate coordinates': 'Remove duplicate coordinates',
            'Import Summary': 'Import Summary',
            'Select Data File': 'Select Data File',
            'Excel files': 'Excel files',
            'CSV files': 'CSV files',
            'All supported': 'All supported',
            'Encoding Error': 'Encoding Error',
            'The CSV file appears to be empty': 'The CSV file appears to be empty',
            'Parsing Error': 'Parsing Error',
            'Please complete all required settings': 'Please complete all required settings',
            'Skip invalid rows': 'Skip invalid rows',
            'Fill missing values': 'Fill missing values',
            'Remove duplicates': 'Remove duplicates',
            
            # Visualization panel
            'Display Options': 'Display Options',
            'Results Only': 'Results Only',
            'Export Image': 'Export Image',
            'Load data to see scatter plot': 'Load data to see scatter plot',
            'Run interpolation to see results': 'Run interpolation to see results',
            'Use toolbar for zoom and pan': 'Use toolbar for zoom and pan',
            'No visualization to export': 'No visualization to export',
            'Export Visualization': 'Export Visualization',
            
            # Export dialog
            'Export Interpolation Results': 'Export Interpolation Results',
            'Export Data Information': 'Export Data Information',
            'Export Format': 'Export Format',
            'Output File': 'Output File',
            'Common Options': 'Common Options',
            'Include metadata and creation information': 'Include metadata and creation information',
            'Overwrite existing files': 'Overwrite existing files',
            'Create output directories if needed': 'Create output directories if needed',
            'CSV Options': 'CSV Options',
            'Include column headers': 'Include column headers',
            'Include row index': 'Include row index',
            'GeoTIFF Options': 'GeoTIFF Options',
            'VTK Options': 'VTK Options',
            'Export Summary': 'Export Summary',
            'Data Preview': 'Data Preview',
            'Export Progress': 'Export Progress',
            'Ready to export': 'Ready to export',
            'No results data available': 'No results data available',
            'Results data structure not recognized': 'Results data structure not recognized',
            'Export Results': 'Export Results',
            'GeoTIFF files': 'GeoTIFF files',
            'TIFF files': 'TIFF files',
            'VTK files': 'VTK files',
            'VTK XML files': 'VTK XML files',
            
            # Error messages from controllers
            'No data loaded': 'No data loaded',
            'Data must contain at least one value column for interpolation': 'Data must contain at least one value column for interpolation',
            'No data loaded for interpolation': 'No data loaded for interpolation',
            'No grid data found in results': 'No grid data found in results',
            'No value columns found in grid data': 'No value columns found in grid data',
            'No data to save': 'No data to save',
            'No data file is currently loaded': 'No data file is currently loaded',
            'CSV file appears to be empty': 'CSV file appears to be empty',
            
            # Visualization panel help text
            'Visualization Panel': 'Visualization Panel',
            'Interpolate': 'Interpolate',
            'Load Project': 'Load Project',
            'Could not retrieve data information': 'Could not retrieve data information'
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
            'records': 'записи',
            
            # Results panel
            'Results': 'Результаты',
            'Method:': 'Метод:',
            'Min Value:': 'Мин. значение:',
            'Max Value:': 'Макс. значение:',
            'Mean Value:': 'Среднее значение:',
            'Grid Information': 'Информация о сетке',
            'Grid Points:': 'Точки сетки:',
            'Cell Size:': 'Размер ячейки:',
            'Grid Extent:': 'Размер сетки:',
            'Sample Results': 'Образцы результатов',
            'X': 'X',
            'Y': 'Y',
            'Value': 'Значение',
            'View All Results': 'Все результаты',
            'Export Results': 'Экспорт результатов',
            'Quality Report': 'Отчет о качестве',
            'All Interpolation Results': 'Все результаты интерполяции',
            'Interpolated Value': 'Интерполированное значение',
            'Close': 'Закрыть',
            'Info': 'Информация',
            'Export results functionality - To be implemented': 'Функция экспорта результатов - будет реализована',
            
            # Analysis and Recommendations
            'Analysis and Recommendations': 'Анализ и рекомендации',
            'Analyze Data': 'Анализировать данные',
            'Apply Recommendations': 'Применить рекомендации',
            'Summary': 'Сводка',
            'Method Comparison': 'Сравнение методов',
            'Optimal Parameters': 'Оптимальные параметры',
            

            'Method comparison will appear here after analysis.': 'Сравнение методов появится здесь после анализа.',
            'Optimal parameters will appear here after analysis.': 'Оптимальные параметры появятся здесь после анализа.',
            'Recommended parameters have been applied.\nYou can now run the interpolation.': 'Рекомендуемые параметры применены.\nТеперь вы можете запустить интерполяцию.',
            
            # Status messages
            'Ready': 'Готов',
            'Interpolation completed successfully': 'Интерполяция завершена успешно',
            'Error occurred': 'Произошла ошибка',
            'Loaded data': 'Данные загружены',
            'rows': 'строк',
            'Running interpolation...': 'Выполняется интерполяция...',
            'No data to save': 'Нет данных для сохранения',
            'Save Project': 'Сохранить проект',
            'Load Project': 'Загрузить проект',
            'Project files': 'Файлы проектов',
            'All files': 'Все файлы',
            'No results to export': 'Нет результатов для экспорта',
            'Please load data first': 'Сначала загрузите данные',
            'Layout reset to default': 'Макет сброшен к значениям по умолчанию',
            'Do you want to quit?': 'Вы хотите выйти?',
            'Quit': 'Выход',
            
            # Dialog titles and labels
            'Data Statistics': 'Статистика данных',
            'Basic Statistics': 'Основная статистика',
            'Spatial Statistics': 'Пространственная статистика',
            'Grid Information': 'Информация о сетке',
            'Grid Details': 'Детали сетки',
            'Memory Estimate': 'Оценка памяти',
            'Preferences dialog - To be implemented': 'Диалог настроек - будет реализован',
            
            # Menu items
            'Project': 'Проект',
            'Reset': 'Сброс',
            'Layout': 'Макет',
            'Show': 'Показать',
            'Data': 'Данные',
            'Panel': 'Панель',
            'Language Settings': 'Настройки языка',
            'Preferences': 'Настройки',
            'User Guide': 'Руководство пользователя',
            'About': 'О программе',
            'Grid Info': 'Информация о сетке',
            
            # Recommendation engine messages
            'No suitable interpolation method found': 'Подходящий метод интерполяции не найден',
            'Data analysis warning:': 'Предупреждение анализа данных:',
            'Quality evaluation failed:': 'Оценка качества не удалась:',
            'No analysis performed yet': 'Анализ еще не выполнен',
            'No recommendations available. Run analyze_and_recommend first.': 'Рекомендации недоступны. Сначала запустите analyze_and_recommend.',
            'Interpolator creation for': 'Создание интерполятора для',
            'not implemented': 'не реализовано',
            
            # Data insights and analysis
            'Very few data points - results may be unreliable': 'Очень мало точек данных - результаты могут быть ненадежными',
            'Limited data points - consider collecting more data': 'Ограниченное количество точек данных - рассмотрите сбор дополнительных данных',
            'Highly clustered data - consider using sectoral search': 'Сильно кластеризованные данные - рассмотрите использование секторного поиска',
            'High proportion of outliers': 'Высокая доля выбросов',
            'detected': 'обнаружено',
            'Very high variability in values - results may be unstable': 'Очень высокая вариабельность значений - результаты могут быть нестабильными',
            
            # Data pattern descriptions
            'Data points are uniformly distributed': 'Точки данных равномерно распределены',
            'Data shows moderate clustering with some regular areas': 'Данные показывают умеренную кластеризацию с некоторыми регулярными областями',
            'Data is highly clustered in specific regions': 'Данные сильно кластеризованы в определенных регионах',
            'Data shows irregular spatial distribution': 'Данные показывают неравномерное пространственное распределение',
            'Values are approximately normally distributed': 'Значения приблизительно нормально распределены',
            'Values are strongly right-skewed': 'Значения сильно смещены вправо',
            'Values are strongly left-skewed': 'Значения сильно смещены влево',
            'with low variability': 'с низкой вариабельностью',
            'with moderate variability': 'с умеренной вариабельностью',
            'with high variability': 'с высокой вариабельностью',
            
            # Quality assessments
            'Excellent - suitable for most interpolation methods': 'Отлично - подходит для большинства методов интерполяции',
            'Good - suitable for interpolation with appropriate parameters': 'Хорошо - подходит для интерполяции с соответствующими параметрами',
            'Fair - interpolation possible but results may be less reliable': 'Удовлетворительно - интерполяция возможна, но результаты могут быть менее надежными',
            'Poor - consider data collection improvements': 'Плохо - рассмотрите улучшение сбора данных',
            
            # Special considerations and best practices
            'Consider detrending or using universal kriging': 'Рассмотрите устранение тренда или использование универсального кригинга',
            'Strong anisotropy detected - use directional parameters': 'Обнаружена сильная анизотропия - используйте направленные параметры',
            '3D interpolation - ensure sufficient vertical sampling': '3D интерполяция - обеспечьте достаточное вертикальное выборочное исследование',
            'Extreme aspect ratio - consider coordinate transformation': 'Экстремальное соотношение сторон - рассмотрите преобразование координат',
            'Always validate results with cross-validation': 'Всегда проверяйте результаты с помощью перекрестной валидации',
            'Visualize interpolation results to check for artifacts': 'Визуализируйте результаты интерполяции для проверки артефактов',
            'With limited data, use conservative parameters': 'При ограниченных данных используйте консервативные параметры',
            'Consider removing or down-weighting outliers': 'Рассмотрите удаление или снижение веса выбросов',
            'Account for trends in your interpretation': 'Учитывайте тренды в вашей интерпретации',
            
            # Parameter guidelines
            'Increase for sparse data, decrease for dense data': 'Увеличьте для разреженных данных, уменьшите для плотных данных',
            'Increase for more local influence (1-4 range typical)': 'Увеличьте для большего локального влияния (обычно диапазон 1-4)',
            'Balance between accuracy (more points) and speed': 'Баланс между точностью (больше точек) и скоростью',
            'Enable for clustered or directional data': 'Включите для кластеризованных или направленных данных',
            'Adjust ratio and angle based on directional patterns': 'Настройте соотношение и угол на основе направленных паттернов',
            
            # Potential issues
            'Too few points may lead to unreliable interpolation': 'Слишком мало точек может привести к ненадежной интерполяции',
            'Many points on edges - extrapolation may be unreliable': 'Много точек на краях - экстраполяция может быть ненадежной',
            'Severe clustering may cause interpolation artifacts': 'Серьезная кластеризация может вызвать артефакты интерполяции',
            
            # Dialog and interface strings
            'Load Data File': 'Загрузить файл данных',
            'File Selection': 'Выбор файла',
            'No file selected': 'Файл не выбран',
            'Recent Files': 'Недавние файлы',
            'File Information': 'Информация о файле',
            'Column Mapping': 'Сопоставление столбцов',
            'Value Columns': 'Столбцы значений',
            'CSV Settings': 'Настройки CSV',
            'Data Validation': 'Проверка данных',
            'Skip rows with invalid coordinates': 'Пропустить строки с неверными координатами',
            'Fill missing values with interpolation': 'Заполнить пропущенные значения интерполяцией',
            'Remove duplicate coordinates': 'Удалить дублирующиеся координаты',
            'Import Summary': 'Сводка импорта',
            'Select Data File': 'Выбрать файл данных',
            'Excel files': 'Файлы Excel',
            'CSV files': 'Файлы CSV',
            'All supported': 'Все поддерживаемые',
            'Encoding Error': 'Ошибка кодировки',
            'The CSV file appears to be empty': 'CSV файл пуст',
            'Parsing Error': 'Ошибка разбора',
            'Please complete all required settings': 'Пожалуйста, заполните все обязательные настройки',
            'Skip invalid rows': 'Пропустить неверные строки',
            'Fill missing values': 'Заполнить пропущенные значения',
            'Remove duplicates': 'Удалить дубликаты',
            
            # Visualization panel
            'Display Options': 'Параметры отображения',
            'Results Only': 'Только результаты',
            'Export Image': 'Экспорт изображения',
            'Load data to see scatter plot': 'Загрузите данные для просмотра диаграммы рассеяния',
            'Run interpolation to see results': 'Запустите интерполяцию для просмотра результатов',
            'Use toolbar for zoom and pan': 'Используйте панель инструментов для масштабирования и панорамирования',
            'No visualization to export': 'Нет визуализации для экспорта',
            'Export Visualization': 'Экспорт визуализации',
            
            # Export dialog
            'Export Interpolation Results': 'Экспорт результатов интерполяции',
            'Export Data Information': 'Информация об экспортируемых данных',
            'Export Format': 'Формат экспорта',
            'Output File': 'Выходной файл',
            'Common Options': 'Общие параметры',
            'Include metadata and creation information': 'Включить метаданные и информацию о создании',
            'Overwrite existing files': 'Перезаписать существующие файлы',
            'Create output directories if needed': 'Создать выходные каталоги при необходимости',
            'CSV Options': 'Параметры CSV',
            'Include column headers': 'Включить заголовки столбцов',
            'Include row index': 'Включить индекс строк',
            'GeoTIFF Options': 'Параметры GeoTIFF',
            'VTK Options': 'Параметры VTK',
            'Export Summary': 'Сводка экспорта',
            'Data Preview': 'Предварительный просмотр данных',
            'Export Progress': 'Прогресс экспорта',
            'Ready to export': 'Готов к экспорту',
            'No results data available': 'Данные результатов недоступны',
            'Results data structure not recognized': 'Структура данных результатов не распознана',
            'Export Results': 'Экспорт результатов',
            'GeoTIFF files': 'Файлы GeoTIFF',
            'TIFF files': 'Файлы TIFF',
            'VTK files': 'Файлы VTK',
            'VTK XML files': 'Файлы VTK XML',
            
            # Error messages from controllers
            'No data loaded': 'Данные не загружены',
            'Data must contain at least one value column for interpolation': 'Данные должны содержать хотя бы один столбец значений для интерполяции',
            'No data loaded for interpolation': 'Данные для интерполяции не загружены',
            'No grid data found in results': 'Данные сетки не найдены в результатах',
            'No value columns found in grid data': 'Столбцы значений не найдены в данных сетки',
            'No data to save': 'Нет данных для сохранения',
            'No data file is currently loaded': 'В настоящее время файл данных не загружен',
            'CSV file appears to be empty': 'CSV файл пуст',
            
            # Visualization panel help text
            'Visualization Panel': 'Панель визуализации',
            'Interpolate': 'Интерполировать',
            'Load Project': 'Загрузить проект',
            'Could not retrieve data information': 'Не удалось получить информацию о данных'
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