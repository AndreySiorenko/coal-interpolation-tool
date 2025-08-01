"""
Unit tests for ExcelReader.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os

# Mock imports for testing without dependencies
try:
    from src.io.readers.excel_reader import ExcelReader
    from src.io.readers.base import DataReadError, ValidationError
except ImportError:
    pytest.skip("Excel reader modules not available", allow_module_level=True)


class TestExcelReader:
    """Test cases for ExcelReader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reader = ExcelReader()
        
        # Create sample Excel data for testing
        np.random.seed(42)
        n_points = 30
        
        self.sample_data = pd.DataFrame({
            'X': np.random.uniform(100, 900, n_points),
            'Y': np.random.uniform(200, 800, n_points),
            'Coal_Thickness': np.random.uniform(1.0, 5.0, n_points),
            'Ash_Content': np.random.uniform(8.0, 25.0, n_points),
            'Moisture': np.random.uniform(2.0, 8.0, n_points)
        })
        
        # Create sample data with Excel-specific issues
        self.problematic_data = pd.DataFrame({
            'Easting': [100.0, 200.0, 300.0, '#N/A', 500.0],
            'Northing': [150.0, 250.0, '#REF!', 450.0, 550.0],
            'Value': [10.5, 12.3, 15.8, 9.2, 11.7],
            'Empty_Col': [None, None, None, None, None],
            'Text_Data': ['Sample A', 'Sample B', 'Sample C', 'Sample D', 'Sample E']
        })
    
    def test_initialization(self):
        """Test ExcelReader initialization."""
        reader = ExcelReader()
        assert reader.data is None
        assert reader.metadata == {}
        assert reader.file_path is None
        assert reader.sheet_names is None
        assert reader.current_sheet is None
        assert reader.engine is None
    
    def test_get_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.reader.get_supported_extensions()
        assert '.xlsx' in extensions
        assert '.xls' in extensions
        assert len(extensions) == 2
    
    def test_detect_engine(self):
        """Test engine detection for different Excel formats."""
        # Test .xlsx files
        engine = self.reader._detect_engine('test_file.xlsx')
        assert engine == 'openpyxl'
        
        # Test .xls files
        engine = self.reader._detect_engine('test_file.xls')
        assert engine == 'xlrd'
        
        # Test unsupported format
        with pytest.raises(DataReadError, match="Unsupported Excel format"):
            self.reader._detect_engine('test_file.txt')
    
    def test_validate_file(self):
        """Test file validation."""
        # Test valid extensions
        with patch('pathlib.Path.exists', return_value=True):
            assert self.reader.validate_file('test.xlsx') is True
            assert self.reader.validate_file('test.xls') is True
            assert self.reader.validate_file('test.txt') is False
        
        # Test non-existent file
        with patch('pathlib.Path.exists', return_value=False):
            assert self.reader.validate_file('nonexistent.xlsx') is False
    
    @patch('pandas.ExcelFile')
    def test_get_sheet_names_success(self, mock_excel_file):
        """Test successful sheet names retrieval."""
        # Setup mock
        mock_file = MagicMock()
        mock_file.sheet_names = ['Sheet1', 'Data', 'Summary']
        mock_excel_file.return_value.__enter__.return_value = mock_file
        
        with patch('pathlib.Path.exists', return_value=True):
            sheet_names = self.reader.get_sheet_names('test.xlsx')
            
        assert sheet_names == ['Sheet1', 'Data', 'Summary']
        assert self.reader.sheet_names == ['Sheet1', 'Data', 'Summary']
        mock_excel_file.assert_called_once_with('test.xlsx', engine='openpyxl')
    
    def test_get_sheet_names_file_not_found(self):
        """Test sheet names retrieval with non-existent file."""
        with pytest.raises(DataReadError, match="File not found"):
            self.reader.get_sheet_names('nonexistent.xlsx')
    
    def test_get_sheet_names_unsupported_format(self):
        """Test sheet names retrieval with unsupported format."""
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(DataReadError, match="Unsupported file type"):
                self.reader.get_sheet_names('test.txt')
    
    @patch('pandas.ExcelFile')
    def test_get_sheet_names_missing_dependency(self, mock_excel_file):
        """Test handling of missing dependencies."""
        mock_excel_file.side_effect = ImportError("No module named 'openpyxl'")
        
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(DataReadError, match="openpyxl library is required"):
                self.reader.get_sheet_names('test.xlsx')
    
    @patch('pandas.read_excel')
    def test_detect_data_range(self, mock_read_excel):
        """Test data range detection."""
        # Create mock data with empty rows/columns
        mock_data = pd.DataFrame({
            0: [None, None, 'X', 100, 200, 300, None],
            1: [None, None, 'Y', 150, 250, 350, None],
            2: [None, None, 'Value', 10, 20, 30, None],
            3: [None, None, None, None, None, None, None]  # Empty column
        })
        mock_read_excel.return_value = mock_data
        
        with patch.object(self.reader, 'get_sheet_names', return_value=['Sheet1']):
            range_info = self.reader.detect_data_range('test.xlsx')
            
        expected = {
            'sheet_name': 'Sheet1',
            'data_start_row': 2,  # First non-empty row
            'data_end_row': 5,    # Last non-empty row
            'data_start_col': 0,  # First non-empty column
            'data_end_col': 2,    # Last non-empty column
            'header_row': 2,      # Detected header row
            'total_rows': 4,      # Total data rows
            'total_cols': 3,      # Total data columns
            'estimated_data_rows': 3  # Data rows excluding header
        }
        
        for key, value in expected.items():
            assert range_info[key] == value
    
    def test_detect_header_row(self):
        """Test header row detection."""
        # Create test data with clear header
        test_data = pd.DataFrame({
            0: [None, 'Project Info', 'X', 100, 200],
            1: [None, 'Survey 2024', 'Y', 150, 250],
            2: [None, 'Coal Data', 'Thickness', 2.5, 3.1]
        })
        
        # Header should be detected at row 2 (0-based)
        header_row = self.reader._detect_header_row(test_data, 1, 4)
        assert header_row == 2
        
        # Test with no clear header (all numeric)
        numeric_data = pd.DataFrame({
            0: [100, 200, 300],
            1: [150, 250, 350],
            2: [2.5, 3.1, 4.2]
        })
        
        header_row = self.reader._detect_header_row(numeric_data, 0, 2)
        assert header_row == 0  # Should default to first row
    
    @patch('pandas.read_excel')
    def test_read_success(self, mock_read_excel):
        """Test successful Excel file reading."""
        mock_read_excel.return_value = self.sample_data
        
        with patch.object(self.reader, 'get_sheet_names', return_value=['Sheet1']):
            with patch('pathlib.Path.exists', return_value=True):
                data = self.reader.read('test.xlsx')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 30
        assert list(data.columns) == ['X', 'Y', 'Coal_Thickness', 'Ash_Content', 'Moisture']
        assert self.reader.current_sheet == 'Sheet1'
        assert self.reader.engine == 'openpyxl'
        
        # Check metadata
        assert 'engine' in self.reader.metadata
        assert 'sheet_name' in self.reader.metadata
        assert 'original_shape' in self.reader.metadata
    
    @patch('pandas.read_excel')
    def test_read_with_sheet_name(self, mock_read_excel):
        """Test reading specific sheet."""
        mock_read_excel.return_value = self.sample_data
        
        with patch.object(self.reader, 'get_sheet_names', return_value=['Sheet1', 'Data']):
            with patch('pathlib.Path.exists', return_value=True):
                data = self.reader.read('test.xlsx', sheet_name='Data')
        
        assert self.reader.current_sheet == 'Data'
        mock_read_excel.assert_called_once()
        call_args = mock_read_excel.call_args[1]
        assert call_args['sheet_name'] == 'Data'
    
    @patch('pandas.read_excel')
    def test_read_with_custom_parameters(self, mock_read_excel):
        """Test reading with custom parameters."""
        mock_read_excel.return_value = self.sample_data
        
        with patch.object(self.reader, 'get_sheet_names', return_value=['Sheet1']):
            with patch('pathlib.Path.exists', return_value=True):
                data = self.reader.read(
                    'test.xlsx',
                    header=1,
                    nrows=10,
                    usecols=['X', 'Y', 'Coal_Thickness']
                )
        
        call_args = mock_read_excel.call_args[1]
        assert call_args['header'] == 1
        assert call_args['nrows'] == 10
        assert call_args['usecols'] == ['X', 'Y', 'Coal_Thickness']
    
    def test_read_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(DataReadError, match="File not found"):
            self.reader.read('nonexistent.xlsx')
    
    def test_read_unsupported_format(self):
        """Test reading unsupported file format."""
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(DataReadError, match="Unsupported file type"):
                self.reader.read('test.txt')
    
    @patch('pandas.read_excel')
    def test_read_empty_file(self, mock_read_excel):
        """Test reading empty Excel file."""
        mock_read_excel.side_effect = Exception("No sheets found")
        
        with patch.object(self.reader, 'get_sheet_names', return_value=[]):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(DataReadError, match="No sheets found"):
                    self.reader.read('empty.xlsx')
    
    @patch('pandas.read_excel')
    def test_read_with_missing_dependency(self, mock_read_excel):
        """Test reading with missing Excel library."""
        mock_read_excel.side_effect = ImportError("No module named 'openpyxl'")
        
        with patch.object(self.reader, 'get_sheet_names', return_value=['Sheet1']):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(DataReadError, match="openpyxl library is required"):
                    self.reader.read('test.xlsx')
    
    @patch('pandas.read_excel')
    def test_preview(self, mock_read_excel):
        """Test data preview functionality."""
        mock_read_excel.return_value = self.sample_data.head(5)
        
        with patch.object(self.reader, 'get_sheet_names', return_value=['Sheet1']):
            with patch('pathlib.Path.exists', return_value=True):
                preview = self.reader.preview('test.xlsx', n_rows=5)
        
        assert len(preview) <= 5
        call_args = mock_read_excel.call_args[1]
        assert call_args['nrows'] == 5
    
    @patch('pandas.read_excel')
    def test_get_file_info(self, mock_read_excel):
        """Test file information retrieval."""
        mock_read_excel.return_value = self.sample_data
        
        with patch.object(self.reader, 'get_sheet_names', return_value=['Sheet1', 'Data']):
            with patch.object(self.reader, 'detect_data_range') as mock_detect:
                mock_detect.return_value = {
                    'sheet_name': 'Sheet1',
                    'total_rows': 30,
                    'total_cols': 5,
                    'estimated_data_rows': 29
                }
                
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 12345
                    
                    file_info = self.reader.get_file_info('test.xlsx')
        
        assert 'file_path' in file_info
        assert 'file_size_bytes' in file_info
        assert file_info['file_size_bytes'] == 12345
        assert 'engine' in file_info
        assert file_info['engine'] == 'openpyxl'
        assert file_info['sheet_names'] == ['Sheet1', 'Data']
        assert file_info['sheet_count'] == 2
        assert 'sheets_info' in file_info
    
    @patch('pandas.read_excel')
    def test_get_sheet_info(self, mock_read_excel):
        """Test specific sheet information retrieval."""
        mock_read_excel.return_value = self.sample_data.head(5)
        
        with patch.object(self.reader, 'detect_data_range') as mock_detect:
            mock_detect.return_value = {
                'sheet_name': 'Sheet1',
                'total_rows': 30,
                'total_cols': 5,
                'estimated_data_rows': 29
            }
            
            sheet_info = self.reader.get_sheet_info('test.xlsx', 'Sheet1')
        
        assert sheet_info['sheet_name'] == 'Sheet1'
        assert 'columns' in sheet_info
        assert 'column_count' in sheet_info
        assert 'preview_data' in sheet_info
    
    def test_validate_coordinates_success(self):
        """Test successful coordinate validation."""
        self.reader.data = self.sample_data
        
        result = self.reader.validate_coordinates('X', 'Y', None)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert 'X' in result['statistics']
        assert 'Y' in result['statistics']
        
        # Check statistics structure
        x_stats = result['statistics']['X']
        assert 'count' in x_stats
        assert 'min' in x_stats
        assert 'max' in x_stats
        assert 'mean' in x_stats
    
    def test_validate_coordinates_missing_columns(self):
        """Test coordinate validation with missing columns."""
        self.reader.data = self.sample_data
        
        result = self.reader.validate_coordinates('Missing', 'Y', None)
        
        assert result['valid'] is False
        assert any('Missing columns' in error for error in result['errors'])
    
    def test_validate_coordinates_no_data(self):
        """Test coordinate validation with no data loaded."""
        with pytest.raises(ValidationError, match="No data loaded"):
            self.reader.validate_coordinates('X', 'Y', None)
    
    def test_validate_coordinates_with_excel_errors(self):
        """Test coordinate validation with Excel error values."""
        self.reader.data = self.problematic_data
        
        result = self.reader.validate_coordinates('Easting', 'Northing', None)
        
        # Should try to convert to numeric and handle Excel errors
        assert result['valid'] is False or len(result['warnings']) > 0
        
        # Check that Excel error detection works
        if result['valid']:
            assert any('error values' in warning for warning in result['warnings'])
    
    def test_validate_coordinates_type_conversion(self):
        """Test coordinate validation with type conversion."""
        # Create data with string numbers
        string_data = pd.DataFrame({
            'X_Str': ['100.0', '200.0', '300.0'],
            'Y_Str': ['150.0', '250.0', '350.0'],
            'Value': [10, 20, 30]
        })
        
        self.reader.data = string_data
        result = self.reader.validate_coordinates('X_Str', 'Y_Str', None)
        
        # Should successfully convert strings to numbers
        assert result['valid'] is True
        
        # Check that data was actually converted
        assert pd.api.types.is_numeric_dtype(self.reader.data['X_Str'])
        assert pd.api.types.is_numeric_dtype(self.reader.data['Y_Str'])
    
    def test_coordinate_validation_with_duplicates(self):
        """Test coordinate validation with duplicate points."""
        # Add duplicate coordinates
        duplicate_data = self.sample_data.copy()
        duplicate_data.loc[len(duplicate_data)] = duplicate_data.iloc[0]
        duplicate_data.loc[len(duplicate_data)] = duplicate_data.iloc[1]
        
        self.reader.data = duplicate_data
        result = self.reader.validate_coordinates('X', 'Y', None)
        
        assert result['valid'] is True
        assert any('duplicate' in warning.lower() for warning in result['warnings'])
    
    def test_get_column_info(self):
        """Test column information retrieval."""
        self.reader.data = self.sample_data
        
        info = self.reader.get_column_info()
        
        assert len(info) == 5
        for col in self.sample_data.columns:
            assert col in info
            assert 'dtype' in info[col]
            assert 'non_null_count' in info[col]
            assert 'null_count' in info[col]
            assert 'unique_count' in info[col]
            
            # Numeric columns should have statistics
            if pd.api.types.is_numeric_dtype(self.sample_data[col]):
                assert 'min' in info[col]
                assert 'max' in info[col]
                assert 'mean' in info[col]
                assert 'std' in info[col]
    
    def test_detect_coordinate_columns(self):
        """Test automatic coordinate column detection."""
        self.reader.data = self.sample_data
        
        detected = self.reader.detect_coordinate_columns()
        
        assert detected['X'] == 'X'
        assert detected['Y'] == 'Y'
        assert detected['Z'] is None
    
    def test_detect_coordinate_columns_alternative_names(self):
        """Test coordinate detection with alternative column names."""
        alt_data = pd.DataFrame({
            'Easting': [100, 200, 300],
            'Northing': [150, 250, 350],
            'Elevation': [500, 600, 700],
            'Coal_Thickness': [2.5, 3.0, 3.5]
        })
        
        self.reader.data = alt_data
        detected = self.reader.detect_coordinate_columns()
        
        assert detected['X'] == 'Easting'
        assert detected['Y'] == 'Northing'
        assert detected['Z'] == 'Elevation'
    
    def test_detect_value_columns(self):
        """Test value column detection."""
        self.reader.data = self.sample_data
        
        value_cols = self.reader.detect_value_columns()
        
        expected_cols = ['Coal_Thickness', 'Ash_Content', 'Moisture']
        assert set(value_cols) == set(expected_cols)
        assert 'X' not in value_cols
        assert 'Y' not in value_cols
    
    def test_get_data_summary(self):
        """Test comprehensive data summary."""
        self.reader.data = self.sample_data
        self.reader.file_path = Path('test.xlsx')
        
        summary = self.reader.get_data_summary()
        
        assert 'file_path' in summary
        assert 'shape' in summary
        assert summary['shape'] == (30, 5)
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'memory_usage' in summary
        assert 'coordinate_columns' in summary
        assert 'value_columns' in summary
        assert 'null_counts' in summary
        assert 'duplicate_rows' in summary


if __name__ == "__main__":
    pytest.main([__file__])