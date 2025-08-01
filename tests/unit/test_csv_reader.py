"""
Unit tests for CSV reader functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.io.readers.csv_reader import CSVReader
from src.io.readers.base import DataReadError, ValidationError


class TestCSVReader:
    """Test cases for CSVReader class."""
    
    def test_initialization(self):
        """Test CSVReader initialization."""
        reader = CSVReader()
        assert reader.data is None
        assert reader.delimiter is None
        assert reader.encoding is None
        assert reader.header_row is None
        assert isinstance(reader.metadata, dict)
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        reader = CSVReader()
        extensions = reader.get_supported_extensions()
        
        assert '.csv' in extensions
        assert '.txt' in extensions  
        assert '.tsv' in extensions
        assert len(extensions) == 3
    
    def test_validate_file_existing(self, temp_csv_file):
        """Test file validation with existing file."""
        reader = CSVReader()
        assert reader.validate_file(temp_csv_file) is True
    
    def test_validate_file_nonexistent(self):
        """Test file validation with non-existent file."""
        reader = CSVReader()
        assert reader.validate_file('nonexistent.csv') is False
    
    def test_validate_file_wrong_extension(self, temp_directory):
        """Test file validation with wrong extension."""
        reader = CSVReader()
        wrong_file = temp_directory / 'test.xyz'
        wrong_file.write_text('test')
        
        assert reader.validate_file(str(wrong_file)) is False
    
    def test_encoding_detection(self, temp_csv_file):
        """Test automatic encoding detection."""
        reader = CSVReader()
        encoding = reader.detect_encoding(temp_csv_file)
        
        # Should return a valid encoding
        assert isinstance(encoding, str)
        assert len(encoding) > 0
    
    def test_delimiter_detection_comma(self, temp_csv_file):
        """Test delimiter detection with comma-separated file."""
        reader = CSVReader()
        delimiter = reader.detect_delimiter(temp_csv_file)
        assert delimiter == ','
    
    def test_delimiter_detection_semicolon(self, different_delimiter_csv):
        """Test delimiter detection with semicolon-separated file."""
        reader = CSVReader()
        delimiter = reader.detect_delimiter(different_delimiter_csv)
        assert delimiter == ';'
    
    def test_header_detection(self, temp_csv_file):
        """Test header row detection."""
        reader = CSVReader()
        encoding = reader.detect_encoding(temp_csv_file)
        delimiter = reader.detect_delimiter(temp_csv_file, encoding)
        header_row = reader.detect_header_row(temp_csv_file, delimiter, encoding)
        
        assert header_row == 0  # Header should be first row
    
    def test_header_detection_with_metadata(self, csv_with_header_issues):
        """Test header detection when file has metadata before headers."""
        reader = CSVReader()
        encoding = reader.detect_encoding(csv_with_header_issues)
        delimiter = reader.detect_delimiter(csv_with_header_issues, encoding)
        header_row = reader.detect_header_row(csv_with_header_issues, delimiter, encoding)
        
        # Should find the actual header row (row with column names)
        assert header_row >= 0
    
    def test_read_basic_csv(self, temp_csv_file):
        """Test basic CSV reading functionality."""
        reader = CSVReader()
        data = reader.read(temp_csv_file)
        
        # Check data was loaded
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert reader.data is not None
        assert reader.is_fitted is False  # Reader doesn't get fitted
        
        # Check metadata was stored
        assert reader.delimiter is not None
        assert reader.encoding is not None
        assert 'delimiter' in reader.metadata
        assert 'encoding' in reader.metadata
    
    def test_read_with_explicit_parameters(self, temp_csv_file):
        """Test reading with explicitly provided parameters."""
        reader = CSVReader()
        data = reader.read(temp_csv_file, delimiter=',', encoding='utf-8', header=0)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert reader.delimiter == ','
        assert reader.encoding == 'utf-8'
        assert reader.header_row == 0
    
    def test_read_nonexistent_file(self):
        """Test reading non-existent file raises error."""
        reader = CSVReader()
        
        with pytest.raises(DataReadError):
            reader.read('nonexistent.csv')
    
    def test_read_empty_file(self, empty_csv_file):
        """Test reading empty file raises error."""
        reader = CSVReader()
        
        with pytest.raises(DataReadError):
            reader.read(empty_csv_file)
    
    def test_read_with_nrows_limit(self, temp_csv_file):
        """Test reading with row limit."""
        reader = CSVReader()
        data = reader.read(temp_csv_file, nrows=5)
        
        assert len(data) <= 5
    
    def test_preview_functionality(self, temp_csv_file):
        """Test preview functionality."""
        reader = CSVReader()
        preview = reader.preview(temp_csv_file, n_rows=3)
        
        assert isinstance(preview, pd.DataFrame)
        assert len(preview) <= 3
    
    def test_get_file_info(self, temp_csv_file):
        """Test file information extraction."""
        reader = CSVReader()
        info = reader.get_file_info(temp_csv_file)
        
        assert 'file_path' in info
        assert 'file_size_bytes' in info
        assert 'encoding' in info
        assert 'delimiter' in info
        assert 'columns' in info
        assert 'total_rows' in info
        assert isinstance(info['columns'], list)
        assert info['total_rows'] > 0
    
    def test_column_detection(self, temp_csv_file):
        """Test automatic column detection."""
        reader = CSVReader()
        data = reader.read(temp_csv_file)
        
        # Test coordinate column detection
        coord_cols = reader.detect_coordinate_columns()
        assert isinstance(coord_cols, dict)
        assert 'X' in coord_cols
        assert 'Y' in coord_cols
        assert 'Z' in coord_cols
        
        # Should detect X and Y columns
        assert coord_cols['X'] is not None
        assert coord_cols['Y'] is not None
    
    def test_value_column_detection(self, temp_csv_file):
        """Test value column detection."""
        reader = CSVReader()
        data = reader.read(temp_csv_file)
        
        value_cols = reader.detect_value_columns()
        assert isinstance(value_cols, list)
        assert len(value_cols) > 0
        
        # Should include numeric columns that aren't coordinates
        expected_cols = ['ASH', 'SULFUR', 'CALORIFIC']
        for col in expected_cols:
            if col in data.columns:
                assert col in value_cols
    
    def test_get_column_info(self, temp_csv_file):
        """Test column information extraction."""
        reader = CSVReader()
        data = reader.read(temp_csv_file)
        
        col_info = reader.get_column_info()
        assert isinstance(col_info, dict)
        
        for col in data.columns:
            assert col in col_info
            assert 'dtype' in col_info[col]
            assert 'non_null_count' in col_info[col]
            assert 'null_count' in col_info[col]
    
    def test_get_data_summary(self, temp_csv_file):
        """Test comprehensive data summary."""
        reader = CSVReader()
        data = reader.read(temp_csv_file)
        
        summary = reader.get_data_summary()
        assert isinstance(summary, dict)
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'coordinate_columns' in summary
        assert 'value_columns' in summary
        assert 'null_counts' in summary
    
    def test_validate_coordinates_valid(self, temp_csv_file):
        """Test coordinate validation with valid data."""
        reader = CSVReader()
        data = reader.read(temp_csv_file)
        
        validation = reader.validate_coordinates('X', 'Y', 'Z')
        assert isinstance(validation, dict)
        assert 'valid' in validation
        assert 'errors' in validation
        assert 'warnings' in validation
        assert 'statistics' in validation
        
        # Should be valid with sample data
        assert validation['valid'] is True
    
    def test_validate_coordinates_missing_column(self, temp_csv_file):
        """Test coordinate validation with missing column."""
        reader = CSVReader()
        data = reader.read(temp_csv_file)
        
        validation = reader.validate_coordinates('X', 'NONEXISTENT', 'Z')
        assert validation['valid'] is False
        assert len(validation['errors']) > 0
    
    def test_validate_coordinates_no_data_loaded(self):
        """Test coordinate validation without loaded data."""
        reader = CSVReader()
        
        with pytest.raises(ValidationError):
            reader.validate_coordinates('X', 'Y', 'Z')
    
    def test_column_name_cleaning(self):
        """Test that column names are cleaned (whitespace stripped)."""
        csv_content = " X , Y , VALUE \n100,200,10\n200,300,20"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            reader = CSVReader()
            data = reader.read(temp_file)
            
            # Column names should be stripped of whitespace
            assert 'X' in data.columns
            assert 'Y' in data.columns  
            assert 'VALUE' in data.columns
            assert ' X ' not in data.columns
            
        finally:
            os.unlink(temp_file)
    
    def test_na_values_handling(self):
        """Test that various NA values are properly handled."""
        csv_content = "X,Y,VALUE\n100,200,10\n200,300,NA\n300,400,null\n400,500,-999"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            reader = CSVReader()
            data = reader.read(temp_file)
            
            # NA and null should be converted to NaN
            assert pd.isna(data.loc[1, 'VALUE'])
            assert pd.isna(data.loc[2, 'VALUE'])
            # -999 should be kept as is (not in default NA values)
            assert data.loc[3, 'VALUE'] == -999
            
        finally:
            os.unlink(temp_file)


class TestCSVReaderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_large_values(self):
        """Test handling of very large numeric values."""
        csv_content = "X,Y,VALUE\n1e10,2e10,1e20\n2e10,3e10,2e20"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            reader = CSVReader()
            data = reader.read(temp_file)
            
            assert len(data) == 2
            assert np.isfinite(data['VALUE']).all()
            
        finally:
            os.unlink(temp_file)
    
    def test_mixed_data_types_in_column(self):
        """Test handling of mixed data types in numeric columns."""
        csv_content = "X,Y,VALUE\n100,200,10.5\n200,300,INVALID\n300,400,30.2"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)  
            temp_file = f.name
        
        try:
            reader = CSVReader()
            data = reader.read(temp_file)
            
            # Invalid value should become NaN
            assert pd.isna(data.loc[1, 'VALUE'])
            assert data.loc[0, 'VALUE'] == 10.5
            assert data.loc[2, 'VALUE'] == 30.2
            
        finally:
            os.unlink(temp_file)
    
    def test_unicode_column_names(self):
        """Test handling of unicode characters in column names.""" 
        csv_content = "X_координата,Y_координата,Зольность\n100,200,15.5\n200,300,18.2"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            reader = CSVReader()
            data = reader.read(temp_file)
            
            assert len(data.columns) == 3
            assert len(data) == 2
            
        finally:
            os.unlink(temp_file)