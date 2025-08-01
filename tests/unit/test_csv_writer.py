"""
Unit tests for CSV writer.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.io.writers.csv_writer import CSVWriter, CSVExportOptions
from src.io.writers.base import GridData, PointData, ExportError


class TestCSVWriter:
    """Test suite for CSV writer."""
    
    def test_init_default_options(self):
        """Test CSV writer initialization with default options."""
        writer = CSVWriter()
        assert isinstance(writer.options, CSVExportOptions)
        assert writer.options.delimiter == ','
        assert writer.options.include_coordinates == True
        assert writer.options.precision == 6
    
    def test_init_custom_options(self):
        """Test CSV writer initialization with custom options."""
        options = CSVExportOptions(
            delimiter=';',
            precision=3,
            include_coordinates=False
        )
        writer = CSVWriter(options)
        assert writer.options.delimiter == ';'
        assert writer.options.precision == 3
        assert writer.options.include_coordinates == False
    
    def test_supported_formats(self):
        """Test supported formats."""
        writer = CSVWriter()
        from src.io.writers.base import ExportFormat
        assert ExportFormat.CSV in writer.supported_formats
    
    def test_file_extensions(self):
        """Test file extensions."""
        writer = CSVWriter()
        assert '.csv' in writer.file_extensions
        assert '.txt' in writer.file_extensions
    
    def test_write_grid_data(self, temp_directory):
        """Test writing grid data to CSV."""
        # Create test grid data
        x_coords = np.array([100, 200, 300])
        y_coords = np.array([100, 200, 300])
        values = np.array([[10, 20, 30], [15, 25, 35], [12, 22, 32]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(100, 300, 100, 300),
            cell_size=100,
            metadata={'test': 'metadata'}
        )
        
        # Write to CSV
        output_file = temp_directory / 'test_grid.csv'
        writer = CSVWriter()
        writer.write_grid(grid_data, output_file)
        
        # Verify file exists and content
        assert output_file.exists()
        
        # Read back and verify
        df = pd.read_csv(output_file)
        assert 'X' in df.columns
        assert 'Y' in df.columns
        assert 'Value' in df.columns
        assert len(df) == 9  # 3x3 grid
    
    def test_write_point_data(self, temp_directory):
        """Test writing point data to CSV."""
        # Create test point data
        coordinates = np.array([[100, 200], [150, 250], [200, 300]])
        values = np.array([10.5, 20.3, 30.1])
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            point_ids=np.array(['P1', 'P2', 'P3']),
            metadata={'test': 'metadata'}
        )
        
        # Write to CSV
        output_file = temp_directory / 'test_points.csv'
        writer = CSVWriter()
        writer.write_points(point_data, output_file)
        
        # Verify file exists and content
        assert output_file.exists()
        
        # Read back and verify
        df = pd.read_csv(output_file)
        assert 'X' in df.columns
        assert 'Y' in df.columns
        assert 'Value' in df.columns
        assert 'Point_ID' in df.columns
        assert len(df) == 3
        
        # Check specific values
        assert df.loc[0, 'X'] == 100
        assert df.loc[0, 'Point_ID'] == 'P1'
    
    def test_write_3d_point_data(self, temp_directory):
        """Test writing 3D point data to CSV."""
        # Create test 3D point data
        coordinates = np.array([[100, 200, 50], [150, 250, 60], [200, 300, 70]])
        values = np.array([10.5, 20.3, 30.1])
        
        point_data = PointData(
            coordinates=coordinates,
            values=values
        )
        
        # Write to CSV
        output_file = temp_directory / 'test_3d_points.csv'
        writer = CSVWriter()
        writer.write_points(point_data, output_file)
        
        # Verify file exists and content
        assert output_file.exists()
        
        # Read back and verify
        df = pd.read_csv(output_file)
        assert 'X' in df.columns
        assert 'Y' in df.columns
        assert 'Z' in df.columns
        assert 'Value' in df.columns
        assert len(df) == 3
    
    def test_custom_delimiter(self, temp_directory):
        """Test writing with custom delimiter."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Write with semicolon delimiter
        options = CSVExportOptions(delimiter=';')
        writer = CSVWriter(options)
        
        output_file = temp_directory / 'test_delimiter.csv'
        writer.write_points(point_data, output_file)
        
        # Read raw content to verify delimiter
        with open(output_file, 'r') as f:
            content = f.read()
            assert ';' in content
            assert ',' not in content.split('\n')[0]  # Header should use semicolon
    
    def test_precision_setting(self, temp_directory):
        """Test numeric precision setting."""
        coordinates = np.array([[100.123456789, 200.987654321]])
        values = np.array([10.123456789])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Write with 2 decimal places
        options = CSVExportOptions(precision=2)
        writer = CSVWriter(options)
        
        output_file = temp_directory / 'test_precision.csv'
        writer.write_points(point_data, output_file)
        
        # Read back and verify precision
        df = pd.read_csv(output_file)
        # Values should be rounded to 2 decimal places
        assert abs(df.loc[0, 'X'] - 100.12) < 0.01
        assert abs(df.loc[0, 'Value'] - 10.12) < 0.01
    
    def test_without_coordinates(self, temp_directory):
        """Test writing without coordinate columns."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        
        point_data = PointData(
            coordinates=coordinates, 
            values=values,
            point_ids=np.array(['P1', 'P2'])
        )
        
        # Write without coordinates
        options = CSVExportOptions(include_coordinates=False)
        writer = CSVWriter(options)
        
        output_file = temp_directory / 'test_no_coords.csv'
        writer.write_points(point_data, output_file)
        
        # Read back and verify
        df = pd.read_csv(output_file)
        assert 'X' not in df.columns
        assert 'Y' not in df.columns
        assert 'Value' in df.columns
        assert 'Point_ID' in df.columns
    
    def test_with_attributes(self, temp_directory):
        """Test writing point data with additional attributes."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        attributes = {
            'ASH': np.array([15.2, 18.7]),
            'SULFUR': np.array([2.1, 2.8])
        }
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            attributes=attributes
        )
        
        output_file = temp_directory / 'test_attributes.csv'
        writer = CSVWriter()
        writer.write_points(point_data, output_file)
        
        # Read back and verify
        df = pd.read_csv(output_file)
        assert 'ASH' in df.columns
        assert 'SULFUR' in df.columns
        assert df.loc[0, 'ASH'] == 15.2
        assert df.loc[1, 'SULFUR'] == 2.8
    
    def test_metadata_comments(self, temp_directory):
        """Test including metadata as comments."""
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            metadata={
                'source': 'Test Suite',
                'date': '2023-01-01',
                'method': 'IDW'
            }
        )
        
        options = CSVExportOptions(include_metadata=True)
        writer = CSVWriter(options)
        
        output_file = temp_directory / 'test_metadata.csv'
        writer.write_points(point_data, output_file)
        
        # Read raw content to check for comments
        with open(output_file, 'r') as f:
            content = f.read()
            assert '# source: Test Suite' in content
            assert '# date: 2023-01-01' in content
            assert '# method: IDW' in content
    
    def test_export_summary_grid(self):
        """Test export summary for grid data."""
        x_coords = np.array([100, 200])
        y_coords = np.array([100, 200])
        values = np.array([[10, 20], [15, 25]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(100, 200, 100, 200),
            cell_size=100
        )
        
        writer = CSVWriter()
        summary = writer.export_summary(grid_data, 'test.csv')
        
        assert summary['format'] == 'CSV'
        assert summary['data_type'] == 'grid'
        assert summary['n_points'] == 4
        assert summary['delimiter'] == ','
    
    def test_export_summary_points(self):
        """Test export summary for point data."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = CSVWriter()
        summary = writer.export_summary(point_data, 'test.csv')
        
        assert summary['format'] == 'CSV'
        assert summary['data_type'] == 'points'
        assert summary['n_points'] == 2
    
    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = CSVWriter()
        
        # Test with directory that doesn't exist (if create_directories is False)
        options = CSVExportOptions(create_directories=False)
        writer = CSVWriter(options)
        
        with pytest.raises(ExportError):
            writer.write_points(point_data, '/nonexistent/path/test.csv')
    
    def test_empty_data(self, temp_directory):
        """Test handling of empty data."""
        # Empty coordinates and values
        coordinates = np.array([]).reshape(0, 2)
        values = np.array([])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = CSVWriter()
        output_file = temp_directory / 'test_empty.csv'
        
        writer.write_points(point_data, output_file)
        
        # File should exist but be empty (except header)
        assert output_file.exists()
        df = pd.read_csv(output_file)
        assert len(df) == 0
        assert 'X' in df.columns  # Header should still be there
    
    def test_nan_values(self, temp_directory):
        """Test handling of NaN values."""
        coordinates = np.array([[100, 200], [150, np.nan], [200, 300]])
        values = np.array([10.5, np.nan, 30.1])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = CSVWriter()
        output_file = temp_directory / 'test_nan.csv'
        writer.write_points(point_data, output_file)
        
        # Read back and verify NaN handling
        df = pd.read_csv(output_file)
        assert len(df) == 3
        assert pd.isna(df.loc[1, 'Y'])
        assert pd.isna(df.loc[1, 'Value'])
    
    def test_overwrite_protection(self, temp_directory):
        """Test file overwrite protection."""
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        point_data = PointData(coordinates=coordinates, values=values)
        
        output_file = temp_directory / 'test_overwrite.csv'
        
        # Create file first
        with open(output_file, 'w') as f:
            f.write('existing content')
        
        # Test with overwrite disabled
        options = CSVExportOptions(overwrite_existing=False)
        writer = CSVWriter(options)
        
        with pytest.raises(ExportError):
            writer.write_points(point_data, output_file)
        
        # Test with overwrite enabled (default)
        writer = CSVWriter()
        writer.write_points(point_data, output_file)  # Should succeed
        
        # Verify file was overwritten
        df = pd.read_csv(output_file)
        assert len(df) == 1