"""
Unit tests for base writer functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from abc import abstractmethod

from src.io.writers.base import (
    BaseWriter, 
    ExportFormat, 
    ExportOptions, 
    GridData, 
    PointData, 
    ExportError,
    UnsupportedFormatError,
    InvalidDataError
)


# Test implementation of BaseWriter for testing abstract methods
class TestWriter(BaseWriter):
    """Test implementation of BaseWriter."""
    
    @property
    def supported_formats(self):
        return [ExportFormat.CSV]
    
    @property
    def file_extensions(self):
        return ['.test']
    
    def write_grid(self, data, filepath, **kwargs):
        # Simple test implementation
        with open(filepath, 'w') as f:
            f.write(f"Grid data: {data.shape}")
    
    def write_points(self, data, filepath, **kwargs):
        # Simple test implementation
        with open(filepath, 'w') as f:
            f.write(f"Point data: {data.n_points}")


class TestExportOptions:
    """Test suite for ExportOptions dataclass."""
    
    def test_default_options(self):
        """Test default export options."""
        options = ExportOptions()
        assert options.overwrite_existing == True
        assert options.create_directories == True
        assert options.compression is None
        assert options.precision == 6
        assert options.include_metadata == True
        assert options.coordinate_system is None
        assert options.custom_attributes == {}
    
    def test_custom_options(self):
        """Test custom export options."""
        custom_attrs = {'test': 'value'}
        options = ExportOptions(
            overwrite_existing=False,
            precision=3,
            coordinate_system='EPSG:4326',
            custom_attributes=custom_attrs
        )
        assert options.overwrite_existing == False
        assert options.precision == 3
        assert options.coordinate_system == 'EPSG:4326'
        assert options.custom_attributes == custom_attrs
    
    def test_post_init(self):
        """Test __post_init__ method."""
        # Test with None custom_attributes
        options = ExportOptions(custom_attributes=None)
        assert options.custom_attributes == {}
        
        # Test with existing custom_attributes
        custom_attrs = {'key': 'value'}
        options = ExportOptions(custom_attributes=custom_attrs)
        assert options.custom_attributes == custom_attrs


class TestGridData:
    """Test suite for GridData dataclass."""
    
    def test_2d_grid_creation(self):
        """Test creation of 2D grid data."""
        x_coords = np.array([0, 1, 2])
        y_coords = np.array([0, 1, 2])
        values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        bounds = (0, 2, 0, 2)
        
        grid = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=bounds,
            cell_size=1.0
        )
        
        assert not grid.is_3d
        assert grid.shape == (3, 3)
        assert grid.n_points == 9
        assert grid.metadata == {}
    
    def test_3d_grid_creation(self):
        """Test creation of 3D grid data."""
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        z_coords = np.array([0, 1])
        values = np.random.random((2, 2, 2))
        bounds = (0, 1, 0, 1, 0, 1)
        
        grid = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
            values=values,
            bounds=bounds,
            cell_size=1.0
        )
        
        assert grid.is_3d
        assert grid.shape == (2, 2, 2)
        assert grid.n_points == 8
    
    def test_grid_with_metadata(self):
        """Test grid data with metadata."""
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        values = np.array([[1, 2], [3, 4]])
        metadata = {'source': 'test', 'method': 'IDW'}
        
        grid = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 1, 0, 1),
            cell_size=1.0,
            metadata=metadata
        )
        
        assert grid.metadata == metadata
    
    def test_post_init(self):
        """Test GridData __post_init__ method."""
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        values = np.array([[1, 2], [3, 4]])
        
        # Test with None metadata
        grid = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 1, 0, 1),
            cell_size=1.0,
            metadata=None
        )
        
        assert grid.metadata == {}


class TestPointData:
    """Test suite for PointData dataclass."""
    
    def test_2d_point_creation(self):
        """Test creation of 2D point data."""
        coordinates = np.array([[100, 200], [150, 250], [200, 300]])
        values = np.array([10.5, 20.3, 30.1])
        
        points = PointData(coordinates=coordinates, values=values)
        
        assert not points.is_3d
        assert points.n_points == 3
        assert points.attributes == {}
        assert points.metadata == {}
        
        bounds = points.bounds
        assert bounds == (100.0, 200.0, 200.0, 300.0)
    
    def test_3d_point_creation(self):
        """Test creation of 3D point data."""
        coordinates = np.array([[100, 200, 50], [150, 250, 60], [200, 300, 70]])
        values = np.array([10.5, 20.3, 30.1])
        
        points = PointData(coordinates=coordinates, values=values)
        
        assert points.is_3d
        assert points.n_points == 3
        
        bounds = points.bounds
        assert len(bounds) == 6  # xmin, xmax, ymin, ymax, zmin, zmax
        assert bounds == (100.0, 200.0, 200.0, 300.0, 50.0, 70.0)
    
    def test_point_data_with_attributes(self):
        """Test point data with additional attributes."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        point_ids = np.array(['P1', 'P2'])
        attributes = {
            'ASH': np.array([15.2, 18.7]),
            'SULFUR': np.array([2.1, 2.8])
        }
        
        points = PointData(
            coordinates=coordinates,
            values=values,
            point_ids=point_ids,
            attributes=attributes
        )
        
        assert np.array_equal(points.point_ids, point_ids)
        assert 'ASH' in points.attributes
        assert 'SULFUR' in points.attributes
        assert np.array_equal(points.attributes['ASH'], np.array([15.2, 18.7]))
    
    def test_invalid_dimensions(self):
        """Test validation of coordinate/value dimensions."""
        coordinates = np.array([[100, 200], [150, 250]])  # 2 points
        values = np.array([10.5, 20.3, 30.1])  # 3 values - mismatch!
        
        with pytest.raises(ValueError, match="Coordinates and values must have same length"):
            PointData(coordinates=coordinates, values=values)
    
    def test_post_init(self):
        """Test PointData __post_init__ method."""
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        
        # Test with None attributes and metadata
        points = PointData(
            coordinates=coordinates,
            values=values,
            attributes=None,
            metadata=None
        )
        
        assert points.attributes == {}
        assert points.metadata == {}


class TestBaseWriter:
    """Test suite for BaseWriter abstract class."""
    
    def test_init_default_options(self):
        """Test initialization with default options."""
        writer = TestWriter()
        assert isinstance(writer.options, ExportOptions)
        assert writer.options.overwrite_existing == True
    
    def test_init_custom_options(self):
        """Test initialization with custom options."""
        options = ExportOptions(precision=3, overwrite_existing=False)
        writer = TestWriter(options)
        assert writer.options.precision == 3
        assert writer.options.overwrite_existing == False
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        
        class IncompleteWriter(BaseWriter):
            pass
        
        # Should not be able to instantiate incomplete writer
        with pytest.raises(TypeError):
            IncompleteWriter()
    
    def test_validate_filepath(self, temp_directory):
        """Test file path validation."""
        writer = TestWriter()
        
        # Test with valid path
        valid_path = temp_directory / 'test.csv'
        validated = writer.validate_filepath(valid_path)
        assert isinstance(validated, Path)
        
        # Test with string path
        string_path = str(temp_directory / 'test2.csv')
        validated = writer.validate_filepath(string_path)
        assert isinstance(validated, Path)
    
    def test_validate_filepath_directory_creation(self, temp_directory):
        """Test directory creation during file path validation."""
        writer = TestWriter()
        
        # Test with non-existent directory
        nested_path = temp_directory / 'nested' / 'dir' / 'test.csv'
        
        # Should create directories by default
        validated = writer.validate_filepath(nested_path)
        assert nested_path.parent.exists()
    
    def test_validate_filepath_no_directory_creation(self, temp_directory):
        """Test validation without directory creation."""
        options = ExportOptions(create_directories=False)
        writer = TestWriter(options)
        
        # Test with non-existent directory
        nested_path = temp_directory / 'nonexistent' / 'test.csv'
        
        with pytest.raises(ExportError, match="Output directory does not exist"):
            writer.validate_filepath(nested_path)
    
    def test_validate_filepath_overwrite_protection(self, temp_directory):
        """Test file overwrite protection."""
        options = ExportOptions(overwrite_existing=False)
        writer = TestWriter(options)
        
        # Create existing file
        existing_file = temp_directory / 'existing.csv'
        existing_file.write_text('existing content')
        
        with pytest.raises(ExportError, match="File already exists"):
            writer.validate_filepath(existing_file)
    
    def test_validate_grid_data_valid(self):
        """Test validation of valid grid data."""
        writer = TestWriter()
        
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        values = np.array([[1, 2], [3, 4]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 1, 0, 1),
            cell_size=1.0
        )
        
        # Should not raise any exception
        writer.validate_grid_data(grid_data)
    
    def test_validate_grid_data_invalid(self):
        """Test validation of invalid grid data."""
        writer = TestWriter()
        
        # Test with None
        with pytest.raises(InvalidDataError, match="Grid data cannot be None"):
            writer.validate_grid_data(None)
        
        # Test with empty coordinates
        x_coords = np.array([])
        y_coords = np.array([])
        values = np.array([])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 1, 0, 1),
            cell_size=1.0
        )
        
        with pytest.raises(InvalidDataError, match="Grid data is empty"):
            writer.validate_grid_data(grid_data)
    
    def test_validate_point_data_valid(self):
        """Test validation of valid point data."""
        writer = TestWriter()
        
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Should not raise any exception
        writer.validate_point_data(point_data)
    
    def test_validate_point_data_invalid(self):
        """Test validation of invalid point data."""
        writer = TestWriter()
        
        # Test with None
        with pytest.raises(InvalidDataError, match="Point data cannot be None"):
            writer.validate_point_data(None)
        
        # Test with empty data
        coordinates = np.array([]).reshape(0, 2)
        values = np.array([])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        with pytest.raises(InvalidDataError, match="Point data is empty"):
            writer.validate_point_data(point_data)
    
    def test_write_methods(self, temp_directory):
        """Test that write methods work with validation."""
        writer = TestWriter()
        
        # Test write_grid
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        values = np.array([[1, 2], [3, 4]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 1, 0, 1),
            cell_size=1.0
        )
        
        grid_file = temp_directory / 'test_grid.test'
        writer.write_grid(grid_data, grid_file)
        assert grid_file.exists()
        
        # Test write_points
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        point_data = PointData(coordinates=coordinates, values=values)
        
        points_file = temp_directory / 'test_points.test'
        writer.write_points(point_data, points_file)
        assert points_file.exists()


class TestExportErrors:
    """Test suite for export-related exceptions."""
    
    def test_export_error(self):
        """Test ExportError exception."""
        error = ExportError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_unsupported_format_error(self):
        """Test UnsupportedFormatError exception."""
        error = UnsupportedFormatError("Unsupported format")
        assert str(error) == "Unsupported format"
        assert isinstance(error, ExportError)
    
    def test_invalid_data_error(self):
        """Test InvalidDataError exception."""
        error = InvalidDataError("Invalid data")
        assert str(error) == "Invalid data"
        assert isinstance(error, ExportError)


class TestExportFormat:
    """Test suite for ExportFormat enum."""
    
    def test_export_format_values(self):
        """Test ExportFormat enum values."""
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.GEOTIFF.value == "geotiff"
        assert ExportFormat.VTK.value == "vtk"
        assert ExportFormat.DXF.value == "dxf"
        assert ExportFormat.JSON.value == "json"
    
    def test_export_format_iteration(self):
        """Test iterating over ExportFormat enum."""
        formats = list(ExportFormat)
        assert len(formats) >= 5  # At least the formats we defined
        assert ExportFormat.CSV in formats
        assert ExportFormat.GEOTIFF in formats


# Integration-style tests for data structures
class TestDataStructureIntegration:
    """Integration tests for data structures with realistic data."""
    
    def test_coal_deposit_grid_data(self):
        """Test grid data with realistic coal deposit values."""
        # Create coal quality grid
        x_coords = np.linspace(100000, 105000, 11)  # UTM coordinates
        y_coords = np.linspace(200000, 205000, 11)
        
        # Simulate ash content values
        ash_values = np.random.normal(15, 3, (11, 11))  # 15% avg ash, 3% std
        ash_values = np.clip(ash_values, 5, 35)  # Realistic range
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=ash_values,
            bounds=(100000, 105000, 200000, 205000),
            cell_size=500,
            coordinate_system='EPSG:32633',
            metadata={
                'parameter': 'ash_content',
                'units': 'percent',
                'method': 'kriging',
                'date': '2023-01-01'
            }
        )
        
        assert grid_data.is_3d == False
        assert grid_data.coordinate_system == 'EPSG:32633'
        assert grid_data.metadata['parameter'] == 'ash_content'
        assert 5 <= np.min(grid_data.values) <= 35
        assert 5 <= np.max(grid_data.values) <= 35
    
    def test_borehole_point_data(self):
        """Test point data with realistic borehole data."""
        # Simulate borehole locations
        n_holes = 50
        x_coords = np.random.uniform(100000, 105000, n_holes)
        y_coords = np.random.uniform(200000, 205000, n_holes)
        z_coords = np.random.uniform(50, 150, n_holes)  # Elevation
        
        coordinates = np.column_stack([x_coords, y_coords, z_coords])
        
        # Coal quality parameters
        ash_content = np.random.normal(15, 5, n_holes)
        sulfur_content = np.random.normal(2.5, 1.0, n_holes)
        calorific_value = 30 - 0.5 * ash_content + np.random.normal(0, 2, n_holes)
        
        # Borehole IDs
        hole_ids = np.array([f'BH{i:03d}' for i in range(n_holes)])
        
        point_data = PointData(
            coordinates=coordinates,
            values=ash_content,
            point_ids=hole_ids,
            attributes={
                'SULFUR': sulfur_content,
                'CALORIFIC': calorific_value,
                'ELEVATION': z_coords
            },
            coordinate_system='EPSG:32633',
            metadata={
                'project': 'Test Coal Mine',
                'survey_date': '2023-01-01',
                'primary_parameter': 'ash_content'
            }
        )
        
        assert point_data.is_3d == True
        assert point_data.n_points == n_holes
        assert len(point_data.point_ids) == n_holes
        assert 'SULFUR' in point_data.attributes
        assert 'CALORIFIC' in point_data.attributes
        
        # Check bounds include all coordinates
        bounds = point_data.bounds
        assert bounds[0] <= np.min(x_coords)  # xmin
        assert bounds[1] >= np.max(x_coords)  # xmax
        assert bounds[4] <= np.min(z_coords)  # zmin
        assert bounds[5] >= np.max(z_coords)  # zmax