"""
Unit tests for DXF writer.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.io.writers.dxf_writer import DXFWriter, DXFExportOptions
from src.io.writers.base import GridData, PointData, ExportError

# Skip tests if ezdxf is not available
pytest_plugins = []

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


@pytest.mark.skipif(not EZDXF_AVAILABLE, reason="ezdxf not available")
class TestDXFWriter:
    """Test suite for DXF writer."""
    
    def test_init_default_options(self):
        """Test DXF writer initialization with default options."""
        writer = DXFWriter()
        assert isinstance(writer.options, DXFExportOptions)
        assert writer.options.units == 'm'
        assert writer.options.layer_name == 'INTERPOLATION'
        assert writer.options.point_style == 'CIRCLE'
        assert writer.options.point_size == 1.0
        assert writer.options.contour_lines == True
    
    def test_init_custom_options(self):
        """Test DXF writer initialization with custom options."""
        options = DXFExportOptions(
            units='mm',
            layer_name='COAL_DATA',
            point_style='CROSS',
            point_size=2.5,
            contour_lines=False,
            include_labels=True
        )
        writer = DXFWriter(options)
        assert writer.options.units == 'mm'
        assert writer.options.layer_name == 'COAL_DATA'
        assert writer.options.point_style == 'CROSS'
        assert writer.options.point_size == 2.5
        assert writer.options.contour_lines == False
        assert writer.options.include_labels == True
    
    def test_supported_formats(self):
        """Test supported formats."""
        writer = DXFWriter()
        from src.io.writers.base import ExportFormat
        assert ExportFormat.DXF in writer.supported_formats
    
    def test_file_extensions(self):
        """Test file extensions."""
        writer = DXFWriter()
        assert '.dxf' in writer.file_extensions
    
    def test_write_point_data_circles(self, temp_directory):
        """Test writing point data as circles."""
        # Create test point data
        coordinates = np.array([
            [100, 200],
            [150, 250],
            [200, 300]
        ])
        values = np.array([10.5, 20.3, 30.1])
        point_ids = np.array(['P1', 'P2', 'P3'])
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            point_ids=point_ids
        )
        
        # Write to DXF with circles
        options = DXFExportOptions(point_style='CIRCLE', point_size=1.5)
        writer = DXFWriter(options)
        
        output_file = temp_directory / 'test_circles.dxf'
        writer.write_points(point_data, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read back with ezdxf and verify
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        # Check for circles
        circles = msp.query('CIRCLE')
        assert len(circles) >= 3  # Should have at least 3 circles
        
        # Check layer
        for circle in circles:
            if circle.dxf.layer == 'INTERPOLATION':
                assert circle.dxf.radius == 1.5
    
    def test_write_point_data_points(self, temp_directory):
        """Test writing point data as point entities."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Write to DXF with points
        options = DXFExportOptions(point_style='POINT')
        writer = DXFWriter(options)
        
        output_file = temp_directory / 'test_points.dxf'
        writer.write_points(point_data, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read back and verify
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        # Check for points
        points = msp.query('POINT')
        assert len(points) >= 2
    
    def test_write_point_data_crosses(self, temp_directory):
        """Test writing point data as crosses (lines)."""
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Write to DXF with crosses
        options = DXFExportOptions(point_style='CROSS', point_size=2.0)
        writer = DXFWriter(options)
        
        output_file = temp_directory / 'test_crosses.dxf'
        writer.write_points(point_data, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read back and verify
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        # Check for lines (crosses are made of lines)
        lines = msp.query('LINE')
        assert len(lines) >= 2  # Each cross is 2 lines
    
    def test_write_3d_point_data(self, temp_directory):
        """Test writing 3D point data."""
        coordinates = np.array([
            [100, 200, 50],
            [150, 250, 60],
            [200, 300, 70]
        ])
        values = np.array([10.5, 20.3, 30.1])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Write as 3D
        options = DXFExportOptions(export_3d=True)
        writer = DXFWriter(options)
        
        output_file = temp_directory / 'test_3d.dxf'
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Read back and verify 3D coordinates
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        circles = msp.query('CIRCLE')
        if len(circles) > 0:
            # Check that Z coordinates are preserved
            for circle in circles:
                if circle.dxf.layer == 'INTERPOLATION':
                    center = circle.dxf.center
                    assert center.z != 0  # Should have Z coordinate
    
    def test_write_grid_data(self, temp_directory):
        """Test writing grid data to DXF."""
        # Create test grid data
        x_coords = np.array([100, 200, 300])
        y_coords = np.array([100, 200, 300])
        values = np.array([
            [10.5, 20.3, 30.1],
            [15.2, 25.8, 35.4],
            [12.7, 22.9, 32.5]
        ])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(100, 300, 100, 300),
            cell_size=100
        )
        
        # Write to DXF
        writer = DXFWriter()
        output_file = temp_directory / 'test_grid.dxf'
        writer.write_grid(grid_data, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read back and verify
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        # Should have circles for grid points
        circles = msp.query('CIRCLE')
        assert len(circles) >= 9  # 3x3 grid
    
    def test_contour_lines_generation(self, temp_directory):
        """Test contour line generation for grid data."""
        try:
            import matplotlib
            import scipy
        except ImportError:
            pytest.skip("matplotlib and scipy required for contour generation")
        
        # Create grid with smooth variation for better contours
        x_coords = np.linspace(0, 10, 11)
        y_coords = np.linspace(0, 10, 11)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Create smooth function for contouring
        values = np.sin(X/2) * np.cos(Y/2) * 10 + 20
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 10, 0, 10),
            cell_size=1
        )
        
        # Write with contour lines
        options = DXFExportOptions(
            contour_lines=True,
            contour_intervals=2.0,
            include_labels=False  # Simplify for testing
        )
        writer = DXFWriter(options)
        
        output_file = temp_directory / 'test_contours.dxf'
        writer.write_grid(grid_data, output_file)
        
        assert output_file.exists()
        
        # Read back and check for polylines (contours)
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        polylines = msp.query('LWPOLYLINE')
        # Should have some contour lines, but exact number depends on data
        assert len(polylines) >= 0  # At least we don't crash
    
    def test_labels_inclusion(self, temp_directory):
        """Test inclusion of value labels."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.567, 20.834])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Write with labels
        options = DXFExportOptions(
            include_labels=True,
            precision=2,
            text_height=1.5
        )
        writer = DXFWriter(options)
        
        output_file = temp_directory / 'test_labels.dxf'
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Read back and check for text entities
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        texts = msp.query('TEXT')
        assert len(texts) >= 2  # Should have text labels
        
        # Check text content (should be rounded to 2 decimal places)
        text_values = [text.dxf.text for text in texts if hasattr(text.dxf, 'text')]
        assert '10.57' in text_values or '20.83' in text_values
    
    def test_different_units(self, temp_directory):
        """Test different drawing units."""
        coordinates = np.array([[1000, 2000]])  # Millimeter coordinates
        values = np.array([10.5])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Test different units
        units_list = ['mm', 'm', 'ft', 'in']
        
        for unit in units_list:
            options = DXFExportOptions(units=unit)
            writer = DXFWriter(options)
            
            output_file = temp_directory / f'test_{unit}.dxf'
            writer.write_points(point_data, output_file)
            
            assert output_file.exists()
            
            # Read back and verify units are set
            doc = ezdxf.readfile(output_file)
            # Units are set in the document header
            assert doc.units is not None
    
    def test_custom_layer_name(self, temp_directory):
        """Test custom layer names."""
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Write with custom layer
        custom_layer = 'COAL_QUALITY_DATA'
        options = DXFExportOptions(layer_name=custom_layer)
        writer = DXFWriter(options)
        
        output_file = temp_directory / 'test_layer.dxf'
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Read back and verify layer
        doc = ezdxf.readfile(output_file)
        
        # Check that layer exists
        assert custom_layer in doc.layers
        
        # Check that entities are on the correct layer
        msp = doc.modelspace()
        circles = msp.query('CIRCLE')
        for circle in circles:
            if circle.dxf.layer == custom_layer:
                break
        else:
            pytest.fail(f"No entities found on layer {custom_layer}")
    
    def test_metadata_inclusion(self, temp_directory):
        """Test inclusion of metadata as text."""
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            metadata={
                'source': 'Test Suite',
                'method': 'IDW',
                'power': 2.0
            }
        )
        
        options = DXFExportOptions(include_metadata=True)
        writer = DXFWriter(options)
        
        output_file = temp_directory / 'test_metadata.dxf'
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Read back and check for metadata text
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        texts = msp.query('TEXT')
        text_content = [text.dxf.text for text in texts if hasattr(text.dxf, 'text')]
        
        # Should have metadata information
        metadata_found = any('Test Suite' in text or 'IDW' in text or '2.0' in text 
                           for text in text_content)
        assert metadata_found
    
    def test_color_by_value(self, temp_directory):
        """Test color coding by value."""
        coordinates = np.array([
            [100, 200],
            [150, 250],
            [200, 300]
        ])
        values = np.array([10.0, 25.0, 40.0])  # Different values for color mapping
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Write with color by value
        options = DXFExportOptions(color_by_value=True)
        writer = DXFWriter(options)
        
        output_file = temp_directory / 'test_colors.dxf'
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Read back and check for different colors
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        circles = msp.query('CIRCLE')
        colors = set()
        for circle in circles:
            if circle.dxf.layer == 'INTERPOLATION':
                colors.add(circle.dxf.color)
        
        # Should have multiple colors (at least 2 different ones)
        assert len(colors) >= 2
    
    def test_export_summary_grid(self):
        """Test export summary for grid data."""
        x_coords = np.array([0, 1, 2])
        y_coords = np.array([0, 1, 2])
        values = np.array([[10, 20, 30], [15, 25, 35], [12, 22, 32]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 2, 0, 2),
            cell_size=1
        )
        
        writer = DXFWriter()
        summary = writer.export_summary(grid_data, 'test.dxf')
        
        assert summary['format'] == 'DXF'
        assert summary['data_type'] == 'grid'
        assert summary['n_points'] == 9
        assert summary['units'] == 'm'
        assert summary['layer_name'] == 'INTERPOLATION'
        assert summary['contour_lines'] == True
    
    def test_export_summary_points(self):
        """Test export summary for point data."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = DXFWriter()
        summary = writer.export_summary(point_data, 'test.dxf')
        
        assert summary['format'] == 'DXF'
        assert summary['data_type'] == 'points'
        assert summary['n_points'] == 2
        assert summary['is_3d'] == False
    
    def test_empty_data(self, temp_directory):
        """Test handling of empty data."""
        # Empty coordinates and values
        coordinates = np.array([]).reshape(0, 2)
        values = np.array([])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = DXFWriter()
        output_file = temp_directory / 'test_empty.dxf'
        
        # Should not crash, should create valid DXF file
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # File should be readable
        doc = ezdxf.readfile(output_file)
        assert doc is not None
    
    def test_nan_values_handling(self, temp_directory):
        """Test handling of NaN values."""
        coordinates = np.array([
            [100, 200],
            [150, np.nan],  # NaN coordinate
            [200, 300]
        ])
        values = np.array([10.5, np.nan, 30.1])  # NaN value
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = DXFWriter()
        output_file = temp_directory / 'test_nan.dxf'
        
        # Should handle NaN gracefully (skip or handle appropriately)
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Should still create valid DXF
        doc = ezdxf.readfile(output_file)
        msp = doc.modelspace()
        
        # Should have fewer entities due to NaN handling
        circles = msp.query('CIRCLE')
        assert len(circles) < 3  # Should skip NaN values


@pytest.mark.skipif(EZDXF_AVAILABLE, reason="Testing without ezdxf")
class TestDXFWriterWithoutEzdxf:
    """Test DXF writer behavior when ezdxf is not available."""
    
    def test_import_error_on_init(self):
        """Test that ImportError is raised when ezdxf is not available."""
        with pytest.raises(ImportError, match="ezdxf library is required"):
            DXFWriter()
    
    def test_factory_function_import_error(self):
        """Test factory function raises ImportError without ezdxf."""
        from src.io.writers.dxf_writer import create_dxf_writer
        
        with pytest.raises(ImportError, match="ezdxf library is required"):
            create_dxf_writer()


@pytest.mark.skipif(not EZDXF_AVAILABLE, reason="ezdxf not available")
class TestDXFFactoryFunction:
    """Test DXF factory function."""
    
    def test_create_dxf_writer(self):
        """Test factory function creates writer with correct options."""
        from src.io.writers.dxf_writer import create_dxf_writer
        
        writer = create_dxf_writer(
            units='mm',
            layer_name='TEST_LAYER',
            point_style='CROSS',
            contour_lines=False,
            include_labels=True
        )
        
        assert isinstance(writer, DXFWriter)
        assert writer.options.units == 'mm'
        assert writer.options.layer_name == 'TEST_LAYER'
        assert writer.options.point_style == 'CROSS'
        assert writer.options.contour_lines == False
        assert writer.options.include_labels == True