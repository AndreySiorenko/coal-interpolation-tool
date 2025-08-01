"""
Unit tests for GeoTIFF writer.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.io.writers.geotiff_writer import GeoTIFFWriter, GeoTIFFExportOptions
from src.io.writers.base import GridData, PointData, ExportError

# Skip tests if rasterio is not available
pytest_plugins = []

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
class TestGeoTIFFWriter:
    """Test suite for GeoTIFF writer."""
    
    def test_init_default_options(self):
        """Test GeoTIFF writer initialization with default options."""
        writer = GeoTIFFWriter()
        assert isinstance(writer.options, GeoTIFFExportOptions)
        assert writer.options.compress == 'lzw'
        assert writer.options.tiled == True
        assert writer.options.dtype == 'float32'
    
    def test_init_custom_options(self):
        """Test GeoTIFF writer initialization with custom options."""
        options = GeoTIFFExportOptions(
            compress='deflate',
            tiled=False,
            dtype='int16',
            nodata_value=-9999
        )
        writer = GeoTIFFWriter(options)
        assert writer.options.compress == 'deflate'
        assert writer.options.tiled == False
        assert writer.options.dtype == 'int16'
        assert writer.options.nodata_value == -9999
    
    def test_supported_formats(self):
        """Test supported formats."""
        writer = GeoTIFFWriter()
        from src.io.writers.base import ExportFormat
        assert ExportFormat.GEOTIFF in writer.supported_formats
    
    def test_file_extensions(self):
        """Test file extensions."""
        writer = GeoTIFFWriter()
        extensions = writer.file_extensions
        assert '.tif' in extensions
        assert '.tiff' in extensions
        assert '.gtiff' in extensions
    
    def test_write_grid_data(self, temp_directory):
        """Test writing grid data to GeoTIFF."""
        # Create test grid data
        x_coords = np.array([100000, 100100, 100200])
        y_coords = np.array([200000, 200100, 200200])
        values = np.array([[10.5, 20.3, 30.1], 
                          [15.2, 25.8, 35.4], 
                          [12.7, 22.9, 32.5]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(100000, 100200, 200000, 200200),
            cell_size=100,
            coordinate_system='EPSG:32633',  # UTM Zone 33N
            metadata={'source': 'Test Suite'}
        )
        
        # Write to GeoTIFF
        output_file = temp_directory / 'test_grid.tif'
        writer = GeoTIFFWriter()
        writer.write_grid(grid_data, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read back with rasterio and verify
        with rasterio.open(output_file) as src:
            assert src.count == 1  # Single band
            assert src.height == 3
            assert src.width == 3
            assert src.dtype == 'float32'
            
            # Check CRS
            assert src.crs is not None
            
            # Check bounds (approximately)
            bounds = src.bounds
            assert abs(bounds.left - 100000) < 100
            assert abs(bounds.right - 100200) < 100
            assert abs(bounds.bottom - 200000) < 100
            assert abs(bounds.top - 200200) < 100
            
            # Read data and verify values
            data = src.read(1)
            assert data.shape == (3, 3)
            assert not np.all(np.isnan(data))  # Should have real values
    
    def test_write_point_data_gridded(self, temp_directory):
        """Test writing point data (automatically gridded) to GeoTIFF."""
        # Create test point data
        coordinates = np.array([
            [100000, 200000],
            [100100, 200000],
            [100200, 200000],
            [100000, 200100],
            [100100, 200100],
            [100200, 200100]
        ])
        values = np.array([10.5, 20.3, 30.1, 15.2, 25.8, 35.4])
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            coordinate_system='EPSG:32633'
        )
        
        # Write to GeoTIFF with custom cell size
        output_file = temp_directory / 'test_points.tif'
        writer = GeoTIFFWriter()
        writer.write_points(point_data, output_file, cell_size=50)
        
        # Verify file exists and can be read
        assert output_file.exists()
        
        with rasterio.open(output_file) as src:
            assert src.count == 1
            assert src.width > 1
            assert src.height > 1
            data = src.read(1)
            assert not np.all(np.isnan(data))
    
    def test_compression_options(self, temp_directory):
        """Test different compression options."""
        # Create simple grid
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        values = np.array([[10, 20], [15, 25]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 1, 0, 1),
            cell_size=1
        )
        
        # Test different compression methods
        compressions = ['lzw', 'deflate', 'none']
        
        for compression in compressions:
            options = GeoTIFFExportOptions(compress=compression)
            writer = GeoTIFFWriter(options)
            
            output_file = temp_directory / f'test_{compression}.tif'
            writer.write_grid(grid_data, output_file)
            
            assert output_file.exists()
            
            # Verify compression setting
            with rasterio.open(output_file) as src:
                if compression != 'none':
                    assert src.compression.name.lower() == compression.lower()
    
    def test_data_types(self, temp_directory):
        """Test different output data types."""
        # Create grid with integer values
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        values = np.array([[10, 20], [15, 25]], dtype=np.float64)
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 1, 0, 1),
            cell_size=1
        )
        
        # Test different data types
        dtypes = ['int16', 'int32', 'float32', 'float64']
        
        for dtype in dtypes:
            options = GeoTIFFExportOptions(dtype=dtype)
            writer = GeoTIFFWriter(options)
            
            output_file = temp_directory / f'test_{dtype}.tif'
            writer.write_grid(grid_data, output_file)
            
            assert output_file.exists()
            
            with rasterio.open(output_file) as src:
                assert str(src.dtype) == dtype
    
    def test_nodata_value(self, temp_directory):
        """Test NoData value handling."""
        # Create grid with NaN values
        x_coords = np.array([0, 1, 2])
        y_coords = np.array([0, 1, 2])
        values = np.array([[10, np.nan, 20], 
                          [15, 25, np.nan], 
                          [np.nan, 30, 35]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 2, 0, 2),
            cell_size=1
        )
        
        # Write with custom NoData value
        nodata_value = -9999
        options = GeoTIFFExportOptions(nodata_value=nodata_value)
        writer = GeoTIFFWriter(options)
        
        output_file = temp_directory / 'test_nodata.tif'
        writer.write_grid(grid_data, output_file)
        
        # Verify NoData handling
        with rasterio.open(output_file) as src:
            assert src.nodata == nodata_value
            data = src.read(1)
            
            # Check that NaN values were replaced with NoData value
            assert np.sum(data == nodata_value) == 3  # Should have 3 NoData pixels
    
    def test_tiling_options(self, temp_directory):
        """Test tiled vs non-tiled output."""
        # Create larger grid for tiling test
        size = 10
        x_coords = np.arange(size)
        y_coords = np.arange(size)
        values = np.random.random((size, size))
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, size-1, 0, size-1),
            cell_size=1
        )
        
        # Test tiled
        options = GeoTIFFExportOptions(tiled=True, blockxsize=4, blockysize=4)
        writer = GeoTIFFWriter(options)
        
        output_file = temp_directory / 'test_tiled.tif'
        writer.write_grid(grid_data, output_file)
        
        with rasterio.open(output_file) as src:
            assert src.is_tiled
            assert src.block_shapes[0] == (4, 4)
        
        # Test non-tiled
        options = GeoTIFFExportOptions(tiled=False)
        writer = GeoTIFFWriter(options)
        
        output_file = temp_directory / 'test_not_tiled.tif'
        writer.write_grid(grid_data, output_file)
        
        with rasterio.open(output_file) as src:
            assert not src.is_tiled
    
    def test_coordinate_systems(self, temp_directory):
        """Test different coordinate systems."""
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        values = np.array([[10, 20], [15, 25]])
        
        # Test with different CRS
        crs_list = ['EPSG:4326', 'EPSG:32633', 'EPSG:3857']
        
        for crs in crs_list:
            grid_data = GridData(
                x_coords=x_coords,
                y_coords=y_coords,
                values=values,
                bounds=(0, 1, 0, 1),
                cell_size=1,
                coordinate_system=crs
            )
            
            writer = GeoTIFFWriter()
            output_file = temp_directory / f'test_{crs.replace(":", "_")}.tif'
            writer.write_grid(grid_data, output_file)
            
            with rasterio.open(output_file) as src:
                assert src.crs is not None
                assert str(src.crs).upper() == crs.upper()
    
    def test_metadata_inclusion(self, temp_directory):
        """Test metadata inclusion in GeoTIFF tags."""
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        values = np.array([[10, 20], [15, 25]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 1, 0, 1),
            cell_size=1,
            metadata={
                'method': 'IDW',
                'power': 2.0,
                'points_used': 100
            }
        )
        
        options = GeoTIFFExportOptions(include_metadata=True)
        writer = GeoTIFFWriter(options)
        
        output_file = temp_directory / 'test_metadata.tif'
        writer.write_grid(grid_data, output_file)
        
        # Check for metadata tags
        with rasterio.open(output_file) as src:
            tags = src.tags()
            assert 'CREATED_BY' in tags
            assert tags['CREATED_BY'] == 'Coal Interpolation Tool'
            assert 'CUSTOM_METHOD' in tags
            assert tags['CUSTOM_METHOD'] == 'IDW'
    
    def test_3d_data_rejection(self, temp_directory):
        """Test rejection of 3D data for single-band GeoTIFF."""
        # Create 3D grid data
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        z_coords = np.array([0, 1])  # This makes it 3D
        values = np.random.random((2, 2, 2))
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
            values=values,
            bounds=(0, 1, 0, 1, 0, 1),
            cell_size=1
        )
        
        writer = GeoTIFFWriter()
        output_file = temp_directory / 'test_3d.tif'
        
        # Should raise ExportError for 3D data
        with pytest.raises(ExportError, match="3D grid data cannot be exported"):
            writer.write_grid(grid_data, output_file)
    
    def test_export_summary(self):
        """Test export summary generation."""
        x_coords = np.array([0, 1, 2])
        y_coords = np.array([0, 1, 2])
        values = np.array([[10, 20, 30], [15, 25, 35], [12, 22, 32]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 2, 0, 2),
            cell_size=1,
            coordinate_system='EPSG:4326'
        )
        
        writer = GeoTIFFWriter()
        summary = writer.export_summary(grid_data, 'test.tif')
        
        assert summary['format'] == 'GeoTIFF'
        assert summary['data_type'] == 'grid'
        assert summary['grid_shape'] == (3, 3)
        assert summary['n_pixels'] == 9
        assert summary['compression'] == 'lzw'
        assert summary['dtype'] == 'float32'
        assert summary['coordinate_system'] == 'EPSG:4326'
    
    def test_invalid_file_extension(self, temp_directory):
        """Test handling of invalid file extensions."""
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        values = np.array([[10, 20], [15, 25]])
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(0, 1, 0, 1),
            cell_size=1
        )
        
        writer = GeoTIFFWriter()
        
        # Should work fine - writer should handle extension validation
        output_file = temp_directory / 'test.xyz'  # Wrong extension
        
        # The writer should either auto-correct or handle this gracefully
        try:
            writer.write_grid(grid_data, output_file)
            # If it succeeds, file should exist
            assert output_file.exists() or (temp_directory / 'test.tif').exists()
        except ExportError:
            # If it fails, that's also acceptable behavior
            pass


@pytest.mark.skipif(RASTERIO_AVAILABLE, reason="Testing without rasterio")
class TestGeoTIFFWriterWithoutRasterio:
    """Test GeoTIFF writer behavior when rasterio is not available."""
    
    def test_import_error_on_init(self):
        """Test that ImportError is raised when rasterio is not available."""
        with pytest.raises(ImportError, match="rasterio library is required"):
            GeoTIFFWriter()
    
    def test_factory_function_import_error(self):
        """Test factory function raises ImportError without rasterio."""
        from src.io.writers.geotiff_writer import create_geotiff_writer
        
        with pytest.raises(ImportError, match="rasterio library is required"):
            create_geotiff_writer()


@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
class TestGeoTIFFFactoryFunction:
    """Test GeoTIFF factory function."""
    
    def test_create_geotiff_writer(self):
        """Test factory function creates writer with correct options."""
        from src.io.writers.geotiff_writer import create_geotiff_writer
        
        writer = create_geotiff_writer(
            crs='EPSG:4326',
            compress='deflate',
            dtype='int16',
            nodata_value=-9999
        )
        
        assert isinstance(writer, GeoTIFFWriter)
        assert writer.options.coordinate_system == 'EPSG:4326'
        assert writer.options.compress == 'deflate'
        assert writer.options.dtype == 'int16'
        assert writer.options.nodata_value == -9999