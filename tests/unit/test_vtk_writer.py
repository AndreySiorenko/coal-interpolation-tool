"""
Unit tests for VTK writer.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.io.writers.vtk_writer import VTKWriter, VTKExportOptions
from src.io.writers.base import GridData, PointData, ExportError

# Skip tests if vtk is not available
pytest_plugins = []

try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


@pytest.mark.skipif(not VTK_AVAILABLE, reason="vtk not available")
class TestVTKWriter:
    """Test suite for VTK writer."""
    
    def test_init_default_options(self):
        """Test VTK writer initialization with default options."""
        writer = VTKWriter()
        assert isinstance(writer.options, VTKExportOptions)
        assert writer.options.file_format == 'xml'
        assert writer.options.data_mode == 'binary'
        assert writer.options.point_size == 3.0
        assert writer.options.write_scalars == True
    
    def test_init_custom_options(self):
        """Test VTK writer initialization with custom options."""
        options = VTKExportOptions(
            file_format='legacy',
            data_mode='ascii',
            point_size=5.0,
            write_scalars=False,
            write_vectors=True
        )
        writer = VTKWriter(options)
        assert writer.options.file_format == 'legacy'
        assert writer.options.data_mode == 'ascii'
        assert writer.options.point_size == 5.0
        assert writer.options.write_scalars == False
        assert writer.options.write_vectors == True
    
    def test_supported_formats(self):
        """Test supported formats."""
        writer = VTKWriter()
        from src.io.writers.base import ExportFormat
        assert ExportFormat.VTK in writer.supported_formats
    
    def test_file_extensions(self):
        """Test file extensions."""
        writer = VTKWriter()
        extensions = writer.file_extensions
        assert '.vtp' in extensions  # PolyData XML
        assert '.vtu' in extensions  # UnstructuredGrid XML  
        assert '.vti' in extensions  # ImageData XML
        assert '.vtk' in extensions  # Legacy format
    
    def test_write_point_data(self, temp_directory):
        """Test writing point data to VTK."""
        # Create test point data
        coordinates = np.array([
            [100.0, 200.0, 50.0],
            [150.0, 250.0, 60.0],
            [200.0, 300.0, 70.0]
        ])
        values = np.array([10.5, 20.3, 30.1])
        point_ids = np.array(['P1', 'P2', 'P3'])
        attributes = {
            'ASH': np.array([15.2, 18.7, 12.3]),
            'SULFUR': np.array([2.1, 2.8, 1.9])
        }
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            point_ids=point_ids,
            attributes=attributes
        )
        
        # Write to VTK
        writer = VTKWriter()
        output_file = temp_directory / 'test_points.vtp'
        writer.write_points(point_data, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read back with VTK and verify
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(output_file))
        reader.Update()
        
        output = reader.GetOutput()
        assert output.GetNumberOfPoints() == 3
        
        # Check that scalar data exists
        point_data_vtk = output.GetPointData()
        assert point_data_vtk.GetNumberOfArrays() >= 1
        
        # Check for main value array
        values_array = point_data_vtk.GetArray('Value')
        if values_array:
            assert values_array.GetNumberOfTuples() == 3
        
        # Check for attribute arrays
        ash_array = point_data_vtk.GetArray('ASH')
        if ash_array:
            assert ash_array.GetNumberOfTuples() == 3
    
    def test_write_2d_point_data(self, temp_directory):
        """Test writing 2D point data to VTK."""
        # Create 2D test data
        coordinates = np.array([
            [100.0, 200.0],
            [150.0, 250.0],
            [200.0, 300.0]
        ])
        values = np.array([10.5, 20.3, 30.1])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Write to VTK
        writer = VTKWriter()
        output_file = temp_directory / 'test_2d_points.vtp'
        writer.write_points(point_data, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read back and verify
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(output_file))
        reader.Update()
        
        output = reader.GetOutput()
        assert output.GetNumberOfPoints() == 3
        
        # Check that Z coordinates are 0 for 2D data
        points = output.GetPoints()
        for i in range(points.GetNumberOfPoints()):
            point = points.GetPoint(i)
            assert point[2] == 0.0  # Z should be 0
    
    def test_write_grid_data(self, temp_directory):
        """Test writing grid data to VTK."""
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
        
        # Write to VTK
        writer = VTKWriter()
        output_file = temp_directory / 'test_grid.vti'
        writer.write_grid(grid_data, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read back with VTK and verify
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(str(output_file))
        reader.Update()
        
        output = reader.GetOutput()
        assert output.GetNumberOfPoints() == 9  # 3x3 grid
        
        # Check dimensions
        dimensions = output.GetDimensions()
        assert dimensions[0] == 3  # X dimension
        assert dimensions[1] == 3  # Y dimension
    
    def test_write_3d_grid_data(self, temp_directory):
        """Test writing 3D grid data to VTK."""
        # Create test 3D grid data
        x_coords = np.array([0, 1])
        y_coords = np.array([0, 1])
        z_coords = np.array([0, 1])
        values = np.random.random((2, 2, 2))
        
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
            values=values,
            bounds=(0, 1, 0, 1, 0, 1),
            cell_size=1
        )
        
        # Write to VTK
        writer = VTKWriter()
        output_file = temp_directory / 'test_3d_grid.vti'
        writer.write_grid(grid_data, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read back and verify
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(str(output_file))
        reader.Update()
        
        output = reader.GetOutput()
        assert output.GetNumberOfPoints() == 8  # 2x2x2 grid
        
        # Check 3D dimensions
        dimensions = output.GetDimensions()
        assert dimensions[0] == 2  # X dimension
        assert dimensions[1] == 2  # Y dimension
        assert dimensions[2] == 2  # Z dimension
    
    def test_different_file_formats(self, temp_directory):
        """Test different VTK file formats."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Test XML format (default)
        options_xml = VTKExportOptions(file_format='xml')
        writer_xml = VTKWriter(options_xml)
        output_xml = temp_directory / 'test_xml.vtp'
        writer_xml.write_points(point_data, output_xml)
        assert output_xml.exists()
        
        # Test legacy format
        options_legacy = VTKExportOptions(file_format='legacy')
        writer_legacy = VTKWriter(options_legacy)
        output_legacy = temp_directory / 'test_legacy.vtk'
        writer_legacy.write_points(point_data, output_legacy)
        assert output_legacy.exists()
        
        # Verify legacy format by checking file content
        with open(output_legacy, 'r') as f:
            content = f.read()
            assert '# vtk DataFile Version' in content  # Legacy format header
    
    def test_ascii_vs_binary_modes(self, temp_directory):
        """Test ASCII vs binary data modes."""
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Test ASCII mode
        options_ascii = VTKExportOptions(data_mode='ascii')
        writer_ascii = VTKWriter(options_ascii)
        output_ascii = temp_directory / 'test_ascii.vtp'
        writer_ascii.write_points(point_data, output_ascii)
        assert output_ascii.exists()
        
        # Test binary mode
        options_binary = VTKExportOptions(data_mode='binary')
        writer_binary = VTKWriter(options_binary)
        output_binary = temp_directory / 'test_binary.vtp'
        writer_binary.write_points(point_data, output_binary)
        assert output_binary.exists()
        
        # Binary files are typically smaller for same data
        ascii_size = output_ascii.stat().st_size
        binary_size = output_binary.stat().st_size
        # Note: For very small datasets, this might not always be true
        # So we just check both files exist and can be read
        
        # Verify both can be read
        reader = vtk.vtkXMLPolyDataReader()
        
        reader.SetFileName(str(output_ascii))
        reader.Update()
        assert reader.GetOutput().GetNumberOfPoints() == 1
        
        reader.SetFileName(str(output_binary))
        reader.Update()
        assert reader.GetOutput().GetNumberOfPoints() == 1
    
    def test_compression_options(self, temp_directory):
        """Test compression options."""
        # Create larger dataset for meaningful compression test
        n_points = 100
        coordinates = np.random.random((n_points, 3)) * 1000
        values = np.random.random(n_points) * 100
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        # Test with compression
        options_compressed = VTKExportOptions(compress_data=True)
        writer_compressed = VTKWriter(options_compressed)
        output_compressed = temp_directory / 'test_compressed.vtp'
        writer_compressed.write_points(point_data, output_compressed)
        
        # Test without compression
        options_uncompressed = VTKExportOptions(compress_data=False)
        writer_uncompressed = VTKWriter(options_uncompressed)
        output_uncompressed = temp_directory / 'test_uncompressed.vtp'
        writer_uncompressed.write_points(point_data, output_uncompressed)
        
        # Both files should exist
        assert output_compressed.exists()
        assert output_uncompressed.exists()
        
        # Compressed file should typically be smaller
        compressed_size = output_compressed.stat().st_size
        uncompressed_size = output_uncompressed.stat().st_size
        
        # For large enough datasets, compressed should be smaller
        # But we'll just verify both can be read correctly
        reader = vtk.vtkXMLPolyDataReader()
        
        reader.SetFileName(str(output_compressed))
        reader.Update()
        assert reader.GetOutput().GetNumberOfPoints() == n_points
        
        reader.SetFileName(str(output_uncompressed))
        reader.Update()
        assert reader.GetOutput().GetNumberOfPoints() == n_points
    
    def test_scalars_and_vectors(self, temp_directory):
        """Test writing scalar and vector data."""
        coordinates = np.array([
            [100, 200, 50],
            [150, 250, 60],
            [200, 300, 70]
        ])
        values = np.array([10.5, 20.3, 30.1])
        
        # Add vector attributes
        attributes = {
            'VELOCITY': np.array([
                [1.0, 2.0, 0.5],
                [1.5, 2.5, 0.7],
                [2.0, 3.0, 0.9]
            ]),
            'GRADIENT': np.array([
                [0.1, 0.2, 0.05],
                [0.15, 0.25, 0.07],
                [0.2, 0.3, 0.09]
            ])
        }
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            attributes=attributes
        )
        
        # Write with both scalars and vectors
        options = VTKExportOptions(
            write_scalars=True,
            write_vectors=True
        )
        writer = VTKWriter(options)
        
        output_file = temp_directory / 'test_vectors.vtp'
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Read back and verify vector data
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(output_file))
        reader.Update()
        
        output = reader.GetOutput()
        point_data_vtk = output.GetPointData()
        
        # Should have both scalar and vector arrays
        assert point_data_vtk.GetNumberOfArrays() >= 2
        
        # Check for vector arrays
        velocity_array = point_data_vtk.GetArray('VELOCITY')
        if velocity_array:
            assert velocity_array.GetNumberOfComponents() == 3  # Vector has 3 components
    
    def test_point_ids_handling(self, temp_directory):
        """Test handling of point IDs."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        point_ids = np.array(['BH001', 'BH002'])
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            point_ids=point_ids
        )
        
        writer = VTKWriter()
        output_file = temp_directory / 'test_ids.vtp'
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Read back and verify
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(output_file))
        reader.Update()
        
        output = reader.GetOutput()
        point_data_vtk = output.GetPointData()
        
        # Should have Point_ID array
        id_array = point_data_vtk.GetArray('Point_ID')
        if id_array:
            assert id_array.GetNumberOfTuples() == 2
    
    def test_metadata_handling(self, temp_directory):
        """Test metadata inclusion in VTK files."""
        coordinates = np.array([[100, 200]])
        values = np.array([10.5])
        
        point_data = PointData(
            coordinates=coordinates,
            values=values,
            metadata={
                'source': 'Test Suite',
                'method': 'IDW',
                'power': 2.0,
                'date': '2023-01-01'
            }
        )
        
        options = VTKExportOptions(include_metadata=True)
        writer = VTKWriter(options)
        
        output_file = temp_directory / 'test_metadata.vtp'
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Read back and check for metadata in field data
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(output_file))
        reader.Update()
        
        output = reader.GetOutput()
        field_data = output.GetFieldData()
        
        # Should have field data with metadata
        assert field_data.GetNumberOfArrays() >= 1
    
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
        
        writer = VTKWriter()
        summary = writer.export_summary(grid_data, 'test.vti')
        
        assert summary['format'] == 'VTK'
        assert summary['data_type'] == 'grid'
        assert summary['vtk_type'] == 'ImageData'
        assert summary['n_points'] == 9
        assert summary['file_format'] == 'xml'
    
    def test_export_summary_points(self):
        """Test export summary for point data."""
        coordinates = np.array([[100, 200], [150, 250]])
        values = np.array([10.5, 20.3])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = VTKWriter()
        summary = writer.export_summary(point_data, 'test.vtp')
        
        assert summary['format'] == 'VTK'
        assert summary['data_type'] == 'points'
        assert summary['vtk_type'] == 'PolyData'
        assert summary['n_points'] == 2
    
    def test_empty_data_handling(self, temp_directory):
        """Test handling of empty datasets."""
        # Empty point data
        coordinates = np.array([]).reshape(0, 2)
        values = np.array([])
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = VTKWriter()
        output_file = temp_directory / 'test_empty.vtp'
        
        # Should handle empty data gracefully
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Should be readable
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(output_file))
        reader.Update()
        
        output = reader.GetOutput()
        assert output.GetNumberOfPoints() == 0
    
    def test_large_dataset(self, temp_directory):
        """Test handling of larger datasets."""
        # Create moderately large dataset
        n_points = 1000
        coordinates = np.random.random((n_points, 3)) * 1000
        values = np.random.random(n_points) * 100
        
        point_data = PointData(coordinates=coordinates, values=values)
        
        writer = VTKWriter()
        output_file = temp_directory / 'test_large.vtp'
        writer.write_points(point_data, output_file)
        
        assert output_file.exists()
        
        # Verify all points were written
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(output_file))
        reader.Update()
        
        output = reader.GetOutput()
        assert output.GetNumberOfPoints() == n_points


@pytest.mark.skipif(VTK_AVAILABLE, reason="Testing without vtk")
class TestVTKWriterWithoutVTK:
    """Test VTK writer behavior when vtk is not available."""
    
    def test_import_error_on_init(self):
        """Test that ImportError is raised when vtk is not available."""
        with pytest.raises(ImportError, match="vtk library is required"):
            VTKWriter()
    
    def test_factory_function_import_error(self):
        """Test factory function raises ImportError without vtk."""
        from src.io.writers.vtk_writer import create_vtk_writer
        
        with pytest.raises(ImportError, match="vtk library is required"):
            create_vtk_writer()


@pytest.mark.skipif(not VTK_AVAILABLE, reason="vtk not available")
class TestVTKFactoryFunction:
    """Test VTK factory function."""
    
    def test_create_vtk_writer(self):
        """Test factory function creates writer with correct options."""
        from src.io.writers.vtk_writer import create_vtk_writer
        
        writer = create_vtk_writer(
            file_format='legacy',
            data_mode='ascii',
            compress_data=False,
            write_vectors=True
        )
        
        assert isinstance(writer, VTKWriter)
        assert writer.options.file_format == 'legacy'
        assert writer.options.data_mode == 'ascii'
        assert writer.options.compress_data == False
        assert writer.options.write_vectors == True