"""
VTK writer for exporting interpolation results to VTK format.
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import warnings

from .base import BaseWriter, ExportFormat, ExportOptions, GridData, PointData, ExportError

# Try to import VTK - it's optional
try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    vtk = None
    numpy_support = None
    VTK_AVAILABLE = False


@dataclass
class VTKExportOptions(ExportOptions):
    """
    VTK-specific export options.
    
    Attributes:
        binary: Use binary format (faster, smaller files)
        ascii: Use ASCII format (human-readable)
        point_data_name: Name for point data arrays
        cell_data_name: Name for cell data arrays
        field_data_name: Name for field data arrays
        ghost_levels: Number of ghost levels for parallel processing
        whole_extent: Whole extent for structured grids
        spacing: Grid spacing (for structured grids)
        origin: Grid origin (for structured grids)
    """
    binary: bool = True
    ascii: bool = False
    point_data_name: str = 'Interpolated_Values'
    cell_data_name: str = 'Cell_Data'
    field_data_name: str = 'Field_Data'
    ghost_levels: int = 0
    whole_extent: Optional[tuple] = None
    spacing: Optional[tuple] = None
    origin: Optional[tuple] = None


class VTKWriter(BaseWriter):
    """
    Writer for exporting data to VTK format.
    
    VTK (Visualization Toolkit) format is widely used for scientific visualization
    and supports both structured and unstructured data. This writer can export
    both grid and point data.
    
    Requires vtk library for functionality.
    """
    
    def __init__(self, options: Optional[VTKExportOptions] = None):
        """
        Initialize VTK writer.
        
        Args:
            options: VTK-specific export options
            
        Raises:
            ImportError: If VTK is not available
        """
        if not VTK_AVAILABLE:
            raise ImportError(
                "VTK library is required for VTK export. "
                "Install it with: pip install vtk"
            )
        
        if options is None:
            options = VTKExportOptions()
        elif not isinstance(options, VTKExportOptions):
            # Convert base options to VTK options
            vtk_options = VTKExportOptions()
            for field in options.__dataclass_fields__:
                if hasattr(options, field):
                    setattr(vtk_options, field, getattr(options, field))
            options = vtk_options
            
        super().__init__(options)
    
    @property
    def supported_formats(self) -> List[ExportFormat]:
        """Return list of formats supported by this writer."""
        return [ExportFormat.VTK]
    
    @property
    def file_extensions(self) -> List[str]:
        """Return list of file extensions for this writer."""
        return ['.vtk', '.vtp', '.vtu', '.vts', '.vtr']
    
    def write_grid(self, 
                   data: GridData, 
                   filepath: Union[str, Path],
                   **kwargs) -> None:
        """
        Write grid data to VTK file.
        
        Creates a structured grid (VTK format) suitable for visualization
        of regularly spaced data.
        
        Args:
            data: Grid data to export
            filepath: Output file path
            **kwargs: Additional VTK options
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Validate inputs
            self.validate_grid_data(data)
            filepath = self.validate_filepath(filepath)
            
            # Update options with kwargs
            vtk_options = self._update_options(**kwargs)
            
            # Create VTK structured grid
            vtk_grid = self._create_structured_grid(data)
            
            # Write to file
            self._write_vtk_data(vtk_grid, filepath, vtk_options)
            
        except Exception as e:
            raise ExportError(f"Failed to export grid data to VTK: {e}")
    
    def write_points(self, 
                     data: PointData, 
                     filepath: Union[str, Path],
                     **kwargs) -> None:
        """
        Write point data to VTK file.
        
        Creates an unstructured grid or polydata suitable for visualization
        of scattered point data.
        
        Args:
            data: Point data to export
            filepath: Output file path
            **kwargs: Additional VTK options
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Validate inputs
            self.validate_point_data(data)
            filepath = self.validate_filepath(filepath)
            
            # Update options with kwargs
            vtk_options = self._update_options(**kwargs)
            
            # Create VTK polydata
            vtk_polydata = self._create_polydata(data)
            
            # Write to file
            self._write_vtk_data(vtk_polydata, filepath, vtk_options)
            
        except Exception as e:
            raise ExportError(f"Failed to export point data to VTK: {e}")
    
    def _create_structured_grid(self, data: GridData) -> 'vtk.vtkStructuredGrid':
        """
        Create VTK structured grid from grid data.
        
        Args:
            data: Grid data to convert
            
        Returns:
            VTK structured grid object
        """
        if data.is_3d:
            return self._create_3d_structured_grid(data)
        else:
            return self._create_2d_structured_grid(data)
    
    def _create_2d_structured_grid(self, data: GridData) -> 'vtk.vtkStructuredGrid':
        """
        Create 2D VTK structured grid.
        
        Args:
            data: 2D grid data
            
        Returns:
            VTK structured grid
        """
        # Create coordinate arrays
        nx, ny = len(data.x_coords), len(data.y_coords)
        
        # Create VTK points
        points = vtk.vtkPoints()
        
        # Add points in VTK order (k varies fastest, then j, then i)
        for j in range(ny):
            for i in range(nx):
                x = data.x_coords[i]
                y = data.y_coords[j]
                z = 0.0  # 2D data, set Z to 0
                points.InsertNextPoint(x, y, z)
        
        # Create structured grid
        structured_grid = vtk.vtkStructuredGrid()
        structured_grid.SetDimensions(nx, ny, 1)
        structured_grid.SetPoints(points)
        
        # Add scalar data
        values_flat = data.values.flatten(order='F')  # Fortran order for VTK
        vtk_array = numpy_support.numpy_to_vtk(values_flat)
        vtk_array.SetName(self.options.point_data_name)
        structured_grid.GetPointData().SetScalars(vtk_array)
        
        return structured_grid
    
    def _create_3d_structured_grid(self, data: GridData) -> 'vtk.vtkStructuredGrid':
        """
        Create 3D VTK structured grid.
        
        Args:
            data: 3D grid data
            
        Returns:
            VTK structured grid
        """
        # Create coordinate arrays
        nx, ny, nz = len(data.x_coords), len(data.y_coords), len(data.z_coords)
        
        # Create VTK points
        points = vtk.vtkPoints()
        
        # Add points in VTK order (k varies fastest, then j, then i)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = data.x_coords[i]
                    y = data.y_coords[j]
                    z = data.z_coords[k]
                    points.InsertNextPoint(x, y, z)
        
        # Create structured grid
        structured_grid = vtk.vtkStructuredGrid()
        structured_grid.SetDimensions(nx, ny, nz)
        structured_grid.SetPoints(points)
        
        # Add scalar data
        values_flat = data.values.flatten(order='F')  # Fortran order for VTK
        vtk_array = numpy_support.numpy_to_vtk(values_flat)
        vtk_array.SetName(self.options.point_data_name)
        structured_grid.GetPointData().SetScalars(vtk_array)
        
        return structured_grid
    
    def _create_polydata(self, data: PointData) -> 'vtk.vtkPolyData':
        """
        Create VTK polydata from point data.
        
        Args:
            data: Point data to convert
            
        Returns:
            VTK polydata object
        """
        # Create VTK points
        points = vtk.vtkPoints()
        
        for i, coord in enumerate(data.coordinates):
            if data.is_3d:
                points.InsertNextPoint(coord[0], coord[1], coord[2])
            else:
                points.InsertNextPoint(coord[0], coord[1], 0.0)  # Add Z=0 for 2D data
        
        # Create vertices for the points
        vertices = vtk.vtkCellArray()
        for i in range(data.n_points):
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            vertices.InsertNextCell(vertex)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        
        # Add scalar data
        vtk_array = numpy_support.numpy_to_vtk(data.values)
        vtk_array.SetName(self.options.point_data_name)
        polydata.GetPointData().SetScalars(vtk_array)
        
        # Add additional attributes if present
        if data.attributes:
            for attr_name, attr_values in data.attributes.items():
                if len(attr_values) == data.n_points:
                    attr_array = numpy_support.numpy_to_vtk(attr_values)
                    attr_array.SetName(attr_name)
                    polydata.GetPointData().AddArray(attr_array)
        
        # Add point IDs if present
        if data.point_ids is not None:
            id_array = numpy_support.numpy_to_vtk(data.point_ids)
            id_array.SetName('Point_IDs')
            polydata.GetPointData().AddArray(id_array)
        
        return polydata
    
    def _write_vtk_data(self, 
                       vtk_data: Union['vtk.vtkStructuredGrid', 'vtk.vtkPolyData'], 
                       filepath: Path, 
                       options: VTKExportOptions) -> None:
        """
        Write VTK data object to file.
        
        Args:
            vtk_data: VTK data object to write
            filepath: Output file path
            options: Export options
        """
        # Determine writer based on data type and file extension
        writer = self._get_appropriate_writer(vtk_data, filepath)
        
        # Configure writer
        writer.SetFileName(str(filepath))
        writer.SetInputData(vtk_data)
        
        # Set format (binary or ASCII)
        if hasattr(writer, 'SetFileTypeToBinary') and options.binary:
            writer.SetFileTypeToBinary()
        elif hasattr(writer, 'SetFileTypeToASCII') and options.ascii:
            writer.SetFileTypeToASCII()
        
        # Add metadata if requested
        if self.options.include_metadata:
            self._add_field_data(vtk_data)
        
        # Write the file
        writer.Write()
    
    def _get_appropriate_writer(self, 
                              vtk_data: Union['vtk.vtkStructuredGrid', 'vtk.vtkPolyData'], 
                              filepath: Path) -> 'vtk.vtkWriter':
        """
        Get appropriate VTK writer based on data type and file extension.
        
        Args:
            vtk_data: VTK data object
            filepath: Output file path
            
        Returns:
            Appropriate VTK writer
        """
        extension = filepath.suffix.lower()
        
        if isinstance(vtk_data, vtk.vtkStructuredGrid):
            if extension == '.vts':
                return vtk.vtkXMLStructuredGridWriter()
            else:
                return vtk.vtkStructuredGridWriter()
        
        elif isinstance(vtk_data, vtk.vtkPolyData):
            if extension == '.vtp':
                return vtk.vtkXMLPolyDataWriter()
            else:
                return vtk.vtkPolyDataWriter()
        
        elif isinstance(vtk_data, vtk.vtkUnstructuredGrid):
            if extension == '.vtu':
                return vtk.vtkXMLUnstructuredGridWriter()
            else:
                return vtk.vtkUnstructuredGridWriter()
        
        else:
            # Default to generic writer
            return vtk.vtkDataSetWriter()
    
    def _add_field_data(self, vtk_data: 'vtk.vtkDataObject') -> None:
        """
        Add metadata as field data to VTK object.
        
        Args:
            vtk_data: VTK data object
        """
        field_data = vtk_data.GetFieldData()
        
        # Add creation info
        creation_info = vtk.vtkStringArray()
        creation_info.SetName("Creation_Info")
        creation_info.SetNumberOfValues(1)
        creation_info.SetValue(0, "Created by Coal Interpolation Tool")
        field_data.AddArray(creation_info)
        
        # Add custom attributes
        if self.options.custom_attributes:
            for key, value in self.options.custom_attributes.items():
                if isinstance(value, str):
                    str_array = vtk.vtkStringArray()
                    str_array.SetName(key)
                    str_array.SetNumberOfValues(1)
                    str_array.SetValue(0, value)
                    field_data.AddArray(str_array)
                elif isinstance(value, (int, float)):
                    num_array = vtk.vtkFloatArray()
                    num_array.SetName(key)
                    num_array.SetNumberOfValues(1)
                    num_array.SetValue(0, float(value))
                    field_data.AddArray(num_array)
    
    def _update_options(self, **kwargs) -> VTKExportOptions:
        """
        Update export options with provided kwargs.
        
        Args:
            **kwargs: Additional options
            
        Returns:
            Updated VTK export options
        """
        # Create a copy of current options
        options = VTKExportOptions()
        
        # Copy current values
        for field in self.options.__dataclass_fields__:
            if hasattr(self.options, field):
                setattr(options, field, getattr(self.options, field))
        
        # Update with kwargs
        for key, value in kwargs.items():
            if hasattr(options, key):
                setattr(options, key, value)
        
        return options
    
    def export_summary(self, 
                      data: Union[GridData, PointData], 
                      filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get summary information about the export.
        
        Args:
            data: Data to be exported
            filepath: Target file path
            
        Returns:
            Dictionary with export summary
        """
        filepath = Path(filepath)
        
        summary = {
            'format': 'VTK',
            'filepath': str(filepath),
            'data_type': 'grid' if isinstance(data, GridData) else 'points',
            'is_3d': data.is_3d,
            'bounds': data.bounds,
            'binary_format': self.options.binary,
            'point_data_name': self.options.point_data_name,
        }
        
        if isinstance(data, GridData):
            summary.update({
                'vtk_type': 'StructuredGrid',
                'grid_shape': data.shape,
                'cell_size': data.cell_size,
                'n_points': data.n_points,
            })
        else:
            summary.update({
                'vtk_type': 'PolyData',
                'n_points': data.n_points,
                'n_attributes': len(data.attributes) if data.attributes else 0,
            })
        
        return summary


def create_vtk_writer(binary: bool = True,
                     point_data_name: str = 'Interpolated_Values',
                     include_metadata: bool = True) -> VTKWriter:
    """
    Factory function to create a VTK writer with common options.
    
    Args:
        binary: Use binary format
        point_data_name: Name for point data arrays
        include_metadata: Include metadata in output
        
    Returns:
        Configured VTK writer
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK library is required for VTK export")
    
    options = VTKExportOptions(
        binary=binary,
        ascii=not binary,
        point_data_name=point_data_name,
        include_metadata=include_metadata
    )
    
    return VTKWriter(options)