"""
Base classes and interfaces for data export functionality.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from enum import Enum


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    GEOTIFF = "geotiff"
    VTK = "vtk"
    DXF = "dxf"
    JSON = "json"


@dataclass
class GridData:
    """
    Data structure for grid-based interpolation results.
    
    Attributes:
        x_coords: X coordinate array
        y_coords: Y coordinate array
        z_coords: Z coordinate array (optional, for 3D grids)
        values: Interpolated values array
        bounds: Grid bounds (xmin, xmax, ymin, ymax, zmin, zmax)
        cell_size: Grid cell size
        coordinate_system: Coordinate reference system (e.g., 'EPSG:4326')
        metadata: Additional metadata dictionary
    """
    x_coords: np.ndarray
    y_coords: np.ndarray
    values: np.ndarray
    bounds: tuple
    cell_size: Union[float, tuple]
    z_coords: Optional[np.ndarray] = None
    coordinate_system: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate grid data after initialization."""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_3d(self) -> bool:
        """Check if this is 3D grid data."""
        return self.z_coords is not None
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the grid."""
        if self.is_3d:
            return (len(self.x_coords), len(self.y_coords), len(self.z_coords))
        else:
            return (len(self.x_coords), len(self.y_coords))
    
    @property
    def n_points(self) -> int:
        """Get total number of grid points."""
        return np.prod(self.shape)


@dataclass
class PointData:
    """
    Data structure for scattered point data.
    
    Attributes:
        coordinates: Array of point coordinates (N x 2 or N x 3)
        values: Array of values at each point
        point_ids: Optional point identifiers
        attributes: Additional point attributes
        coordinate_system: Coordinate reference system
        metadata: Additional metadata dictionary
    """
    coordinates: np.ndarray
    values: np.ndarray
    point_ids: Optional[np.ndarray] = None
    attributes: Optional[Dict[str, np.ndarray]] = None
    coordinate_system: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate point data after initialization."""
        if self.attributes is None:
            self.attributes = {}
        if self.metadata is None:
            self.metadata = {}
        
        # Validate dimensions
        if len(self.coordinates) != len(self.values):
            raise ValueError("Coordinates and values must have same length")
    
    @property
    def is_3d(self) -> bool:
        """Check if this is 3D point data."""
        return self.coordinates.shape[1] == 3
    
    @property
    def n_points(self) -> int:
        """Get number of points."""
        return len(self.coordinates)
    
    @property
    def bounds(self) -> tuple:
        """Get data bounds."""
        if self.is_3d:
            return (
                float(self.coordinates[:, 0].min()),
                float(self.coordinates[:, 0].max()),
                float(self.coordinates[:, 1].min()),
                float(self.coordinates[:, 1].max()),
                float(self.coordinates[:, 2].min()),
                float(self.coordinates[:, 2].max())
            )
        else:
            return (
                float(self.coordinates[:, 0].min()),
                float(self.coordinates[:, 0].max()),
                float(self.coordinates[:, 1].min()),
                float(self.coordinates[:, 1].max())
            )


@dataclass
class ExportOptions:
    """
    Base class for export options.
    
    Attributes:
        overwrite_existing: Whether to overwrite existing files
        create_directories: Whether to create output directories if they don't exist
        compression: Compression method (format-specific)
        precision: Numeric precision for output
        include_metadata: Whether to export metadata
        coordinate_system: Target coordinate system
        custom_attributes: Custom attributes to include
    """
    overwrite_existing: bool = True
    create_directories: bool = True
    compression: Optional[str] = None
    precision: int = 6
    include_metadata: bool = True
    coordinate_system: Optional[str] = None
    custom_attributes: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize custom attributes if None."""
        if self.custom_attributes is None:
            self.custom_attributes = {}


class ExportError(Exception):
    """Base exception for export-related errors."""
    pass


class UnsupportedFormatError(ExportError):
    """Raised when attempting to export to an unsupported format."""
    pass


class InvalidDataError(ExportError):
    """Raised when input data is invalid for export."""
    pass


class BaseWriter(ABC):
    """
    Abstract base class for all data writers.
    
    This class defines the common interface that all export writers must implement.
    Writers handle the conversion of interpolation results to specific file formats.
    """
    
    def __init__(self, options: Optional[ExportOptions] = None):
        """
        Initialize the writer with export options.
        
        Args:
            options: Export options specific to this writer
        """
        self.options = options or ExportOptions()
        
    @property
    @abstractmethod
    def supported_formats(self) -> List[ExportFormat]:
        """Return list of formats supported by this writer."""
        pass
    
    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Return list of file extensions for this writer."""
        pass
    
    @abstractmethod
    def write_grid(self, 
                   data: GridData, 
                   filepath: Union[str, Path],
                   **kwargs) -> None:
        """
        Write grid data to file.
        
        Args:
            data: Grid data to export
            filepath: Output file path
            **kwargs: Additional format-specific options
            
        Raises:
            ExportError: If export fails
        """
        pass
    
    @abstractmethod
    def write_points(self, 
                     data: PointData, 
                     filepath: Union[str, Path],
                     **kwargs) -> None:
        """
        Write point data to file.
        
        Args:
            data: Point data to export
            filepath: Output file path
            **kwargs: Additional format-specific options
            
        Raises:
            ExportError: If export fails
        """
        pass
    
    def can_export_format(self, format_type: ExportFormat) -> bool:
        """
        Check if this writer can export the given format.
        
        Args:
            format_type: Format to check
            
        Returns:
            True if format is supported
        """
        return format_type in self.supported_formats
    
    def validate_filepath(self, filepath: Union[str, Path]) -> Path:
        """
        Validate and prepare output file path.
        
        Args:
            filepath: Output file path
            
        Returns:
            Validated Path object
            
        Raises:
            ExportError: If path validation fails
        """
        filepath = Path(filepath)
        
        # Check if file extension is supported
        if filepath.suffix.lower().lstrip('.') not in [ext.lstrip('.') for ext in self.file_extensions]:
            raise ExportError(f"Unsupported file extension: {filepath.suffix}")
        
        # Create parent directories if needed
        if self.options.create_directories:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and handle overwrite
        if filepath.exists() and not self.options.overwrite_existing:
            raise ExportError(f"File already exists and overwrite is disabled: {filepath}")
        
        return filepath
    
    def validate_grid_data(self, data: GridData) -> None:
        """
        Validate grid data before export.
        
        Args:
            data: Grid data to validate
            
        Raises:
            InvalidDataError: If data is invalid
        """
        if data.values is None or len(data.values) == 0:
            raise InvalidDataError("Grid data values cannot be empty")
        
        if data.x_coords is None or len(data.x_coords) == 0:
            raise InvalidDataError("Grid X coordinates cannot be empty")
        
        if data.y_coords is None or len(data.y_coords) == 0:
            raise InvalidDataError("Grid Y coordinates cannot be empty")
        
        # Check dimensions consistency
        expected_shape = (len(data.y_coords), len(data.x_coords))
        if data.is_3d:
            expected_shape = (len(data.z_coords), len(data.y_coords), len(data.x_coords))
        
        if data.values.shape != expected_shape:
            raise InvalidDataError(
                f"Grid values shape {data.values.shape} doesn't match "
                f"coordinate shape {expected_shape}"
            )
    
    def validate_point_data(self, data: PointData) -> None:
        """
        Validate point data before export.
        
        Args:
            data: Point data to validate
            
        Raises:
            InvalidDataError: If data is invalid
        """
        if data.coordinates is None or len(data.coordinates) == 0:
            raise InvalidDataError("Point coordinates cannot be empty")
        
        if data.values is None or len(data.values) == 0:
            raise InvalidDataError("Point values cannot be empty")
        
        if len(data.coordinates) != len(data.values):
            raise InvalidDataError("Coordinates and values must have same length")
        
        # Check coordinate dimensions
        if data.coordinates.ndim != 2 or data.coordinates.shape[1] not in [2, 3]:
            raise InvalidDataError("Coordinates must be N x 2 or N x 3 array")
    
    def get_export_info(self) -> Dict[str, Any]:
        """
        Get information about this writer's capabilities.
        
        Returns:
            Dictionary with writer information
        """
        return {
            'name': self.__class__.__name__,
            'supported_formats': [fmt.value for fmt in self.supported_formats],
            'file_extensions': self.file_extensions,
            'supports_grid': hasattr(self, 'write_grid'),
            'supports_points': hasattr(self, 'write_points'),
            'supports_3d': getattr(self, 'supports_3d', False),
        }


class WriterRegistry:
    """
    Registry for managing available writers.
    
    This class maintains a registry of available writers and provides
    methods to find appropriate writers for specific formats.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._writers: Dict[ExportFormat, List[BaseWriter]] = {}
    
    def register_writer(self, writer_class: type, options: Optional[ExportOptions] = None):
        """
        Register a writer class.
        
        Args:
            writer_class: Writer class to register
            options: Default options for this writer
        """
        writer = writer_class(options)
        
        for fmt in writer.supported_formats:
            if fmt not in self._writers:
                self._writers[fmt] = []
            self._writers[fmt].append(writer)
    
    def get_writer(self, format_type: ExportFormat) -> Optional[BaseWriter]:
        """
        Get a writer for the specified format.
        
        Args:
            format_type: Desired export format
            
        Returns:
            Writer instance or None if not available
        """
        writers = self._writers.get(format_type, [])
        return writers[0] if writers else None
    
    def get_writers(self, format_type: ExportFormat) -> List[BaseWriter]:
        """
        Get all writers for the specified format.
        
        Args:
            format_type: Desired export format
            
        Returns:
            List of writer instances
        """
        return self._writers.get(format_type, [])
    
    def get_supported_formats(self) -> List[ExportFormat]:
        """
        Get list of all supported formats.
        
        Returns:
            List of supported export formats
        """
        return list(self._writers.keys())
    
    def get_file_extensions(self, format_type: ExportFormat) -> List[str]:
        """
        Get file extensions for a specific format.
        
        Args:
            format_type: Export format
            
        Returns:
            List of file extensions
        """
        writers = self._writers.get(format_type, [])
        extensions = []
        for writer in writers:
            extensions.extend(writer.file_extensions)
        return list(set(extensions))  # Remove duplicates


# Global writer registry instance
writer_registry = WriterRegistry()