"""
GeoTIFF writer for exporting interpolation results to GeoTIFF format.
"""

from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import warnings
import pandas as pd

from .base import BaseWriter, ExportFormat, ExportOptions, GridData, PointData, ExportError

# Try to import rasterio - it's optional
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    RASTERIO_AVAILABLE = False


@dataclass
class GeoTIFFExportOptions(ExportOptions):
    """
    GeoTIFF-specific export options.
    
    Attributes:
        crs: Coordinate Reference System (EPSG code or WKT string)
        nodata_value: Value to use for NoData pixels
        compress: Compression method ('none', 'lzw', 'jpeg', 'deflate')
        tiled: Create tiled TIFF
        blockxsize: Tile width (if tiled)
        blockysize: Tile height (if tiled)
        bigtiff: Create BigTIFF format for large files
        photometric: Photometric interpretation
        predictor: Predictor for compression
        interleave: Band interleaving ('pixel', 'band')
        dtype: Output data type (numpy dtype)
    """
    crs: Optional[str] = None
    nodata_value: Optional[float] = None
    compress: str = 'lzw'
    tiled: bool = True
    blockxsize: int = 512
    blockysize: int = 512
    bigtiff: bool = False
    photometric: Optional[str] = None
    predictor: Optional[int] = None
    interleave: str = 'band'
    dtype: str = 'float32'


class GeoTIFFWriter(BaseWriter):
    """
    Writer for exporting grid data to GeoTIFF format.
    
    GeoTIFF is a georeferenced raster format widely used in GIS applications.
    This writer supports only grid data as point data cannot be directly
    represented in raster format.
    
    Requires rasterio library for functionality.
    """
    
    def __init__(self, options: Optional[GeoTIFFExportOptions] = None):
        """
        Initialize GeoTIFF writer.
        
        Args:
            options: GeoTIFF-specific export options
            
        Raises:
            ImportError: If rasterio is not available
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError(
                "rasterio library is required for GeoTIFF export. "
                "Install it with: pip install rasterio"
            )
        
        if options is None:
            options = GeoTIFFExportOptions()
        elif not isinstance(options, GeoTIFFExportOptions):
            # Convert base options to GeoTIFF options
            geotiff_options = GeoTIFFExportOptions()
            for field in options.__dataclass_fields__:
                if hasattr(options, field):
                    setattr(geotiff_options, field, getattr(options, field))
            options = geotiff_options
            
        super().__init__(options)
    
    @property
    def supported_formats(self) -> List[ExportFormat]:
        """Return list of formats supported by this writer."""
        return [ExportFormat.GEOTIFF]
    
    @property
    def file_extensions(self) -> List[str]:
        """Return list of file extensions for this writer."""
        return ['.tif', '.tiff', '.gtiff']
    
    def write_grid(self, 
                   data: GridData, 
                   filepath: Union[str, Path],
                   **kwargs) -> None:
        """
        Write grid data to GeoTIFF file.
        
        Args:
            data: Grid data to export
            filepath: Output file path
            **kwargs: Additional GeoTIFF options
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Validate inputs
            self.validate_grid_data(data)
            filepath = self.validate_filepath(filepath)
            
            # Update options with kwargs
            geotiff_options = self._update_options(**kwargs)
            
            # Check if 3D data (not supported for single-band GeoTIFF)
            if data.is_3d:
                raise ExportError("3D grid data cannot be exported to single-band GeoTIFF")
            
            # Prepare raster data and metadata
            raster_data, profile = self._prepare_raster_data(data, geotiff_options)
            
            # Write to GeoTIFF
            self._write_geotiff(raster_data, profile, filepath)
            
        except Exception as e:
            raise ExportError(f"Failed to export grid data to GeoTIFF: {e}")
    
    def write_points(self, 
                     data: PointData, 
                     filepath: Union[str, Path],
                     **kwargs) -> None:
        """
        Write point data to GeoTIFF file.
        
        Point data is first interpolated to a regular grid before export.
        This is a convenience method that performs gridding automatically.
        
        Args:
            data: Point data to export
            filepath: Output file path
            **kwargs: Additional options including gridding parameters
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Extract gridding parameters
            cell_size = kwargs.get('cell_size', self._estimate_cell_size(data))
            buffer_factor = kwargs.get('buffer_factor', 0.1)
            
            # Convert points to grid
            grid_data = self._points_to_grid(data, cell_size, buffer_factor)
            
            # Export the grid
            self.write_grid(grid_data, filepath, **kwargs)
            
        except Exception as e:
            raise ExportError(f"Failed to export point data to GeoTIFF: {e}")
    
    def _prepare_raster_data(self, 
                           data: GridData, 
                           options: GeoTIFFExportOptions) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare raster data and metadata for writing.
        
        Args:
            data: Grid data to prepare
            options: Export options
            
        Returns:
            Tuple of (raster_data, rasterio_profile)
        """
        # Get raster array - rasterio expects (bands, height, width)
        # Our grid data is (height, width) so we add band dimension
        raster_data = data.values[np.newaxis, :, :]  # Add band dimension
        
        # Handle data type conversion
        if options.dtype:
            try:
                raster_data = raster_data.astype(options.dtype)
            except (ValueError, TypeError) as e:
                warnings.warn(f"Could not convert to {options.dtype}, using float32: {e}")
                raster_data = raster_data.astype('float32')
        
        # Handle NoData values
        if options.nodata_value is not None:
            # Replace NaN values with NoData value
            raster_data = np.where(np.isnan(raster_data), options.nodata_value, raster_data)
        
        # Calculate geotransform
        transform = self._calculate_transform(data)
        
        # Determine CRS
        crs = self._determine_crs(data, options)
        
        # Create rasterio profile
        height, width = data.values.shape
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,  # Single band
            'dtype': raster_data.dtype,
            'crs': crs,
            'transform': transform,
        }
        
        # Add compression and tiling options
        if options.compress and options.compress != 'none':
            profile['compress'] = options.compress
            
        if options.tiled:
            profile['tiled'] = True
            profile['blockxsize'] = min(options.blockxsize, width)
            profile['blockysize'] = min(options.blockysize, height)
        
        if options.bigtiff:
            profile['BIGTIFF'] = 'YES'
        
        if options.photometric:
            profile['photometric'] = options.photometric
            
        if options.predictor:
            profile['predictor'] = options.predictor
            
        profile['interleave'] = options.interleave
        
        # Add NoData value to profile if specified
        if options.nodata_value is not None:
            profile['nodata'] = options.nodata_value
        
        return raster_data, profile
    
    def _calculate_transform(self, data: GridData) -> 'rasterio.Affine':
        """
        Calculate the affine transform for the grid.
        
        Args:
            data: Grid data
            
        Returns:
            Rasterio affine transform
        """
        # Get bounds
        xmin, xmax, ymin, ymax = data.bounds[:4]
        
        # Calculate cell size
        if isinstance(data.cell_size, (list, tuple)):
            cell_size_x, cell_size_y = data.cell_size
        else:
            cell_size_x = cell_size_y = data.cell_size
        
        # Create transform using rasterio
        # Note: rasterio expects (width, height) but we have (len(x), len(y))
        height, width = data.values.shape
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
        
        return transform
    
    def _determine_crs(self, data: GridData, options: GeoTIFFExportOptions) -> Optional['CRS']:
        """
        Determine the coordinate reference system.
        
        Args:
            data: Grid data
            options: Export options
            
        Returns:
            Rasterio CRS object or None
        """
        # Priority: options > data.coordinate_system > None
        crs_string = options.coordinate_system or data.coordinate_system
        
        if crs_string:
            try:
                return CRS.from_string(crs_string)
            except Exception as e:
                warnings.warn(f"Could not parse CRS '{crs_string}': {e}")
                return None
        
        return None
    
    def _write_geotiff(self, 
                      raster_data: np.ndarray, 
                      profile: Dict[str, Any], 
                      filepath: Path) -> None:
        """
        Write raster data to GeoTIFF file.
        
        Args:
            raster_data: Raster data array
            profile: Rasterio profile
            filepath: Output file path
        """
        with rasterio.open(filepath, 'w', **profile) as dst:
            # Write the data
            dst.write(raster_data)
            
            # Add metadata if available
            if self.options.include_metadata:
                metadata = self._prepare_metadata()
                if metadata:
                    dst.update_tags(**metadata)
    
    def _prepare_metadata(self) -> Dict[str, str]:
        """
        Prepare metadata for GeoTIFF tags.
        
        Returns:
            Dictionary of metadata tags
        """
        metadata = {
            'CREATED_BY': 'Coal Interpolation Tool',
            'CREATION_DATE': pd.Timestamp.now().isoformat(),
        }
        
        # Add custom attributes as metadata
        if self.options.custom_attributes:
            for key, value in self.options.custom_attributes.items():
                if isinstance(value, (str, int, float)):
                    metadata[f'CUSTOM_{key.upper()}'] = str(value)
        
        return metadata
    
    def _points_to_grid(self, 
                       data: PointData, 
                       cell_size: float, 
                       buffer_factor: float = 0.1) -> GridData:
        """
        Convert point data to grid data using simple gridding.
        
        Args:
            data: Point data to grid
            cell_size: Grid cell size
            buffer_factor: Buffer around data bounds
            
        Returns:
            GridData object
        """
        # This is a simplified gridding - in practice, you might want
        # to use more sophisticated interpolation methods
        
        # Calculate bounds with buffer
        bounds = data.bounds
        if data.is_3d:
            xmin, xmax, ymin, ymax = bounds[:4]
        else:
            xmin, xmax, ymin, ymax = bounds
        
        # Add buffer
        x_buffer = (xmax - xmin) * buffer_factor
        y_buffer = (ymax - ymin) * buffer_factor
        xmin -= x_buffer
        xmax += x_buffer
        ymin -= y_buffer
        ymax += y_buffer
        
        # Create grid coordinates
        x_coords = np.arange(xmin, xmax + cell_size, cell_size)
        y_coords = np.arange(ymin, ymax + cell_size, cell_size)
        
        # Create empty grid
        grid_values = np.full((len(y_coords), len(x_coords)), np.nan)
        
        # Simple nearest-neighbor gridding
        for i, point in enumerate(data.coordinates):
            x, y = point[:2]
            
            # Find nearest grid cell
            x_idx = np.argmin(np.abs(x_coords - x))
            y_idx = np.argmin(np.abs(y_coords - y))
            
            if 0 <= x_idx < len(x_coords) and 0 <= y_idx < len(y_coords):
                grid_values[y_idx, x_idx] = data.values[i]
        
        # Create GridData object
        return GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=grid_values,
            bounds=(xmin, xmax, ymin, ymax),
            cell_size=cell_size,
            coordinate_system=data.coordinate_system,
            metadata=data.metadata.copy() if data.metadata else {}
        )
    
    def _estimate_cell_size(self, data: PointData) -> float:
        """
        Estimate appropriate cell size for gridding point data.
        
        Args:
            data: Point data
            
        Returns:
            Estimated cell size
        """
        bounds = data.bounds
        if data.is_3d:
            x_range = bounds[1] - bounds[0]
            y_range = bounds[3] - bounds[2]
        else:
            x_range = bounds[1] - bounds[0]
            y_range = bounds[3] - bounds[2]
        
        # Use ~1/100 of the smaller dimension
        return min(x_range, y_range) / 100.0
    
    def _update_options(self, **kwargs) -> GeoTIFFExportOptions:
        """
        Update export options with provided kwargs.
        
        Args:
            **kwargs: Additional options
            
        Returns:
            Updated GeoTIFF export options
        """
        # Create a copy of current options
        options = GeoTIFFExportOptions()
        
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
            'format': 'GeoTIFF',
            'filepath': str(filepath),
            'data_type': 'grid' if isinstance(data, GridData) else 'points',
            'bounds': data.bounds,
            'compression': self.options.compress,
            'dtype': self.options.dtype,
            'tiled': self.options.tiled,
        }
        
        if isinstance(data, GridData):
            summary.update({
                'grid_shape': data.shape,
                'cell_size': data.cell_size,
                'n_pixels': data.n_points,
            })
        else:
            summary.update({
                'n_points': data.n_points,
                'gridded': True,  # Points will be gridded
                'estimated_cell_size': self._estimate_cell_size(data),
            })
        
        if hasattr(data, 'coordinate_system') and data.coordinate_system:
            summary['coordinate_system'] = data.coordinate_system or self.options.coordinate_system
        
        return summary


def create_geotiff_writer(crs: Optional[str] = None,
                         compress: str = 'lzw',
                         dtype: str = 'float32',
                         nodata_value: Optional[float] = None) -> GeoTIFFWriter:
    """
    Factory function to create a GeoTIFF writer with common options.
    
    Args:
        crs: Coordinate reference system
        compress: Compression method
        dtype: Output data type
        nodata_value: NoData value
        
    Returns:
        Configured GeoTIFF writer
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio library is required for GeoTIFF export")
    
    options = GeoTIFFExportOptions(
        coordinate_system=crs,
        compress=compress,
        dtype=dtype,
        nodata_value=nodata_value
    )
    
    return GeoTIFFWriter(options)