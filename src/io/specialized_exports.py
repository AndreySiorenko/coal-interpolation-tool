"""
Specialized export formats for geological and mining software.

Provides export capabilities for:
- Surfer grid files (.grd)
- Golden Software formats
- NetCDF files (.nc)
- Other industry-standard formats
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
import logging
from pathlib import Path
from abc import ABC, abstractmethod
import struct
from datetime import datetime


class BaseExporter(ABC):
    """Base class for specialized exporters."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def export(self, data: Union[pd.DataFrame, np.ndarray], 
               file_path: str, **kwargs):
        """Export data to specialized format."""
        pass
    
    @abstractmethod
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate data format for export."""
        pass


class SurferExporter(BaseExporter):
    """
    Exporter for Surfer grid files (.grd).
    
    Surfer is a popular contouring and 3D surface mapping software.
    Supports both ASCII and binary Surfer 6 format.
    """
    
    def __init__(self, format_type: str = 'binary'):
        super().__init__()
        self.format_type = format_type.lower()
        if self.format_type not in ['ascii', 'binary']:
            raise ValueError("Format type must be 'ascii' or 'binary'")
    
    def export(self, data: Union[pd.DataFrame, np.ndarray], 
               file_path: str, 
               x_col: str = 'x',
               y_col: str = 'y', 
               z_col: str = 'z',
               grid_method: str = 'auto',
               **kwargs):
        """
        Export data to Surfer grid format.
        
        Args:
            data: DataFrame with coordinates and values or 2D array
            file_path: Output file path
            x_col: X coordinate column name
            y_col: Y coordinate column name  
            z_col: Z value column name
            grid_method: Gridding method ('auto', 'regular', 'irregular')
            **kwargs: Additional parameters
        """
        self.logger.info(f"Exporting Surfer grid to: {file_path}")
        
        if not self.validate_data(data):
            raise ValueError("Invalid data format for Surfer export")
        
        if isinstance(data, pd.DataFrame):
            grid_data = self._prepare_grid_from_dataframe(data, x_col, y_col, z_col, grid_method)
        else:
            grid_data = self._prepare_grid_from_array(data)
        
        if self.format_type == 'ascii':
            self._export_ascii(grid_data, file_path)
        else:
            self._export_binary(grid_data, file_path)
        
        self.logger.info("Surfer export completed successfully")
    
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate data for Surfer export."""
        if isinstance(data, pd.DataFrame):
            return len(data) > 0 and not data.empty
        elif isinstance(data, np.ndarray):
            return data.size > 0 and data.ndim == 2
        return False
    
    def _prepare_grid_from_dataframe(self, df: pd.DataFrame, 
                                   x_col: str, y_col: str, z_col: str,
                                   grid_method: str) -> Dict[str, Any]:
        """Prepare grid data from DataFrame."""
        
        if grid_method == 'regular':
            # Assume data is on regular grid
            x_unique = np.sort(df[x_col].unique())
            y_unique = np.sort(df[y_col].unique())
            
            nx, ny = len(x_unique), len(y_unique)
            
            # Create grid
            z_grid = np.full((ny, nx), np.nan)
            
            for _, row in df.iterrows():
                x_idx = np.searchsorted(x_unique, row[x_col])
                y_idx = np.searchsorted(y_unique, row[y_col])
                if x_idx < nx and y_idx < ny:
                    z_grid[y_idx, x_idx] = row[z_col]
        
        else:
            # Auto or irregular - create regular grid through interpolation
            x_min, x_max = df[x_col].min(), df[x_col].max()
            y_min, y_max = df[y_col].min(), df[y_col].max()
            
            # Determine grid resolution
            n_points = len(df)
            grid_size = min(100, max(20, int(np.sqrt(n_points))))
            
            x_unique = np.linspace(x_min, x_max, grid_size)
            y_unique = np.linspace(y_min, y_max, grid_size)
            nx, ny = len(x_unique), len(y_unique)
            
            # Simple gridding using nearest neighbor
            X, Y = np.meshgrid(x_unique, y_unique)
            z_grid = np.full((ny, nx), np.nan)
            
            from scipy.spatial import cKDTree
            
            # Build tree for data points
            data_points = df[[x_col, y_col]].values
            data_values = df[z_col].values
            tree = cKDTree(data_points)
            
            # Interpolate to grid
            for i in range(ny):
                for j in range(nx):
                    grid_point = [X[i, j], Y[i, j]]
                    distance, idx = tree.query(grid_point)
                    if distance < (x_max - x_min) / grid_size * 2:  # Reasonable distance
                        z_grid[i, j] = data_values[idx]
        
        return {
            'x': x_unique,
            'y': y_unique,
            'z': z_grid,
            'nx': nx,
            'ny': ny,
            'x_min': float(x_unique[0]),
            'x_max': float(x_unique[-1]),
            'y_min': float(y_unique[0]),
            'y_max': float(y_unique[-1]),
            'z_min': float(np.nanmin(z_grid)),
            'z_max': float(np.nanmax(z_grid))
        }
    
    def _prepare_grid_from_array(self, array: np.ndarray) -> Dict[str, Any]:
        """Prepare grid data from 2D array."""
        ny, nx = array.shape
        
        # Assume unit spacing
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)
        
        return {
            'x': x,
            'y': y,
            'z': array,
            'nx': nx,
            'ny': ny,
            'x_min': 0.0,
            'x_max': float(nx - 1),
            'y_min': 0.0,
            'y_max': float(ny - 1),
            'z_min': float(np.nanmin(array)),
            'z_max': float(np.nanmax(array))
        }
    
    def _export_ascii(self, grid_data: Dict[str, Any], file_path: str):
        """Export ASCII Surfer format."""
        with open(file_path, 'w') as f:
            # Header
            f.write("DSAA\n")  # Surfer ASCII grid identifier
            f.write(f"{grid_data['nx']} {grid_data['ny']}\n")
            f.write(f"{grid_data['x_min']:.6f} {grid_data['x_max']:.6f}\n")
            f.write(f"{grid_data['y_min']:.6f} {grid_data['y_max']:.6f}\n")
            f.write(f"{grid_data['z_min']:.6f} {grid_data['z_max']:.6f}\n")
            
            # Data (row by row, bottom to top)
            z_grid = grid_data['z']
            for i in range(grid_data['ny']):
                row = z_grid[grid_data['ny'] - 1 - i, :]  # Flip Y
                for j in range(grid_data['nx']):
                    if np.isnan(row[j]):
                        f.write("1.70141e+038 ")  # Surfer blank value
                    else:
                        f.write(f"{row[j]:.6f} ")
                f.write("\n")
    
    def _export_binary(self, grid_data: Dict[str, Any], file_path: str):
        """Export binary Surfer format."""
        with open(file_path, 'wb') as f:
            # Header (Surfer 6 binary format)
            f.write(b'DSBB')  # Binary grid identifier
            
            # Grid dimensions
            f.write(struct.pack('<HH', grid_data['nx'], grid_data['ny']))
            
            # Extents
            f.write(struct.pack('<dddd', 
                              grid_data['x_min'], grid_data['x_max'],
                              grid_data['y_min'], grid_data['y_max']))
            
            # Z range
            f.write(struct.pack('<dd', grid_data['z_min'], grid_data['z_max']))
            
            # Data (as float32)
            z_grid = grid_data['z']
            blank_value = 1.70141e+038
            
            for i in range(grid_data['ny']):
                row = z_grid[grid_data['ny'] - 1 - i, :]  # Flip Y
                for j in range(grid_data['nx']):
                    if np.isnan(row[j]):
                        f.write(struct.pack('<f', blank_value))
                    else:
                        f.write(struct.pack('<f', float(row[j])))


class NetCDFExporter(BaseExporter):
    """
    Exporter for NetCDF format.
    
    NetCDF is a scientific data format commonly used in atmospheric
    and oceanic sciences, and increasingly in earth sciences.
    """
    
    def __init__(self):
        super().__init__()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import netCDF4
            import xarray as xr
            self.netCDF4 = netCDF4
            self.xr = xr  
            self.dependencies_available = True
        except ImportError:
            self.netCDF4 = None
            self.xr = None
            self.dependencies_available = False
            self.logger.warning("netCDF4 or xarray not available - NetCDF export limited")
    
    def export(self, data: Union[pd.DataFrame, np.ndarray],
               file_path: str,
               x_col: str = 'x',
               y_col: str = 'y',
               z_col: str = 'z',
               time_col: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None,
               **kwargs):
        """
        Export data to NetCDF format.
        
        Args:
            data: DataFrame with coordinates and values
            file_path: Output file path
            x_col: X coordinate column name
            y_col: Y coordinate column name
            z_col: Z value column name
            time_col: Time column name (optional)
            metadata: Additional metadata
            **kwargs: Additional parameters
        """
        if not self.dependencies_available:
            raise ImportError("netCDF4 and xarray are required for NetCDF export")
        
        self.logger.info(f"Exporting NetCDF to: {file_path}")
        
        if not self.validate_data(data):
            raise ValueError("Invalid data format for NetCDF export")
        
        if isinstance(data, pd.DataFrame):
            dataset = self._create_dataset_from_dataframe(data, x_col, y_col, z_col, time_col, metadata)
        else:
            dataset = self._create_dataset_from_array(data, metadata)
        
        # Write to file
        dataset.to_netcdf(file_path)
        dataset.close()
        
        self.logger.info("NetCDF export completed successfully")
    
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate data for NetCDF export."""
        if isinstance(data, pd.DataFrame):
            return len(data) > 0 and not data.empty
        elif isinstance(data, np.ndarray):
            return data.size > 0
        return False
    
    def _create_dataset_from_dataframe(self, df: pd.DataFrame,
                                     x_col: str, y_col: str, z_col: str,
                                     time_col: Optional[str],
                                     metadata: Optional[Dict[str, Any]]) -> 'xr.Dataset':
        """Create xarray Dataset from DataFrame."""
        
        # Get unique coordinates
        x_coords = np.sort(df[x_col].unique())
        y_coords = np.sort(df[y_col].unique())
        
        coords = {'x': x_coords, 'y': y_coords}
        dims = ['y', 'x']
        
        # Handle time dimension
        if time_col and time_col in df.columns:
            time_coords = pd.to_datetime(df[time_col]).unique()
            time_coords = np.sort(time_coords)
            coords['time'] = time_coords
            dims = ['time'] + dims
        
        # Create data arrays
        data_vars = {}
        
        if time_col and time_col in df.columns:
            # 3D data with time
            shape = (len(coords['time']), len(y_coords), len(x_coords))
            z_array = np.full(shape, np.nan)
            
            for _, row in df.iterrows():
                x_idx = np.searchsorted(x_coords, row[x_col])
                y_idx = np.searchsorted(y_coords, row[y_col])
                time_idx = np.searchsorted(coords['time'], pd.to_datetime(row[time_col]))
                
                if x_idx < len(x_coords) and y_idx < len(y_coords) and time_idx < len(coords['time']):
                    z_array[time_idx, y_idx, x_idx] = row[z_col]
        else:
            # 2D data
            shape = (len(y_coords), len(x_coords))
            z_array = np.full(shape, np.nan)
            
            for _, row in df.iterrows():
                x_idx = np.searchsorted(x_coords, row[x_col])
                y_idx = np.searchsorted(y_coords, row[y_col])
                
                if x_idx < len(x_coords) and y_idx < len(y_coords):
                    z_array[y_idx, x_idx] = row[z_col]
        
        # Create data variable
        data_vars[z_col] = (dims, z_array)
        
        # Add other columns as data variables
        other_cols = [col for col in df.columns if col not in [x_col, y_col, z_col, time_col]]
        for col in other_cols:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Numeric data - grid it
                col_array = np.full(shape, np.nan)
                
                if time_col and time_col in df.columns:
                    for _, row in df.iterrows():
                        x_idx = np.searchsorted(x_coords, row[x_col])
                        y_idx = np.searchsorted(y_coords, row[y_col])
                        time_idx = np.searchsorted(coords['time'], pd.to_datetime(row[time_col]))
                        
                        if (x_idx < len(x_coords) and y_idx < len(y_coords) and 
                            time_idx < len(coords['time'])):
                            col_array[time_idx, y_idx, x_idx] = row[col]
                else:
                    for _, row in df.iterrows():
                        x_idx = np.searchsorted(x_coords, row[x_col])
                        y_idx = np.searchsorted(y_coords, row[y_col])
                        
                        if x_idx < len(x_coords) and y_idx < len(y_coords):
                            col_array[y_idx, x_idx] = row[col]
                
                data_vars[col] = (dims, col_array)
        
        # Create dataset
        dataset = self.xr.Dataset(data_vars, coords=coords)
        
        # Add metadata
        if metadata:
            dataset.attrs.update(metadata)
        
        # Add standard attributes
        dataset.attrs.update({
            'title': 'Coal deposit interpolation data',
            'institution': 'Coal Interpolation Tool',
            'created': datetime.now().isoformat(),
            'conventions': 'CF-1.6'
        })
        
        # Add coordinate attributes
        if 'x' in dataset.coords:
            dataset.x.attrs.update({
                'long_name': 'Easting',
                'units': 'meters',
                'axis': 'X'
            })
        
        if 'y' in dataset.coords:
            dataset.y.attrs.update({
                'long_name': 'Northing', 
                'units': 'meters',
                'axis': 'Y'
            })
        
        if 'time' in dataset.coords:
            dataset.time.attrs.update({
                'long_name': 'Time',
                'axis': 'T'
            })
        
        return dataset
    
    def _create_dataset_from_array(self, array: np.ndarray,
                                 metadata: Optional[Dict[str, Any]]) -> 'xr.Dataset':
        """Create xarray Dataset from numpy array."""
        
        dims = ['y', 'x'] if array.ndim == 2 else ['z', 'y', 'x']
        
        # Create coordinates
        coords = {}
        for i, dim in enumerate(dims):
            coords[dim] = np.arange(array.shape[i])
        
        # Create dataset
        data_vars = {'data': (dims, array)}
        dataset = self.xr.Dataset(data_vars, coords=coords)
        
        # Add metadata
        if metadata:
            dataset.attrs.update(metadata)
        
        return dataset


class GoldenSoftwareExporter(BaseExporter):
    """
    Exporter for Golden Software formats (Voxler, Grapher).
    
    Supports various Golden Software data formats.
    """
    
    def __init__(self, software: str = 'voxler'):
        super().__init__()
        self.software = software.lower()
        if self.software not in ['voxler', 'grapher']:
            raise ValueError("Software must be 'voxler' or 'grapher'")
    
    def export(self, data: Union[pd.DataFrame, np.ndarray],
               file_path: str,
               **kwargs):
        """
        Export data to Golden Software format.
        
        Args:
            data: Data to export
            file_path: Output file path
            **kwargs: Additional parameters
        """
        self.logger.info(f"Exporting {self.software} format to: {file_path}")
        
        if not self.validate_data(data):
            raise ValueError("Invalid data format for Golden Software export")
        
        if self.software == 'voxler':
            self._export_voxler(data, file_path, **kwargs)
        else:
            self._export_grapher(data, file_path, **kwargs)
        
        self.logger.info("Golden Software export completed successfully")
    
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate data for Golden Software export."""
        if isinstance(data, pd.DataFrame):
            return len(data) > 0 and not data.empty
        elif isinstance(data, np.ndarray):
            return data.size > 0
        return False
    
    def _export_voxler(self, data: Union[pd.DataFrame, np.ndarray], 
                      file_path: str, **kwargs):
        """Export Voxler data format."""
        if isinstance(data, pd.DataFrame):
            # Export as CSV with specific headers for Voxler
            data.to_csv(file_path, index=False)
        else:
            # Export array data
            np.savetxt(file_path, data, delimiter=',')
    
    def _export_grapher(self, data: Union[pd.DataFrame, np.ndarray],
                       file_path: str, **kwargs):
        """Export Grapher data format."""
        if isinstance(data, pd.DataFrame):
            # Export as tab-delimited for Grapher
            data.to_csv(file_path, sep='\t', index=False)
        else:
            # Export array data
            np.savetxt(file_path, data, delimiter='\t')


# Factory function for creating exporters
def create_exporter(format_type: str, **kwargs) -> BaseExporter:
    """
    Create appropriate exporter based on format type.
    
    Args:
        format_type: Export format ('surfer', 'netcdf', 'golden_software')
        **kwargs: Additional parameters for exporter
        
    Returns:
        Appropriate exporter instance
    """
    format_type = format_type.lower()
    
    if format_type in ['surfer', 'grd']:
        return SurferExporter(**kwargs)
    elif format_type in ['netcdf', 'nc']:
        return NetCDFExporter(**kwargs)
    elif format_type in ['golden_software', 'voxler', 'grapher']:
        software = kwargs.get('software', 'voxler')
        return GoldenSoftwareExporter(software=software)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")


def export_data(data: Union[pd.DataFrame, np.ndarray],
                file_path: str,
                format_type: str,
                **kwargs):
    """
    Convenience function to export data to any supported format.
    
    Args:
        data: Data to export
        file_path: Output file path
        format_type: Export format type
        **kwargs: Additional parameters
    """
    exporter = create_exporter(format_type, **kwargs)
    exporter.export(data, file_path, **kwargs)