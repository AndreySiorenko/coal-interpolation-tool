"""
CSV writer for exporting interpolation results to CSV format.
"""

import csv
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .base import BaseWriter, ExportFormat, ExportOptions, GridData, PointData, ExportError


@dataclass
class CSVExportOptions(ExportOptions):
    """
    CSV-specific export options.
    
    Attributes:
        delimiter: Field delimiter (default: ',')
        line_terminator: Line terminator (default: system default)
        quote_char: Quote character (default: '"')
        quoting: Quoting behavior (csv.QUOTE_MINIMAL, etc.)
        header: Include column headers
        index: Include row index
        encoding: File encoding (default: 'utf-8')
        decimal: Decimal separator (default: '.')
        float_format: Format string for floats
        na_rep: String representation of NaN values
    """
    delimiter: str = ','
    line_terminator: Optional[str] = None
    quote_char: str = '"'
    quoting: int = csv.QUOTE_MINIMAL
    header: bool = True
    index: bool = False
    encoding: str = 'utf-8'
    decimal: str = '.'
    float_format: Optional[str] = None
    na_rep: str = ''
    
    def __post_init__(self):
        """Set default float format based on precision."""
        super().__post_init__()
        if self.float_format is None:
            self.float_format = f'%.{self.precision}f'


class CSVWriter(BaseWriter):
    """
    Writer for exporting data to CSV format.
    
    CSV export supports both grid and point data, creating tabular representations
    suitable for spreadsheet applications and further processing.
    """
    
    def __init__(self, options: Optional[CSVExportOptions] = None):
        """
        Initialize CSV writer.
        
        Args:
            options: CSV-specific export options
        """
        if options is None:
            options = CSVExportOptions()
        elif not isinstance(options, CSVExportOptions):
            # Convert base options to CSV options
            csv_options = CSVExportOptions()
            for field in options.__dataclass_fields__:
                if hasattr(options, field):
                    setattr(csv_options, field, getattr(options, field))
            options = csv_options
            
        super().__init__(options)
    
    @property
    def supported_formats(self) -> List[ExportFormat]:
        """Return list of formats supported by this writer."""
        return [ExportFormat.CSV]
    
    @property
    def file_extensions(self) -> List[str]:
        """Return list of file extensions for this writer."""
        return ['.csv', '.txt']
    
    def write_grid(self, 
                   data: GridData, 
                   filepath: Union[str, Path],
                   **kwargs) -> None:
        """
        Write grid data to CSV file.
        
        Creates a CSV file with columns for coordinates and interpolated values.
        For 2D grids: X, Y, Value
        For 3D grids: X, Y, Z, Value
        
        Args:
            data: Grid data to export
            filepath: Output file path
            **kwargs: Additional CSV options
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Validate inputs
            self.validate_grid_data(data)
            filepath = self.validate_filepath(filepath)
            
            # Update options with kwargs
            csv_options = self._update_options(**kwargs)
            
            # Generate grid points
            df = self._grid_to_dataframe(data)
            
            # Write to CSV
            self._write_dataframe(df, filepath, csv_options)
            
        except Exception as e:
            raise ExportError(f"Failed to export grid data to CSV: {e}")
    
    def write_points(self, 
                     data: PointData, 
                     filepath: Union[str, Path],
                     **kwargs) -> None:
        """
        Write point data to CSV file.
        
        Creates a CSV file with columns for coordinates, values, and any additional attributes.
        
        Args:
            data: Point data to export
            filepath: Output file path
            **kwargs: Additional CSV options
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Validate inputs
            self.validate_point_data(data)
            filepath = self.validate_filepath(filepath)
            
            # Update options with kwargs
            csv_options = self._update_options(**kwargs)
            
            # Convert to DataFrame
            df = self._points_to_dataframe(data)
            
            # Write to CSV
            self._write_dataframe(df, filepath, csv_options)
            
        except Exception as e:
            raise ExportError(f"Failed to export point data to CSV: {e}")
    
    def _grid_to_dataframe(self, data: GridData) -> pd.DataFrame:
        """
        Convert grid data to pandas DataFrame.
        
        Args:
            data: Grid data to convert
            
        Returns:
            DataFrame with coordinate and value columns
        """
        # Create coordinate meshgrids
        if data.is_3d:
            X, Y, Z = np.meshgrid(data.x_coords, data.y_coords, data.z_coords, indexing='ij')
            # Flatten all arrays
            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z.flatten()
            values_flat = data.values.flatten()
            
            # Create DataFrame
            df_data = {
                'X': x_flat,
                'Y': y_flat,
                'Z': z_flat,
                'Value': values_flat
            }
        else:
            X, Y = np.meshgrid(data.x_coords, data.y_coords, indexing='ij')
            # Flatten arrays
            x_flat = X.flatten()
            y_flat = Y.flatten()
            values_flat = data.values.flatten()
            
            # Create DataFrame
            df_data = {
                'X': x_flat,
                'Y': y_flat,
                'Value': values_flat
            }
        
        df = pd.DataFrame(df_data)
        
        # Add metadata as additional columns if requested
        if self.options.include_metadata and data.metadata:
            for key, value in data.metadata.items():
                if isinstance(value, (str, int, float)):
                    df[f'meta_{key}'] = value
        
        # Filter out NaN values if requested
        if not self.options.custom_attributes.get('include_nan', True):
            df = df.dropna(subset=['Value'])
        
        return df
    
    def _points_to_dataframe(self, data: PointData) -> pd.DataFrame:
        """
        Convert point data to pandas DataFrame.
        
        Args:
            data: Point data to convert
            
        Returns:
            DataFrame with coordinate, value, and attribute columns
        """
        # Base coordinate columns
        if data.is_3d:
            df_data = {
                'X': data.coordinates[:, 0],
                'Y': data.coordinates[:, 1],
                'Z': data.coordinates[:, 2],
                'Value': data.values
            }
        else:
            df_data = {
                'X': data.coordinates[:, 0],
                'Y': data.coordinates[:, 1],
                'Value': data.values
            }
        
        # Add point IDs if available
        if data.point_ids is not None:
            df_data['ID'] = data.point_ids
        
        # Add additional attributes
        if data.attributes:
            for key, values in data.attributes.items():
                if len(values) == data.n_points:
                    df_data[key] = values
        
        df = pd.DataFrame(df_data)
        
        # Add metadata as additional columns if requested
        if self.options.include_metadata and data.metadata:
            for key, value in data.metadata.items():
                if isinstance(value, (str, int, float)):
                    df[f'meta_{key}'] = value
        
        return df
    
    def _write_dataframe(self, 
                        df: pd.DataFrame, 
                        filepath: Path, 
                        csv_options: CSVExportOptions) -> None:
        """
        Write DataFrame to CSV file.
        
        Args:
            df: DataFrame to write
            filepath: Output file path
            csv_options: CSV writing options
        """
        # Prepare pandas to_csv arguments
        csv_args = {
            'sep': csv_options.delimiter,
            'header': csv_options.header,
            'index': csv_options.index,
            'encoding': csv_options.encoding,
            'decimal': csv_options.decimal,
            'float_format': csv_options.float_format,
            'na_rep': csv_options.na_rep,
            'quotechar': csv_options.quote_char,
            'quoting': csv_options.quoting,
        }
        
        # Add line terminator if specified
        if csv_options.line_terminator:
            csv_args['line_terminator'] = csv_options.line_terminator
        
        # Write to file
        df.to_csv(filepath, **csv_args)
    
    def _update_options(self, **kwargs) -> CSVExportOptions:
        """
        Update export options with provided kwargs.
        
        Args:
            **kwargs: Additional options
            
        Returns:
            Updated CSV export options
        """
        # Create a copy of current options
        options = CSVExportOptions()
        
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
            'format': 'CSV',
            'filepath': str(filepath),
            'data_type': 'grid' if isinstance(data, GridData) else 'points',
            'n_points': data.n_points,
            'is_3d': data.is_3d,
            'bounds': data.bounds,
            'delimiter': self.options.delimiter,
            'encoding': self.options.encoding,
            'precision': self.options.precision,
        }
        
        if isinstance(data, GridData):
            summary.update({
                'grid_shape': data.shape,
                'cell_size': data.cell_size,
            })
        
        if hasattr(data, 'coordinate_system') and data.coordinate_system:
            summary['coordinate_system'] = data.coordinate_system
        
        return summary


def create_csv_writer(delimiter: str = ',',
                     precision: int = 6,
                     header: bool = True,
                     encoding: str = 'utf-8') -> CSVWriter:
    """
    Factory function to create a CSV writer with common options.
    
    Args:
        delimiter: Field delimiter
        precision: Decimal precision
        header: Include column headers
        encoding: File encoding
        
    Returns:
        Configured CSV writer
    """
    options = CSVExportOptions(
        delimiter=delimiter,
        precision=precision,
        header=header,
        encoding=encoding
    )
    
    return CSVWriter(options)