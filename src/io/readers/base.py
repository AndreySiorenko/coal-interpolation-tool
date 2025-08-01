"""
Base abstract class for data readers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


class DataReadError(Exception):
    """Exception raised when data reading fails."""
    pass


class ValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class BaseReader(ABC):
    """
    Abstract base class for all data readers.
    
    Provides common interface for reading various data formats
    used in coal deposit interpolation projects.
    """
    
    def __init__(self):
        """Initialize the reader."""
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.file_path: Optional[Path] = None
        
    @abstractmethod
    def read(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Read data from file.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional reader-specific parameters
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            DataReadError: If file cannot be read
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions (e.g., ['.csv', '.txt'])
        """
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if file can be read by this reader.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file is supported
        """
        path = Path(file_path)
        if not path.exists():
            return False
            
        return path.suffix.lower() in self.get_supported_extensions()
    
    def preview(self, file_path: str, n_rows: int = 5, **kwargs) -> pd.DataFrame:
        """
        Preview first n rows of the file.
        
        Args:
            file_path: Path to the data file
            n_rows: Number of rows to preview
            **kwargs: Additional reader parameters
            
        Returns:
            DataFrame with preview data
        """
        # Default implementation - subclasses can override for efficiency
        full_data = self.read(file_path, **kwargs)
        return full_data.head(n_rows)
    
    def get_column_info(self) -> Dict[str, Any]:
        """
        Get information about columns in loaded data.
        
        Returns:
            Dictionary with column information
        """
        if self.data is None:
            return {}
            
        info = {}
        for col in self.data.columns:
            info[col] = {
                'dtype': str(self.data[col].dtype),
                'non_null_count': self.data[col].count(),
                'null_count': self.data[col].isnull().sum(),
                'unique_count': self.data[col].nunique(),
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(self.data[col]):
                info[col].update({
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                })
                
        return info
    
    def detect_coordinate_columns(self) -> Dict[str, Optional[str]]:
        """
        Attempt to automatically detect coordinate columns.
        
        Returns:
            Dictionary with detected column names for X, Y, Z coordinates
        """
        if self.data is None:
            return {'X': None, 'Y': None, 'Z': None}
            
        columns = self.data.columns.str.lower()
        
        # Common patterns for coordinate columns
        x_patterns = ['x', 'easting', 'east', 'longitude', 'lon', 'long']
        y_patterns = ['y', 'northing', 'north', 'latitude', 'lat']
        z_patterns = ['z', 'elevation', 'elev', 'depth', 'altitude', 'alt']
        
        detected = {'X': None, 'Y': None, 'Z': None}
        
        for col in columns:
            if any(pattern in col for pattern in x_patterns) and detected['X'] is None:
                detected['X'] = self.data.columns[columns.tolist().index(col)]
            elif any(pattern in col for pattern in y_patterns) and detected['Y'] is None:
                detected['Y'] = self.data.columns[columns.tolist().index(col)]
            elif any(pattern in col for pattern in z_patterns) and detected['Z'] is None:
                detected['Z'] = self.data.columns[columns.tolist().index(col)]
                
        return detected
    
    def detect_value_columns(self) -> List[str]:
        """
        Detect columns that likely contain interpolation values.
        
        Returns:
            List of column names that appear to be value columns
        """
        if self.data is None:
            return []
            
        # Get coordinate columns to exclude them
        coord_cols = set(self.detect_coordinate_columns().values())
        coord_cols.discard(None)
        
        # Find numeric columns that are not coordinates
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        value_cols = [col for col in numeric_cols if col not in coord_cols]
        
        return value_cols
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of loaded data.
        
        Returns:
            Dictionary with data summary information
        """
        if self.data is None:
            return {}
            
        return {
            'file_path': str(self.file_path) if self.file_path else None,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'coordinate_columns': self.detect_coordinate_columns(),
            'value_columns': self.detect_value_columns(),
            'null_counts': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
        }