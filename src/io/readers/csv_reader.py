"""
CSV data reader for coal deposit interpolation data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import csv
import chardet

from .base import BaseReader, DataReadError, ValidationError


class CSVReader(BaseReader):
    """
    CSV file reader optimized for geological survey data.
    
    Supports automatic delimiter detection, encoding detection,
    and handles common issues in geological datasets.
    """
    
    def __init__(self):
        """Initialize CSV reader."""
        super().__init__()
        self.delimiter: Optional[str] = None
        self.encoding: Optional[str] = None
        self.header_row: Optional[int] = None
        
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.csv', '.txt', '.tsv']
    
    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding using chardet.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Detected encoding string
        """
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'  # Fallback to UTF-8
    
    def detect_delimiter(self, file_path: str, encoding: str = 'utf-8') -> str:
        """
        Detect CSV delimiter automatically.
        
        Args:
            file_path: Path to the CSV file
            encoding: File encoding
            
        Returns:
            Detected delimiter character
        """
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                # Read first few lines to detect delimiter
                sample = file.read(8192)
                
            # Use csv.Sniffer to detect delimiter
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            # Validate delimiter makes sense
            common_delimiters = [',', ';', '\t', '|', ' ']
            if delimiter in common_delimiters:
                return delimiter
                
        except Exception:
            pass
            
        # Fallback: count occurrences of common delimiters
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                first_line = file.readline()
                
            delimiter_counts = {
                ',': first_line.count(','),
                ';': first_line.count(';'),
                '\t': first_line.count('\t'),
                '|': first_line.count('|'),
            }
            
            # Return delimiter with highest count
            return max(delimiter_counts, key=delimiter_counts.get)
            
        except Exception:
            return ','  # Ultimate fallback
    
    def detect_header_row(self, file_path: str, delimiter: str, encoding: str = 'utf-8') -> int:
        """
        Detect which row contains the header.
        
        Args:
            file_path: Path to the CSV file
            delimiter: CSV delimiter
            encoding: File encoding
            
        Returns:
            Row number containing headers (0-based)
        """
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                reader = csv.reader(file, delimiter=delimiter)
                
                for i, row in enumerate(reader):
                    if i > 10:  # Don't check beyond first 10 rows
                        break
                        
                    # Check if row contains mostly non-numeric values (likely headers)
                    if len(row) > 1:
                        non_numeric_count = 0
                        for cell in row:
                            try:
                                float(cell.strip())
                            except (ValueError, AttributeError):
                                non_numeric_count += 1
                                
                        # If more than half are non-numeric, likely a header
                        if non_numeric_count > len(row) / 2:
                            return i
                            
        except Exception:
            pass
            
        return 0  # Default to first row
    
    def read(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Read CSV file with automatic parameter detection.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Optional parameters:
                - delimiter: CSV delimiter (auto-detected if not provided)
                - encoding: File encoding (auto-detected if not provided)
                - header: Header row number (auto-detected if not provided)
                - skiprows: Rows to skip
                - nrows: Maximum number of rows to read
                
        Returns:
            DataFrame with loaded data
            
        Raises:
            DataReadError: If file cannot be read
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise DataReadError(f"File not found: {file_path}")
                
            if not self.validate_file(file_path):
                raise DataReadError(f"Unsupported file type: {path.suffix}")
            
            # Auto-detect or use provided parameters
            encoding = kwargs.get('encoding') or self.detect_encoding(file_path)
            delimiter = kwargs.get('delimiter') or self.detect_delimiter(file_path, encoding)
            header_row = kwargs.get('header')
            
            if header_row is None:
                header_row = self.detect_header_row(file_path, delimiter, encoding)
            
            # Store detected parameters
            self.encoding = encoding
            self.delimiter = delimiter
            self.header_row = header_row
            self.file_path = path
            
            # Read the CSV file
            read_kwargs = {
                'filepath_or_buffer': file_path,
                'delimiter': delimiter,
                'encoding': encoding,
                'header': header_row,
                'skipinitialspace': True,  # Remove leading whitespace
                'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN', '-'],
            }
            
            # Add optional parameters
            if 'skiprows' in kwargs:
                read_kwargs['skiprows'] = kwargs['skiprows']  
            if 'nrows' in kwargs:
                read_kwargs['nrows'] = kwargs['nrows']
                
            self.data = pd.read_csv(**read_kwargs)
            
            # Clean column names
            self.data.columns = self.data.columns.str.strip()
            
            # Store metadata
            self.metadata = {
                'delimiter': delimiter,
                'encoding': encoding,
                'header_row': header_row,
                'original_shape': self.data.shape,
                'file_size_bytes': path.stat().st_size,
            }
            
            return self.data
            
        except pd.errors.EmptyDataError:
            raise DataReadError(f"CSV file is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise DataReadError(f"Error parsing CSV file: {e}")
        except Exception as e:
            raise DataReadError(f"Unexpected error reading CSV file: {e}")
    
    def preview(self, file_path: str, n_rows: int = 5, **kwargs) -> pd.DataFrame:
        """
        Preview first n rows efficiently.
        
        Args:
            file_path: Path to CSV file
            n_rows: Number of rows to preview
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with preview data
        """
        kwargs['nrows'] = n_rows
        return self.read(file_path, **kwargs)
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about CSV file without loading all data.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            encoding = self.detect_encoding(file_path)
            delimiter = self.detect_delimiter(file_path, encoding)
            
            # Count total rows
            with open(file_path, 'r', encoding=encoding) as file:
                total_rows = sum(1 for _ in file)
            
            # Get column names from header
            with open(file_path, 'r', encoding=encoding) as file:
                reader = csv.reader(file, delimiter=delimiter)
                header_row = self.detect_header_row(file_path, delimiter, encoding)
                
                # Skip to header row
                for _ in range(header_row):
                    next(reader)
                    
                columns = next(reader, [])
            
            return {
                'file_path': str(path),
                'file_size_bytes': path.stat().st_size,
                'encoding': encoding,
                'delimiter': delimiter,
                'header_row': header_row,
                'total_rows': total_rows,
                'estimated_data_rows': total_rows - header_row - 1,
                'columns': columns,
                'column_count': len(columns),
            }
            
        except Exception as e:
            raise DataReadError(f"Error getting file info: {e}")
    
    def validate_coordinates(self, x_col: str, y_col: str, z_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate coordinate columns in loaded data.
        
        Args:
            x_col: Name of X coordinate column
            y_col: Name of Y coordinate column  
            z_col: Name of Z coordinate column (optional)
            
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValidationError("No data loaded")
            
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check if columns exist
        missing_cols = []
        for col_name, col in [('X', x_col), ('Y', y_col), ('Z', z_col)]:
            if col and col not in self.data.columns:
                missing_cols.append(col)
                
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {missing_cols}")
            return results
        
        # Validate coordinate columns
        for coord_name, col_name in [('X', x_col), ('Y', y_col), ('Z', z_col)]:
            if not col_name:
                continue
                
            col_data = self.data[col_name]
            
            # Check if numeric
            if not pd.api.types.is_numeric_dtype(col_data):
                results['valid'] = False
                results['errors'].append(f"{coord_name} coordinate column '{col_name}' is not numeric")
                continue
            
            # Check for null values
            null_count = col_data.isnull().sum()
            if null_count > 0:
                results['warnings'].append(f"{coord_name} coordinate has {null_count} null values")
            
            # Basic statistics
            results['statistics'][coord_name] = {
                'column': col_name,
                'count': len(col_data),
                'null_count': null_count,
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'std': col_data.std(),
            }
            
        # Check for duplicate coordinates
        coord_cols = [col for col in [x_col, y_col, z_col] if col]
        duplicate_count = self.data[coord_cols].duplicated().sum()
        if duplicate_count > 0:
            results['warnings'].append(f"Found {duplicate_count} duplicate coordinate points")
            
        return results