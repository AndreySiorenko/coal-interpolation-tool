"""
Excel data reader for coal deposit interpolation data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from .base import BaseReader, DataReadError, ValidationError

# Set up logger
logger = logging.getLogger(__name__)


class ExcelReader(BaseReader):
    """
    Excel file reader optimized for geological survey data.
    
    Supports both .xlsx and .xls formats, handles multiple worksheets,
    and provides Excel-specific functionality for geological datasets.
    """
    
    def __init__(self):
        """Initialize Excel reader."""
        super().__init__()
        self.sheet_names: Optional[List[str]] = None
        self.current_sheet: Optional[str] = None
        self.engine: Optional[str] = None
        
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.xlsx', '.xls']
    
    def _detect_engine(self, file_path: str) -> str:
        """
        Detect appropriate engine for Excel file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Engine name ('openpyxl' for .xlsx, 'xlrd' for .xls)
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.xlsx':
            return 'openpyxl'
        elif extension == '.xls':
            return 'xlrd'
        else:
            raise DataReadError(f"Unsupported Excel format: {extension}")
    
    def get_sheet_names(self, file_path: str) -> List[str]:
        """
        Get list of sheet names in Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            List of sheet names
            
        Raises:
            DataReadError: If file cannot be read
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise DataReadError(f"File not found: {file_path}")
                
            if not self.validate_file(file_path):
                raise DataReadError(f"Unsupported file type: {path.suffix}")
            
            engine = self._detect_engine(file_path)
            
            # Read Excel file to get sheet names
            with pd.ExcelFile(file_path, engine=engine) as excel_file:
                sheet_names = excel_file.sheet_names
                
            self.sheet_names = sheet_names
            return sheet_names
            
        except ImportError as e:
            if 'openpyxl' in str(e):
                raise DataReadError(
                    "openpyxl library is required to read .xlsx files. "
                    "Install it with: pip install openpyxl"
                )
            elif 'xlrd' in str(e):
                raise DataReadError(
                    "xlrd library is required to read .xls files. "
                    "Install it with: pip install xlrd"
                )
            else:
                raise DataReadError(f"Missing dependency: {e}")
        except Exception as e:
            raise DataReadError(f"Error reading Excel file: {e}")
    
    def detect_data_range(self, file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect the actual data range in Excel sheet (excluding empty rows/columns).
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name (uses first sheet if None)
            
        Returns:
            Dictionary with data range information
        """
        try:
            engine = self._detect_engine(file_path)
            
            if sheet_name is None:
                sheet_names = self.get_sheet_names(file_path)
                sheet_name = sheet_names[0] if sheet_names else None
                
            if not sheet_name:
                raise DataReadError("No sheets found in Excel file")
            
            # Read entire sheet to analyze structure
            df = pd.read_excel(
                file_path, 
                sheet_name=sheet_name,
                engine=engine,
                header=None  # Don't assume header location
            )
            
            # Find first and last non-empty rows
            non_empty_rows = df.dropna(how='all').index
            if len(non_empty_rows) == 0:
                raise DataReadError(f"Sheet '{sheet_name}' is empty")
                
            first_data_row = non_empty_rows.min()
            last_data_row = non_empty_rows.max()
            
            # Find first and last non-empty columns
            non_empty_cols = df.dropna(how='all', axis=1).columns
            first_data_col = non_empty_cols.min()
            last_data_col = non_empty_cols.max()
            
            # Detect header row by analyzing content
            header_row = self._detect_header_row(df, first_data_row, last_data_row)
            
            return {
                'sheet_name': sheet_name,
                'data_start_row': first_data_row,
                'data_end_row': last_data_row,
                'data_start_col': first_data_col,
                'data_end_col': last_data_col,
                'header_row': header_row,
                'total_rows': last_data_row - first_data_row + 1,
                'total_cols': last_data_col - first_data_col + 1,
                'estimated_data_rows': last_data_row - header_row if header_row else last_data_row - first_data_row,
            }
            
        except Exception as e:
            raise DataReadError(f"Error detecting data range: {e}")
    
    def _detect_header_row(self, df: pd.DataFrame, start_row: int, end_row: int) -> Optional[int]:
        """
        Detect which row contains the header in Excel data.
        
        Args:
            df: DataFrame with Excel data
            start_row: First data row index
            end_row: Last data row index
            
        Returns:
            Row index containing headers (None if no clear header)
        """
        # Check first few rows after start_row
        max_check_rows = min(5, end_row - start_row + 1)
        
        for row_idx in range(start_row, start_row + max_check_rows):
            if row_idx >= len(df):
                break
                
            row_data = df.iloc[row_idx].dropna()
            if len(row_data) == 0:
                continue
            
            # Check if row contains mostly text (likely headers)
            text_count = 0
            for value in row_data:
                try:
                    # Try to convert to number
                    float(str(value).strip())
                except (ValueError, TypeError):
                    text_count += 1
            
            # If more than 50% are text, likely a header
            if text_count > len(row_data) * 0.5:
                return row_idx
                
        # If no clear header found, assume first data row is header
        return start_row
    
    def read(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Read Excel file with automatic parameter detection.
        
        Args:
            file_path: Path to Excel file
            **kwargs: Optional parameters:
                - sheet_name: Sheet to read (uses first sheet if not provided)
                - header: Header row number (auto-detected if not provided)
                - skiprows: Rows to skip
                - nrows: Maximum number of rows to read
                - usecols: Columns to use
                
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
            
            engine = self._detect_engine(file_path)
            self.engine = engine
            
            # Get sheet name
            sheet_name = kwargs.get('sheet_name')
            if sheet_name is None:
                sheet_names = self.get_sheet_names(file_path)
                sheet_name = sheet_names[0] if sheet_names else None
                
            if not sheet_name:
                raise DataReadError("No sheets found in Excel file")
                
            self.current_sheet = sheet_name
            
            # Detect data range if header not provided
            header_row = kwargs.get('header')
            if header_row is None:
                range_info = self.detect_data_range(file_path, sheet_name)
                header_row = range_info['header_row']
            
            # Prepare read parameters
            read_kwargs = {
                'io': file_path,
                'sheet_name': sheet_name,
                'engine': engine,
                'header': header_row,
                'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN', '-', '#N/A', '#VALUE!', '#REF!'],
            }
            
            # Add optional parameters
            if 'skiprows' in kwargs:
                read_kwargs['skiprows'] = kwargs['skiprows']
            if 'nrows' in kwargs:
                read_kwargs['nrows'] = kwargs['nrows']
            if 'usecols' in kwargs:
                read_kwargs['usecols'] = kwargs['usecols']
                
            # Read Excel file
            self.data = pd.read_excel(**read_kwargs)
            self.file_path = path
            
            # Clean column names
            if isinstance(self.data.columns[0], str):
                self.data.columns = self.data.columns.str.strip()
            else:
                # Handle numeric column names (convert to string)
                self.data.columns = [str(col).strip() for col in self.data.columns]
            
            # Remove completely empty rows and columns
            self.data = self.data.dropna(how='all').dropna(how='all', axis=1)
            
            # Store metadata
            self.metadata = {
                'engine': engine,
                'sheet_name': sheet_name,
                'header_row': header_row,
                'original_shape': self.data.shape,
                'file_size_bytes': path.stat().st_size,
                'available_sheets': self.sheet_names or [],
            }
            
            return self.data
            
        except ImportError as e:
            if 'openpyxl' in str(e):
                raise DataReadError(
                    "openpyxl library is required to read .xlsx files. "
                    "Install it with: pip install openpyxl"
                )
            elif 'xlrd' in str(e):
                raise DataReadError(
                    "xlrd library is required to read .xls files. "
                    "Install it with: pip install xlrd"
                )
            else:
                raise DataReadError(f"Missing dependency: {e}")
        except Exception as e:
            raise DataReadError(f"Error reading Excel file: {e}")
    
    def preview(self, file_path: str, n_rows: int = 5, **kwargs) -> pd.DataFrame:
        """
        Preview first n rows of Excel file efficiently.
        
        Args:
            file_path: Path to Excel file
            n_rows: Number of rows to preview
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with preview data
        """
        kwargs['nrows'] = n_rows
        return self.read(file_path, **kwargs)
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about Excel file without loading all data.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            sheet_names = self.get_sheet_names(file_path)
            
            # Get info for each sheet
            sheets_info = {}
            for sheet_name in sheet_names:
                try:
                    range_info = self.detect_data_range(file_path, sheet_name)
                    sheets_info[sheet_name] = range_info
                except Exception as e:
                    logger.warning(f"Could not analyze sheet '{sheet_name}': {e}")
                    sheets_info[sheet_name] = {'error': str(e)}
            
            return {
                'file_path': str(path),
                'file_size_bytes': path.stat().st_size,
                'engine': self._detect_engine(file_path),
                'sheet_names': sheet_names,
                'sheet_count': len(sheet_names),
                'sheets_info': sheets_info,
            }
            
        except Exception as e:
            raise DataReadError(f"Error getting file info: {e}")
    
    def get_sheet_info(self, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific sheet.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of the sheet
            
        Returns:
            Dictionary with sheet information
        """
        try:
            range_info = self.detect_data_range(file_path, sheet_name)
            
            # Read a preview to get column information
            preview_df = self.preview(file_path, n_rows=5, sheet_name=sheet_name)
            
            range_info.update({
                'columns': list(preview_df.columns),
                'column_count': len(preview_df.columns),
                'preview_data': preview_df.to_dict('records'),
            })
            
            return range_info
            
        except Exception as e:
            raise DataReadError(f"Error getting sheet info: {e}")
    
    def validate_coordinates(self, x_col: str, y_col: str, z_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate coordinate columns in loaded Excel data.
        
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
                # Try to convert to numeric (Excel sometimes stores numbers as text)
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    if numeric_data.isnull().sum() > len(col_data) * 0.1:  # More than 10% conversion failures
                        results['valid'] = False
                        results['errors'].append(f"{coord_name} coordinate column '{col_name}' is not numeric")
                        continue
                    else:
                        # Update the data with converted values
                        self.data[col_name] = numeric_data
                        col_data = numeric_data
                except Exception:
                    results['valid'] = False
                    results['errors'].append(f"{coord_name} coordinate column '{col_name}' cannot be converted to numeric")
                    continue
            
            # Check for null values
            null_count = col_data.isnull().sum()
            if null_count > 0:
                results['warnings'].append(f"{coord_name} coordinate has {null_count} null values")
            
            # Check for Excel-specific issues
            if col_data.dtype == 'object':
                # Check for Excel error values
                error_values = col_data.astype(str).str.contains('#', na=False).sum()
                if error_values > 0:
                    results['warnings'].append(f"{coord_name} coordinate has {error_values} Excel error values")
            
            # Basic statistics
            try:
                results['statistics'][coord_name] = {
                    'column': col_name,
                    'count': len(col_data),
                    'null_count': null_count,
                    'min': float(col_data.min()) if not col_data.empty else None,
                    'max': float(col_data.max()) if not col_data.empty else None,
                    'mean': float(col_data.mean()) if not col_data.empty else None,
                    'std': float(col_data.std()) if not col_data.empty else None,
                }
            except Exception as e:
                results['warnings'].append(f"Could not calculate statistics for {coord_name}: {e}")
                
        # Check for duplicate coordinates
        coord_cols = [col for col in [x_col, y_col, z_col] if col]
        if coord_cols:
            duplicate_count = self.data[coord_cols].duplicated().sum()
            if duplicate_count > 0:
                results['warnings'].append(f"Found {duplicate_count} duplicate coordinate points")
                
        return results