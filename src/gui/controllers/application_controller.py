"""
Application controller managing the interaction between GUI and backend components.
"""

import json
import pandas as pd
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path
import threading
import time

# Import backend components
from ...io.readers.csv_reader import CSVReader
from ...io.readers.excel_reader import ExcelReader
from ...io.writers.csv_writer import CSVWriter
from ...io.writers.geotiff_writer import GeoTIFFWriter
from ...io.writers.vtk_writer import VTKWriter
from ...io.writers.dxf_writer import DXFWriter
from ...io.writers.base import ExportFormat, GridData, PointData
from ...core.interpolation.idw import IDWInterpolator, IDWParameters
from ...core.interpolation.rbf import RBFInterpolator, RBFParameters, RBFKernel
from ...core.interpolation.kriging import KrigingInterpolator, KrigingParameters, KrigingType, VariogramModel
from ...core.interpolation.base import SearchParameters
from ...core.grid import GridGenerator, GridParameters


class ApplicationController:
    """
    Controller class managing the application logic and backend integration.
    
    This class serves as the bridge between the GUI components and the backend
    interpolation system. It handles:
    - Data loading and validation
    - Interpolation parameter management
    - Interpolation execution with progress tracking
    - Results management and export
    - Project save/load functionality
    - Event notifications to GUI components
    """
    
    def __init__(self):
        """Initialize the application controller."""
        self.current_data = None
        self.current_filename = None
        self.current_file_path = None
        self.current_parameters = None
        self.interpolation_results = None
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Backend components
        self.csv_reader = CSVReader()
        self.excel_reader = ExcelReader()
        self.grid_generator = GridGenerator()
        
    def bind_event(self, event_name: str, handler: Callable):
        """
        Bind an event handler to an event.
        
        Args:
            event_name: Name of the event
            handler: Callback function to handle the event
        """
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)
        
    def emit_event(self, event_name: str, *args, **kwargs):
        """
        Emit an event to all registered handlers.
        
        Args:
            event_name: Name of the event
            *args, **kwargs: Arguments to pass to handlers
        """
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    print(f"Error in event handler for {event_name}: {e}")
                    
    def load_data_file(self, file_path: str):
        """
        Load a data file.
        
        Args:
            file_path: Path to the data file
        """
        try:
            # Determine file type and load accordingly
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                self.current_data = self.csv_reader.read(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                self.current_data = self.excel_reader.read(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
            self.current_filename = Path(file_path).name
            self.current_file_path = file_path
            
            # Validate data structure
            self._validate_data_structure()
            
            # Emit data loaded event
            data_info = {
                'filename': self.current_filename,
                'file_path': self.current_file_path,
                'rows': len(self.current_data),
                'columns': list(self.current_data.columns),
                'bounds': self._calculate_data_bounds()
            }
            
            self.emit_event('data_loaded', data_info)
            
        except Exception as e:
            self.emit_event('error_occurred', f"Failed to load data: {str(e)}")
            raise
            
    def load_data_with_settings(self, settings: Dict[str, Any]):
        """
        Load a data file with advanced settings from the data loader dialog.
        
        Args:
            settings: Dictionary containing file path and loading settings
        """
        try:
            file_path = settings['file_path']
            
            # Load data with custom settings
            import pandas as pd
            
            # Apply custom CSV reading settings
            df = pd.read_csv(
                file_path,
                delimiter=settings['delimiter'],
                encoding=settings['encoding'],
                header=settings['header_row'] if settings['header_row'] > 0 else 0
            )
            
            # Apply column mapping
            column_mapping = {
                settings['x_column']: 'X',
                settings['y_column']: 'Y'
            }
            
            # Create new dataframe with mapped columns
            mapped_data = pd.DataFrame()
            mapped_data['X'] = df[settings['x_column']]
            mapped_data['Y'] = df[settings['y_column']]
            
            # Add selected value columns
            for col in settings['value_columns']:
                mapped_data[col] = df[col]
            
            # Apply validation options
            if settings['skip_invalid_rows']:
                # Remove rows with invalid coordinates
                mapped_data = mapped_data.dropna(subset=['X', 'Y'])
                
            if settings['remove_duplicates']:
                # Remove duplicate coordinates
                mapped_data = mapped_data.drop_duplicates(subset=['X', 'Y'])
                
            if settings['fill_missing_values']:
                # Fill missing values (simple forward fill for now)
                mapped_data = mapped_data.ffill()
            
            self.current_data = mapped_data
            self.current_filename = Path(file_path).name
            self.current_file_path = file_path
            
            # Validate data structure
            self._validate_data_structure()
            
            # Emit data loaded event
            data_info = {
                'filename': self.current_filename,
                'file_path': self.current_file_path,
                'rows': len(self.current_data),
                'columns': list(self.current_data.columns),
                'bounds': self._calculate_data_bounds(),
                'load_settings': settings
            }
            
            self.emit_event('data_loaded', data_info)
            
        except Exception as e:
            self.emit_event('error_occurred', f"Failed to load data with settings: {str(e)}")
            raise
            
    def _validate_data_structure(self):
        """Validate that the loaded data has the required structure."""
        if self.current_data is None:
            raise ValueError("No data loaded")
            
        required_columns = ['X', 'Y']
        missing_columns = [col for col in required_columns if col not in self.current_data.columns]
        
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")
            
        # Check for at least one value column
        value_columns = [col for col in self.current_data.columns if col not in ['X', 'Y']]
        if not value_columns:
            raise ValueError("Data must contain at least one value column for interpolation")
            
    def _calculate_data_bounds(self) -> Dict[str, float]:
        """Calculate the spatial bounds of the loaded data."""
        if self.current_data is None:
            return {}
            
        return {
            'min_x': float(self.current_data['X'].min()),
            'max_x': float(self.current_data['X'].max()),
            'min_y': float(self.current_data['Y'].min()),
            'max_y': float(self.current_data['Y'].max())
        }
        
    def get_data_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently loaded data.
        
        Returns:
            Dictionary with data information or None if no data loaded
        """
        if self.current_data is None:
            return None
            
        return {
            'filename': self.current_filename,
            'rows': len(self.current_data),
            'columns': list(self.current_data.columns),
            'bounds': self._calculate_data_bounds(),
            'data_types': self.current_data.dtypes.to_dict(),
            'missing_values': self.current_data.isnull().sum().to_dict()
        }
        
    def get_value_columns(self) -> List[str]:
        """
        Get list of columns that can be used for interpolation.
        
        Returns:
            List of column names excluding X, Y coordinates
        """
        if self.current_data is None:
            return []
            
        return [col for col in self.current_data.columns if col not in ['X', 'Y']]
        
    def run_interpolation(self, parameters: Dict[str, Any]):
        """
        Run interpolation with given parameters.
        
        Args:
            parameters: Dictionary containing interpolation parameters
        """
        if self.current_data is None:
            raise ValueError("No data loaded for interpolation")
            
        self.current_parameters = parameters
        
        # Run interpolation in a separate thread to avoid blocking the GUI
        thread = threading.Thread(target=self._run_interpolation_worker, args=(parameters,))
        thread.daemon = True
        thread.start()
        
    def _run_interpolation_worker(self, parameters: Dict[str, Any]):
        """
        Worker function to run interpolation in a separate thread.
        
        Args:
            parameters: Interpolation parameters
        """
        try:
            self.emit_event('interpolation_started')
            
            # Extract parameters
            value_column = parameters.get('value_column', self.get_value_columns()[0])
            method = parameters.get('method', 'IDW')
            
            # Grid parameters
            grid_params = GridParameters(
                cell_size=parameters.get('cell_size', 50.0),
                buffer=parameters.get('buffer', 0.1)
            )
            
            # Search parameters
            search_params = SearchParameters(
                search_radius=parameters.get('search_radius', 1000.0),
                min_points=parameters.get('min_points', 1),
                max_points=parameters.get('max_points', 12),
                use_sectors=parameters.get('use_sectors', False)
            )
            
            # Update progress
            self.emit_event('interpolation_progress', 10)
            
            # Create grid
            self.grid_generator = GridGenerator(grid_params)
            grid = self.grid_generator.create_regular_grid(data=self.current_data)
            
            self.emit_event('interpolation_progress', 30)
            
            # Setup interpolator based on method
            if method == 'RBF':
                # Convert string kernel parameter to enum
                kernel_str = parameters.get('rbf_kernel', 'multiquadric')
                try:
                    kernel = RBFKernel(kernel_str.lower())
                except ValueError:
                    # Fallback to default if invalid kernel
                    kernel = RBFKernel.MULTIQUADRIC
                
                # RBF parameters
                rbf_params = RBFParameters(
                    kernel=kernel,
                    shape_parameter=parameters.get('rbf_shape_parameter', 1.0),
                    regularization=parameters.get('rbf_regularization', 1e-12),
                    polynomial_degree=parameters.get('rbf_polynomial_degree', -1),
                    use_global=parameters.get('rbf_use_global', True)
                )
                interpolator = RBFInterpolator(search_params, rbf_params)
            elif method == 'Kriging':
                # Convert string parameters to enum types
                kriging_type_str = parameters.get('kriging_type', 'ordinary')
                variogram_model_str = parameters.get('kriging_variogram_model', 'spherical')
                
                # Map string values to enum types
                kriging_type = KrigingType(kriging_type_str.lower())
                variogram_model = VariogramModel(variogram_model_str.lower())
                
                # Kriging parameters
                kriging_params = KrigingParameters(
                    kriging_type=kriging_type,
                    variogram_model=variogram_model,
                    nugget=parameters.get('kriging_nugget', 0.0),
                    sill=parameters.get('kriging_sill', 1.0),
                    range_param=parameters.get('kriging_range', 1000.0),
                    use_global=parameters.get('kriging_use_global', True),
                    auto_fit_variogram=parameters.get('kriging_auto_fit', True)
                )
                interpolator = KrigingInterpolator(search_params, kriging_params)
            else:  # Default to IDW
                # IDW parameters
                idw_params = IDWParameters(
                    power=parameters.get('power', 2.0),
                    smoothing=parameters.get('smoothing', 0.0)
                )
                interpolator = IDWInterpolator(search_params, idw_params)
            
            # Fit the interpolator
            # All interpolators now use the same interface: DataFrame and column names
            interpolator.fit(self.current_data, 'X', 'Y', value_column)
            
            self.emit_event('interpolation_progress', 50)
            
            # Simulate progress during prediction
            prediction_points = grid[['X', 'Y']]
            total_points = len(prediction_points)
            
            # For demonstration, we'll predict in chunks to show progress
            chunk_size = max(1, total_points // 10)
            all_predictions = []
            
            for i in range(0, total_points, chunk_size):
                chunk = prediction_points.iloc[i:i+chunk_size]
                chunk_predictions = interpolator.predict(chunk)
                all_predictions.extend(chunk_predictions)
                
                progress = 50 + (i / total_points) * 40
                self.emit_event('interpolation_progress', progress)
                
                # Small delay to simulate computation time
                time.sleep(0.1)
                
            # Combine results
            results_grid = grid.copy()
            results_grid[f'{value_column}_interpolated'] = all_predictions
            
            # Calculate statistics
            stats = {
                'min_value': min(all_predictions),
                'max_value': max(all_predictions),
                'mean_value': sum(all_predictions) / len(all_predictions),
                'grid_points': len(all_predictions),
                'method': method,
                'parameters': parameters
            }
            
            # Store results
            self.interpolation_results = {
                'grid': results_grid,
                'statistics': stats,
                'parameters': parameters,
                'source_data': self.current_data.copy()
            }
            
            self.emit_event('interpolation_progress', 100)
            self.emit_event('interpolation_completed', self.interpolation_results)
            
        except Exception as e:
            self.emit_event('error_occurred', f"Interpolation failed: {str(e)}")
            
    def get_interpolation_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the current interpolation results.
        
        Returns:
            Dictionary with interpolation results or None if no results available
        """
        return self.interpolation_results
        
    def export_results(self, file_path: str, results: Dict[str, Any]):
        """
        Export interpolation results to a file.
        
        Args:
            file_path: Path where to save the results
            results: Results dictionary to export
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                # Export grid data as CSV
                results['grid'].to_csv(file_path, index=False)
                
            elif file_ext == '.json':
                # Export full results as JSON
                exportable_results = {
                    'statistics': results['statistics'],
                    'parameters': results['parameters'],
                    'grid_data': results['grid'].to_dict('records')
                }
                
                with open(file_path, 'w') as f:
                    json.dump(exportable_results, f, indent=2, default=str)
                    
            else:
                raise ValueError(f"Unsupported export format: {file_ext}")
                
        except Exception as e:
            self.emit_event('error_occurred', f"Export failed: {str(e)}")
            raise
            
    def export_results_with_settings(self, export_settings: Dict[str, Any]):
        """
        Export interpolation results using advanced export settings from export dialog.
        
        Args:
            export_settings: Dictionary containing export settings from export dialog
        """
        try:
            file_path = export_settings['file_path']
            export_format = export_settings['format']
            export_options = export_settings['options']
            results_data = export_settings['results_data']
            
            # Convert results data to appropriate format for writers
            grid_data = self._convert_to_grid_data(results_data)
            
            # Create appropriate writer based on format
            if export_format == ExportFormat.CSV:
                writer = CSVWriter(export_options)
                writer.write_grid(grid_data, file_path)
                
            elif export_format == ExportFormat.GEOTIFF:
                writer = GeoTIFFWriter(export_options)
                writer.write_grid(grid_data, file_path)
                
            elif export_format == ExportFormat.VTK:
                writer = VTKWriter(export_options)
                writer.write_grid(grid_data, file_path)
                
            elif export_format == ExportFormat.DXF:
                writer = DXFWriter(export_options)
                writer.write_grid(grid_data, file_path)
                
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            self.emit_event('error_occurred', f"Advanced export failed: {str(e)}")
            raise
            
    def _convert_to_grid_data(self, results_data: Dict[str, Any]) -> GridData:
        """
        Convert results data to GridData format for writers.
        
        Args:
            results_data: Results data from interpolation
            
        Returns:
            GridData object suitable for export writers
        """
        import numpy as np
        
        # Extract grid information from results
        grid_df = results_data.get('grid')
        if grid_df is None:
            raise ValueError("No grid data found in results")
            
        # Get unique coordinates to determine grid structure
        x_coords = np.sort(grid_df['X'].unique())
        y_coords = np.sort(grid_df['Y'].unique())
        
        # Get value column (find interpolated column)
        value_cols = [col for col in grid_df.columns if col.endswith('_interpolated')]
        if not value_cols:
            # Fallback to first non-coordinate column
            value_cols = [col for col in grid_df.columns if col not in ['X', 'Y']]
            
        if not value_cols:
            raise ValueError("No value columns found in grid data")
            
        value_col = value_cols[0]
        
        # Reshape values into grid
        n_x, n_y = len(x_coords), len(y_coords)
        values = np.zeros((n_y, n_x))
        
        for _, row in grid_df.iterrows():
            try:
                x_idx = np.where(x_coords == row['X'])[0][0]
                y_idx = np.where(y_coords == row['Y'])[0][0]
                
                # Convert value to float, handle string values
                val = row[value_col]
                if isinstance(val, str):
                    try:
                        val = float(val)
                    except ValueError:
                        val = np.nan  # Use NaN for non-numeric strings
                
                values[y_idx, x_idx] = val
            except (IndexError, ValueError, TypeError) as e:
                print(f"Warning: Error processing row {row.name}: {e}")
                continue
            
        # Calculate bounds and cell size
        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())
        
        # Estimate cell size
        cell_size_x = (x_max - x_min) / (n_x - 1) if n_x > 1 else 1.0
        cell_size_y = (y_max - y_min) / (n_y - 1) if n_y > 1 else 1.0
        cell_size = (cell_size_x + cell_size_y) / 2
        
        # Create GridData object
        grid_data = GridData(
            x_coords=x_coords,
            y_coords=y_coords,
            values=values,
            bounds=(x_min, x_max, y_min, y_max),
            cell_size=cell_size,
            coordinate_system=None,  # Could be extracted from metadata if available
            metadata={
                'source': 'Coal Interpolation Tool',
                'method': results_data.get('statistics', {}).get('method', 'Unknown'),
                'value_column': value_col
            }
        )
        
        return grid_data
            
    def save_project(self, file_path: str):
        """
        Save the current project to a file.
        
        Args:
            file_path: Path where to save the project
        """
        if self.current_data is None:
            raise ValueError("No data to save")
            
        project_data = {
            'version': '0.4.0',
            'data_filename': self.current_filename,
            'data': self.current_data.to_dict('records'),
            'parameters': self.current_parameters,
            'has_results': self.interpolation_results is not None
        }
        
        if self.interpolation_results:
            project_data['results'] = {
                'statistics': self.interpolation_results['statistics'],
                'grid_data': self.interpolation_results['grid'].to_dict('records')
            }
            
        try:
            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2, default=str)
        except Exception as e:
            self.emit_event('error_occurred', f"Failed to save project: {str(e)}")
            raise
            
    def load_project(self, file_path: str):
        """
        Load a project from a file.
        
        Args:
            file_path: Path to the project file
        """
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)
                
            # Restore data
            self.current_data = pd.DataFrame(project_data['data'])
            self.current_filename = project_data['data_filename']
            self.current_parameters = project_data.get('parameters')
            
            # Restore results if available
            if project_data.get('has_results') and 'results' in project_data:
                results_data = project_data['results']
                self.interpolation_results = {
                    'grid': pd.DataFrame(results_data['grid_data']),
                    'statistics': results_data['statistics'],
                    'parameters': self.current_parameters,
                    'source_data': self.current_data.copy()
                }
                
            # Emit events
            data_info = {
                'filename': self.current_filename,
                'rows': len(self.current_data),
                'columns': list(self.current_data.columns),
                'bounds': self._calculate_data_bounds()
            }
            
            self.emit_event('data_loaded', data_info)
            
            if self.interpolation_results:
                self.emit_event('interpolation_completed', self.interpolation_results)
                
        except Exception as e:
            self.emit_event('error_occurred', f"Failed to load project: {str(e)}")
            raise
            
    def has_data(self) -> bool:
        """
        Check if data is currently loaded.
        
        Returns:
            True if data is loaded, False otherwise
        """
        return self.current_data is not None
        
    def has_results(self) -> bool:
        """
        Check if interpolation results are available.
        
        Returns:
            True if results are available, False otherwise
        """
        return self.interpolation_results is not None
    
    def get_excel_sheet_names(self, file_path: str) -> List[str]:
        """
        Get available sheet names from Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            List of sheet names
        """
        return self.excel_reader.get_sheet_names(file_path)
    
    def get_excel_sheet_info(self, file_path: str, sheet_name: str) -> Dict[str, Any]:
        """
        Get information about specific Excel sheet.
           
        Args:
            file_path: Path to Excel file
            sheet_name: Name of the sheet
            
        Returns:
            Dictionary with sheet information
        """
        return self.excel_reader.get_sheet_info(file_path, sheet_name)
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the currently loaded data.
        
        Returns:
            DataFrame with current data or None if no data loaded
        """
        return self.current_data
        
    def reload_current_data(self):
        """
        Reload the currently loaded data file.
        
        Raises:
            ValueError: If no data file is currently loaded
            Exception: If the file cannot be reloaded
        """
        if not self.current_file_path:
            raise ValueError("No data file is currently loaded")
            
        # Reload using the stored file path
        self.load_data_file(self.current_file_path)