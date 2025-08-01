"""
Base abstract class for interpolation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import warnings


@dataclass
class InterpolationParameters:
    """Base class for interpolation parameters."""
    pass


@dataclass
class SearchParameters:
    """Parameters for neighborhood search in interpolation."""
    search_radius: float = 1000.0  # Search radius in map units
    min_points: int = 1            # Minimum points required
    max_points: int = 12           # Maximum points to use
    use_sectors: bool = False      # Use sectorial search
    n_sectors: int = 4             # Number of sectors
    min_per_sector: int = 1        # Minimum points per sector
    max_per_sector: int = 4        # Maximum points per sector
    
    # Anisotropy parameters
    anisotropy_ratio: float = 1.0  # Ratio of minor to major axis
    anisotropy_angle: float = 0.0  # Angle of major axis in degrees


class InterpolationError(Exception):
    """Base exception for interpolation errors."""
    pass


class FittingError(InterpolationError):
    """Exception raised when interpolator cannot be fitted."""
    pass


class PredictionError(InterpolationError):
    """Exception raised during prediction."""
    pass


class BaseInterpolator(ABC):
    """
    Abstract base class for all interpolation methods.
    
    Provides common interface and functionality for interpolation algorithms
    used in coal deposit modeling.
    """
    
    def __init__(self, search_params: Optional[SearchParameters] = None):
        """
        Initialize interpolator.
        
        Args:
            search_params: Search parameters for neighborhood selection
        """
        self.search_params = search_params or SearchParameters()
        self.is_fitted = False
        self.training_data: Optional[pd.DataFrame] = None
        self.coordinate_columns: Dict[str, str] = {}
        self.value_column: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def fit(self, 
            data: pd.DataFrame,
            x_col: str,
            y_col: str, 
            value_col: str,
            z_col: Optional[str] = None,
            **kwargs) -> 'BaseInterpolator':
        """
        Fit the interpolator to training data.
        
        Args:
            data: Training data DataFrame
            x_col: Name of X coordinate column
            y_col: Name of Y coordinate column  
            value_col: Name of value column to interpolate
            z_col: Name of Z coordinate column (for 3D interpolation)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            FittingError: If fitting fails
        """
        pass
    
    @abstractmethod
    def predict(self, 
                points: Union[np.ndarray, pd.DataFrame, List[Tuple[float, float]]],
                **kwargs) -> np.ndarray:
        """
        Predict values at given points.
        
        Args:
            points: Points to predict at. Can be:
                - np.ndarray of shape (n_points, 2) or (n_points, 3) 
                - DataFrame with coordinate columns
                - List of (x, y) or (x, y, z) tuples
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predicted values
            
        Raises:
            PredictionError: If prediction fails
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """
        Get human-readable name of interpolation method.
        
        Returns:
            Method name string
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameters of the interpolator.
        
        Returns:
            Dictionary of parameter names and values
        """
        pass
    
    def set_parameters(self, **params) -> 'BaseInterpolator':
        """
        Set interpolator parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        # Update search parameters if provided
        if hasattr(self, 'search_params'):
            for key, value in params.items():
                if hasattr(self.search_params, key):
                    setattr(self.search_params, key, value)
                    
        return self
    
    def _validate_training_data(self, 
                               data: pd.DataFrame,
                               x_col: str,
                               y_col: str,
                               value_col: str,
                               z_col: Optional[str] = None) -> pd.DataFrame:
        """
        Validate and prepare training data.
        
        Args:
            data: Input DataFrame
            x_col: X coordinate column name
            y_col: Y coordinate column name
            value_col: Value column name
            z_col: Z coordinate column name (optional)
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            FittingError: If data validation fails
        """
        # Check if columns exist
        required_cols = [x_col, y_col, value_col]
        if z_col:
            required_cols.append(z_col)
            
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise FittingError(f"Missing columns: {missing_cols}")
        
        # Create working copy
        work_data = data[required_cols].copy()
        
        # Check for non-numeric columns
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(work_data[col]):
                try:
                    work_data[col] = pd.to_numeric(work_data[col], errors='coerce')
                except Exception:
                    raise FittingError(f"Column '{col}' cannot be converted to numeric")
        
        # Remove rows with any NaN values
        initial_count = len(work_data)
        work_data = work_data.dropna()
        final_count = len(work_data)
        
        if final_count == 0:
            raise FittingError("No valid data points after removing NaN values")
            
        if final_count < initial_count:
            warnings.warn(f"Removed {initial_count - final_count} rows containing NaN values")
        
        # Check for minimum number of points
        min_required = max(3, self.search_params.min_points)
        if final_count < min_required:
            raise FittingError(f"Insufficient data points. Need at least {min_required}, got {final_count}")
        
        # Check for duplicate coordinates
        coord_cols = [x_col, y_col]
        if z_col:
            coord_cols.append(z_col)
            
        duplicate_mask = work_data[coord_cols].duplicated(keep='first')
        duplicate_count = duplicate_mask.sum()
        
        if duplicate_count > 0:
            warnings.warn(f"Found {duplicate_count} duplicate coordinates. Keeping first occurrence.")
            work_data = work_data[~duplicate_mask]
        
        return work_data
    
    def _parse_prediction_points(self, 
                                points: Union[np.ndarray, pd.DataFrame, List[Tuple]]) -> np.ndarray:
        """
        Parse prediction points into standardized numpy array.
        
        Args:
            points: Points in various formats
            
        Returns:
            numpy array of shape (n_points, n_dims)
            
        Raises:
            PredictionError: If points cannot be parsed
        """
        try:
            if isinstance(points, pd.DataFrame):
                # Use stored coordinate column names
                x_col = self.coordinate_columns.get('X')
                y_col = self.coordinate_columns.get('Y') 
                z_col = self.coordinate_columns.get('Z')
                
                if not x_col or not y_col:
                    raise PredictionError("Coordinate column names not set. Fit interpolator first.")
                
                coord_cols = [x_col, y_col]
                if z_col and z_col in points.columns:
                    coord_cols.append(z_col)
                    
                return points[coord_cols].values
                
            elif isinstance(points, (list, tuple)):
                return np.array(points)
                
            elif isinstance(points, np.ndarray):
                return points
                
            else:
                raise PredictionError(f"Unsupported points type: {type(points)}")
                
        except Exception as e:
            raise PredictionError(f"Error parsing prediction points: {e}")
    
    def _find_neighbors(self, 
                       target_point: np.ndarray,
                       training_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find neighboring points for interpolation.
        
        Args:
            target_point: Point to interpolate at (x, y, [z])
            training_points: Training data points
            
        Returns:
            Tuple of (neighbor_indices, distances)
        """
        # Calculate distances
        if len(target_point) == 2:  # 2D
            distances = np.sqrt(
                (training_points[:, 0] - target_point[0])**2 + 
                (training_points[:, 1] - target_point[1])**2
            )
        else:  # 3D
            distances = np.sqrt(
                (training_points[:, 0] - target_point[0])**2 + 
                (training_points[:, 1] - target_point[1])**2 +
                (training_points[:, 2] - target_point[2])**2
            )
        
        # Filter by search radius
        within_radius = distances <= self.search_params.search_radius
        candidate_indices = np.where(within_radius)[0]
        candidate_distances = distances[within_radius]
        
        if len(candidate_indices) < self.search_params.min_points:
            # If not enough points within radius, use closest points
            sorted_indices = np.argsort(distances)
            candidate_indices = sorted_indices[:self.search_params.max_points]
            candidate_distances = distances[candidate_indices]
        
        # Sort by distance and limit to max_points
        sorted_order = np.argsort(candidate_distances)
        n_select = min(len(sorted_order), self.search_params.max_points)
        
        selected_indices = candidate_indices[sorted_order[:n_select]]
        selected_distances = candidate_distances[sorted_order[:n_select]]
        
        return selected_indices, selected_distances
    
    def predict_grid(self, 
                    x_range: Tuple[float, float],
                    y_range: Tuple[float, float], 
                    grid_size: Union[int, Tuple[int, int]],
                    z_value: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict values on a regular grid.
        
        Args:
            x_range: (min_x, max_x) range
            y_range: (min_y, max_y) range  
            grid_size: Grid resolution (n_points) or (nx, ny)
            z_value: Fixed Z value for 3D interpolation
            
        Returns:
            Tuple of (x_grid, y_grid, predicted_values)
        """
        if not self.is_fitted:
            raise PredictionError("Interpolator must be fitted before prediction")
        
        # Handle grid size
        if isinstance(grid_size, int):
            nx = ny = grid_size
        else:
            nx, ny = grid_size
        
        # Create coordinate grids
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)
        
        # Create prediction points
        if z_value is not None and 'Z' in self.coordinate_columns:
            points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_value)])
        else:
            points = np.column_stack([X.ravel(), Y.ravel()])
        
        # Predict values
        predicted = self.predict(points)
        
        # Reshape to grid
        predicted_grid = predicted.reshape(X.shape)
        
        return X, Y, predicted_grid
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training data.
        
        Returns:
            Dictionary with training data summary
        """
        if not self.is_fitted or self.training_data is None:
            return {}
        
        summary = {
            'n_points': len(self.training_data),
            'coordinate_columns': self.coordinate_columns,
            'value_column': self.value_column,
            'method': self.get_method_name(),
            'parameters': self.get_parameters(),
        }
        
        # Add coordinate statistics
        for coord_name, col_name in self.coordinate_columns.items():
            if col_name in self.training_data.columns:
                coord_data = self.training_data[col_name]
                summary[f'{coord_name}_range'] = (coord_data.min(), coord_data.max())
                summary[f'{coord_name}_mean'] = coord_data.mean()
        
        # Add value statistics
        if self.value_column and self.value_column in self.training_data.columns:
            value_data = self.training_data[self.value_column]
            summary['value_range'] = (value_data.min(), value_data.max())
            summary['value_mean'] = value_data.mean()
            summary['value_std'] = value_data.std()
        
        return summary