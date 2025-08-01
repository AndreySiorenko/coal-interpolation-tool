"""
IDW (Inverse Distance Weighted) interpolation implementation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Union, List
from dataclasses import dataclass
from scipy.spatial import cKDTree
import warnings

from .base import BaseInterpolator, InterpolationParameters, SearchParameters
from .base import FittingError, PredictionError


@dataclass
class IDWParameters(InterpolationParameters):
    """Parameters specific to IDW interpolation."""
    power: float = 2.0                # Power parameter (p in 1/d^p)
    smoothing: float = 0.0            # Smoothing parameter to avoid division by zero
    use_kdtree: bool = True           # Use KD-tree for neighbor search


class IDWInterpolator(BaseInterpolator):
    """
    Inverse Distance Weighted (IDW) interpolation.
    
    IDW is a deterministic spatial interpolation method that estimates values
    at unmeasured locations using a weighted average of values from nearby
    measured locations. Weights are inversely proportional to the distance
    raised to a power parameter.
    
    The IDW formula:
        Z(x0) = Σ(wi * zi) / Σ(wi)
        where wi = 1 / (di^p + ε)
        
    Parameters:
        search_params: Search parameters for finding neighbors
        idw_params: IDW-specific parameters including power and smoothing
    """
    
    def __init__(self, 
                 search_params: Optional[SearchParameters] = None,
                 idw_params: Optional[IDWParameters] = None):
        """Initialize IDW interpolator with given parameters."""
        super().__init__(search_params)
        self.idw_params = idw_params or IDWParameters()
        self.kdtree = None
        self.training_points = None
        self.training_values = None
        
    def get_method_name(self) -> str:
        """Return human-readable name of the interpolation method."""
        return "Inverse Distance Weighted (IDW)"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters of the interpolator."""
        params = {
            'power': self.idw_params.power,
            'smoothing': self.idw_params.smoothing,
            'use_kdtree': self.idw_params.use_kdtree,
            'search_radius': self.search_params.search_radius,
            'min_points': self.search_params.min_points,
            'max_points': self.search_params.max_points,
            'use_sectors': self.search_params.use_sectors,
            'n_sectors': self.search_params.n_sectors,
            'anisotropy_ratio': self.search_params.anisotropy_ratio,
            'anisotropy_angle': self.search_params.anisotropy_angle
        }
        return params
    
    def set_parameters(self, **params) -> 'IDWInterpolator':
        """Set interpolator parameters."""
        # Handle IDW-specific parameters
        for key in ['power', 'smoothing', 'use_kdtree']:
            if key in params:
                setattr(self.idw_params, key, params[key])
        
        # Let parent handle search parameters
        super().set_parameters(**params)
        
        # Rebuild KD-tree if necessary
        if self.is_fitted and self.idw_params.use_kdtree:
            self._build_kdtree()
            
        return self
    
    def fit(self, 
            data: pd.DataFrame,
            x_col: str,
            y_col: str,
            value_col: str,
            z_col: Optional[str] = None,
            **kwargs) -> 'IDWInterpolator':
        """
        Fit the IDW interpolator to training data.
        
        Args:
            data: Training data DataFrame
            x_col: Name of X coordinate column
            y_col: Name of Y coordinate column
            value_col: Name of value column to interpolate
            z_col: Name of Z coordinate column (for 3D interpolation)
            **kwargs: Additional parameters (e.g., power, smoothing)
            
        Returns:
            Self for method chaining
            
        Raises:
            FittingError: If fitting fails
        """
        # Update parameters if provided
        if kwargs:
            self.set_parameters(**kwargs)
        
        # Validate and prepare data
        self.training_data = self._validate_training_data(data, x_col, y_col, value_col, z_col)
        
        # Store column names
        self.coordinate_columns = {'X': x_col, 'Y': y_col}
        if z_col:
            self.coordinate_columns['Z'] = z_col
        self.value_column = value_col
        
        # Extract coordinates and values
        coord_cols = [x_col, y_col]
        if z_col:
            coord_cols.append(z_col)
            
        self.training_points = self.training_data[coord_cols].values
        self.training_values = self.training_data[value_col].values
        
        # Apply coordinate transformation for anisotropy if needed
        if self.search_params.anisotropy_ratio != 1.0 or self.search_params.anisotropy_angle != 0.0:
            self.training_points = self._apply_anisotropy_transform(self.training_points)
        
        # Build KD-tree for efficient neighbor search
        if self.idw_params.use_kdtree:
            self._build_kdtree()
        
        self.is_fitted = True
        
        # Store metadata
        self.metadata = {
            'n_dimensions': len(coord_cols),
            'data_bounds': {
                coord_name: (self.training_data[col_name].min(), 
                            self.training_data[col_name].max())
                for coord_name, col_name in self.coordinate_columns.items()
            }
        }
        
        return self
    
    def predict(self, 
                points: Union[np.ndarray, pd.DataFrame, List[Tuple[float, float]]],
                **kwargs) -> np.ndarray:
        """
        Predict values at given points using IDW interpolation.
        
        Args:
            points: Points to predict at
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predicted values
            
        Raises:
            PredictionError: If prediction fails
        """
        if not self.is_fitted:
            raise PredictionError("Interpolator must be fitted before prediction")
        
        # Parse prediction points
        pred_points = self._parse_prediction_points(points)
        
        # Apply anisotropy transformation if needed
        if self.search_params.anisotropy_ratio != 1.0 or self.search_params.anisotropy_angle != 0.0:
            pred_points = self._apply_anisotropy_transform(pred_points)
        
        # Initialize results array
        n_points = len(pred_points)
        results = np.full(n_points, np.nan)
        
        # Predict for each point
        for i, point in enumerate(pred_points):
            try:
                results[i] = self._predict_single_point(point)
            except Exception as e:
                warnings.warn(f"Failed to predict point {i}: {e}")
                
        return results
    
    def _predict_single_point(self, point: np.ndarray) -> float:
        """
        Predict value at a single point.
        
        Args:
            point: Coordinates of the point
            
        Returns:
            Interpolated value
        """
        # Find neighbors
        if self.search_params.use_sectors:
            neighbor_indices, distances = self._find_neighbors_sectoral(point)
        else:
            neighbor_indices, distances = self._find_neighbors_standard(point)
        
        # Check if we have enough neighbors
        if len(neighbor_indices) < self.search_params.min_points:
            if len(neighbor_indices) == 0:
                return np.nan
            # If not enough points but some exist, use what we have
            warnings.warn(f"Only {len(neighbor_indices)} neighbors found, "
                         f"minimum required is {self.search_params.min_points}")
        
        # Check for exact matches (distance = 0)
        exact_match = np.where(distances < 1e-10)[0]
        if len(exact_match) > 0:
            # Return the value of the exact match (or average if multiple)
            return np.mean(self.training_values[neighbor_indices[exact_match]])
        
        # Calculate weights
        weights = self._calculate_weights(distances)
        
        # Calculate weighted average
        neighbor_values = self.training_values[neighbor_indices]
        weighted_sum = np.sum(weights * neighbor_values)
        weight_sum = np.sum(weights)
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return np.nan
    
    def _calculate_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Calculate IDW weights from distances.
        
        Args:
            distances: Array of distances
            
        Returns:
            Array of weights
        """
        # Add smoothing to avoid division by zero
        adjusted_distances = distances + self.idw_params.smoothing
        
        # Calculate weights: w = 1 / d^p
        weights = 1.0 / np.power(adjusted_distances, self.idw_params.power)
        
        return weights
    
    def _find_neighbors_standard(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find neighbors using standard search (all directions).
        
        Args:
            point: Target point coordinates
            
        Returns:
            Tuple of (neighbor_indices, distances)
        """
        if self.idw_params.use_kdtree and self.kdtree is not None:
            # Use KD-tree for efficient search
            distances, indices = self.kdtree.query(
                point.reshape(1, -1),
                k=min(self.search_params.max_points, len(self.training_points)),
                distance_upper_bound=self.search_params.search_radius
            )
            
            # Remove invalid indices (those beyond search radius)
            valid_mask = distances[0] < np.inf
            neighbor_indices = indices[0][valid_mask]
            neighbor_distances = distances[0][valid_mask]
            
        else:
            # Use parent's method for brute force search
            neighbor_indices, neighbor_distances = self._find_neighbors(point, self.training_points)
        
        return neighbor_indices, neighbor_distances
    
    def _find_neighbors_sectoral(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find neighbors using sectoral search.
        
        Args:
            point: Target point coordinates
            
        Returns:
            Tuple of (neighbor_indices, distances)
        """
        # Calculate angles to all training points (2D only for now)
        if len(point) > 2:
            warnings.warn("Sectoral search is currently only supported for 2D data. "
                         "Falling back to standard search.")
            return self._find_neighbors_standard(point)
        
        # Calculate vectors from target point to all training points
        vectors = self.training_points[:, :2] - point[:2]
        distances = np.sqrt(np.sum(vectors**2, axis=1))
        
        # Filter by search radius
        within_radius = distances <= self.search_params.search_radius
        if not np.any(within_radius):
            # If no points within radius, use closest points
            sorted_indices = np.argsort(distances)
            n_use = min(self.search_params.max_points, len(distances))
            return sorted_indices[:n_use], distances[sorted_indices[:n_use]]
        
        # Calculate angles (in radians, 0 to 2π)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles[angles < 0] += 2 * np.pi
        
        # Divide into sectors
        sector_size = 2 * np.pi / self.search_params.n_sectors
        sectors = (angles / sector_size).astype(int)
        
        # Select points from each sector
        selected_indices = []
        selected_distances = []
        
        for sector in range(self.search_params.n_sectors):
            # Find points in this sector that are within radius
            sector_mask = (sectors == sector) & within_radius
            sector_indices = np.where(sector_mask)[0]
            
            if len(sector_indices) > 0:
                # Sort by distance within sector
                sector_distances = distances[sector_indices]
                sorted_order = np.argsort(sector_distances)
                
                # Select up to max_per_sector points
                n_select = min(self.search_params.max_per_sector, len(sorted_order))
                selected = sector_indices[sorted_order[:n_select]]
                
                selected_indices.extend(selected)
                selected_distances.extend(distances[selected])
        
        # Convert to arrays
        selected_indices = np.array(selected_indices)
        selected_distances = np.array(selected_distances)
        
        # Check minimum points requirement
        if len(selected_indices) < self.search_params.min_points:
            # Add more points from closest regardless of sector
            remaining_mask = ~np.isin(np.arange(len(distances)), selected_indices)
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_indices) > 0:
                remaining_distances = distances[remaining_indices]
                sorted_order = np.argsort(remaining_distances)
                n_add = self.search_params.min_points - len(selected_indices)
                n_add = min(n_add, len(remaining_indices))
                
                additional_indices = remaining_indices[sorted_order[:n_add]]
                selected_indices = np.concatenate([selected_indices, additional_indices])
                selected_distances = np.concatenate([selected_distances, distances[additional_indices]])
        
        return selected_indices, selected_distances
    
    def _apply_anisotropy_transform(self, points: np.ndarray) -> np.ndarray:
        """
        Apply anisotropy transformation to coordinates.
        
        Args:
            points: Original coordinates
            
        Returns:
            Transformed coordinates
        """
        if self.search_params.anisotropy_ratio == 1.0 and self.search_params.anisotropy_angle == 0.0:
            return points
        
        # Only transform X-Y coordinates (first 2 dimensions)
        transformed = points.copy()
        
        # Convert angle to radians
        angle_rad = np.radians(self.search_params.anisotropy_angle)
        
        # Create rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate to align with anisotropy axes
        x_rot = points[:, 0] * cos_a + points[:, 1] * sin_a
        y_rot = -points[:, 0] * sin_a + points[:, 1] * cos_a
        
        # Apply anisotropy scaling
        y_rot *= self.search_params.anisotropy_ratio
        
        # Rotate back
        transformed[:, 0] = x_rot * cos_a - y_rot * sin_a
        transformed[:, 1] = x_rot * sin_a + y_rot * cos_a
        
        return transformed
    
    def _build_kdtree(self):
        """Build KD-tree for efficient neighbor search."""
        if self.training_points is not None:
            self.kdtree = cKDTree(self.training_points)