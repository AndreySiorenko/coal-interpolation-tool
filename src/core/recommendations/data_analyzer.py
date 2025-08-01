"""
Data analyzer for geological survey data.

Analyzes spatial characteristics of borehole data to provide insights
for optimal interpolation method selection and parameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from scipy import spatial, stats
import warnings


@dataclass
class DataCharacteristics:
    """Container for data analysis results."""
    n_points: int
    dimensions: int
    bounds: Dict[str, Tuple[float, float]]
    density: float  # points per unit area/volume
    distribution_uniformity: float  # 0-1, higher is more uniform
    has_trend: bool
    trend_type: Optional[str]  # 'linear', 'quadratic', None
    anisotropy_ratio: float
    anisotropy_angle: float  # degrees
    statistics: Dict[str, Any]
    outlier_indices: List[int]
    clustering_score: float  # 0-1, higher means more clustered
    nearest_neighbor_stats: Dict[str, float]


class DataAnalyzer:
    """
    Analyzes spatial and statistical characteristics of geological data.
    
    Provides comprehensive analysis including:
    - Spatial density and distribution
    - Trend detection
    - Anisotropy analysis
    - Outlier detection
    - Clustering metrics
    """
    
    def __init__(self):
        """Initialize data analyzer."""
        self.data: Optional[pd.DataFrame] = None
        self.coordinates: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.kdtree: Optional[spatial.cKDTree] = None
        
    def analyze(self, 
                data: pd.DataFrame,
                x_col: str,
                y_col: str,
                value_col: str,
                z_col: Optional[str] = None) -> DataCharacteristics:
        """
        Perform comprehensive data analysis.
        
        Args:
            data: Input DataFrame
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column to analyze
            z_col: Z coordinate column (optional for 3D)
            
        Returns:
            DataCharacteristics object with analysis results
        """
        # Store data
        self.data = data
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
            
        self.coordinates = data[coord_cols].values
        self.values = data[value_col].values
        self.dimensions = len(coord_cols)
        
        # Build KD-tree for spatial analysis
        self.kdtree = spatial.cKDTree(self.coordinates)
        
        # Perform analyses
        bounds = self._calculate_bounds(coord_cols)
        density = self._calculate_density(bounds)
        uniformity = self._analyze_distribution()
        trend_info = self._detect_trends()
        aniso_ratio, aniso_angle = self._analyze_anisotropy()
        statistics = self._calculate_statistics()
        outliers = self._detect_outliers()
        clustering = self._analyze_clustering()
        nn_stats = self._nearest_neighbor_analysis()
        
        return DataCharacteristics(
            n_points=len(data),
            dimensions=self.dimensions,
            bounds=bounds,
            density=density,
            distribution_uniformity=uniformity,
            has_trend=trend_info['has_trend'],
            trend_type=trend_info['trend_type'],
            anisotropy_ratio=aniso_ratio,
            anisotropy_angle=aniso_angle,
            statistics=statistics,
            outlier_indices=outliers,
            clustering_score=clustering,
            nearest_neighbor_stats=nn_stats
        )
    
    def _calculate_bounds(self, coord_cols: List[str]) -> Dict[str, Tuple[float, float]]:
        """Calculate coordinate bounds."""
        bounds = {}
        for i, col in enumerate(coord_cols):
            coord_data = self.coordinates[:, i]
            bounds[col] = (float(coord_data.min()), float(coord_data.max()))
        return bounds
    
    def _calculate_density(self, bounds: Dict[str, Tuple[float, float]]) -> float:
        """Calculate point density per unit area/volume."""
        # Calculate area or volume
        ranges = [b[1] - b[0] for b in bounds.values()]
        if self.dimensions == 2:
            area = ranges[0] * ranges[1]
            return len(self.data) / area
        else:  # 3D
            volume = ranges[0] * ranges[1] * ranges[2]
            return len(self.data) / volume
    
    def _analyze_distribution(self) -> float:
        """
        Analyze spatial distribution uniformity.
        
        Returns:
            Uniformity score (0-1), higher means more uniform
        """
        # Divide space into grid cells and count points
        n_bins = int(np.sqrt(len(self.data) / 5))  # Adaptive binning
        n_bins = max(3, min(n_bins, 20))  # Reasonable limits
        
        if self.dimensions == 2:
            hist, _, _ = np.histogram2d(
                self.coordinates[:, 0],
                self.coordinates[:, 1],
                bins=n_bins
            )
            expected_per_cell = len(self.data) / (n_bins * n_bins)
        else:  # 3D
            hist, _ = np.histogramdd(self.coordinates, bins=n_bins)
            expected_per_cell = len(self.data) / (n_bins ** 3)
        
        # Calculate uniformity using coefficient of variation
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0]  # Non-empty cells
        
        if len(hist_flat) == 0:
            return 0.0
            
        cv = np.std(hist_flat) / np.mean(hist_flat) if np.mean(hist_flat) > 0 else 1.0
        uniformity = 1.0 / (1.0 + cv)  # Convert to 0-1 scale
        
        return float(uniformity)
    
    def _detect_trends(self) -> Dict[str, Any]:
        """
        Detect spatial trends in the data.
        
        Returns:
            Dictionary with trend information
        """
        if len(self.data) < 10:
            return {'has_trend': False, 'trend_type': None, 'r_squared': 0.0}
        
        # Fit linear trend
        if self.dimensions == 2:
            X = self.coordinates
        else:  # Use only X,Y for trend in 3D
            X = self.coordinates[:, :2]
            
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Linear regression
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, self.values, rcond=None)
            
            # Calculate R-squared
            ss_total = np.sum((self.values - np.mean(self.values)) ** 2)
            ss_residual = residuals[0] if len(residuals) > 0 else 0
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
            # Determine if trend is significant
            has_trend = r_squared > 0.1  # 10% variance explained
            
            # Check for quadratic trend if linear is significant
            trend_type = 'linear' if has_trend else None
            
            if has_trend and len(self.data) > 20:
                # Try quadratic terms
                X_quad = np.column_stack([
                    X_with_intercept,
                    X[:, 0]**2,  # x^2
                    X[:, 1]**2,  # y^2  
                    X[:, 0] * X[:, 1]  # xy
                ])
                
                coeffs_quad, residuals_quad, rank_quad, s_quad = np.linalg.lstsq(
                    X_quad, self.values, rcond=None
                )
                
                ss_residual_quad = residuals_quad[0] if len(residuals_quad) > 0 else 0
                r_squared_quad = 1 - (ss_residual_quad / ss_total) if ss_total > 0 else 0
                
                # If quadratic significantly better
                if r_squared_quad > r_squared * 1.2:  # 20% improvement
                    trend_type = 'quadratic'
                    r_squared = r_squared_quad
                    
        except Exception:
            return {'has_trend': False, 'trend_type': None, 'r_squared': 0.0}
            
        return {
            'has_trend': has_trend,
            'trend_type': trend_type,
            'r_squared': float(r_squared)
        }
    
    def _analyze_anisotropy(self) -> Tuple[float, float]:
        """
        Analyze directional variation (anisotropy).
        
        Returns:
            Tuple of (anisotropy_ratio, anisotropy_angle)
        """
        if self.dimensions != 2 or len(self.data) < 30:
            return 1.0, 0.0
            
        # Calculate variogram in different directions
        n_directions = 8
        angles = np.linspace(0, 180, n_directions, endpoint=False)
        max_ranges = []
        
        for angle in angles:
            # Rotate coordinates
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            rotated_x = (self.coordinates[:, 0] * cos_a + 
                         self.coordinates[:, 1] * sin_a)
            
            # Calculate range in this direction
            range_val = np.ptp(rotated_x)
            max_ranges.append(range_val)
        
        max_ranges = np.array(max_ranges)
        
        # Find principal directions
        max_idx = np.argmax(max_ranges)
        min_idx = np.argmin(max_ranges)
        
        anisotropy_ratio = max_ranges[min_idx] / max_ranges[max_idx] if max_ranges[max_idx] > 0 else 1.0
        anisotropy_angle = angles[max_idx]
        
        return float(anisotropy_ratio), float(anisotropy_angle)
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistical properties of values."""
        stats_dict = {
            'mean': float(np.mean(self.values)),
            'std': float(np.std(self.values)),
            'min': float(np.min(self.values)),
            'max': float(np.max(self.values)),
            'median': float(np.median(self.values)),
            'q25': float(np.percentile(self.values, 25)),
            'q75': float(np.percentile(self.values, 75)),
            'iqr': float(np.percentile(self.values, 75) - np.percentile(self.values, 25)),
            'cv': float(np.std(self.values) / np.mean(self.values)) if np.mean(self.values) != 0 else 0.0
        }
        
        # Skewness and kurtosis
        try:
            stats_dict['skewness'] = float(stats.skew(self.values))
            stats_dict['kurtosis'] = float(stats.kurtosis(self.values))
        except Exception:
            stats_dict['skewness'] = 0.0
            stats_dict['kurtosis'] = 0.0
            
        return stats_dict
    
    def _detect_outliers(self) -> List[int]:
        """
        Detect outliers using IQR method.
        
        Returns:
            List of outlier indices
        """
        q1 = np.percentile(self.values, 25)
        q3 = np.percentile(self.values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (self.values < lower_bound) | (self.values > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        
        return outlier_indices
    
    def _analyze_clustering(self) -> float:
        """
        Analyze spatial clustering using nearest neighbor distances.
        
        Returns:
            Clustering score (0-1), higher means more clustered
        """
        if len(self.data) < 10:
            return 0.5
            
        # Get nearest neighbor distances
        distances, _ = self.kdtree.query(self.coordinates, k=2)
        nn_distances = distances[:, 1]  # Second column is nearest neighbor
        
        # Calculate expected distance for uniform distribution
        if self.dimensions == 2:
            area = np.prod([b[1] - b[0] for b in list(self._calculate_bounds(
                ['x', 'y']).values())[:2]])
            expected_distance = 0.5 * np.sqrt(area / len(self.data))
        else:
            volume = np.prod([b[1] - b[0] for b in self._calculate_bounds(
                ['x', 'y', 'z']).values()])
            expected_distance = 0.5 * (volume / len(self.data)) ** (1/3)
        
        # Calculate clustering index
        actual_mean_distance = np.mean(nn_distances)
        clustering_index = expected_distance / actual_mean_distance if actual_mean_distance > 0 else 1.0
        
        # Convert to 0-1 scale
        clustering_score = 1.0 - 1.0 / (1.0 + clustering_index)
        
        return float(np.clip(clustering_score, 0, 1))
    
    def _nearest_neighbor_analysis(self) -> Dict[str, float]:
        """Analyze nearest neighbor statistics."""
        if len(self.data) < 2:
            return {'mean_distance': 0.0, 'std_distance': 0.0, 'min_distance': 0.0}
            
        distances, _ = self.kdtree.query(self.coordinates, k=2)
        nn_distances = distances[:, 1]
        
        return {
            'mean_distance': float(np.mean(nn_distances)),
            'std_distance': float(np.std(nn_distances)),
            'min_distance': float(np.min(nn_distances)),
            'max_distance': float(np.max(nn_distances))
        }