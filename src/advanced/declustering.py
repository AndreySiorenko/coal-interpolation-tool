"""
Declustering module for coal deposit data.

Provides methods for reducing spatial clustering bias in data:
- Cell declustering (regular grid-based)
- Polygon declustering (irregular polygon-based)
- Distance-based declustering
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
import logging
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon


@dataclass
class DeclusteringResult:
    """Container for declustering results."""
    declustered_data: pd.DataFrame
    weights: np.ndarray
    original_count: int
    effective_count: float  # Sum of weights
    declustering_method: str
    quality_metrics: Dict[str, float]
    declustering_info: Dict[str, Any]


class CellDeclusterer:
    """
    Cell-based declustering for spatial data.
    
    Implements regular grid-based declustering to reduce spatial clustering
    bias by assigning weights based on sample density in grid cells.
    """
    
    def __init__(self, 
                 cell_size: Optional[float] = None,
                 min_cells_x: int = 10,
                 min_cells_y: int = 10,
                 max_cells: int = 1000):
        """
        Initialize cell declusterer.
        
        Args:
            cell_size: Fixed cell size (if None, calculated automatically)
            min_cells_x: Minimum number of cells in X direction
            min_cells_y: Minimum number of cells in Y direction
            max_cells: Maximum total number of cells
        """
        self.cell_size = cell_size
        self.min_cells_x = min_cells_x
        self.min_cells_y = min_cells_y
        self.max_cells = max_cells
        self.logger = logging.getLogger(__name__)
    
    def decluster(self,
                 data: pd.DataFrame,
                 x_col: str,
                 y_col: str,
                 value_col: str,
                 z_col: Optional[str] = None) -> DeclusteringResult:
        """
        Perform cell-based declustering.
        
        Args:
            data: Input data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column for statistics
            z_col: Z coordinate column (optional, for 3D declustering)
            
        Returns:
            DeclusteringResult with weights and declustered data
        """
        self.logger.info("Starting cell-based declustering")
        
        # Get coordinates
        x = data[x_col].values
        y = data[y_col].values
        
        # Determine cell size if not provided
        if self.cell_size is None:
            cell_size = self._calculate_optimal_cell_size(x, y)
        else:
            cell_size = self.cell_size
        
        self.logger.info(f"Using cell size: {cell_size:.2f}")
        
        # Create grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Expand slightly to ensure all points are included
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.01
        x_max += x_range * 0.01
        y_min -= y_range * 0.01
        y_max += y_range * 0.01
        
        # Grid dimensions
        n_cells_x = max(self.min_cells_x, int(np.ceil((x_max - x_min) / cell_size)))
        n_cells_y = max(self.min_cells_y, int(np.ceil((y_max - y_min) / cell_size)))
        
        # Limit total cells
        if n_cells_x * n_cells_y > self.max_cells:
            scale_factor = np.sqrt(self.max_cells / (n_cells_x * n_cells_y))
            n_cells_x = int(n_cells_x * scale_factor)
            n_cells_y = int(n_cells_y * scale_factor)
            cell_size = min((x_max - x_min) / n_cells_x, (y_max - y_min) / n_cells_y)
        
        # Assign points to cells
        cell_x = np.floor((x - x_min) / cell_size).astype(int)
        cell_y = np.floor((y - y_min) / cell_size).astype(int)
        
        # Ensure points are within bounds
        cell_x = np.clip(cell_x, 0, n_cells_x - 1)
        cell_y = np.clip(cell_y, 0, n_cells_y - 1)
        
        # Create cell IDs
        cell_ids = cell_y * n_cells_x + cell_x
        
        # Count samples per cell
        unique_cells, cell_counts = np.unique(cell_ids, return_counts=True)
        cell_count_dict = dict(zip(unique_cells, cell_counts))
        
        # Calculate weights (inverse of cell count)
        weights = np.array([1.0 / cell_count_dict[cell_id] for cell_id in cell_ids])
        
        # Normalize weights to sum to original count
        weights = weights * len(data) / weights.sum()
        
        # Create declustered dataset
        declustered_data = data.copy()
        declustered_data['cell_x'] = cell_x
        declustered_data['cell_y'] = cell_y
        declustered_data['cell_id'] = cell_ids
        declustered_data['decluster_weight'] = weights
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            data, declustered_data, weights, x_col, y_col, value_col
        )
        
        # Declustering info
        declustering_info = {
            'cell_size': cell_size,
            'n_cells_x': n_cells_x,
            'n_cells_y': n_cells_y,
            'total_cells': n_cells_x * n_cells_y,
            'occupied_cells': len(unique_cells),
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max)
        }
        
        return DeclusteringResult(
            declustered_data=declustered_data,
            weights=weights,
            original_count=len(data),
            effective_count=weights.sum(),
            declustering_method='cell_based',
            quality_metrics=quality_metrics,
            declustering_info=declustering_info
        )
    
    def _calculate_optimal_cell_size(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate optimal cell size based on data distribution."""
        # Use various methods and take median
        methods = []
        
        # Method 1: Based on data extent and target cell count
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        area = x_range * y_range
        target_cells = min(self.max_cells, len(x) // 2)  # Target 2 samples per cell on average
        cell_area = area / target_cells
        methods.append(np.sqrt(cell_area))
        
        # Method 2: Based on average nearest neighbor distance
        if len(x) > 1:
            coords = np.column_stack([x, y])
            kdtree = cKDTree(coords)
            distances, _ = kdtree.query(coords, k=2)  # k=2 to exclude self
            avg_nn_distance = np.mean(distances[:, 1])
            methods.append(avg_nn_distance * 2)  # Cells should be ~2x NN distance
        
        # Method 3: Based on sample density
        density = len(x) / area if area > 0 else 1
        optimal_samples_per_cell = 3  # Target samples per cell
        cell_area = optimal_samples_per_cell / density
        methods.append(np.sqrt(cell_area))
        
        # Take median of methods
        optimal_size = np.median(methods)
        
        # Apply constraints
        min_size = min(x_range, y_range) / 50  # At least 50 cells in smallest dimension
        max_size = min(x_range, y_range) / 5   # At most 5 cells in smallest dimension
        
        return np.clip(optimal_size, min_size, max_size)
    
    def _calculate_quality_metrics(self,
                                 original_data: pd.DataFrame,
                                 declustered_data: pd.DataFrame,
                                 weights: np.ndarray,
                                 x_col: str,
                                 y_col: str,
                                 value_col: str) -> Dict[str, float]:
        """Calculate declustering quality metrics."""
        metrics = {}
        
        # Weight distribution statistics
        metrics['min_weight'] = float(weights.min())
        metrics['max_weight'] = float(weights.max())
        metrics['mean_weight'] = float(weights.mean())
        metrics['std_weight'] = float(weights.std())
        metrics['weight_cv'] = float(weights.std() / weights.mean())
        
        # Effective sample size
        effective_n = (weights.sum() ** 2) / (weights ** 2).sum()
        metrics['effective_sample_size'] = float(effective_n)
        metrics['efficiency'] = float(effective_n / len(weights))
        
        # Spatial clustering metrics
        coords = original_data[[x_col, y_col]].values
        if len(coords) > 1:
            kdtree = cKDTree(coords)
            distances, _ = kdtree.query(coords, k=min(6, len(coords)))  # Up to 5 neighbors
            
            # Average nearest neighbor distance
            avg_nn_dist = np.mean(distances[:, 1])
            metrics['avg_nearest_neighbor_distance'] = float(avg_nn_dist)
            
            # Clark-Evans clustering index
            area = ((coords[:, 0].max() - coords[:, 0].min()) * 
                   (coords[:, 1].max() - coords[:, 1].min()))
            density = len(coords) / area if area > 0 else 0
            expected_nn_dist = 0.5 / np.sqrt(density) if density > 0 else 0
            
            if expected_nn_dist > 0:
                clark_evans = avg_nn_dist / expected_nn_dist
                metrics['clark_evans_index'] = float(clark_evans)
            else:
                metrics['clark_evans_index'] = 1.0
        
        # Value statistics comparison
        if value_col in original_data.columns:
            original_values = original_data[value_col].dropna()
            if len(original_values) > 0:
                # Original statistics
                orig_mean = original_values.mean()
                orig_std = original_values.std()
                
                # Weighted statistics
                valid_mask = ~pd.isna(declustered_data[value_col])
                if valid_mask.any():
                    weighted_values = declustered_data.loc[valid_mask, value_col].values
                    sample_weights = weights[valid_mask]
                    
                    weighted_mean = np.average(weighted_values, weights=sample_weights)
                    weighted_var = np.average((weighted_values - weighted_mean) ** 2, 
                                            weights=sample_weights)
                    weighted_std = np.sqrt(weighted_var)
                    
                    metrics['original_mean'] = float(orig_mean)
                    metrics['weighted_mean'] = float(weighted_mean)
                    metrics['mean_change_pct'] = float((weighted_mean - orig_mean) / orig_mean * 100)
                    
                    metrics['original_std'] = float(orig_std)
                    metrics['weighted_std'] = float(weighted_std)
                    if orig_std > 0:
                        metrics['std_change_pct'] = float((weighted_std - orig_std) / orig_std * 100)
        
        return metrics


class PolygonDeclusterer:
    """
    Polygon-based declustering for spatial data.
    
    Uses Voronoi polygons or custom polygons to assign weights
    based on polygon areas and sample densities.
    """
    
    def __init__(self, 
                 polygon_method: str = 'voronoi',
                 boundary_buffer: float = 0.1):
        """
        Initialize polygon declusterer.
        
        Args:
            polygon_method: Method for polygon generation ('voronoi', 'custom')
            boundary_buffer: Buffer fraction for boundary extension
        """
        self.polygon_method = polygon_method
        self.boundary_buffer = boundary_buffer
        self.logger = logging.getLogger(__name__)
    
    def decluster(self,
                 data: pd.DataFrame,
                 x_col: str,
                 y_col: str,
                 value_col: str,
                 boundary_polygon: Optional[np.ndarray] = None) -> DeclusteringResult:
        """
        Perform polygon-based declustering.
        
        Args:
            data: Input data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column for statistics
            boundary_polygon: Optional boundary polygon vertices
            
        Returns:
            DeclusteringResult with weights and declustered data
        """
        self.logger.info(f"Starting polygon-based declustering ({self.polygon_method})")
        
        coords = data[[x_col, y_col]].values
        
        if self.polygon_method == 'voronoi':
            weights = self._voronoi_declustering(coords, boundary_polygon)
        else:
            raise ValueError(f"Unknown polygon method: {self.polygon_method}")
        
        # Normalize weights
        weights = weights * len(data) / weights.sum()
        
        # Create declustered dataset
        declustered_data = data.copy()
        declustered_data['decluster_weight'] = weights
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            data, declustered_data, weights, x_col, y_col, value_col
        )
        
        # Declustering info
        declustering_info = {
            'polygon_method': self.polygon_method,
            'boundary_buffer': self.boundary_buffer,
            'has_boundary': boundary_polygon is not None
        }
        
        return DeclusteringResult(
            declustered_data=declustered_data,
            weights=weights,
            original_count=len(data),
            effective_count=weights.sum(),
            declustering_method='polygon_based',
            quality_metrics=quality_metrics,
            declustering_info=declustering_info
        )
    
    def _voronoi_declustering(self,
                            coords: np.ndarray,
                            boundary_polygon: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform Voronoi-based declustering."""
        if len(coords) < 3:
            return np.ones(len(coords))
        
        # Create Voronoi diagram
        vor = Voronoi(coords)
        
        # Calculate polygon areas
        areas = []
        
        for i, point_region in enumerate(vor.point_region):
            region = vor.regions[point_region]
            
            if not region or -1 in region:
                # Unbounded region - assign median area
                areas.append(np.nan)
            else:
                # Calculate polygon area
                polygon_vertices = vor.vertices[region]
                area = self._polygon_area(polygon_vertices)
                areas.append(area)
        
        areas = np.array(areas)
        
        # Handle unbounded regions
        finite_areas = areas[~np.isnan(areas)]
        if len(finite_areas) > 0:
            median_area = np.median(finite_areas)
            areas[np.isnan(areas)] = median_area
        else:
            # All regions unbounded - use equal weights
            areas.fill(1.0)
        
        # Convert areas to weights (larger area = higher weight)
        if np.std(areas) > 0:
            weights = areas / np.mean(areas)
        else:
            weights = np.ones_like(areas)
        
        return weights
    
    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate area of polygon using shoelace formula."""
        if len(vertices) < 3:
            return 0.0
        
        x = vertices[:, 0]
        y = vertices[:, 1]
        
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                           for i in range(-1, len(x) - 1)))
    
    def _calculate_quality_metrics(self,
                                 original_data: pd.DataFrame,
                                 declustered_data: pd.DataFrame,
                                 weights: np.ndarray,
                                 x_col: str,
                                 y_col: str,
                                 value_col: str) -> Dict[str, float]:
        """Calculate declustering quality metrics."""
        metrics = {}
        
        # Weight distribution statistics
        metrics['min_weight'] = float(weights.min())
        metrics['max_weight'] = float(weights.max())
        metrics['mean_weight'] = float(weights.mean())
        metrics['std_weight'] = float(weights.std())
        metrics['weight_cv'] = float(weights.std() / weights.mean())
        
        # Effective sample size
        effective_n = (weights.sum() ** 2) / (weights ** 2).sum()
        metrics['effective_sample_size'] = float(effective_n)
        metrics['efficiency'] = float(effective_n / len(weights))
        
        # Value statistics comparison
        if value_col in original_data.columns:
            original_values = original_data[value_col].dropna()
            if len(original_values) > 0:
                orig_mean = original_values.mean()
                orig_std = original_values.std()
                
                # Weighted statistics
                valid_mask = ~pd.isna(declustered_data[value_col])
                if valid_mask.any():
                    weighted_values = declustered_data.loc[valid_mask, value_col].values
                    sample_weights = weights[valid_mask]
                    
                    weighted_mean = np.average(weighted_values, weights=sample_weights)
                    weighted_var = np.average((weighted_values - weighted_mean) ** 2, 
                                            weights=sample_weights)
                    weighted_std = np.sqrt(weighted_var)
                    
                    metrics['original_mean'] = float(orig_mean)
                    metrics['weighted_mean'] = float(weighted_mean)
                    metrics['mean_change_pct'] = float((weighted_mean - orig_mean) / orig_mean * 100)
                    
                    metrics['original_std'] = float(orig_std)
                    metrics['weighted_std'] = float(weighted_std)
                    if orig_std > 0:
                        metrics['std_change_pct'] = float((weighted_std - orig_std) / orig_std * 100)
        
        return metrics


class DistanceDeclusterer:
    """
    Distance-based declustering using various distance metrics.
    
    Assigns weights based on local point density calculated using
    distance to nearest neighbors.
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 distance_power: float = 2.0):
        """
        Initialize distance declusterer.
        
        Args:
            n_neighbors: Number of nearest neighbors to consider
            distance_power: Power for distance weighting
        """
        self.n_neighbors = n_neighbors
        self.distance_power = distance_power
        self.logger = logging.getLogger(__name__)
    
    def decluster(self,
                 data: pd.DataFrame,
                 x_col: str,
                 y_col: str,
                 value_col: str) -> DeclusteringResult:
        """
        Perform distance-based declustering.
        
        Args:
            data: Input data
            x_col: X coordinate column
            y_col: Y coordinate column  
            value_col: Value column for statistics
            
        Returns:
            DeclusteringResult with weights and declustered data
        """
        self.logger.info("Starting distance-based declustering")
        
        coords = data[[x_col, y_col]].values
        
        if len(coords) <= self.n_neighbors:
            # Too few points for meaningful declustering
            weights = np.ones(len(coords))
        else:
            # Build KDTree
            kdtree = cKDTree(coords)
            
            # Find k nearest neighbors (k+1 to exclude self)
            distances, _ = kdtree.query(coords, k=self.n_neighbors + 1)
            
            # Use distances to neighbors (exclude self at index 0)
            neighbor_distances = distances[:, 1:]
            
            # Calculate local density (inverse of average distance)
            avg_distances = np.mean(neighbor_distances, axis=1)
            densities = 1.0 / (avg_distances ** self.distance_power + 1e-10)
            
            # Convert densities to weights (inverse of density)
            weights = 1.0 / (densities + 1e-10)
        
        # Normalize weights
        weights = weights * len(data) / weights.sum()
        
        # Create declustered dataset
        declustered_data = data.copy()
        declustered_data['decluster_weight'] = weights
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            data, declustered_data, weights, x_col, y_col, value_col
        )
        
        # Declustering info
        declustering_info = {
            'n_neighbors': self.n_neighbors,
            'distance_power': self.distance_power
        }
        
        return DeclusteringResult(
            declustered_data=declustered_data,
            weights=weights,
            original_count=len(data),
            effective_count=weights.sum(),
            declustering_method='distance_based',
            quality_metrics=quality_metrics,
            declustering_info=declustering_info
        )
    
    def _calculate_quality_metrics(self,
                                 original_data: pd.DataFrame,
                                 declustered_data: pd.DataFrame,
                                 weights: np.ndarray,
                                 x_col: str,
                                 y_col: str,
                                 value_col: str) -> Dict[str, float]:
        """Calculate declustering quality metrics."""
        metrics = {}
        
        # Weight distribution statistics
        metrics['min_weight'] = float(weights.min())
        metrics['max_weight'] = float(weights.max())
        metrics['mean_weight'] = float(weights.mean())
        metrics['std_weight'] = float(weights.std())
        metrics['weight_cv'] = float(weights.std() / weights.mean())
        
        # Effective sample size
        effective_n = (weights.sum() ** 2) / (weights ** 2).sum()
        metrics['effective_sample_size'] = float(effective_n)
        metrics['efficiency'] = float(effective_n / len(weights))
        
        return metrics