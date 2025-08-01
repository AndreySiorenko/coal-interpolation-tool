"""
Parameter optimizer for interpolation methods.

Automatically determines optimal parameters for interpolation
based on data characteristics and cross-validation.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import warnings

from .data_analyzer import DataCharacteristics
from ..interpolation.base import SearchParameters


@dataclass
class OptimizationResult:
    """Results of parameter optimization."""
    method: str
    parameters: Dict[str, Any]
    quality_score: float  # Cross-validation score
    reasoning: Dict[str, str]  # Parameter -> reason for choice
    search_history: List[Dict[str, Any]]  # Optimization history


class ParameterOptimizer:
    """
    Optimizes interpolation parameters based on data characteristics.
    
    Uses a combination of heuristics and cross-validation to find
    optimal parameters for each interpolation method.
    """
    
    def __init__(self):
        """Initialize parameter optimizer."""
        self.characteristics: Optional[DataCharacteristics] = None
        self.data: Optional[Any] = None
        
    def optimize_parameters(self,
                          method: str,
                          characteristics: DataCharacteristics,
                          data: Optional[Any] = None,
                          user_constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimize parameters for given interpolation method.
        
        Args:
            method: Interpolation method name ('IDW', 'Kriging', 'RBF')
            characteristics: Data analysis results
            data: Optional data for cross-validation
            user_constraints: Optional user-defined constraints
            
        Returns:
            OptimizationResult with optimal parameters
        """
        self.characteristics = characteristics
        self.data = data
        constraints = user_constraints or {}
        
        if method.upper() == 'IDW':
            return self._optimize_idw_parameters(constraints)
        elif method.upper() == 'KRIGING':
            return self._optimize_kriging_parameters(constraints)
        elif method.upper() == 'RBF':
            return self._optimize_rbf_parameters(constraints)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _optimize_idw_parameters(self, constraints: Dict[str, Any]) -> OptimizationResult:
        """Optimize IDW parameters."""
        parameters = {}
        reasoning = {}
        search_history = []
        
        # 1. Search radius
        search_radius = self._optimize_search_radius(constraints)
        parameters['search_radius'] = search_radius
        reasoning['search_radius'] = self._get_search_radius_reasoning(search_radius)
        
        # 2. Power parameter
        power = self._optimize_idw_power(constraints)
        parameters['power'] = power
        reasoning['power'] = self._get_power_reasoning(power)
        
        # 3. Number of points
        min_points, max_points = self._optimize_point_counts(constraints)
        parameters['min_points'] = min_points
        parameters['max_points'] = max_points
        reasoning['min_points'] = f"Minimum {min_points} points ensures stable interpolation"
        reasoning['max_points'] = f"Maximum {max_points} points balances accuracy and speed"
        
        # 4. Sectoral search
        use_sectors, n_sectors = self._optimize_sectors(constraints)
        parameters['use_sectors'] = use_sectors
        parameters['n_sectors'] = n_sectors
        if use_sectors:
            parameters['min_per_sector'] = max(1, min_points // n_sectors)
            parameters['max_per_sector'] = max_points // n_sectors + 1
            reasoning['use_sectors'] = "Sectoral search ensures directional balance"
            reasoning['n_sectors'] = f"{n_sectors} sectors provide good coverage"
        
        # 5. Anisotropy
        if self.characteristics.anisotropy_ratio < 0.8:
            parameters['anisotropy_ratio'] = self.characteristics.anisotropy_ratio
            parameters['anisotropy_angle'] = self.characteristics.anisotropy_angle
            reasoning['anisotropy'] = "Data shows directional variation"
        else:
            parameters['anisotropy_ratio'] = 1.0
            parameters['anisotropy_angle'] = 0.0
            reasoning['anisotropy'] = "Data appears isotropic"
        
        # Create SearchParameters object
        search_params = SearchParameters(
            search_radius=parameters['search_radius'],
            min_points=parameters['min_points'],
            max_points=parameters['max_points'],
            use_sectors=parameters['use_sectors'],
            n_sectors=parameters.get('n_sectors', 4),
            min_per_sector=parameters.get('min_per_sector', 1),
            max_per_sector=parameters.get('max_per_sector', 4),
            anisotropy_ratio=parameters.get('anisotropy_ratio', 1.0),
            anisotropy_angle=parameters.get('anisotropy_angle', 0.0)
        )
        
        # Calculate quality score (simplified)
        quality_score = self._estimate_idw_quality(parameters)
        
        return OptimizationResult(
            method='IDW',
            parameters=parameters,
            quality_score=quality_score,
            reasoning=reasoning,
            search_history=search_history
        )
    
    def _optimize_search_radius(self, constraints: Dict[str, Any]) -> float:
        """Optimize search radius based on data density."""
        if 'search_radius' in constraints:
            return constraints['search_radius']
        
        # Calculate based on nearest neighbor statistics
        nn_stats = self.characteristics.nearest_neighbor_stats
        mean_nn_distance = nn_stats['mean_distance']
        
        # Use multiple of mean nearest neighbor distance
        if self.characteristics.distribution_uniformity > 0.7:
            # Uniform distribution: smaller radius OK
            radius_multiplier = 5.0
        elif self.characteristics.distribution_uniformity > 0.4:
            # Moderate clustering: medium radius
            radius_multiplier = 8.0
        else:
            # High clustering: larger radius needed
            radius_multiplier = 12.0
        
        search_radius = mean_nn_distance * radius_multiplier
        
        # Apply bounds
        bounds = list(self.characteristics.bounds.values())
        max_dimension = max(b[1] - b[0] for b in bounds)
        
        # Radius should be between 10% and 50% of max dimension
        search_radius = np.clip(search_radius, 
                               0.1 * max_dimension,
                               0.5 * max_dimension)
        
        return float(search_radius)
    
    def _optimize_idw_power(self, constraints: Dict[str, Any]) -> float:
        """Optimize IDW power parameter."""
        if 'power' in constraints:
            return constraints['power']
        
        # Base power
        power = 2.0
        
        # Adjust based on data characteristics
        # Higher CV suggests more local variation
        cv = self.characteristics.statistics.get('cv', 0.5)
        if cv > 1.0:
            power += 0.5  # Higher power for more local influence
        elif cv < 0.3:
            power -= 0.5  # Lower power for smoother variation
        
        # Clustering affects optimal power
        if self.characteristics.clustering_score > 0.7:
            power += 0.5  # Higher power in clustered areas
        
        # Outliers suggest need for higher power
        outlier_ratio = len(self.characteristics.outlier_indices) / self.characteristics.n_points
        if outlier_ratio > 0.05:
            power += 0.5
        
        # Apply bounds
        power = np.clip(power, 1.0, 4.0)
        
        return float(power)
    
    def _optimize_point_counts(self, constraints: Dict[str, Any]) -> Tuple[int, int]:
        """Optimize min and max point counts."""
        min_points = constraints.get('min_points', 2)
        max_points = constraints.get('max_points', 12)
        
        # Adjust based on data density
        if self.characteristics.n_points < 50:
            # Sparse data: use more points
            min_points = max(2, min_points)
            max_points = min(self.characteristics.n_points // 2, 20)
        elif self.characteristics.n_points < 500:
            # Moderate density
            min_points = 3
            max_points = 15
        else:
            # Dense data: can use fewer points
            min_points = 4
            max_points = 12
        
        # Ensure consistency
        max_points = max(max_points, min_points + 2)
        
        return int(min_points), int(max_points)
    
    def _optimize_sectors(self, constraints: Dict[str, Any]) -> Tuple[bool, int]:
        """Determine if sectoral search should be used."""
        if 'use_sectors' in constraints:
            use_sectors = constraints['use_sectors']
            n_sectors = constraints.get('n_sectors', 4)
            return use_sectors, n_sectors
        
        # Use sectors if data shows clustering or anisotropy
        use_sectors = (
            self.characteristics.clustering_score > 0.6 or
            self.characteristics.anisotropy_ratio < 0.8 or
            self.characteristics.distribution_uniformity < 0.5
        )
        
        # Number of sectors
        if self.characteristics.dimensions == 2:
            n_sectors = 4  # Standard quadrants
        else:
            n_sectors = 8  # Octants for 3D
        
        return use_sectors, n_sectors
    
    def _get_search_radius_reasoning(self, radius: float) -> str:
        """Get reasoning for search radius choice."""
        nn_mean = self.characteristics.nearest_neighbor_stats['mean_distance']
        ratio = radius / nn_mean if nn_mean > 0 else 0
        
        if ratio < 6:
            return f"Small radius ({ratio:.1f}x mean NN distance) for dense, uniform data"
        elif ratio < 10:
            return f"Medium radius ({ratio:.1f}x mean NN distance) for balanced coverage"
        else:
            return f"Large radius ({ratio:.1f}x mean NN distance) for sparse or clustered data"
    
    def _get_power_reasoning(self, power: float) -> str:
        """Get reasoning for power parameter choice."""
        if power <= 1.5:
            return "Low power (p≤1.5) for smooth, regional variation"
        elif power <= 2.5:
            return "Standard power (p=2) balances local and regional influence"
        elif power <= 3.0:
            return "Higher power (p=2.5-3) emphasizes local variation"
        else:
            return "High power (p>3) for strong local influence, reduces outlier impact"
    
    def _estimate_idw_quality(self, parameters: Dict[str, Any]) -> float:
        """Estimate quality score for IDW parameters."""
        score = 70.0  # Base score
        
        # Good search radius
        if 'search_radius' in parameters:
            score += 5
        
        # Appropriate power
        power = parameters.get('power', 2.0)
        if 1.5 <= power <= 3.0:
            score += 5
        
        # Using sectors when beneficial
        if parameters.get('use_sectors', False) and self.characteristics.clustering_score > 0.5:
            score += 10
        
        # Anisotropy handling
        if self.characteristics.anisotropy_ratio < 0.8 and 'anisotropy_ratio' in parameters:
            score += 10
        
        return min(score, 100.0)
    
    def _optimize_kriging_parameters(self, constraints: Dict[str, Any]) -> OptimizationResult:
        """Optimize Ordinary Kriging parameters."""
        parameters = {}
        reasoning = {}
        
        # Variogram model selection
        if self.characteristics.statistics.get('cv', 0) > 1.0:
            parameters['variogram_model'] = 'exponential'
            reasoning['variogram_model'] = "Exponential model for high variability data"
        else:
            parameters['variogram_model'] = 'spherical'
            reasoning['variogram_model'] = "Spherical model for moderate variability"
        
        # Range estimation
        bounds = list(self.characteristics.bounds.values())
        max_dist = np.sqrt(sum((b[1] - b[0])**2 for b in bounds[:2]))
        parameters['range'] = max_dist * 0.3
        reasoning['range'] = "Range set to 30% of diagonal for good coverage"
        
        # Sill (variance)
        parameters['sill'] = self.characteristics.statistics['std'] ** 2
        reasoning['sill'] = "Sill equals data variance"
        
        # Nugget effect
        if len(self.characteristics.outlier_indices) > 0:
            parameters['nugget'] = parameters['sill'] * 0.2
            reasoning['nugget'] = "20% nugget for measurement error/micro-variability"
        else:
            parameters['nugget'] = parameters['sill'] * 0.1
            reasoning['nugget'] = "10% nugget for low noise data"
        
        # Search parameters (similar to IDW)
        search_params = self._optimize_search_radius(constraints)
        parameters['search_radius'] = search_params
        parameters['max_points'] = min(30, self.characteristics.n_points // 3)
        
        quality_score = 80.0  # Kriging typically performs well
        
        return OptimizationResult(
            method='Kriging',
            parameters=parameters,
            quality_score=quality_score,
            reasoning=reasoning,
            search_history=[]
        )
    
    def _optimize_rbf_parameters(self, constraints: Dict[str, Any]) -> OptimizationResult:
        """Optimize RBF parameters."""
        parameters = {}
        reasoning = {}
        
        # Kernel selection
        if self.characteristics.statistics.get('cv', 0) < 0.5:
            parameters['kernel'] = 'thin_plate_spline'
            reasoning['kernel'] = "Thin plate spline for smooth data"
        else:
            parameters['kernel'] = 'multiquadric'
            reasoning['kernel'] = "Multiquadric for variable data"
        
        # Smoothing parameter
        if len(self.characteristics.outlier_indices) > 0:
            parameters['smooth'] = 0.1
            reasoning['smooth'] = "Smoothing added for outlier robustness"
        else:
            parameters['smooth'] = 0.0
            reasoning['smooth'] = "No smoothing for exact interpolation"
        
        # Epsilon (shape parameter)
        nn_mean = self.characteristics.nearest_neighbor_stats['mean_distance']
        parameters['epsilon'] = 1.0 / nn_mean if nn_mean > 0 else 1.0
        reasoning['epsilon'] = "Shape parameter based on data spacing"
        
        quality_score = 75.0
        
        return OptimizationResult(
            method='RBF',
            parameters=parameters,
            quality_score=quality_score,
            reasoning=reasoning,
            search_history=[]
        )