"""
Data compositing module for coal deposit data.

Provides methods for combining multiple data points into composite values:
- Interval-based compositing (combining adjacent measurements)
- Statistical compositing (weighted averaging methods)
- Spatial compositing (combining nearby points)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import warnings
import logging
from scipy.spatial import cKDTree
from scipy import stats


@dataclass
class CompositingResult:
    """Container for data compositing results."""
    composited_data: pd.DataFrame
    original_count: int
    composited_count: int
    compositing_method: str
    quality_metrics: Dict[str, float]
    compositing_info: Dict[str, Any]


class DataCompositor:
    """
    Advanced data compositing engine for coal deposit data.
    
    Provides various methods for combining multiple data points into
    representative composite values, essential for geological modeling.
    """
    
    def __init__(self, 
                 min_composite_length: float = 1.0,
                 max_composite_length: float = 10.0,
                 quality_threshold: float = 0.8):
        """
        Initialize data compositor.
        
        Args:
            min_composite_length: Minimum length for interval compositing
            max_composite_length: Maximum length for interval compositing
            quality_threshold: Quality threshold for composite acceptance
        """
        self.min_composite_length = min_composite_length
        self.max_composite_length = max_composite_length
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(__name__)
    
    def interval_based_compositing(self,
                                 data: pd.DataFrame,
                                 hole_id_col: str,
                                 from_col: str,
                                 to_col: str,
                                 value_cols: List[str],
                                 composite_length: float,
                                 method: str = 'length_weighted',
                                 min_recovery: float = 0.5) -> CompositingResult:
        """
        Perform interval-based compositing on drill hole data.
        
        Args:
            data: Input drill hole data
            hole_id_col: Column with hole IDs
            from_col: From depth column
            to_col: To depth column
            value_cols: Columns to composite
            composite_length: Target composite length
            method: Compositing method ('length_weighted', 'simple_average')
            min_recovery: Minimum recovery ratio for valid composite
            
        Returns:
            CompositingResult with composited data
        """
        self.logger.info(f"Starting interval-based compositing with {composite_length}m intervals")
        
        composited_rows = []
        quality_stats = {'valid_composites': 0, 'rejected_composites': 0}
        
        # Group by hole ID
        for hole_id, hole_data in data.groupby(hole_id_col):
            hole_data = hole_data.sort_values(from_col).copy()
            
            # Get hole extent
            hole_start = hole_data[from_col].min()
            hole_end = hole_data[to_col].max()
            
            # Generate composite intervals
            composite_starts = np.arange(hole_start, hole_end, composite_length)
            
            for comp_start in composite_starts:
                comp_end = comp_start + composite_length
                
                # Find overlapping intervals
                overlapping = hole_data[
                    (hole_data[from_col] < comp_end) & 
                    (hole_data[to_col] > comp_start)
                ].copy()
                
                if len(overlapping) == 0:
                    continue
                
                # Calculate overlap lengths
                overlapping['overlap_start'] = np.maximum(overlapping[from_col], comp_start)
                overlapping['overlap_end'] = np.minimum(overlapping[to_col], comp_end)
                overlapping['overlap_length'] = overlapping['overlap_end'] - overlapping['overlap_start']
                
                # Calculate recovery
                total_overlap = overlapping['overlap_length'].sum()
                recovery = total_overlap / composite_length
                
                if recovery < min_recovery:
                    quality_stats['rejected_composites'] += 1
                    continue
                
                # Composite values
                composite_values = {}
                
                if method == 'length_weighted':
                    # Length-weighted average
                    weights = overlapping['overlap_length'] / total_overlap
                    for col in value_cols:
                        if col in overlapping.columns:
                            valid_mask = ~pd.isna(overlapping[col])
                            if valid_mask.any():
                                composite_values[col] = np.average(
                                    overlapping.loc[valid_mask, col],
                                    weights=weights[valid_mask]
                                )
                            else:
                                composite_values[col] = np.nan
                
                elif method == 'simple_average':
                    # Simple average
                    for col in value_cols:
                        if col in overlapping.columns:
                            composite_values[col] = overlapping[col].mean()
                
                # Create composite record
                composite_record = {
                    hole_id_col: hole_id,
                    from_col: comp_start,
                    to_col: comp_end,
                    'composite_length': composite_length,
                    'recovery': recovery,
                    'n_intervals': len(overlapping)
                }
                composite_record.update(composite_values)
                
                composited_rows.append(composite_record)
                quality_stats['valid_composites'] += 1
        
        # Create results DataFrame
        if composited_rows:
            composited_data = pd.DataFrame(composited_rows)
        else:
            # Empty result
            columns = [hole_id_col, from_col, to_col, 'composite_length', 'recovery', 'n_intervals'] + value_cols
            composited_data = pd.DataFrame(columns=columns)
        
        # Calculate quality metrics
        quality_metrics = {
            'original_intervals': len(data),
            'composited_intervals': len(composited_data),
            'reduction_ratio': len(composited_data) / len(data) if len(data) > 0 else 0,
            'average_recovery': composited_data['recovery'].mean() if len(composited_data) > 0 else 0,
            'rejection_rate': quality_stats['rejected_composites'] / 
                            (quality_stats['valid_composites'] + quality_stats['rejected_composites'])
                            if (quality_stats['valid_composites'] + quality_stats['rejected_composites']) > 0 else 0
        }
        
        compositing_info = {
            'method': method,
            'composite_length': composite_length,
            'min_recovery': min_recovery,
            'holes_processed': data[hole_id_col].nunique()
        }
        
        return CompositingResult(
            composited_data=composited_data,
            original_count=len(data),
            composited_count=len(composited_data),
            compositing_method='interval_based',
            quality_metrics=quality_metrics,
            compositing_info=compositing_info
        )
    
    def statistical_compositing(self,
                              data: pd.DataFrame,
                              x_col: str,
                              y_col: str,
                              value_cols: List[str],
                              composite_radius: float,
                              z_col: Optional[str] = None,
                              method: str = 'inverse_distance',
                              power: float = 2.0,
                              min_samples: int = 2,
                              max_samples: int = 10) -> CompositingResult:
        """
        Perform statistical compositing based on spatial proximity.
        
        Args:
            data: Input data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_cols: Value columns to composite
            composite_radius: Search radius for compositing
            z_col: Z coordinate column (optional)
            method: Compositing method ('inverse_distance', 'average', 'median')
            power: Power for inverse distance weighting
            min_samples: Minimum samples for valid composite
            max_samples: Maximum samples to use
            
        Returns:
            CompositingResult with composited data
        """
        self.logger.info(f"Starting statistical compositing with {composite_radius}m radius")
        
        # Prepare coordinates
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        coordinates = data[coord_cols].values
        n_points = len(data)
        
        # Build spatial index
        if len(coord_cols) == 2:
            kdtree = cKDTree(coordinates)
        else:
            kdtree = cKDTree(coordinates)
        
        # Find composite points (grid-based approach)
        x_min, x_max = data[x_col].min(), data[x_col].max()
        y_min, y_max = data[y_col].min(), data[y_col].max()
        
        # Create grid
        grid_spacing = composite_radius / 2
        x_grid = np.arange(x_min, x_max + grid_spacing, grid_spacing)
        y_grid = np.arange(y_min, y_max + grid_spacing, grid_spacing)
        
        composite_points = []
        
        for x in x_grid:
            for y in y_grid:
                if z_col:
                    # For 3D, use average Z of nearby points
                    nearby_indices = kdtree.query_ball_point([x, y, 0], composite_radius)
                    if len(nearby_indices) >= min_samples:
                        avg_z = data.iloc[nearby_indices][z_col].mean()
                        composite_points.append([x, y, avg_z])
                else:
                    composite_points.append([x, y])
        
        if not composite_points:
            # Return empty result
            columns = coord_cols + value_cols + ['n_samples', 'composite_quality']
            composited_data = pd.DataFrame(columns=columns)
            return CompositingResult(
                composited_data=composited_data,
                original_count=len(data),
                composited_count=0,
                compositing_method='statistical',
                quality_metrics={'reduction_ratio': 0},
                compositing_info={'method': method}
            )
        
        composite_points = np.array(composite_points)
        
        # Build KDTree for composite points
        comp_kdtree = cKDTree(composite_points)
        
        composited_rows = []
        
        for i, comp_point in enumerate(composite_points):
            # Find nearby original points
            if len(coord_cols) == 2:
                nearby_indices = kdtree.query_ball_point(comp_point, composite_radius)
            else:
                nearby_indices = kdtree.query_ball_point(comp_point, composite_radius)
            
            if len(nearby_indices) < min_samples:
                continue
            
            # Limit to max_samples
            if len(nearby_indices) > max_samples:
                distances, indices = kdtree.query(comp_point, k=max_samples)
                nearby_indices = indices.tolist()
            
            nearby_data = data.iloc[nearby_indices]
            nearby_coords = coordinates[nearby_indices]
            
            # Calculate distances for weighting
            distances = np.sqrt(np.sum((nearby_coords - comp_point) ** 2, axis=1))
            
            # Composite values
            composite_record = {}
            
            # Add coordinates
            for j, col in enumerate(coord_cols):
                composite_record[col] = comp_point[j]
            
            # Composite each value column
            for col in value_cols:
                if col in nearby_data.columns:
                    valid_mask = ~pd.isna(nearby_data[col])
                    if not valid_mask.any():
                        composite_record[col] = np.nan
                        continue
                    
                    valid_values = nearby_data.loc[valid_mask, col].values
                    valid_distances = distances[valid_mask.values]
                    
                    if method == 'inverse_distance':
                        # Avoid division by zero
                        weights = 1 / (valid_distances + 1e-10) ** power
                        weights /= weights.sum()
                        composite_record[col] = np.average(valid_values, weights=weights)
                    
                    elif method == 'average':
                        composite_record[col] = np.mean(valid_values)
                    
                    elif method == 'median':
                        composite_record[col] = np.median(valid_values)
            
            # Add metadata
            composite_record['n_samples'] = len(nearby_indices)
            composite_record['composite_quality'] = min(1.0, len(nearby_indices) / max_samples)
            
            composited_rows.append(composite_record)
        
        # Create results DataFrame
        if composited_rows:
            composited_data = pd.DataFrame(composited_rows)
        else:
            columns = coord_cols + value_cols + ['n_samples', 'composite_quality']
            composited_data = pd.DataFrame(columns=columns)
        
        # Calculate quality metrics
        quality_metrics = {
            'original_points': len(data),
            'composited_points': len(composited_data),
            'reduction_ratio': len(composited_data) / len(data) if len(data) > 0 else 0,
            'average_samples_per_composite': composited_data['n_samples'].mean() if len(composited_data) > 0 else 0,
            'average_quality': composited_data['composite_quality'].mean() if len(composited_data) > 0 else 0
        }
        
        compositing_info = {
            'method': method,
            'composite_radius': composite_radius,
            'min_samples': min_samples,
            'max_samples': max_samples,
            'power': power if method == 'inverse_distance' else None
        }
        
        return CompositingResult(
            composited_data=composited_data,
            original_count=len(data),
            composited_count=len(composited_data),
            compositing_method='statistical',
            quality_metrics=quality_metrics,
            compositing_info=compositing_info
        )
    
    def domain_based_compositing(self,
                               data: pd.DataFrame,
                               domain_col: str,
                               value_cols: List[str],
                               method: str = 'domain_weighted') -> CompositingResult:
        """
        Perform domain-based compositing (by geological domains).
        
        Args:
            data: Input data
            domain_col: Column with domain/zone information
            value_cols: Value columns to composite
            method: Compositing method ('domain_weighted', 'simple_average')
            
        Returns:
            CompositingResult with domain composites
        """
        self.logger.info("Starting domain-based compositing")
        
        composited_rows = []
        
        for domain, domain_data in data.groupby(domain_col):
            if len(domain_data) == 0:
                continue
            
            composite_record = {domain_col: domain}
            
            for col in value_cols:
                if col in domain_data.columns:
                    valid_data = domain_data[col].dropna()
                    
                    if len(valid_data) == 0:
                        composite_record[col] = np.nan
                        continue
                    
                    if method == 'domain_weighted':
                        # Weight by frequency or other domain-specific criteria
                        composite_record[col] = valid_data.mean()
                        composite_record[f'{col}_std'] = valid_data.std()
                        composite_record[f'{col}_count'] = len(valid_data)
                    
                    elif method == 'simple_average':
                        composite_record[col] = valid_data.mean()
            
            # Add domain statistics
            composite_record['domain_sample_count'] = len(domain_data)
            
            composited_rows.append(composite_record)
        
        # Create results DataFrame
        if composited_rows:
            composited_data = pd.DataFrame(composited_rows)
        else:
            columns = [domain_col] + value_cols + ['domain_sample_count']
            composited_data = pd.DataFrame(columns=columns)
        
        # Calculate quality metrics
        quality_metrics = {
            'original_samples': len(data),
            'domains_composited': len(composited_data),
            'average_samples_per_domain': len(data) / len(composited_data) if len(composited_data) > 0 else 0,
            'reduction_ratio': len(composited_data) / len(data) if len(data) > 0 else 0
        }
        
        compositing_info = {
            'method': method,
            'domains': data[domain_col].unique().tolist()
        }
        
        return CompositingResult(
            composited_data=composited_data,
            original_count=len(data),
            composited_count=len(composited_data),
            compositing_method='domain_based',
            quality_metrics=quality_metrics,
            compositing_info=compositing_info
        )
    
    def validate_compositing_quality(self, result: CompositingResult) -> Dict[str, Any]:
        """
        Validate compositing quality and provide recommendations.
        
        Args:
            result: CompositingResult to validate
            
        Returns:
            Dictionary with quality assessment and recommendations
        """
        quality_assessment = {
            'overall_quality': 'Unknown',
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        metrics = result.quality_metrics
        
        # Evaluate reduction ratio
        reduction_ratio = metrics.get('reduction_ratio', 0)
        if reduction_ratio > 0.8:
            quality_assessment['issues'].append("Low data reduction - compositing may not be effective")
            quality_assessment['recommendations'].append("Consider increasing composite length/radius")
        elif reduction_ratio < 0.1:
            quality_assessment['issues'].append("Very high data reduction - may lose important details")
            quality_assessment['recommendations'].append("Consider decreasing composite length/radius")
        
        # Evaluate based on method
        if result.compositing_method == 'interval_based':
            rejection_rate = metrics.get('rejection_rate', 0)
            if rejection_rate > 0.3:
                quality_assessment['issues'].append("High composite rejection rate")
                quality_assessment['recommendations'].append("Consider reducing minimum recovery threshold")
            
            avg_recovery = metrics.get('average_recovery', 0)
            if avg_recovery < 0.7:
                quality_assessment['issues'].append("Low average recovery")
                quality_assessment['recommendations'].append("Check data quality and composite parameters")
        
        elif result.compositing_method == 'statistical':
            avg_samples = metrics.get('average_samples_per_composite', 0)
            if avg_samples < 3:
                quality_assessment['issues'].append("Low sample density per composite")
                quality_assessment['recommendations'].append("Consider increasing search radius")
        
        # Calculate overall quality score
        score_components = []
        
        if 0.2 <= reduction_ratio <= 0.7:
            score_components.append(0.4)  # Good reduction
        elif 0.1 <= reduction_ratio < 0.2 or 0.7 < reduction_ratio <= 0.8:
            score_components.append(0.2)  # Acceptable reduction
        else:
            score_components.append(0.0)  # Poor reduction
        
        # Add method-specific quality components
        if result.compositing_method == 'interval_based':
            rejection_rate = metrics.get('rejection_rate', 0)
            if rejection_rate < 0.2:
                score_components.append(0.3)
            elif rejection_rate < 0.4:
                score_components.append(0.15)
            else:
                score_components.append(0.0)
            
            avg_recovery = metrics.get('average_recovery', 0)
            if avg_recovery > 0.8:
                score_components.append(0.3)
            elif avg_recovery > 0.6:
                score_components.append(0.15)
            else:
                score_components.append(0.0)
        
        quality_assessment['quality_score'] = sum(score_components)
        
        # Overall quality classification
        if quality_assessment['quality_score'] >= 0.8:
            quality_assessment['overall_quality'] = 'Excellent'
        elif quality_assessment['quality_score'] >= 0.6:
            quality_assessment['overall_quality'] = 'Good'
        elif quality_assessment['quality_score'] >= 0.4:
            quality_assessment['overall_quality'] = 'Fair'
        else:
            quality_assessment['overall_quality'] = 'Poor'
        
        return quality_assessment