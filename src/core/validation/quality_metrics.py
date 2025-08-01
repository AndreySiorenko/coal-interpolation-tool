"""
Quality metrics module for interpolation validation.

Provides comprehensive metrics for assessing interpolation quality including:
- Basic error metrics (RMSE, MAE, MAPE)
- Advanced metrics (Nash-Sutcliffe, Willmott's d)
- Spatial error analysis
- Directional error analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats, spatial
import warnings


@dataclass
class MetricsResult:
    """Container for comprehensive metrics results."""
    basic_metrics: Dict[str, float]
    advanced_metrics: Dict[str, float]
    spatial_metrics: Dict[str, Any]
    directional_metrics: Dict[str, Any]
    error_distribution: Dict[str, Any]
    summary: Dict[str, Any]


class QualityMetrics:
    """
    Comprehensive quality metrics calculator for interpolation validation.
    
    Provides a wide range of metrics for assessing interpolation quality,
    including error metrics, correlation measures, and spatial analysis.
    """
    
    def __init__(self):
        """Initialize quality metrics calculator."""
        self.epsilon = 1e-10  # Small value to avoid division by zero
    
    def calculate_metrics(self,
                         actual: np.ndarray,
                         predicted: np.ndarray,
                         coordinates: Optional[np.ndarray] = None,
                         weights: Optional[np.ndarray] = None) -> MetricsResult:
        """
        Calculate comprehensive quality metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            coordinates: Spatial coordinates (optional)
            weights: Sample weights (optional)
            
        Returns:
            MetricsResult with all calculated metrics
        """
        # Ensure arrays are numpy arrays
        actual = np.asarray(actual).flatten()
        predicted = np.asarray(predicted).flatten()
        
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted arrays must have same length")
        
        # Calculate all metric categories
        basic_metrics = self._calculate_basic_metrics(actual, predicted, weights)
        advanced_metrics = self._calculate_advanced_metrics(actual, predicted, weights)
        error_distribution = self._analyze_error_distribution(actual, predicted)
        
        # Spatial metrics if coordinates provided
        spatial_metrics = {}
        directional_metrics = {}
        
        if coordinates is not None:
            spatial_metrics = self._calculate_spatial_metrics(
                actual, predicted, coordinates
            )
            if coordinates.shape[1] >= 2:
                directional_metrics = self._calculate_directional_metrics(
                    actual, predicted, coordinates
                )
        
        # Generate summary
        summary = self._generate_summary(
            basic_metrics, advanced_metrics, spatial_metrics
        )
        
        return MetricsResult(
            basic_metrics=basic_metrics,
            advanced_metrics=advanced_metrics,
            spatial_metrics=spatial_metrics,
            directional_metrics=directional_metrics,
            error_distribution=error_distribution,
            summary=summary
        )
    
    def _calculate_basic_metrics(self,
                               actual: np.ndarray,
                               predicted: np.ndarray,
                               weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic error metrics."""
        errors = predicted - actual
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        if weights is not None:
            # Weighted metrics
            n_eff = np.sum(weights)
            mae = np.sum(weights * abs_errors) / n_eff
            mse = np.sum(weights * squared_errors) / n_eff
            bias = np.sum(weights * errors) / n_eff
        else:
            # Unweighted metrics
            n_eff = len(actual)
            mae = np.mean(abs_errors)
            mse = np.mean(squared_errors)
            bias = np.mean(errors)
        
        rmse = np.sqrt(mse)
        
        # Relative metrics
        mean_actual = np.mean(actual)
        range_actual = np.ptp(actual)
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = np.abs(actual) > self.epsilon
        if np.any(mask):
            mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100
        else:
            mape = np.inf
        
        # Normalized metrics
        nmae = mae / mean_actual if mean_actual != 0 else np.inf
        nrmse = rmse / mean_actual if mean_actual != 0 else np.inf
        rnrmse = rmse / range_actual if range_actual != 0 else np.inf
        
        # Percent bias
        pbias = (bias / mean_actual * 100) if mean_actual != 0 else np.inf
        
        return {
            'n_samples': int(n_eff),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'bias': float(bias),
            'mape': float(mape),
            'nmae': float(nmae),
            'nrmse': float(nrmse),
            'rnrmse': float(rnrmse),
            'pbias': float(pbias),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'std_error': float(np.std(errors, ddof=1))
        }
    
    def _calculate_advanced_metrics(self,
                                  actual: np.ndarray,
                                  predicted: np.ndarray,
                                  weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate advanced quality metrics."""
        errors = predicted - actual
        mean_actual = np.mean(actual)
        
        # R-squared (coefficient of determination)
        ss_total = np.sum((actual - mean_actual) ** 2)
        ss_residual = np.sum(errors ** 2)
        
        if ss_total > self.epsilon:
            r_squared = 1 - (ss_residual / ss_total)
        else:
            r_squared = 0.0
        
        # Adjusted R-squared (placeholder - needs degrees of freedom)
        # For interpolation, we'll use standard R-squared
        adj_r_squared = r_squared
        
        # Pearson correlation coefficient
        if np.std(actual) > 0 and np.std(predicted) > 0:
            pearson_r, pearson_p = stats.pearsonr(actual, predicted)
        else:
            pearson_r, pearson_p = 0.0, 1.0
        
        # Spearman rank correlation
        spearman_r, spearman_p = stats.spearmanr(actual, predicted)
        
        # Nash-Sutcliffe Efficiency (NSE)
        if ss_total > self.epsilon:
            nse = 1 - (ss_residual / ss_total)
        else:
            nse = -np.inf
        
        # Modified Nash-Sutcliffe Efficiency (emphasis on peaks)
        if ss_total > self.epsilon:
            # Weight errors by actual values
            weighted_errors = errors * np.abs(actual - mean_actual)
            ss_weighted_residual = np.sum(weighted_errors ** 2)
            ss_weighted_total = np.sum((actual - mean_actual) ** 4)
            if ss_weighted_total > self.epsilon:
                modified_nse = 1 - (ss_weighted_residual / ss_weighted_total)
            else:
                modified_nse = -np.inf
        else:
            modified_nse = -np.inf
        
        # Willmott's Index of Agreement (d)
        numerator = np.sum(errors ** 2)
        denominator = np.sum((np.abs(predicted - mean_actual) + 
                            np.abs(actual - mean_actual)) ** 2)
        
        if denominator > self.epsilon:
            willmott_d = 1 - (numerator / denominator)
        else:
            willmott_d = 0.0
        
        # Modified Willmott's d (d1)
        numerator_d1 = np.sum(np.abs(errors))
        denominator_d1 = np.sum(np.abs(predicted - mean_actual) + 
                               np.abs(actual - mean_actual))
        
        if denominator_d1 > self.epsilon:
            willmott_d1 = 1 - (numerator_d1 / denominator_d1)
        else:
            willmott_d1 = 0.0
        
        # Kling-Gupta Efficiency (KGE)
        if np.std(actual) > 0 and np.std(predicted) > 0:
            # Correlation component
            r = pearson_r
            # Bias component
            beta = np.mean(predicted) / mean_actual if mean_actual != 0 else 0
            # Variability component
            gamma = np.std(predicted) / np.std(actual)
            # KGE
            kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
        else:
            kge = -np.inf
        
        # Volumetric Efficiency (VE)
        ve = 1 - np.sum(np.abs(errors)) / np.sum(np.abs(actual - mean_actual))
        
        return {
            'r_squared': float(r_squared),
            'adj_r_squared': float(adj_r_squared),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'nash_sutcliffe': float(nse),
            'modified_nse': float(modified_nse),
            'willmott_d': float(willmott_d),
            'willmott_d1': float(willmott_d1),
            'kling_gupta': float(kge),
            'volumetric_efficiency': float(ve)
        }
    
    def _analyze_error_distribution(self,
                                  actual: np.ndarray,
                                  predicted: np.ndarray) -> Dict[str, Any]:
        """Analyze error distribution characteristics."""
        errors = predicted - actual
        abs_errors = np.abs(errors)
        
        # Basic distribution statistics
        distribution = {
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors, ddof=1)),
            'skewness': float(stats.skew(errors)),
            'kurtosis': float(stats.kurtosis(errors)),
            'iqr': float(np.percentile(errors, 75) - np.percentile(errors, 25))
        }
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        distribution['percentiles'] = {
            f'p{p}': float(np.percentile(errors, p)) for p in percentiles
        }
        
        # Error categories
        rmse = np.sqrt(np.mean(errors ** 2))
        distribution['error_categories'] = {
            'within_0.5_rmse': float(np.sum(abs_errors <= 0.5 * rmse) / len(errors)),
            'within_1_rmse': float(np.sum(abs_errors <= rmse) / len(errors)),
            'within_2_rmse': float(np.sum(abs_errors <= 2 * rmse) / len(errors)),
            'outliers_3_rmse': float(np.sum(abs_errors > 3 * rmse) / len(errors))
        }
        
        # Normality test
        if len(errors) >= 8:
            stat, p_value = stats.normaltest(errors)
            distribution['normality_test'] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        
        return distribution
    
    def _calculate_spatial_metrics(self,
                                 actual: np.ndarray,
                                 predicted: np.ndarray,
                                 coordinates: np.ndarray) -> Dict[str, Any]:
        """Calculate spatial error metrics."""
        errors = predicted - actual
        abs_errors = np.abs(errors)
        
        # Build KDTree for spatial analysis
        if coordinates.shape[1] >= 2:
            kdtree = spatial.cKDTree(coordinates[:, :2])
        else:
            return {}
        
        spatial_metrics = {}
        
        # Spatial autocorrelation of errors (Moran's I approximation)
        n = len(errors)
        if n > 10:
            # Create spatial weights matrix (k-nearest neighbors)
            k = min(10, n - 1)
            distances, indices = kdtree.query(coordinates[:, :2], k=k+1)
            
            # Calculate Moran's I
            mean_error = np.mean(errors)
            numerator = 0
            denominator = np.sum((errors - mean_error) ** 2)
            w_sum = 0
            
            for i in range(n):
                neighbors = indices[i, 1:]  # Exclude self
                neighbor_distances = distances[i, 1:]
                
                # Inverse distance weights
                weights = 1 / (neighbor_distances + self.epsilon)
                weights /= np.sum(weights)
                
                for j, w in zip(neighbors, weights):
                    numerator += w * (errors[i] - mean_error) * (errors[j] - mean_error)
                    w_sum += w
            
            if w_sum > 0 and denominator > 0:
                morans_i = (n / w_sum) * (numerator / denominator)
            else:
                morans_i = 0.0
            
            spatial_metrics['morans_i_errors'] = float(morans_i)
        
        # Local error indicators
        if n > 20:
            # Calculate local error relative to neighbors
            k = min(5, n - 1)
            _, indices = kdtree.query(coordinates[:, :2], k=k+1)
            
            local_error_ratios = []
            for i in range(n):
                neighbors = indices[i, 1:]
                neighbor_errors = abs_errors[neighbors]
                if np.mean(neighbor_errors) > self.epsilon:
                    ratio = abs_errors[i] / np.mean(neighbor_errors)
                    local_error_ratios.append(ratio)
            
            if local_error_ratios:
                spatial_metrics['mean_local_error_ratio'] = float(np.mean(local_error_ratios))
                spatial_metrics['std_local_error_ratio'] = float(np.std(local_error_ratios))
        
        # Error clustering
        if n > 30:
            # Identify spatial clusters of high errors
            threshold = np.percentile(abs_errors, 90)
            high_error_points = coordinates[abs_errors > threshold]
            
            if len(high_error_points) > 3:
                # Calculate nearest neighbor distances for high error points
                high_error_tree = spatial.cKDTree(high_error_points[:, :2])
                nn_distances, _ = high_error_tree.query(high_error_points[:, :2], k=2)
                mean_nn_dist = np.mean(nn_distances[:, 1])
                
                # Compare to random expectation
                total_area = (np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])) * \
                           (np.max(coordinates[:, 1]) - np.min(coordinates[:, 1]))
                expected_nn_dist = 0.5 * np.sqrt(total_area / len(high_error_points))
                
                if expected_nn_dist > 0:
                    clustering_ratio = mean_nn_dist / expected_nn_dist
                else:
                    clustering_ratio = 1.0
                
                spatial_metrics['error_clustering_ratio'] = float(clustering_ratio)
                spatial_metrics['high_error_threshold'] = float(threshold)
        
        return spatial_metrics
    
    def _calculate_directional_metrics(self,
                                     actual: np.ndarray,
                                     predicted: np.ndarray,
                                     coordinates: np.ndarray) -> Dict[str, Any]:
        """Calculate directional error metrics."""
        errors = predicted - actual
        x, y = coordinates[:, 0], coordinates[:, 1]
        
        directional_metrics = {}
        
        # Directional bias analysis
        n_directions = 8
        angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
        directional_biases = []
        directional_rmses = []
        
        for angle in angles:
            # Project coordinates onto direction
            direction = np.array([np.cos(angle), np.sin(angle)])
            projections = x * direction[0] + y * direction[1]
            
            # Sort by projection
            sort_idx = np.argsort(projections)
            sorted_errors = errors[sort_idx]
            
            # Calculate bias and RMSE for this direction
            directional_biases.append(np.mean(sorted_errors))
            directional_rmses.append(np.sqrt(np.mean(sorted_errors ** 2)))
        
        directional_metrics['directional_biases'] = directional_biases
        directional_metrics['directional_rmses'] = directional_rmses
        directional_metrics['angles_deg'] = (angles * 180 / np.pi).tolist()
        
        # Anisotropy in errors
        max_rmse = max(directional_rmses)
        min_rmse = min(directional_rmses)
        if max_rmse > 0:
            error_anisotropy_ratio = min_rmse / max_rmse
        else:
            error_anisotropy_ratio = 1.0
        
        directional_metrics['error_anisotropy_ratio'] = float(error_anisotropy_ratio)
        directional_metrics['max_error_direction'] = float(angles[np.argmax(directional_rmses)] * 180 / np.pi)
        directional_metrics['min_error_direction'] = float(angles[np.argmin(directional_rmses)] * 180 / np.pi)
        
        # Gradient analysis
        if len(errors) > 10:
            # Fit linear trend to errors
            A = np.column_stack([np.ones(len(x)), x, y])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, errors, rcond=None)
                error_gradient = np.array([coeffs[1], coeffs[2]])
                gradient_magnitude = np.linalg.norm(error_gradient)
                gradient_direction = np.arctan2(error_gradient[1], error_gradient[0]) * 180 / np.pi
                
                directional_metrics['error_gradient_magnitude'] = float(gradient_magnitude)
                directional_metrics['error_gradient_direction'] = float(gradient_direction)
            except:
                pass
        
        return directional_metrics
    
    def _generate_summary(self,
                         basic_metrics: Dict[str, float],
                         advanced_metrics: Dict[str, float],
                         spatial_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of quality assessment."""
        summary = {
            'overall_quality': self._calculate_overall_quality_score(
                basic_metrics, advanced_metrics
            ),
            'key_metrics': {
                'rmse': basic_metrics['rmse'],
                'r_squared': advanced_metrics['r_squared'],
                'nash_sutcliffe': advanced_metrics['nash_sutcliffe'],
                'willmott_d': advanced_metrics['willmott_d']
            }
        }
        
        # Quality classification
        quality_score = summary['overall_quality']
        if quality_score >= 0.9:
            quality_class = 'Excellent'
        elif quality_score >= 0.8:
            quality_class = 'Good'
        elif quality_score >= 0.7:
            quality_class = 'Fair'
        elif quality_score >= 0.6:
            quality_class = 'Poor'
        else:
            quality_class = 'Very Poor'
        
        summary['quality_classification'] = quality_class
        
        # Recommendations
        recommendations = []
        
        if basic_metrics['pbias'] > 10:
            recommendations.append("High positive bias detected - model overestimates")
        elif basic_metrics['pbias'] < -10:
            recommendations.append("High negative bias detected - model underestimates")
        
        if basic_metrics['mape'] > 20:
            recommendations.append("High relative errors - consider model refinement")
        
        if spatial_metrics.get('morans_i_errors', 0) > 0.3:
            recommendations.append("Spatial clustering of errors - check for spatial trends")
        
        summary['recommendations'] = recommendations
        
        return summary
    
    def _calculate_overall_quality_score(self,
                                       basic_metrics: Dict[str, float],
                                       advanced_metrics: Dict[str, float]) -> float:
        """Calculate overall quality score (0-1)."""
        # Combine multiple metrics into single score
        scores = []
        
        # R-squared component (0-1)
        r2 = advanced_metrics['r_squared']
        scores.append(max(0, r2))
        
        # Nash-Sutcliffe component (convert to 0-1)
        nse = advanced_metrics['nash_sutcliffe']
        nse_score = max(0, min(1, (nse + 1) / 2))  # Map [-inf,1] to [0,1]
        scores.append(nse_score)
        
        # Willmott's d component (0-1)
        willmott = advanced_metrics['willmott_d']
        scores.append(max(0, willmott))
        
        # MAPE component (inverse, normalized)
        mape = basic_metrics['mape']
        if mape < 100:
            mape_score = 1 - (mape / 100)
        else:
            mape_score = 0
        scores.append(mape_score)
        
        # Average all component scores
        overall_score = np.mean(scores)
        
        return float(overall_score)