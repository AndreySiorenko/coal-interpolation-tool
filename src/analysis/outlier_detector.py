"""
Advanced outlier detection for geological data.

Provides multiple outlier detection methods including statistical,
spatial, and machine learning approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class OutlierResults:
    """Container for outlier detection results."""
    statistical_outliers: Dict[str, Any]
    spatial_outliers: Dict[str, Any]
    multivariate_outliers: Dict[str, Any]
    ensemble_results: Dict[str, Any]
    summary: Dict[str, Any]


class OutlierDetector:
    """
    Advanced outlier detector for geological data.
    
    Provides multiple outlier detection methods:
    - Statistical methods (IQR, Z-score, Modified Z-score)  
    - Spatial methods (spatial distance-based)
    - Multivariate methods (Mahalanobis distance, Isolation Forest)
    - Ensemble methods (combining multiple approaches)
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize outlier detector.
        
        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5)
        """
        self.contamination = max(0.001, min(0.5, contamination))
        self.data: Optional[pd.DataFrame] = None
        self.coordinates: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
    
    def detect_outliers(self,
                       data: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       value_col: str,
                       z_col: Optional[str] = None,
                       methods: Optional[List[str]] = None) -> OutlierResults:
        """
        Detect outliers using multiple methods.
        
        Args:
            data: Input DataFrame
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            z_col: Z coordinate column (optional)
            methods: List of methods to use (default: all available)
            
        Returns:
            OutlierResults with comprehensive outlier analysis
        """
        # Store data
        self.data = data
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        self.coordinates = data[coord_cols].values
        self.values = data[value_col].values
        
        if methods is None:
            methods = ['statistical', 'spatial', 'multivariate', 'ensemble']
        
        results = {}
        
        # Statistical outlier detection
        if 'statistical' in methods:
            results['statistical_outliers'] = self._statistical_outlier_detection()
        
        # Spatial outlier detection  
        if 'spatial' in methods:
            results['spatial_outliers'] = self._spatial_outlier_detection()
        
        # Multivariate outlier detection
        if 'multivariate' in methods:
            results['multivariate_outliers'] = self._multivariate_outlier_detection()
        
        # Ensemble outlier detection
        if 'ensemble' in methods:
            results['ensemble_results'] = self._ensemble_outlier_detection(results)
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return OutlierResults(**results)
    
    def _statistical_outlier_detection(self) -> Dict[str, Any]:
        """Detect outliers using statistical methods."""
        results = {}
        
        # IQR Method
        results['iqr_method'] = self._iqr_outliers()
        
        # Z-Score Method
        results['zscore_method'] = self._zscore_outliers()
        
        # Modified Z-Score Method
        results['modified_zscore'] = self._modified_zscore_outliers()
        
        # Grubbs Test
        results['grubbs_test'] = self._grubbs_test()
        
        # Tukey Fences (variations of IQR)
        results['tukey_fences'] = self._tukey_fences()
        
        return results
    
    def _spatial_outlier_detection(self) -> Dict[str, Any]:
        """Detect outliers using spatial methods."""
        results = {}
        
        # Distance-based outliers
        results['distance_based'] = self._distance_based_outliers()
        
        # Local outlier factor (spatial version)
        results['spatial_lof'] = self._spatial_local_outlier_factor()
        
        # Spatial residuals
        results['spatial_residuals'] = self._spatial_residual_outliers()
        
        # Convex hull outliers
        results['convex_hull'] = self._convex_hull_outliers()
        
        return results
    
    def _multivariate_outlier_detection(self) -> Dict[str, Any]:
        """Detect outliers using multivariate methods."""
        results = {}
        
        # Mahalanobis distance
        results['mahalanobis'] = self._mahalanobis_outliers()
        
        if SKLEARN_AVAILABLE:
            # Isolation Forest
            results['isolation_forest'] = self._isolation_forest_outliers()
            
            # One-Class SVM
            results['one_class_svm'] = self._one_class_svm_outliers()
            
            # Local Outlier Factor
            results['local_outlier_factor'] = self._local_outlier_factor_outliers()
            
            # Elliptic Envelope
            results['elliptic_envelope'] = self._elliptic_envelope_outliers()
        else:
            results['sklearn_note'] = 'Scikit-learn not available for advanced methods'
        
        return results
    
    def _ensemble_outlier_detection(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple outlier detection methods."""
        if len(previous_results) == 0:
            return {'error': 'No previous results to ensemble'}
        
        # Collect all outlier indices from different methods
        all_outlier_indices = []
        method_names = []
        
        # Extract indices from statistical methods
        if 'statistical_outliers' in previous_results:
            stat_results = previous_results['statistical_outliers']
            for method, result in stat_results.items():
                if isinstance(result, dict) and 'outlier_indices' in result:
                    all_outlier_indices.append(set(result['outlier_indices']))
                    method_names.append(f'stat_{method}')
        
        # Extract indices from spatial methods
        if 'spatial_outliers' in previous_results:
            spatial_results = previous_results['spatial_outliers']
            for method, result in spatial_results.items():
                if isinstance(result, dict) and 'outlier_indices' in result:
                    all_outlier_indices.append(set(result['outlier_indices']))
                    method_names.append(f'spatial_{method}')
        
        # Extract indices from multivariate methods
        if 'multivariate_outliers' in previous_results:
            mv_results = previous_results['multivariate_outliers']
            for method, result in mv_results.items():
                if isinstance(result, dict) and 'outlier_indices' in result:
                    all_outlier_indices.append(set(result['outlier_indices']))
                    method_names.append(f'mv_{method}')
        
        if len(all_outlier_indices) == 0:
            return {'error': 'No outlier indices found in previous results'}
        
        n_points = len(self.data)
        n_methods = len(all_outlier_indices)
        
        # Count votes for each point
        vote_counts = np.zeros(n_points)
        for outlier_set in all_outlier_indices:
            for idx in outlier_set:
                if 0 <= idx < n_points:
                    vote_counts[idx] += 1
        
        # Different ensemble strategies
        ensemble_results = {}
        
        # Majority vote (>50% of methods)
        majority_threshold = n_methods / 2
        majority_outliers = np.where(vote_counts > majority_threshold)[0].tolist()
        
        # Consensus (all methods agree)
        consensus_outliers = np.where(vote_counts == n_methods)[0].tolist()
        
        # Top percentile by votes
        vote_percentile = np.percentile(vote_counts, 100 * (1 - self.contamination))
        percentile_outliers = np.where(vote_counts >= vote_percentile)[0].tolist()
        
        # Union (any method identifies as outlier)
        union_outliers = list(set().union(*all_outlier_indices))
        
        ensemble_results = {
            'majority_vote': {
                'outlier_indices': majority_outliers,
                'n_outliers': len(majority_outliers),
                'threshold_votes': majority_threshold
            },
            'consensus': {
                'outlier_indices': consensus_outliers,
                'n_outliers': len(consensus_outliers),
                'agreement': 'all_methods'
            },
            'percentile_based': {
                'outlier_indices': percentile_outliers,
                'n_outliers': len(percentile_outliers),
                'vote_threshold': float(vote_percentile)
            },
            'union': {
                'outlier_indices': union_outliers,
                'n_outliers': len(union_outliers),
                'agreement': 'any_method'
            },
            'vote_statistics': {
                'vote_counts': vote_counts.tolist(),
                'mean_votes': float(np.mean(vote_counts)),
                'max_votes': int(np.max(vote_counts)),
                'n_methods_used': n_methods,
                'method_names': method_names
            }
        }
        
        return ensemble_results
    
    def _iqr_outliers(self) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range method."""
        q1, q3 = np.percentile(self.values, [25, 75])
        iqr = q3 - q1
        
        # Standard IQR outliers (1.5 * IQR)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (self.values < lower_bound) | (self.values > upper_bound)
        
        # Extreme outliers (3.0 * IQR)
        extreme_lower = q1 - 3.0 * iqr
        extreme_upper = q3 + 3.0 * iqr
        extreme_mask = (self.values < extreme_lower) | (self.values > extreme_upper)
        
        return {
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'extreme_indices': np.where(extreme_mask)[0].tolist(),
            'n_outliers': int(np.sum(outlier_mask)),
            'n_extreme': int(np.sum(extreme_mask)),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_percentage': float(np.sum(outlier_mask) / len(self.values) * 100)
        }
    
    def _zscore_outliers(self) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(self.values))
        
        # Different thresholds
        outlier_2sigma = z_scores > 2
        outlier_2_5sigma = z_scores > 2.5
        outlier_3sigma = z_scores > 3
        
        return {
            'outlier_indices': np.where(outlier_3sigma)[0].tolist(),  # Default: 3-sigma
            'outlier_2sigma_indices': np.where(outlier_2sigma)[0].tolist(),
            'outlier_2_5sigma_indices': np.where(outlier_2_5sigma)[0].tolist(),
            'z_scores': z_scores.tolist(),
            'max_z_score': float(np.max(z_scores)),
            'n_outliers': int(np.sum(outlier_3sigma)),
            'outlier_percentage': float(np.sum(outlier_3sigma) / len(self.values) * 100)
        }
    
    def _modified_zscore_outliers(self) -> Dict[str, Any]:
        """Detect outliers using Modified Z-score (based on median)."""
        median = np.median(self.values)
        mad = stats.median_abs_deviation(self.values)
        
        if mad == 0:
            return {'error': 'Median Absolute Deviation is zero'}
        
        modified_z_scores = 0.6745 * (self.values - median) / mad
        threshold = 3.5  # Common threshold for modified Z-score
        
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        return {
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'modified_z_scores': modified_z_scores.tolist(),
            'threshold': threshold,
            'n_outliers': int(np.sum(outlier_mask)),
            'outlier_percentage': float(np.sum(outlier_mask) / len(self.values) * 100)
        }
    
    def _grubbs_test(self) -> Dict[str, Any]:
        """Perform Grubbs test for outliers (single outlier)."""
        if len(self.values) < 3:
            return {'error': 'Insufficient data for Grubbs test'}
        
        n = len(self.values)
        mean = np.mean(self.values)
        std = np.std(self.values, ddof=1)
        
        if std == 0:
            return {'error': 'Standard deviation is zero'}
        
        # Calculate Grubbs statistic for each point
        grubbs_stats = np.abs(self.values - mean) / std
        max_grubbs = np.max(grubbs_stats)
        max_idx = np.argmax(grubbs_stats)
        
        # Critical value (approximate, for alpha=0.05)
        t_critical = stats.t.ppf(1 - 0.05/(2*n), df=n-2)
        critical_value = ((n-1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n-2 + t_critical**2))
        
        is_outlier = max_grubbs > critical_value
        
        return {
            'outlier_indices': [max_idx] if is_outlier else [],
            'grubbs_statistic': float(max_grubbs),
            'critical_value': float(critical_value),
            'is_outlier': bool(is_outlier),
            'suspected_outlier_index': int(max_idx),
            'suspected_outlier_value': float(self.values[max_idx])
        }
    
    def _tukey_fences(self) -> Dict[str, Any]:
        """Detect outliers using Tukey fences (variations of IQR)."""
        q1, q3 = np.percentile(self.values, [25, 75])
        iqr = q3 - q1
        
        results = {}
        
        # Different fence multipliers
        multipliers = [1.5, 2.0, 2.5, 3.0]
        for mult in multipliers:
            lower_bound = q1 - mult * iqr
            upper_bound = q3 + mult * iqr
            outlier_mask = (self.values < lower_bound) | (self.values > upper_bound)
            
            results[f'fence_{mult}'] = {
                'outlier_indices': np.where(outlier_mask)[0].tolist(),
                'n_outliers': int(np.sum(outlier_mask)),
                'multiplier': mult,
                'bounds': [float(lower_bound), float(upper_bound)]
            }
        
        return results
    
    def _distance_based_outliers(self) -> Dict[str, Any]:
        """Detect outliers based on distance to neighbors."""
        if len(self.coordinates) < 5:
            return {'error': 'Insufficient points for distance-based detection'}
        
        # Calculate distances to k nearest neighbors
        from scipy import spatial
        kdtree = spatial.cKDTree(self.coordinates)
        
        k = min(10, len(self.coordinates) - 1)
        distances, _ = kdtree.query(self.coordinates, k=k+1)
        neighbor_distances = distances[:, 1:]  # Exclude self
        
        # Average distance to k neighbors
        avg_distances = np.mean(neighbor_distances, axis=1)
        
        # Outliers are points with unusually large average distances
        threshold = np.percentile(avg_distances, 100 * (1 - self.contamination))
        outlier_mask = avg_distances > threshold
        
        return {
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'average_distances': avg_distances.tolist(),
            'threshold': float(threshold),
            'n_outliers': int(np.sum(outlier_mask)),
            'k_neighbors': k
        }
    
    def _spatial_local_outlier_factor(self) -> Dict[str, Any]:
        """Spatial version of Local Outlier Factor."""
        if not SKLEARN_AVAILABLE or len(self.coordinates) < 5:
            return {'error': 'Insufficient data or sklearn not available'}
        
        try:
            # Use coordinates for spatial LOF
            lof = LocalOutlierFactor(
                n_neighbors=min(20, len(self.coordinates) - 1),
                contamination=self.contamination
            )
            outlier_labels = lof.fit_predict(self.coordinates)
            outlier_scores = -lof.negative_outlier_factor_
            
            outlier_indices = np.where(outlier_labels == -1)[0].tolist()
            
            return {
                'outlier_indices': outlier_indices,
                'outlier_scores': outlier_scores.tolist(),
                'n_outliers': len(outlier_indices),
                'threshold': float(np.percentile(outlier_scores, 100 * (1 - self.contamination)))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _spatial_residual_outliers(self) -> Dict[str, Any]:
        """Detect outliers based on spatial residuals."""
        if len(self.coordinates) < 10:
            return {'error': 'Insufficient data for spatial residual analysis'}
        
        # Simple spatial trend removal
        try:
            # Fit polynomial surface (degree 1)
            x, y = self.coordinates[:, 0], self.coordinates[:, 1]
            
            # Design matrix for linear trend
            A = np.column_stack([np.ones(len(x)), x, y])
            coeffs, residuals, rank, s = np.linalg.lstsq(A, self.values, rcond=None)
            
            # Calculate fitted values and residuals
            fitted = A @ coeffs
            spatial_residuals = self.values - fitted
            
            # Detect outliers in residuals
            residual_std = np.std(spatial_residuals)
            threshold = 2.5 * residual_std
            outlier_mask = np.abs(spatial_residuals) > threshold
            
            return {
                'outlier_indices': np.where(outlier_mask)[0].tolist(),
                'spatial_residuals': spatial_residuals.tolist(),
                'fitted_values': fitted.tolist(),
                'threshold': float(threshold),
                'n_outliers': int(np.sum(outlier_mask)),
                'residual_std': float(residual_std)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _convex_hull_outliers(self) -> Dict[str, Any]:
        """Detect outliers based on convex hull distance."""
        if len(self.coordinates) < 4:
            return {'error': 'Insufficient points for convex hull analysis'}
        
        try:
            from scipy.spatial import ConvexHull
            
            # Only works for 2D
            if self.coordinates.shape[1] != 2:
                return {'error': 'Convex hull outlier detection only supports 2D data'}
            
            hull = ConvexHull(self.coordinates)
            hull_indices = set(hull.vertices)
            
            # Points on convex hull might be spatial outliers
            # Also check for points far from hull
            
            outlier_indices = []
            
            # Points on convex hull boundary
            boundary_outliers = list(hull_indices)
            
            # Points far from centroid
            centroid = np.mean(self.coordinates, axis=0)
            distances_from_centroid = np.sqrt(np.sum((self.coordinates - centroid)**2, axis=1))
            far_threshold = np.percentile(distances_from_centroid, 95)
            far_outliers = np.where(distances_from_centroid > far_threshold)[0].tolist()
            
            return {
                'boundary_outlier_indices': boundary_outliers,
                'far_outlier_indices': far_outliers,
                'outlier_indices': list(set(boundary_outliers + far_outliers)),
                'hull_vertices': hull.vertices.tolist(),
                'n_boundary_outliers': len(boundary_outliers),
                'n_far_outliers': len(far_outliers)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _mahalanobis_outliers(self) -> Dict[str, Any]:
        """Detect outliers using Mahalanobis distance."""
        # Combine coordinates and values
        if len(self.coordinates[0]) == 2:
            features = np.column_stack([self.coordinates, self.values])
        else:
            features = np.column_stack([self.coordinates, self.values])
        
        try:
            # Calculate covariance matrix
            cov_matrix = np.cov(features.T)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            
            # Calculate Mahalanobis distances
            mean_vector = np.mean(features, axis=0)
            mahal_distances = []
            
            for point in features:
                diff = point - mean_vector
                mahal_dist = np.sqrt(diff @ inv_cov_matrix @ diff.T)
                mahal_distances.append(mahal_dist)
            
            mahal_distances = np.array(mahal_distances)
            
            # Outliers based on chi-square distribution
            from scipy.stats import chi2
            threshold = chi2.ppf(1 - self.contamination, df=features.shape[1])
            outlier_mask = mahal_distances**2 > threshold
            
            return {
                'outlier_indices': np.where(outlier_mask)[0].tolist(),
                'mahalanobis_distances': mahal_distances.tolist(),
                'threshold': float(np.sqrt(threshold)),
                'n_outliers': int(np.sum(outlier_mask)),
                'outlier_percentage': float(np.sum(outlier_mask) / len(mahal_distances) * 100)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _isolation_forest_outliers(self) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest."""
        features = np.column_stack([self.coordinates, self.values])
        
        try:
            iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            outlier_labels = iso_forest.fit_predict(features)
            outlier_scores = iso_forest.score_samples(features)
            
            outlier_indices = np.where(outlier_labels == -1)[0].tolist()
            
            return {
                'outlier_indices': outlier_indices,
                'outlier_scores': outlier_scores.tolist(),
                'n_outliers': len(outlier_indices),
                'contamination_used': self.contamination
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _one_class_svm_outliers(self) -> Dict[str, Any]:
        """Detect outliers using One-Class SVM."""
        features = np.column_stack([self.coordinates, self.values])
        
        try:
            # Standardize features
            features_std = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
            
            svm = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
            outlier_labels = svm.fit_predict(features_std)
            
            outlier_indices = np.where(outlier_labels == -1)[0].tolist()
            
            return {
                'outlier_indices': outlier_indices,
                'n_outliers': len(outlier_indices),
                'nu_parameter': self.contamination
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _local_outlier_factor_outliers(self) -> Dict[str, Any]:
        """Detect outliers using Local Outlier Factor."""
        features = np.column_stack([self.coordinates, self.values])
        
        try:
            lof = LocalOutlierFactor(
                n_neighbors=min(20, len(features) - 1),
                contamination=self.contamination
            )
            outlier_labels = lof.fit_predict(features)
            outlier_scores = -lof.negative_outlier_factor_
            
            outlier_indices = np.where(outlier_labels == -1)[0].tolist()
            
            return {
                'outlier_indices': outlier_indices,
                'outlier_scores': outlier_scores.tolist(),
                'n_outliers': len(outlier_indices)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _elliptic_envelope_outliers(self) -> Dict[str, Any]:
        """Detect outliers using Elliptic Envelope (robust covariance)."""
        features = np.column_stack([self.coordinates, self.values])
        
        try:
            envelope = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
            outlier_labels = envelope.fit_predict(features)
            
            outlier_indices = np.where(outlier_labels == -1)[0].tolist()
            
            return {
                'outlier_indices': outlier_indices,
                'n_outliers': len(outlier_indices),
                'contamination_used': self.contamination
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of outlier detection results."""
        all_outlier_sets = []
        method_counts = {}
        
        # Collect all outlier indices
        for category, methods in results.items():
            if category == 'summary':
                continue
                
            if isinstance(methods, dict):
                for method, result in methods.items():
                    if isinstance(result, dict) and 'outlier_indices' in result:
                        outliers = set(result['outlier_indices'])
                        all_outlier_sets.append(outliers)
                        method_counts[f"{category}_{method}"] = len(outliers)
        
        # Calculate union and intersection
        if all_outlier_sets:
            union_outliers = set().union(*all_outlier_sets)
            intersection_outliers = set.intersection(*all_outlier_sets) if len(all_outlier_sets) > 1 else all_outlier_sets[0]
        else:
            union_outliers = set()
            intersection_outliers = set()
        
        # Statistics
        n_total_points = len(self.data)
        
        summary = {
            'total_points': n_total_points,
            'methods_used': len(method_counts),
            'union_outliers': list(union_outliers),
            'intersection_outliers': list(intersection_outliers),
            'n_union_outliers': len(union_outliers),
            'n_intersection_outliers': len(intersection_outliers),
            'union_percentage': float(len(union_outliers) / n_total_points * 100) if n_total_points > 0 else 0,
            'intersection_percentage': float(len(intersection_outliers) / n_total_points * 100) if n_total_points > 0 else 0,
            'method_agreement': float(len(intersection_outliers) / len(union_outliers)) if len(union_outliers) > 0 else 0,
            'outliers_per_method': method_counts
        }
        
        # Recommendations
        if len(intersection_outliers) > 0:
            summary['recommendation'] = 'High confidence outliers found (intersection of methods)'
        elif len(union_outliers) > n_total_points * 0.2:
            summary['recommendation'] = 'Many potential outliers detected - consider data quality'
        elif len(union_outliers) == 0:
            summary['recommendation'] = 'No outliers detected by any method'
        else:
            summary['recommendation'] = 'Some outliers detected - review individual methods'
        
        return summary