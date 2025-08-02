"""
Advanced spatial analysis for geological data.

Provides comprehensive spatial analysis including density estimation,
clustering analysis, and spatial pattern recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from scipy import spatial, stats
from scipy.spatial.distance import pdist, squareform
import warnings

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.neighbors import KernelDensity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class SpatialResults:
    """Container for spatial analysis results."""
    density_analysis: Dict[str, Any]
    clustering_analysis: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    anisotropy_analysis: Dict[str, Any]
    neighborhood_analysis: Dict[str, Any]
    spatial_statistics: Dict[str, Any]


class SpatialAnalyzer:
    """
    Advanced spatial analyzer for geological data.
    
    Provides comprehensive spatial analysis including:
    - Kernel density estimation
    - Clustering analysis (K-means, DBSCAN)
    - Spatial pattern recognition
    - Anisotropy analysis
    - Neighborhood analysis
    - Spatial statistics
    """
    
    def __init__(self):
        """Initialize spatial analyzer."""
        self.coordinates: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.kdtree: Optional[spatial.cKDTree] = None
        self.dimensions: int = 2
    
    def analyze(self, 
                data: pd.DataFrame,
                x_col: str,
                y_col: str,
                value_col: str,
                z_col: Optional[str] = None) -> SpatialResults:
        """
        Perform comprehensive spatial analysis.
        
        Args:
            data: Input DataFrame
            x_col: X coordinate column
            y_col: Y coordinate column  
            value_col: Value column
            z_col: Z coordinate column (optional for 3D)
            
        Returns:
            SpatialResults object with complete analysis
        """
        # Prepare coordinates
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        self.coordinates = data[coord_cols].values
        self.values = data[value_col].values
        self.dimensions = len(coord_cols)
        
        # Build KD-tree
        self.kdtree = spatial.cKDTree(self.coordinates)
        
        # Perform analyses
        density = self._density_analysis()
        clustering = self._clustering_analysis()
        patterns = self._pattern_analysis()
        anisotropy = self._anisotropy_analysis()
        neighborhood = self._neighborhood_analysis()
        spatial_stats = self._spatial_statistics()
        
        return SpatialResults(
            density_analysis=density,
            clustering_analysis=clustering,
            pattern_analysis=patterns,
            anisotropy_analysis=anisotropy,
            neighborhood_analysis=neighborhood,
            spatial_statistics=spatial_stats
        )
    
    def _density_analysis(self) -> Dict[str, Any]:
        """Analyze point density patterns."""
        analysis = {}
        
        # Basic density metrics
        if self.dimensions == 2:
            area = self._calculate_convex_hull_area()
            analysis['points_per_unit_area'] = len(self.coordinates) / area if area > 0 else 0
        else:  # 3D
            volume = self._calculate_convex_hull_volume()
            analysis['points_per_unit_volume'] = len(self.coordinates) / volume if volume > 0 else 0
        
        # Nearest neighbor density
        if len(self.coordinates) > 1:
            distances, _ = self.kdtree.query(self.coordinates, k=2)
            nn_distances = distances[:, 1]  # First neighbor distances
            
            analysis['nearest_neighbor'] = {
                'mean_distance': float(np.mean(nn_distances)),
                'std_distance': float(np.std(nn_distances)),
                'min_distance': float(np.min(nn_distances)),
                'max_distance': float(np.max(nn_distances)),
                'clark_evans_index': self._clark_evans_index(nn_distances)
            }
        
        # Grid-based density variation
        analysis['density_variation'] = self._analyze_density_variation()
        
        # Kernel density estimation
        if SKLEARN_AVAILABLE and len(self.coordinates) >= 5:
            analysis['kde_analysis'] = self._kernel_density_analysis()
        
        return analysis
    
    def _clustering_analysis(self) -> Dict[str, Any]:
        """Analyze spatial clustering patterns."""
        analysis = {}
        
        if not SKLEARN_AVAILABLE or len(self.coordinates) < 4:
            analysis['error'] = 'Insufficient data or sklearn not available'
            return analysis
        
        # K-means clustering
        analysis['kmeans'] = self._kmeans_analysis()
        
        # DBSCAN clustering
        analysis['dbscan'] = self._dbscan_analysis()
        
        # Cluster validation metrics
        analysis['validation'] = self._cluster_validation()
        
        return analysis
    
    def _pattern_analysis(self) -> Dict[str, Any]:
        """Analyze spatial patterns and regularity."""
        analysis = {}
        
        if len(self.coordinates) < 3:
            return {'error': 'Insufficient points for pattern analysis'}
        
        # Ripley's K function approximation
        analysis['ripley_k'] = self._ripley_k_analysis()
        
        # Quadrat analysis
        analysis['quadrat_analysis'] = self._quadrat_analysis()
        
        # Spatial autocorrelation
        analysis['spatial_autocorrelation'] = self._spatial_autocorrelation()
        
        # Pattern regularity index
        analysis['regularity_index'] = self._pattern_regularity()
        
        return analysis
    
    def _anisotropy_analysis(self) -> Dict[str, Any]:
        """Analyze directional variations (anisotropy)."""
        if self.dimensions != 2 or len(self.coordinates) < 10:
            return {'error': 'Insufficient data or not 2D'}
        
        analysis = {}
        
        # Directional variograms
        directions = np.linspace(0, 180, 8, endpoint=False)
        directional_ranges = []
        directional_variances = []
        
        for angle in directions:
            range_val, variance = self._directional_variogram(angle)
            directional_ranges.append(range_val)
            directional_variances.append(variance)
        
        analysis['directional_analysis'] = {
            'directions': directions.tolist(),
            'ranges': directional_ranges,
            'variances': directional_variances,
            'max_range_direction': float(directions[np.argmax(directional_ranges)]),
            'min_range_direction': float(directions[np.argmin(directional_ranges)]),
            'anisotropy_ratio': float(np.min(directional_ranges) / np.max(directional_ranges)) if np.max(directional_ranges) > 0 else 1.0
        }
        
        # Ellipse of anisotropy
        analysis['anisotropy_ellipse'] = self._anisotropy_ellipse()
        
        return analysis
    
    def _neighborhood_analysis(self) -> Dict[str, Any]:
        """Analyze neighborhood characteristics."""
        analysis = {}
        
        if len(self.coordinates) < 5:
            return {'error': 'Insufficient points for neighborhood analysis'}
        
        # Multi-scale neighborhood analysis
        k_values = [3, 5, 10, min(20, len(self.coordinates)-1)]
        for k in k_values:
            if k < len(self.coordinates):
                distances, indices = self.kdtree.query(self.coordinates, k=k+1)
                neighbor_distances = distances[:, 1:]  # Exclude self
                
                analysis[f'k{k}_neighbors'] = {
                    'mean_distance': float(np.mean(neighbor_distances)),
                    'std_distance': float(np.std(neighbor_distances)),
                    'max_distance': float(np.max(neighbor_distances)),
                    'distance_distribution': np.histogram(neighbor_distances.flatten(), bins=10)[0].tolist()
                }
        
        # Local density variations
        analysis['local_density'] = self._local_density_analysis()
        
        return analysis
    
    def _spatial_statistics(self) -> Dict[str, Any]:
        """Calculate various spatial statistics."""
        stats_dict = {}
        
        if len(self.coordinates) < 2:
            return {'error': 'Insufficient points'}
        
        # Centroid and spread
        centroid = np.mean(self.coordinates, axis=0)
        stats_dict['centroid'] = centroid.tolist()
        
        # Standard distance (spatial standard deviation)
        distances_from_centroid = np.sqrt(np.sum((self.coordinates - centroid)**2, axis=1))
        stats_dict['standard_distance'] = float(np.std(distances_from_centroid))
        stats_dict['mean_distance_from_centroid'] = float(np.mean(distances_from_centroid))
        
        # Bounding box and area
        bounds = {
            'min': np.min(self.coordinates, axis=0).tolist(),
            'max': np.max(self.coordinates, axis=0).tolist(),
            'range': np.ptp(self.coordinates, axis=0).tolist()
        }
        stats_dict['bounds'] = bounds
        
        if self.dimensions == 2:
            stats_dict['bounding_box_area'] = float(bounds['range'][0] * bounds['range'][1])
            stats_dict['convex_hull_area'] = self._calculate_convex_hull_area()
        
        # Compactness measures
        stats_dict['compactness'] = self._calculate_compactness()
        
        return stats_dict
    
    def _calculate_convex_hull_area(self) -> float:
        """Calculate convex hull area for 2D data."""
        if self.dimensions != 2 or len(self.coordinates) < 3:
            return 0.0
        
        try:
            hull = spatial.ConvexHull(self.coordinates)
            return float(hull.volume)  # In 2D, volume is area
        except Exception:
            return 0.0
    
    def _calculate_convex_hull_volume(self) -> float:
        """Calculate convex hull volume for 3D data."""
        if self.dimensions != 3 or len(self.coordinates) < 4:
            return 0.0
        
        try:
            hull = spatial.ConvexHull(self.coordinates)
            return float(hull.volume)
        except Exception:
            return 0.0
    
    def _clark_evans_index(self, nn_distances: np.ndarray) -> float:
        """Calculate Clark-Evans clustering index."""
        observed_mean = np.mean(nn_distances)
        
        # Expected mean distance for random distribution
        if self.dimensions == 2:
            area = self._calculate_convex_hull_area()
            density = len(self.coordinates) / area if area > 0 else 1
            expected_mean = 0.5 / np.sqrt(density)
        else:
            # 3D approximation
            volume = self._calculate_convex_hull_volume()
            density = len(self.coordinates) / volume if volume > 0 else 1
            expected_mean = 0.5 * (1 / density) ** (1/3)
        
        return float(observed_mean / expected_mean) if expected_mean > 0 else 1.0
    
    def _analyze_density_variation(self) -> Dict[str, Any]:
        """Analyze spatial variation in point density."""
        # Create grid and count points per cell
        n_bins = max(3, min(int(np.sqrt(len(self.coordinates) / 2)), 15))
        
        if self.dimensions == 2:
            hist, x_edges, y_edges = np.histogram2d(
                self.coordinates[:, 0],
                self.coordinates[:, 1],
                bins=n_bins
            )
            
            # Calculate density statistics
            non_zero_cells = hist[hist > 0]
            
            return {
                'grid_size': n_bins,
                'total_cells': int(n_bins ** 2),
                'occupied_cells': int(len(non_zero_cells)),
                'occupancy_rate': float(len(non_zero_cells) / (n_bins ** 2)),
                'mean_points_per_cell': float(np.mean(non_zero_cells)) if len(non_zero_cells) > 0 else 0,
                'std_points_per_cell': float(np.std(non_zero_cells)) if len(non_zero_cells) > 0 else 0,
                'max_points_per_cell': int(np.max(hist)),
                'density_coefficient_variation': float(np.std(non_zero_cells) / np.mean(non_zero_cells)) if len(non_zero_cells) > 0 and np.mean(non_zero_cells) > 0 else 0
            }
        
        return {'error': '3D density variation not implemented'}
    
    def _kernel_density_analysis(self) -> Dict[str, Any]:
        """Perform kernel density estimation analysis."""
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        try:
            # Fit KDE with different bandwidths
            bandwidths = np.logspace(-2, 1, 10)
            best_bandwidth = None
            best_score = -np.inf
            
            for bw in bandwidths:
                kde = KernelDensity(bandwidth=bw, kernel='gaussian')
                kde.fit(self.coordinates)
                score = kde.score(self.coordinates)
                if score > best_score:
                    best_score = score
                    best_bandwidth = bw
            
            # Fit final KDE
            kde = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
            kde.fit(self.coordinates)
            
            # Calculate density at each point
            log_densities = kde.score_samples(self.coordinates)
            densities = np.exp(log_densities)
            
            return {
                'optimal_bandwidth': float(best_bandwidth),
                'log_likelihood': float(best_score),
                'density_statistics': {
                    'mean': float(np.mean(densities)),
                    'std': float(np.std(densities)),
                    'min': float(np.min(densities)),
                    'max': float(np.max(densities)),
                    'cv': float(np.std(densities) / np.mean(densities)) if np.mean(densities) > 0 else 0
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _kmeans_analysis(self) -> Dict[str, Any]:
        """Perform K-means clustering analysis."""
        results = {}
        
        # Test different numbers of clusters
        max_k = min(10, len(self.coordinates) // 2)
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.coordinates)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score if sklearn has it
            try:
                from sklearn.metrics import silhouette_score
                score = silhouette_score(self.coordinates, labels)
                silhouette_scores.append(score)
            except ImportError:
                silhouette_scores.append(0.0)
        
        # Find optimal k using elbow method
        if len(inertias) > 2:
            # Simple elbow detection
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            optimal_k = np.argmax(second_diffs) + 2  # +2 because we started from k=2
        else:
            optimal_k = 2
        
        results['optimal_k'] = int(optimal_k)
        results['inertias'] = inertias
        results['silhouette_scores'] = silhouette_scores
        results['k_range'] = list(range(2, max_k + 1))
        
        # Perform final clustering with optimal k
        if optimal_k <= max_k:
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_labels = final_kmeans.fit_predict(self.coordinates)
            
            results['final_clustering'] = {
                'labels': final_labels.tolist(),
                'centers': final_kmeans.cluster_centers_.tolist(),
                'inertia': float(final_kmeans.inertia_)
            }
        
        return results
    
    def _dbscan_analysis(self) -> Dict[str, Any]:
        """Perform DBSCAN clustering analysis."""
        # Estimate eps using k-distance graph
        if len(self.coordinates) >= 4:
            distances, _ = self.kdtree.query(self.coordinates, k=4)
            k_distances = np.sort(distances[:, 3])  # 4th nearest neighbor
            eps_estimate = np.percentile(k_distances, 95)  # Use 95th percentile
        else:
            eps_estimate = 0.1
        
        # Try different eps values around the estimate
        eps_values = [eps_estimate * factor for factor in [0.5, 0.75, 1.0, 1.25, 1.5]]
        min_samples_values = [3, 5, max(3, len(self.coordinates) // 20)]
        
        best_result = None
        best_score = -1
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                if min_samples >= len(self.coordinates):
                    continue
                    
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.coordinates)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # Score based on number of clusters and noise points
                if n_clusters > 0:
                    score = n_clusters / (1 + n_noise / len(self.coordinates))
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'labels': labels.tolist(),
                            'noise_ratio': n_noise / len(self.coordinates)
                        }
        
        if best_result is None:
            return {'error': 'No suitable clustering found'}
        
        return best_result
    
    def _cluster_validation(self) -> Dict[str, Any]:
        """Validate clustering results."""
        # This would contain various clustering validation metrics
        # For now, return placeholder
        return {'note': 'Cluster validation metrics would be calculated here'}
    
    def _ripley_k_analysis(self) -> Dict[str, Any]:
        """Simplified Ripley's K function analysis."""
        # Calculate K function for several distance values
        max_distance = np.percentile(pdist(self.coordinates), 95)
        distances = np.linspace(max_distance/10, max_distance, 10)
        
        k_values = []
        for d in distances:
            # Count pairs within distance d
            pair_distances = pdist(self.coordinates)
            pairs_within_d = np.sum(pair_distances <= d)
            
            # Estimate area (2D) or volume (3D)
            if self.dimensions == 2:
                area = self._calculate_convex_hull_area()
                lambda_hat = len(self.coordinates) / area if area > 0 else 1
                k_d = pairs_within_d / (len(self.coordinates) * lambda_hat)
            else:
                volume = self._calculate_convex_hull_volume()
                lambda_hat = len(self.coordinates) / volume if volume > 0 else 1
                k_d = pairs_within_d / (len(self.coordinates) * lambda_hat)
            
            k_values.append(k_d)
        
        return {
            'distances': distances.tolist(),
            'k_values': k_values,
            'interpretation': 'Ripley K analysis - values > expected indicate clustering'
        }
    
    def _quadrat_analysis(self) -> Dict[str, Any]:
        """Perform quadrat analysis for spatial randomness."""
        # Divide area into quadrats and count points
        n_quadrats = max(4, min(16, len(self.coordinates) // 3))
        
        if self.dimensions == 2:
            hist, _, _ = np.histogram2d(
                self.coordinates[:, 0],
                self.coordinates[:, 1],
                bins=int(np.sqrt(n_quadrats))
            )
            
            observed_counts = hist.flatten()
            mean_count = np.mean(observed_counts)
            variance = np.var(observed_counts)
            
            # Variance-to-mean ratio
            vmr = variance / mean_count if mean_count > 0 else 1.0
            
            return {
                'n_quadrats': len(observed_counts),
                'mean_count_per_quadrat': float(mean_count),
                'variance': float(variance),  
                'variance_to_mean_ratio': float(vmr),
                'interpretation': 'VMR > 1: clustered, VMR = 1: random, VMR < 1: regular'
            }
        
        return {'error': '3D quadrat analysis not implemented'}
    
    def _spatial_autocorrelation(self) -> Dict[str, Any]:
        """Calculate spatial autocorrelation of values."""
        if len(self.values) < 5:
            return {'error': 'Insufficient data'}
        
        # Calculate distance matrix
        distances = squareform(pdist(self.coordinates))
        
        # Create simple spatial weights (inverse distance with cutoff)
        max_distance = np.percentile(distances[distances > 0], 75)
        # Use safe division to avoid zero division
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_distances = np.divide(1, distances, out=np.zeros_like(distances), where=distances!=0)
        weights = np.where((distances <= max_distance) & (distances > 0), inv_distances, 0)
        np.fill_diagonal(weights, 0)  # No self-weights
        
        # Calculate Moran's I approximation
        n = len(self.values)
        mean_val = np.mean(self.values)
        
        numerator = 0
        denominator = 0
        w_sum = 0
        
        for i in range(n):
            for j in range(n):
                if weights[i, j] > 0:
                    numerator += weights[i, j] * (self.values[i] - mean_val) * (self.values[j] - mean_val)
                    w_sum += weights[i, j]
            denominator += (self.values[i] - mean_val) ** 2
        
        if w_sum > 0 and denominator > 0:
            morans_i = (n / w_sum) * (numerator / denominator)
        else:
            morans_i = 0.0
        
        return {
            'morans_i': float(morans_i),
            'interpretation': 'I > 0: positive autocorr, I < 0: negative autocorr, I ≈ 0: no autocorr'
        }
    
    def _pattern_regularity(self) -> float:
        """Calculate pattern regularity index."""
        if len(self.coordinates) < 3:
            return 0.0
        
        # Based on nearest neighbor distance uniformity
        distances, _ = self.kdtree.query(self.coordinates, k=2)
        nn_distances = distances[:, 1]
        
        cv = np.std(nn_distances) / np.mean(nn_distances) if np.mean(nn_distances) > 0 else 1.0
        regularity = 1.0 / (1.0 + cv)  # Convert to 0-1 scale
        
        return float(regularity)
    
    def _directional_variogram(self, angle_degrees: float) -> Tuple[float, float]:
        """Calculate directional variogram properties."""
        angle_rad = np.radians(angle_degrees)
        
        # Project coordinates onto direction
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        projections = np.dot(self.coordinates[:, :2], direction_vector)
        
        # Calculate range and variance in this direction
        range_val = np.ptp(projections)
        
        # Simple variance estimate
        if len(projections) > 1:
            variance = np.var(self.values)
        else:
            variance = 0.0
        
        return float(range_val), float(variance)
    
    def _anisotropy_ellipse(self) -> Dict[str, Any]:
        """Calculate anisotropy ellipse parameters."""
        if self.dimensions != 2:
            return {'error': 'Only 2D supported'}
        
        # Calculate covariance matrix of coordinates
        cov_matrix = np.cov(self.coordinates.T)
        
        # Eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Calculate ellipse parameters
        major_axis = float(2 * np.sqrt(eigenvals[0]))
        minor_axis = float(2 * np.sqrt(eigenvals[1]))
        orientation = float(np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])))
        eccentricity = float(np.sqrt(1 - (minor_axis/major_axis)**2)) if major_axis > 0 else 0.0
        
        return {
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'orientation_degrees': orientation,
            'eccentricity': eccentricity,
            'anisotropy_ratio': float(minor_axis / major_axis) if major_axis > 0 else 1.0
        }
    
    def _local_density_analysis(self) -> Dict[str, Any]:
        """Analyze local density variations."""
        if len(self.coordinates) < 5:
            return {'error': 'Insufficient data'}
        
        # Calculate local density for each point using k-nearest neighbors
        k = min(10, len(self.coordinates) - 1)
        distances, _ = self.kdtree.query(self.coordinates, k=k+1)
        neighbor_distances = distances[:, 1:]  # Exclude self
        
        # Local density = k / (area of circle containing k neighbors)
        max_distances = np.max(neighbor_distances, axis=1)
        if self.dimensions == 2:
            local_densities = k / (np.pi * max_distances**2)
        else:  # 3D
            local_densities = k / ((4/3) * np.pi * max_distances**3)
        
        return {
            'local_densities': local_densities.tolist(),
            'mean_local_density': float(np.mean(local_densities)),
            'std_local_density': float(np.std(local_densities)),
            'density_variation': float(np.std(local_densities) / np.mean(local_densities)) if np.mean(local_densities) > 0 else 0,
            'k_neighbors_used': k
        }
    
    def _calculate_compactness(self) -> float:
        """Calculate spatial compactness measure."""
        if self.dimensions == 2:
            area = self._calculate_convex_hull_area()
            perimeter = self._calculate_convex_hull_perimeter()
            
            if perimeter > 0:
                # Isoperimetric quotient
                compactness = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness = 0.0
        else:
            # 3D compactness approximation
            volume = self._calculate_convex_hull_volume()
            surface_area = self._calculate_convex_hull_surface_area()
            
            if surface_area > 0:
                compactness = (36 * np.pi * volume**2) ** (1/3) / surface_area
            else:
                compactness = 0.0
        
        return float(compactness)
    
    def _calculate_convex_hull_perimeter(self) -> float:
        """Calculate convex hull perimeter for 2D data."""
        if self.dimensions != 2 or len(self.coordinates) < 3:
            return 0.0
        
        try:
            hull = spatial.ConvexHull(self.coordinates)
            perimeter = 0.0
            vertices = self.coordinates[hull.vertices]
            
            for i in range(len(vertices)):
                j = (i + 1) % len(vertices)
                perimeter += np.linalg.norm(vertices[j] - vertices[i])
            
            return float(perimeter)
        except Exception:
            return 0.0
    
    def _calculate_convex_hull_surface_area(self) -> float:
        """Calculate convex hull surface area for 3D data."""
        if self.dimensions != 3 or len(self.coordinates) < 4:
            return 0.0
        
        try:
            hull = spatial.ConvexHull(self.coordinates)
            return float(hull.area)  # In 3D, area is surface area
        except Exception:
            return 0.0