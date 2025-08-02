#!/usr/bin/env python3
"""
Test spatial analysis functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

def test_spatial_analysis():
    """Test the spatial analysis functionality."""
    print("=== Testing Spatial Analysis ===")
    
    try:
        from src.analysis.spatial_analyzer import SpatialAnalyzer
        print("OK: SpatialAnalyzer imports successfully")
        
        # Create test data with spatial patterns
        np.random.seed(42)
        
        # Create clustered data
        cluster1 = np.random.normal([100, 100], [10, 10], (20, 2))
        cluster2 = np.random.normal([200, 200], [15, 15], (15, 2))
        scattered = np.random.uniform([50, 50], [250, 250], (10, 2))
        
        coordinates = np.vstack([cluster1, cluster2, scattered])
        values = np.concatenate([
            np.random.normal(50, 5, 20),  # Cluster 1 values
            np.random.normal(30, 3, 15),  # Cluster 2 values  
            np.random.normal(40, 10, 10)  # Scattered values
        ])
        
        # Create DataFrame
        df = pd.DataFrame({
            'X': coordinates[:, 0],
            'Y': coordinates[:, 1],
            'Value': values
        })
        
        print(f"OK: Test data created: {len(df)} points with clusters")
        
        # Initialize analyzer
        analyzer = SpatialAnalyzer()
        
        # Test full analysis
        results = analyzer.analyze(df, 'X', 'Y', 'Value')
        print("OK: Complete spatial analysis completed")
        
        # Test individual components
        print(f"OK: Density analysis: {len(results.density_analysis)} metrics")
        print(f"OK: Clustering analysis: {len(results.clustering_analysis)} metrics")
        print(f"OK: Pattern analysis: {len(results.pattern_analysis)} metrics")
        print(f"OK: Anisotropy analysis: {len(results.anisotropy_analysis)} metrics")
        print(f"OK: Neighborhood analysis: {len(results.neighborhood_analysis)} metrics")
        print(f"OK: Spatial statistics: {len(results.spatial_statistics)} metrics")
        
        # Print some key results
        if 'nearest_neighbor' in results.density_analysis:
            nn = results.density_analysis['nearest_neighbor']
            print(f"  Mean nearest neighbor distance: {nn['mean_distance']:.2f}")
            print(f"  Clark-Evans index: {nn['clark_evans_index']:.3f}")
        
        if 'kmeans' in results.clustering_analysis:
            kmeans = results.clustering_analysis['kmeans']
            print(f"  Optimal K-means clusters: {kmeans.get('optimal_k', 'N/A')}")
        
        if 'dbscan' in results.clustering_analysis:
            dbscan = results.clustering_analysis['dbscan']
            if 'n_clusters' in dbscan:
                print(f"  DBSCAN clusters found: {dbscan['n_clusters']}")
        
        if 'directional_analysis' in results.anisotropy_analysis:
            aniso = results.anisotropy_analysis['directional_analysis']
            print(f"  Anisotropy ratio: {aniso['anisotropy_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Spatial analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_density_analysis():
    """Test density analysis specifically."""
    print("\n=== Testing Density Analysis ===")
    
    try:
        from src.analysis.spatial_analyzer import SpatialAnalyzer
        
        # Create regular grid
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        xx, yy = np.meshgrid(x, y)
        coordinates = np.column_stack([xx.ravel(), yy.ravel()])
        values = np.random.normal(0, 1, len(coordinates))
        
        df = pd.DataFrame({
            'X': coordinates[:, 0],
            'Y': coordinates[:, 1], 
            'Value': values
        })
        
        analyzer = SpatialAnalyzer()
        results = analyzer.analyze(df, 'X', 'Y', 'Value')
        
        density = results.density_analysis
        
        print(f"Points per unit area: {density.get('points_per_unit_area', 'N/A'):.4f}")
        if 'nearest_neighbor' in density:
            nn = density['nearest_neighbor']
            print(f"Mean NN distance: {nn['mean_distance']:.3f}")
            print(f"Clark-Evans index: {nn['clark_evans_index']:.3f} (1.0 = random)")
        
        if 'density_variation' in density:
            dv = density['density_variation']
            print(f"Density CV: {dv.get('density_coefficient_variation', 'N/A'):.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Density analysis test failed: {e}")
        return False

def test_clustering_analysis():
    """Test clustering analysis."""
    print("\n=== Testing Clustering Analysis ===")
    
    try:
        from src.analysis.spatial_analyzer import SpatialAnalyzer
        
        # Create data with clear clusters
        np.random.seed(123)
        cluster1 = np.random.normal([0, 0], [1, 1], (25, 2))
        cluster2 = np.random.normal([10, 10], [1, 1], (25, 2))
        
        coordinates = np.vstack([cluster1, cluster2])
        values = np.concatenate([
            np.random.normal(100, 5, 25),
            np.random.normal(200, 5, 25)
        ])
        
        df = pd.DataFrame({
            'X': coordinates[:, 0],
            'Y': coordinates[:, 1],
            'Value': values
        })
        
        analyzer = SpatialAnalyzer()
        results = analyzer.analyze(df, 'X', 'Y', 'Value')
        
        clustering = results.clustering_analysis
        
        if 'kmeans' in clustering:
            kmeans = clustering['kmeans']
            print(f"Optimal K: {kmeans.get('optimal_k', 'N/A')}")
            if 'final_clustering' in kmeans:
                print(f"Final inertia: {kmeans['final_clustering']['inertia']:.2f}")
        
        if 'dbscan' in clustering:
            dbscan = clustering['dbscan']
            if 'n_clusters' in dbscan:
                print(f"DBSCAN clusters: {dbscan['n_clusters']}")
                print(f"Noise ratio: {dbscan['noise_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Clustering analysis test failed: {e}")
        return False

def test_anisotropy_analysis():
    """Test anisotropy analysis."""
    print("\n=== Testing Anisotropy Analysis ===")
    
    try:
        from src.analysis.spatial_analyzer import SpatialAnalyzer
        
        # Create elongated pattern (anisotropic)
        np.random.seed(42)
        x = np.random.normal(0, 10, 50)  # Wide spread in X
        y = np.random.normal(0, 2, 50)   # Narrow spread in Y
        values = np.random.normal(0, 1, 50)
        
        df = pd.DataFrame({'X': x, 'Y': y, 'Value': values})
        
        analyzer = SpatialAnalyzer()
        results = analyzer.analyze(df, 'X', 'Y', 'Value')
        
        aniso = results.anisotropy_analysis
        
        if 'directional_analysis' in aniso:
            da = aniso['directional_analysis']
            print(f"Anisotropy ratio: {da['anisotropy_ratio']:.3f}")
            print(f"Max range direction: {da['max_range_direction']:.1f} degrees")
            print(f"Min range direction: {da['min_range_direction']:.1f} degrees")
        
        if 'anisotropy_ellipse' in aniso:
            ellipse = aniso['anisotropy_ellipse']
            if 'major_axis' in ellipse:
                print(f"Ellipse ratio: {ellipse['anisotropy_ratio']:.3f}")
                print(f"Orientation: {ellipse['orientation_degrees']:.1f} degrees")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Anisotropy analysis test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== SPATIAL ANALYSIS TESTS ===")
    
    success = True
    success &= test_spatial_analysis()
    success &= test_density_analysis()
    success &= test_clustering_analysis()
    success &= test_anisotropy_analysis()
    
    if success:
        print("\nOK: All spatial analysis tests passed!")
    else:
        print("\nERROR: Some tests failed!")
    
    print("\n=== TESTS COMPLETED ===")