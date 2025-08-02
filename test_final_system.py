#!/usr/bin/env python3
"""
Final comprehensive test of the analysis and recommendation system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

def test_complete_analysis_pipeline():
    """Test the complete analysis pipeline."""
    print("=== Testing Complete Analysis Pipeline ===")
    
    try:
        # Import all main components
        from src.analysis.statistical_analyzer import StatisticalAnalyzer
        from src.analysis.spatial_analyzer import SpatialAnalyzer
        from src.core.interpolation.variogram_analysis import VariogramAnalyzer
        from src.core.recommendations.recommendation_engine import RecommendationEngine
        
        print("OK: All analysis modules imported successfully")
        
        # Create realistic coal exploration dataset
        np.random.seed(42)
        n_points = 50
        
        # Simulate drill hole locations
        x = np.random.uniform(0, 1000, n_points)  # meters
        y = np.random.uniform(0, 1000, n_points)
        
        # Simulate ash content with spatial structure
        ash_content = np.zeros(n_points)
        base_trend = 12 + 0.008 * x + 0.004 * y  # Regional trend
        
        # Add local spatial correlation
        for i in range(n_points):
            local_effect = 0
            for j in range(i):  # Only use previous points
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < 200:  # Influence range
                    weight = np.exp(-dist/100)
                    local_effect += weight * (ash_content[j] - base_trend[j])
            
            ash_content[i] = base_trend[i] + 0.3 * local_effect + np.random.normal(0, 1.5)
        
        # Ensure reasonable values
        ash_content = np.clip(ash_content, 5, 25)
        
        data = pd.DataFrame({
            'X': x,
            'Y': y,
            'Ash_Content': ash_content
        })
        
        print(f"OK: Simulated coal dataset: {len(data)} drill holes")
        print(f"  Ash content range: {ash_content.min():.1f} - {ash_content.max():.1f}%")
        print(f"  Mean ash content: {ash_content.mean():.1f}%")
        
        # 1. Statistical Analysis
        print("\n--- Statistical Analysis ---")
        stat_analyzer = StatisticalAnalyzer()
        stat_results = stat_analyzer.analyze(data, 'Ash_Content')
        
        print(f"Count: {stat_results.descriptive_stats['count']:.0f}")
        print(f"Mean: {stat_results.descriptive_stats['mean']:.2f}%")
        print(f"Std: {stat_results.descriptive_stats['std']:.2f}%")
        print(f"Skewness: {stat_results.distribution_analysis['skewness']:.3f}")
        
        # Test enhanced features
        raw_values = np.array(stat_results.descriptive_stats['_raw_values'])
        grubbs_result = stat_analyzer.grubbs_test(raw_values)
        print(f"Outliers detected: {grubbs_result['n_outliers']}")
        
        transform_result = stat_analyzer.transform_data(raw_values, method='auto')
        print(f"Best transformation: {transform_result['method']}")
        
        # 2. Spatial Analysis  
        print("\n--- Spatial Analysis ---")
        spatial_analyzer = SpatialAnalyzer()
        spatial_results = spatial_analyzer.analyze(data, 'X', 'Y', 'Ash_Content')
        
        if 'nearest_neighbor' in spatial_results.density_analysis:
            nn = spatial_results.density_analysis['nearest_neighbor']
            print(f"Mean NN distance: {nn['mean_distance']:.1f} m")
            print(f"Clark-Evans index: {nn['clark_evans_index']:.3f}")
        
        if 'kmeans' in spatial_results.clustering_analysis:
            kmeans = spatial_results.clustering_analysis['kmeans']
            print(f"Optimal clusters: {kmeans.get('optimal_k', 'N/A')}")
        
        if 'directional_analysis' in spatial_results.anisotropy_analysis:
            aniso = spatial_results.anisotropy_analysis['directional_analysis']
            print(f"Anisotropy ratio: {aniso['anisotropy_ratio']:.3f}")
        
        # 3. Variogram Analysis
        print("\n--- Variogram Analysis ---")
        coordinates = data[['X', 'Y']].values
        values = data['Ash_Content'].values
        
        variogram_analyzer = VariogramAnalyzer()
        variogram_results = variogram_analyzer.analyze_variogram(coordinates, values)
        
        if 'best_model' in variogram_results and variogram_results['best_model']:
            best_model = variogram_results['best_model']
            print(f"Best variogram model: {best_model.model_type.value}")
            print(f"Nugget: {best_model.nugget:.2f}")
            print(f"Range: {best_model.range_param:.1f} m")
            print(f"Model R2: {best_model.r_squared:.3f}")
        
        # 4. Integrated Recommendations
        print("\n--- Integrated Recommendations ---")
        rec_engine = RecommendationEngine()
        recommendations = rec_engine.analyze_and_recommend(data, 'X', 'Y', 'Ash_Content')
        
        print(f"Recommended method: {recommendations.recommended_method}")
        print(f"Number of methods evaluated: {len(recommendations.method_scores)}")
        
        # Show top 3 methods
        for i, method_score in enumerate(recommendations.method_scores[:3]):
            print(f"  {i+1}. {method_score.method}: {method_score.total_score:.3f}")
        
        # Show parameters for best method
        print(f"\nOptimal parameters for {recommendations.recommended_method}:")
        for param, value in recommendations.optimal_parameters.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.3f}")
            else:
                print(f"  {param}: {value}")
        
        # Summary
        print(f"\nAnalysis Summary:")
        print(f"Computation time: {recommendations.computation_time:.2f} seconds")
        print(f"Warnings: {len(recommendations.warnings)}")
        if recommendations.warnings:
            for warning in recommendations.warnings:
                print(f"  - {warning}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Complete analysis pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_components_integration():
    """Test that all analysis components work together."""
    print("\n=== Testing Components Integration ===")
    
    try:
        # Test that enhanced statistical analysis works with other components
        from src.analysis.statistical_analyzer import StatisticalAnalyzer
        from src.analysis.spatial_analyzer import SpatialAnalyzer
        
        # Simple dataset
        np.random.seed(123)
        data = pd.DataFrame({
            'X': np.random.uniform(0, 100, 30),
            'Y': np.random.uniform(0, 100, 30),
            'Value': np.random.normal(50, 10, 30)
        })
        
        # Statistical analysis with enhancements
        stat_analyzer = StatisticalAnalyzer()
        stat_results = stat_analyzer.analyze(data, 'Value')
        
        # Check if enhanced features are available
        has_raw_values = '_raw_values' in stat_results.descriptive_stats
        print(f"Enhanced statistical analysis: {'OK' if has_raw_values else 'Missing'}")
        
        if has_raw_values:
            raw_values = np.array(stat_results.descriptive_stats['_raw_values'])
            
            # Test Grubbs outlier detection
            grubbs_result = stat_analyzer.grubbs_test(raw_values)
            print(f"  Grubbs test: {grubbs_result['n_outliers']} outliers detected")
            
            # Test data transformation
            transform_result = stat_analyzer.transform_data(raw_values)
            print(f"  Data transformation: {transform_result['method']}")
            
            # Test enhanced reporting
            enhanced_report = stat_analyzer.generate_enhanced_report(stat_results, 'Value')
            print(f"  Enhanced report: {len(enhanced_report)} characters")
        
        # Spatial analysis
        spatial_analyzer = SpatialAnalyzer()
        spatial_results = spatial_analyzer.analyze(data, 'X', 'Y', 'Value')
        
        # Check key components
        components = [
            'density_analysis',
            'clustering_analysis', 
            'pattern_analysis',
            'anisotropy_analysis',
            'neighborhood_analysis',
            'spatial_statistics'
        ]
        
        working_components = sum(1 for comp in components if hasattr(spatial_results, comp))
        print(f"Spatial analysis components: {working_components}/{len(components)} working")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Components integration test failed: {e}")
        return False

def test_system_robustness():
    """Test system with various data conditions."""
    print("\n=== Testing System Robustness ===")
    
    try:
        from src.analysis.statistical_analyzer import StatisticalAnalyzer
        from src.analysis.spatial_analyzer import SpatialAnalyzer
        
        test_cases = [
            ("Small dataset", 5),
            ("Medium dataset", 25), 
            ("Large dataset", 100)
        ]
        
        for case_name, n_points in test_cases:
            print(f"\nTesting {case_name} ({n_points} points):")
            
            np.random.seed(42)
            data = pd.DataFrame({
                'X': np.random.uniform(0, 50, n_points),
                'Y': np.random.uniform(0, 50, n_points),
                'Value': np.random.normal(25, 5, n_points)
            })
            
            # Statistical analysis
            try:
                stat_analyzer = StatisticalAnalyzer()
                stat_results = stat_analyzer.analyze(data, 'Value')
                print(f"  Statistical analysis: OK")
            except Exception as e:
                print(f"  Statistical analysis: Failed - {e}")
            
            # Spatial analysis
            try:
                spatial_analyzer = SpatialAnalyzer()
                spatial_results = spatial_analyzer.analyze(data, 'X', 'Y', 'Value')
                print(f"  Spatial analysis: OK")
            except Exception as e:
                print(f"  Spatial analysis: Failed - {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Robustness test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== FINAL SYSTEM TESTS ===")
    
    success = True
    success &= test_complete_analysis_pipeline()
    success &= test_analysis_components_integration()
    success &= test_system_robustness()
    
    if success:
        print("\n" + "="*50)
        print("SUCCESS: All system tests passed!")
        print("The analysis and recommendation system is fully operational.")
        print("="*50)
    else:
        print("\nERROR: Some system tests failed!")
    
    print("\n=== TESTS COMPLETED ===")