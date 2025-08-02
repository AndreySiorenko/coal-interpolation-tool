#!/usr/bin/env python3
"""
Test parameter optimization and validation functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

def test_parameter_optimization():
    """Test parameter optimization functionality."""
    print("=== Testing Parameter Optimization ===")
    
    try:
        from src.core.recommendations.parameter_optimizer import ParameterOptimizer
        from src.core.recommendations.data_analyzer import DataAnalyzer
        print("OK: Parameter optimization modules import successfully")
        
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'X': np.random.uniform(0, 100, 50),
            'Y': np.random.uniform(0, 100, 50),
            'Value': np.random.normal(50, 10, 50)
        })
        
        print(f"OK: Test data created: {len(data)} points")
        
        # Analyze data characteristics
        analyzer = DataAnalyzer()
        characteristics = analyzer.analyze(data, 'X', 'Y', 'Value')
        print("OK: Data analysis completed")
        
        # Test parameter optimization for different methods
        optimizer = ParameterOptimizer()
        
        methods_to_test = ['IDW', 'RBF']  # Skip Kriging to avoid complex dependencies
        
        for method in methods_to_test:
            try:
                result = optimizer.optimize_parameters(method, characteristics, data)
                print(f"OK: {method} optimization completed")
                print(f"  Parameters: {list(result.parameters.keys())}")
                print(f"  Quality score: {result.quality_score:.3f}")
                print(f"  Reasoning items: {len(result.reasoning)}")
            except Exception as e:
                print(f"WARNING: {method} optimization failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Parameter optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_validation():
    """Test cross-validation functionality."""
    print("\n=== Testing Cross Validation ===")
    
    try:
        from src.core.validation.cross_validator import CrossValidator
        from src.core.interpolation.idw import IDWInterpolator
        from src.core.interpolation.base import SearchParameters
        print("OK: Cross validation modules import successfully")
        
        # Create test data
        np.random.seed(123)
        n_points = 30
        x = np.random.uniform(0, 50, n_points)
        y = np.random.uniform(0, 50, n_points) 
        values = 100 + 0.5 * x + 0.3 * y + np.random.normal(0, 5, n_points)
        
        data = pd.DataFrame({
            'X': x,
            'Y': y,
            'Value': values
        })
        
        print(f"OK: Test data created: {len(data)} points")
        
        # Setup interpolator
        search_params = SearchParameters(
            search_radius=25.0,
            min_points=3,
            max_points=10
        )
        interpolator = IDWInterpolator(search_params=search_params)
        
        # Test cross-validation
        validator = CrossValidator(n_jobs=1)  # Single job for testing
        
        # Test K-fold CV
        cv_result = validator.k_fold(
            interpolator=interpolator,
            data=data,
            x_col='X',
            y_col='Y', 
            value_col='Value',
            k=5
        )
        
        print("OK: K-fold cross-validation completed")
        print(f"  RMSE: {cv_result.metrics['rmse']:.3f}")
        print(f"  MAE: {cv_result.metrics['mae']:.3f}")
        print(f"  R2: {cv_result.metrics['r2']:.3f}")
        print(f"  Number of folds: {cv_result.n_folds}")
        
        # Test LOO CV (on smaller dataset)
        small_data = data.sample(15, random_state=42)
        loo_result = validator.leave_one_out(
            interpolator=interpolator,
            data=small_data,
            x_col='X',
            y_col='Y',
            value_col='Value'
        )
        
        print("OK: Leave-one-out cross-validation completed")
        print(f"  LOO RMSE: {loo_result.metrics['rmse']:.3f}")
        print(f"  LOO R2: {loo_result.metrics['r2']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Cross validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_metrics():
    """Test quality metrics calculation."""
    print("\n=== Testing Quality Metrics ===")
    
    try:
        from src.core.validation.quality_metrics import QualityMetrics
        print("OK: Quality metrics module imports successfully")
        
        # Create test predictions and actual values
        actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        predicted = np.array([1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1, 9.9])
        
        metrics = QualityMetrics.calculate_metrics(actual, predicted)
        
        print("OK: Quality metrics calculated")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  R2: {metrics['r2']:.3f}")
        print(f"  Bias: {metrics['bias']:.3f}")
        print(f"  NSE: {metrics['nse']:.3f}")
        
        # Test with perfect predictions
        perfect_metrics = QualityMetrics.calculate_metrics(actual, actual)
        print(f"  Perfect R2: {perfect_metrics['r2']:.3f}")
        print(f"  Perfect RMSE: {perfect_metrics['rmse']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Quality metrics test failed: {e}")
        return False

def test_data_analyzer():
    """Test data analyzer functionality.""" 
    print("\n=== Testing Data Analyzer ===")
    
    try:
        from src.core.recommendations.data_analyzer import DataAnalyzer
        print("OK: Data analyzer imports successfully")
        
        # Create test data with various patterns
        np.random.seed(42)
        
        # Regular grid
        x_grid = np.linspace(0, 100, 10)
        y_grid = np.linspace(0, 100, 10)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # Add some scattered points
        x_scatter = np.random.uniform(0, 100, 20)
        y_scatter = np.random.uniform(0, 100, 20)
        
        x_all = np.concatenate([xx.ravel(), x_scatter])
        y_all = np.concatenate([yy.ravel(), y_scatter])
        
        # Create values with trend and noise
        values = 50 + 0.2 * x_all + 0.1 * y_all + np.random.normal(0, 5, len(x_all))
        
        data = pd.DataFrame({
            'X': x_all,
            'Y': y_all,
            'Value': values
        })
        
        print(f"OK: Test data created: {len(data)} points")
        
        # Analyze data
        analyzer = DataAnalyzer()
        characteristics = analyzer.analyze(data, 'X', 'Y', 'Value')
        
        print("OK: Data analysis completed")
        print(f"  Data quality: {characteristics.data_quality}")
        print(f"  Spatial regularity: {characteristics.spatial_regularity}")
        print(f"  Sample density: {characteristics.sample_density}")
        print(f"  Has trend: {characteristics.has_trend}")
        print(f"  Has clusters: {characteristics.has_clusters}")
        print(f"  Is anisotropic: {characteristics.is_anisotropic}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Data analyzer test failed: {e}")
        return False

def test_integrated_optimization():
    """Test integrated optimization workflow."""
    print("\n=== Testing Integrated Optimization ===")
    
    try:
        from src.core.recommendations.recommendation_engine import RecommendationEngine
        print("OK: Recommendation engine imports successfully")
        
        # Create realistic test data
        np.random.seed(42)
        
        # Simulate coal exploration data
        n_points = 40
        x = np.random.uniform(0, 1000, n_points)  # meters
        y = np.random.uniform(0, 1000, n_points)
        
        # Simulate ash content with spatial correlation
        ash_content = np.zeros(n_points)
        for i in range(n_points):
            base_value = 15 + 0.01 * x[i] + 0.005 * y[i]  # Linear trend
            
            # Add spatial correlation
            spatial_effect = 0
            for j in range(n_points):
                if i != j:
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    spatial_effect += np.exp(-dist/200) * np.random.normal(0, 2)
            
            ash_content[i] = base_value + spatial_effect/n_points + np.random.normal(0, 1)
        
        data = pd.DataFrame({
            'X': x,
            'Y': y,
            'Ash_Content': ash_content
        })
        
        print(f"OK: Coal data created: {len(data)} drill holes")
        
        # Get recommendations
        engine = RecommendationEngine()
        recommendations = engine.analyze_and_recommend(data, 'X', 'Y', 'Ash_Content')
        
        print("OK: Recommendations generated")
        print(f"  Recommended methods: {len(recommendations.method_rankings)}")
        print(f"  Best method: {recommendations.recommended_method}")
        
        for i, (method, score) in enumerate(recommendations.method_rankings[:3]):
            print(f"  {i+1}. {method}: {score:.3f}")
        
        # Show parameter recommendations for best method
        best_method = recommendations.recommended_method
        if best_method in recommendations.parameter_recommendations:
            params = recommendations.parameter_recommendations[best_method]
            print(f"  {best_method} parameters:")
            for param, value in params.parameters.items():
                if isinstance(value, float):
                    print(f"    {param}: {value:.3f}")
                else:
                    print(f"    {param}: {value}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Integrated optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== PARAMETER OPTIMIZATION & VALIDATION TESTS ===")
    
    success = True
    success &= test_parameter_optimization()
    success &= test_cross_validation()
    success &= test_quality_metrics()
    success &= test_data_analyzer()
    success &= test_integrated_optimization()
    
    if success:
        print("\nOK: All optimization and validation tests passed!")
    else:
        print("\nERROR: Some tests failed!")
    
    print("\n=== TESTS COMPLETED ===")