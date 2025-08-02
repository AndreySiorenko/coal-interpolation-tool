#!/usr/bin/env python3
"""
Test variogram analysis functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

def test_variogram_analysis():
    """Test the variogram analysis functionality."""
    print("=== Testing Variogram Analysis ===")
    
    try:
        from src.core.interpolation.variogram_analysis import VariogramAnalyzer, VariogramAnalysisOptions
        print("OK: VariogramAnalyzer imports successfully")
        
        # Create test data with spatial correlation
        np.random.seed(42)
        
        # Create structured spatial data
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        xx, yy = np.meshgrid(x[::2], y[::2])  # Reduce to 10x10 grid
        coordinates = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Create spatially correlated values using a simple trend
        values = 50 + 0.3 * coordinates[:, 0] + 0.2 * coordinates[:, 1] + np.random.normal(0, 5, len(coordinates))
        
        print(f"OK: Test data created: {len(coordinates)} points")
        
        # Setup analysis options
        options = VariogramAnalysisOptions(
            n_lags=10,
            n_directions=4,
            create_plots=False  # Disable plots for testing
        )
        
        # Initialize analyzer
        analyzer = VariogramAnalyzer(options)
        
        # Test full analysis
        results = analyzer.analyze_variogram(coordinates, values)
        print("OK: Complete variogram analysis completed")
        
        # Check results structure
        expected_keys = ['experimental_variogram', 'directional_variograms', 'anisotropy', 'fitted_models', 'best_model']
        found_keys = [key for key in expected_keys if key in results]
        print(f"OK: Found {len(found_keys)}/{len(expected_keys)} expected result categories")
        
        # Test experimental variogram
        exp_var = results['experimental_variogram']
        print(f"  Experimental variogram: {len(exp_var.distances)} lags")
        print(f"  Max semivariance: {np.max(exp_var.semivariances):.2f}")
        
        # Test directional variograms
        if 'directional_variograms' in results:
            dir_vars = results['directional_variograms']
            print(f"  Directional variograms: {len(dir_vars)} directions")
        
        # Test fitted models
        if 'fitted_models' in results:
            models = results['fitted_models']
            print(f"  Fitted models: {len(models)}")
            for model in models[:3]:  # Show top 3
                print(f"    {model.model_type.value}: R2={model.r_squared:.3f}, RMSE={model.rmse:.2f}")
        
        # Test best model
        if 'best_model' in results:
            best = results['best_model']
            print(f"  Best model: {best.model_type.value}")
            print(f"    Nugget: {best.nugget:.2f}, Sill: {best.sill:.2f}, Range: {best.range_param:.2f}")
        
        # Test anisotropy
        if 'anisotropy' in results:
            aniso = results['anisotropy']
            print(f"  Anisotropy detected: {aniso.is_anisotropic}")
            if aniso.is_anisotropic:
                print(f"    Ratio: {aniso.anisotropy_ratio:.2f}")
                print(f"    Major direction: {aniso.major_direction:.0f}°")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Variogram analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_experimental_variogram():
    """Test experimental variogram calculation."""
    print("\n=== Testing Experimental Variogram ===")
    
    try:
        from src.core.interpolation.variogram_analysis import VariogramAnalyzer
        
        # Create simple linear trend data
        coordinates = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        values = np.array([0, 1, 2, 3, 4])  # Perfect linear trend
        
        analyzer = VariogramAnalyzer()
        exp_var = analyzer.calculate_experimental_variogram(coordinates, values)
        
        print(f"Lag distances: {exp_var.distances}")
        print(f"Semivariances: {exp_var.semivariances}")
        print(f"Number of pairs: {exp_var.n_pairs}")
        
        # For linear trend, semivariance should increase with distance
        if len(exp_var.semivariances) > 1:
            is_increasing = np.all(np.diff(exp_var.semivariances) >= 0)
            print(f"Semivariance increases with distance: {is_increasing}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Experimental variogram test failed: {e}")
        return False

def test_model_fitting():
    """Test variogram model fitting."""
    print("\n=== Testing Model Fitting ===")
    
    try:
        from src.core.interpolation.variogram_analysis import (
            VariogramAnalyzer, VariogramModel, VariogramAnalysisOptions
        )
        
        # Create data with known spatial structure
        np.random.seed(123)
        coordinates = np.random.uniform(0, 50, (30, 2))
        
        # Add spatial correlation using distance-based model
        n_points = len(coordinates)
        values = np.zeros(n_points)
        
        for i in range(n_points):
            # Base value plus spatially correlated component
            spatial_effect = 0
            for j in range(n_points):
                if i != j:
                    dist = np.linalg.norm(coordinates[i] - coordinates[j])
                    spatial_effect += np.exp(-dist/10) * np.random.normal(0, 1)
            values[i] = 100 + spatial_effect/n_points + np.random.normal(0, 2)
        
        # Test different models
        options = VariogramAnalysisOptions(create_plots=False, cross_validate=False)
        analyzer = VariogramAnalyzer(options)
        
        exp_var = analyzer.calculate_experimental_variogram(coordinates, values)
        
        # Test fitting individual models
        models_to_test = [VariogramModel.SPHERICAL, VariogramModel.EXPONENTIAL, VariogramModel.GAUSSIAN]
        
        for model_type in models_to_test:
            try:
                fitted = analyzer.fit_variogram_model(exp_var, model_type)
                print(f"{model_type.value}: R2={fitted.r_squared:.3f}, AIC={fitted.aic:.1f}")
            except Exception as e:
                print(f"{model_type.value}: Failed - {e}")
        
        # Test automatic model selection
        analyzer.experimental_variograms['omnidirectional'] = exp_var
        all_models = analyzer.fit_all_models()
        best_model = analyzer.select_best_model(all_models)
        
        if best_model:
            print(f"Best model: {best_model.model_type.value} (R2={best_model.r_squared:.3f})")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Model fitting test failed: {e}")
        return False

def test_directional_analysis():
    """Test directional variogram analysis."""
    print("\n=== Testing Directional Analysis ===")
    
    try:
        from src.core.interpolation.variogram_analysis import VariogramAnalyzer, VariogramAnalysisOptions
        
        # Create anisotropic data (elongated in X direction)
        np.random.seed(42)
        x = np.random.normal(0, 20, 40)  # Wide spread in X
        y = np.random.normal(0, 5, 40)   # Narrow spread in Y
        coordinates = np.column_stack([x, y])
        
        # Values correlated with position
        values = 0.5 * x + 0.1 * y + np.random.normal(0, 3, 40)
        
        options = VariogramAnalysisOptions(
            n_directions=4,
            detect_anisotropy=True,
            create_plots=False
        )
        
        analyzer = VariogramAnalyzer(options)
        
        # Calculate directional variograms
        directional_vars = analyzer.calculate_directional_variograms(coordinates, values)
        print(f"Calculated {len(directional_vars)} directional variograms")
        
        # Store in analyzer for anisotropy detection
        analyzer.experimental_variograms.update(directional_vars)
        
        # Test anisotropy detection
        aniso_result = analyzer.detect_anisotropy()
        print(f"Anisotropy detected: {aniso_result.is_anisotropic}")
        
        if aniso_result.is_anisotropic:
            print(f"  Anisotropy ratio: {aniso_result.anisotropy_ratio:.2f}")
            print(f"  Major direction: {aniso_result.major_direction:.0f}°")
            print(f"  Minor direction: {aniso_result.minor_direction:.0f}°")
            print(f"  Major range: {aniso_result.major_range:.1f}")
            print(f"  Minor range: {aniso_result.minor_range:.1f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Directional analysis test failed: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions."""
    print("\n=== Testing Convenience Functions ===")
    
    try:
        from src.core.interpolation.variogram_analysis import (
            analyze_spatial_structure, quick_variogram_fit, VariogramModel
        )
        
        # Simple test data
        coordinates = np.random.uniform(0, 10, (20, 2))
        values = np.random.normal(0, 1, 20)
        
        # Test analyze_spatial_structure
        results = analyze_spatial_structure(coordinates, values)
        print(f"Spatial structure analysis completed: {len(results)} result categories")
        
        # Test quick_variogram_fit
        quick_model = quick_variogram_fit(coordinates, values)
        print(f"Quick fit: {quick_model.model_type.value} (R2={quick_model.r_squared:.3f})")
        
        # Test specific model fitting
        spherical_model = quick_variogram_fit(coordinates, values, VariogramModel.SPHERICAL)
        print(f"Spherical fit: R2={spherical_model.r_squared:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Convenience functions test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== VARIOGRAM ANALYSIS TESTS ===")
    
    success = True
    success &= test_variogram_analysis()
    success &= test_experimental_variogram()
    success &= test_model_fitting()
    success &= test_directional_analysis()
    success &= test_convenience_functions()
    
    if success:
        print("\nOK: All variogram analysis tests passed!")
    else:
        print("\nERROR: Some tests failed!")
    
    print("\n=== TESTS COMPLETED ===")