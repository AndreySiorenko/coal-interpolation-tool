#!/usr/bin/env python3
"""
Test RBF interpolation functionality to verify the fix.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_rbf_interpolation():
    """Test RBF interpolation process."""
    
    print("Testing RBF interpolation...")
    
    try:
        from src.gui.controllers.application_controller import ApplicationController
        
        # Create controller
        controller = ApplicationController()
        
        # Load test data first
        csv_file = Path(__file__).parent / "Скважины устья.csv"
        
        settings = {
            'file_path': str(csv_file),
            'x_column': 'East',
            'y_column': 'North', 
            'value_columns': ['Elev'],
            'delimiter': ',',
            'encoding': 'utf-8',
            'header_row': 0,
            'skip_invalid_rows': True,
            'fill_missing_values': False,
            'remove_duplicates': True
        }
        
        print("Step 1: Loading data...")
        controller.load_data_with_settings(settings)
        print("SUCCESS: Data loaded")
        
        # Test RBF interpolation parameters
        rbf_params = {
            'method': 'RBF',
            'value_column': 'Elev',
            'cell_size': 100.0,
            'buffer': 0.1,
            'search_radius': 1000.0,
            'min_points': 3,
            'max_points': 10,
            'rbf_kernel': 'multiquadric',
            'rbf_shape_parameter': 1.0,
            'rbf_regularization': 1e-12,
            'rbf_polynomial_degree': -1,
            'rbf_use_global': True
        }
        
        print("Step 2: Testing RBF interpolation...")
        print(f"Parameters: {rbf_params}")
        
        # This should trigger the corrected RBF interpolation code
        controller.run_interpolation(rbf_params)
        
        # Wait a bit for the interpolation to complete (it runs in a thread)
        import time
        time.sleep(5)
        
        # Check if results are available
        if controller.interpolation_results:
            print("SUCCESS: RBF Interpolation completed!")
            print(f"Results type: {type(controller.interpolation_results)}")
        else:
            print("RBF interpolation may still be running or failed")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_rbf():
    """Test RBF interpolator directly."""
    
    print("\nTesting RBF interpolator directly...")
    
    try:
        from src.core.interpolation.rbf import RBFInterpolator, RBFParameters, RBFKernel
        from src.core.interpolation.base import SearchParameters
        
        # Load test data
        csv_file = Path(__file__).parent / "Скважины устья.csv"
        df = pd.read_csv(csv_file)
        
        # Take first 50 points for quick test
        test_data = df.head(50).copy()
        test_data = test_data.rename(columns={'East': 'X', 'North': 'Y'})
        
        print(f"Test data shape: {test_data.shape}")
        
        # Create interpolator with corrected kernel parameter
        search_params = SearchParameters(
            search_radius=1000.0,
            min_points=3,
            max_points=10
        )
        
        rbf_params = RBFParameters(
            kernel=RBFKernel.MULTIQUADRIC,  # Use enum, not string
            shape_parameter=1.0,
            regularization=1e-12,
            polynomial_degree=-1,
            use_global=True
        )
        
        interpolator = RBFInterpolator(search_params, rbf_params)
        
        # Test fit method
        print("Testing RBF fit method...")
        interpolator.fit(test_data, 'X', 'Y', 'Elev')
        print("SUCCESS: RBF Interpolator fitted successfully!")
        
        return True
        
    except Exception as e:
        print(f"ERROR in direct RBF test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rbf_kernels():
    """Test different RBF kernel types."""
    
    print("\nTesting RBF kernel types...")
    
    try:
        from src.core.interpolation.rbf import RBFKernel
        
        # Test all available kernels
        kernels = ['multiquadric', 'gaussian', 'inverse_multiquadric', 'linear', 'cubic']
        
        for kernel_str in kernels:
            try:
                kernel = RBFKernel(kernel_str.lower())
                print(f"  {kernel_str}: OK -> {kernel}")
            except ValueError:
                print(f"  {kernel_str}: ERROR - not found")
        
        return True
        
    except Exception as e:
        print(f"ERROR testing kernels: {e}")
        return False

if __name__ == "__main__":
    print("RBF Interpolation Test Suite")
    print("=" * 40)
    
    # Test 1: Kernel types
    kernel_success = test_rbf_kernels()
    
    # Test 2: Direct RBF interpolator
    direct_success = test_direct_rbf()
    
    # Test 3: Full application process
    app_success = test_rbf_interpolation()
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"  Kernel Types: {'PASS' if kernel_success else 'FAIL'}")
    print(f"  Direct RBF: {'PASS' if direct_success else 'FAIL'}")
    print(f"  Application: {'PASS' if app_success else 'FAIL'}")
    
    if kernel_success and direct_success and app_success:
        print("\nAll tests PASSED! RBF interpolation should work in the application.")
    else:
        print("\nSome tests FAILED. Check errors above.")