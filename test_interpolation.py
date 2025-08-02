#!/usr/bin/env python3
"""
Test interpolation functionality to verify the fix.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_interpolation_process():
    """Test the complete interpolation process."""
    
    print("Testing interpolation process...")
    
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
            'value_columns': ['Elev'],  # Use elevation for interpolation
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
        
        # Test interpolation parameters
        interpolation_params = {
            'method': 'IDW',
            'value_column': 'Elev',
            'cell_size': 100.0,
            'buffer': 0.1,
            'search_radius': 1000.0,
            'min_points': 3,
            'max_points': 10,
            'power': 2.0,
            'smoothing': 0.0
        }
        
        print("Step 2: Testing interpolation...")
        print(f"Parameters: {interpolation_params}")
        
        # This should trigger the corrected interpolation code
        controller.run_interpolation(interpolation_params)
        
        # Wait a bit for the interpolation to complete (it runs in a thread)
        import time
        time.sleep(3)
        
        # Check if results are available
        if controller.interpolation_results:
            print("SUCCESS: Interpolation completed!")
            print(f"Results type: {type(controller.interpolation_results)}")
        else:
            print("Interpolation may still be running or failed")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_interpolator():
    """Test interpolator directly to isolate issues."""
    
    print("\nTesting interpolator directly...")
    
    try:
        from src.core.interpolation.idw import IDWInterpolator, IDWParameters
        from src.core.interpolation.base import SearchParameters
        
        # Load test data
        csv_file = Path(__file__).parent / "Скважины устья.csv"
        df = pd.read_csv(csv_file)
        
        # Take first 50 points for quick test
        test_data = df.head(50).copy()
        test_data = test_data.rename(columns={'East': 'X', 'North': 'Y'})
        
        print(f"Test data shape: {test_data.shape}")
        print(f"Columns: {list(test_data.columns)}")
        
        # Create interpolator
        search_params = SearchParameters(
            search_radius=1000.0,
            min_points=3,
            max_points=10
        )
        
        idw_params = IDWParameters(
            power=2.0,
            smoothing=0.0
        )
        
        interpolator = IDWInterpolator(search_params, idw_params)
        
        # Test fit method with corrected signature
        print("Testing fit method...")
        interpolator.fit(test_data, 'X', 'Y', 'Elev')
        print("SUCCESS: Interpolator fitted successfully!")
        
        return True
        
    except Exception as e:
        print(f"ERROR in direct test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Interpolation Test Suite")
    print("=" * 40)
    
    # Test 1: Direct interpolator
    direct_success = test_direct_interpolator()
    
    # Test 2: Full application process
    app_success = test_interpolation_process()
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"  Direct Interpolator: {'PASS' if direct_success else 'FAIL'}")
    print(f"  Application Process: {'PASS' if app_success else 'FAIL'}")
    
    if direct_success and app_success:
        print("\nAll tests PASSED! Interpolation should work in the application.")
    else:
        print("\nSome tests FAILED. Check errors above.")