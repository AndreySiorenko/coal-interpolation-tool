#!/usr/bin/env python3
"""
Test file loading functionality to check if the DataFrame error is fixed.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_application_loading():
    """Test the complete application file loading process."""
    
    print("Testing application file loading...")
    
    try:
        from src.gui.controllers.application_controller import ApplicationController
        
        # Create controller
        controller = ApplicationController()
        
        # Simulate the exact settings from the data loader dialog
        csv_file = Path(__file__).parent / "Скважины устья.csv"
        
        settings = {
            'file_path': str(csv_file),
            'x_column': 'East',
            'y_column': 'North', 
            'value_columns': ['Dhid', 'Elev', 'TD'],
            'delimiter': ',',
            'encoding': 'utf-8',
            'header_row': 0,
            'skip_invalid_rows': True,
            'fill_missing_values': False,
            'remove_duplicates': True
        }
        
        print(f"Loading file: {csv_file}")
        print(f"Settings: {settings}")
        
        # This should trigger the same code path as the GUI
        controller.load_data_with_settings(settings)
        
        print("SUCCESS: File loaded successfully!")
        
        # Get data info
        data_info = controller.get_data_info()
        if data_info:
            print(f"Data info: {data_info}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_reader_directly():
    """Test CSV reader directly to isolate any issues."""
    
    print("\nTesting CSV reader directly...")
    
    try:
        from src.io.readers.csv_reader import CSVReader
        
        csv_file = Path(__file__).parent / "Скважины устья.csv"
        reader = CSVReader()
        
        # Test reading
        df = reader.read(str(csv_file))
        print(f"CSV Reader SUCCESS: {df.shape}")
        
        # Test validation
        validation = reader.validate_coordinates('East', 'North')
        print(f"Validation result: {validation['valid']}")
        
        return True
        
    except Exception as e:
        print(f"CSV Reader ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("File Loading Test Suite")
    print("=" * 40)
    
    # Test 1: CSV Reader
    csv_success = test_csv_reader_directly()
    
    # Test 2: Application Controller
    app_success = test_application_loading()
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"  CSV Reader: {'PASS' if csv_success else 'FAIL'}")
    print(f"  Application: {'PASS' if app_success else 'FAIL'}")
    
    if csv_success and app_success:
        print("\nAll tests PASSED! The DataFrame error appears to be fixed.")
    else:
        print("\nSome tests FAILED. Further investigation needed.")