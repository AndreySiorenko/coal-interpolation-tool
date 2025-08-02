#!/usr/bin/env python3
"""
Debug script to locate DataFrame truth value error.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_data_loading():
    """Test the data loading process step by step."""
    
    print("Testing data loading process...")
    
    # Test 1: Basic CSV reading
    csv_file = Path(__file__).parent / "Скважины устья.csv"
    print(f"\nStep 1: Reading CSV file: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"SUCCESS: CSV loaded with shape {df.shape}")
    except Exception as e:
        print(f"ERROR in CSV reading: {e}")
        return False
    
    # Test 2: Data processing
    print("\nStep 2: Testing data processing...")
    try:
        # Apply column mapping like in the application
        mapped_data = pd.DataFrame()
        mapped_data['X'] = df['East']  # Use East as X
        mapped_data['Y'] = df['North']  # Use North as Y
        mapped_data['Dhid'] = df['Dhid']
        mapped_data['Elev'] = df['Elev']
        
        print(f"SUCCESS: Data mapping completed with shape {mapped_data.shape}")
    except Exception as e:
        print(f"ERROR in data processing: {e}")
        return False
    
    # Test 3: Testing logic that might cause the error
    print("\nStep 3: Testing potentially problematic logic...")
    
    try:
        # Test various DataFrame checks
        print(f"  mapped_data is None: {mapped_data is None}")
        print(f"  mapped_data.empty: {mapped_data.empty}")
        print(f"  len(mapped_data): {len(mapped_data)}")
        
        # Test that would cause error
        # result = mapped_data and True  # This would fail
        
        # Test validation like in the app
        if mapped_data is not None and not mapped_data.empty:
            print("  Validation check passed")
        
        # Test dropna
        cleaned_data = mapped_data.dropna(subset=['X', 'Y'])
        print(f"  After dropna: {cleaned_data.shape}")
        
        # Test duplicates
        no_dups = cleaned_data.drop_duplicates(subset=['X', 'Y'])
        print(f"  After removing duplicates: {no_dups.shape}")
        
    except Exception as e:
        print(f"ERROR in logic testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test the actual application controller
    print("\nStep 4: Testing application controller...")
    try:
        from src.gui.controllers.application_controller import ApplicationController
        
        controller = ApplicationController()
        
        # Test the load_data_with_settings method
        settings = {
            'file_path': str(csv_file),
            'x_column': 'East',
            'y_column': 'North',
            'value_columns': ['Dhid', 'Elev'],
            'delimiter': ',',
            'encoding': 'utf-8',
            'header_row': 0,
            'skip_invalid_rows': True,
            'fill_missing_values': False,
            'remove_duplicates': True
        }
        
        print("  Attempting to load data with application controller...")
        controller.load_data_with_settings(settings)
        print("  SUCCESS: Application controller loaded data")
        
    except Exception as e:
        print(f"ERROR in application controller: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nAll tests passed!")
    return True

if __name__ == "__main__":
    print("DataFrame Truth Value Error Debug Tool")
    print("=" * 50)
    
    success = test_data_loading()
    
    if not success:
        print("\nDebugging completed - found error location above")
    else:
        print("\nNo errors found in data loading process")