#!/usr/bin/env python3
"""
Simple test for CSV file loading without emoji characters.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_csv_file():
    """Test loading the specific CSV file."""
    
    csv_file = Path(__file__).parent / "Скважины устья.csv"
    
    print("Testing CSV file...")
    print(f"File: {csv_file}")
    print(f"Exists: {csv_file.exists()}")
    
    if not csv_file.exists():
        print("ERROR: File not found!")
        return False
    
    print(f"File size: {csv_file.stat().st_size} bytes")
    
    # Test 1: Basic pandas read
    print("\nTest 1: Basic pandas read_csv...")
    try:
        df = pd.read_csv(csv_file)
        print(f"SUCCESS! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("First 3 rows:")
        print(df.head(3))
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Error type: {type(e)}")
    
    # Test 2: With explicit encoding
    print("\nTest 2: With UTF-8 encoding...")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"SUCCESS! Shape: {df.shape}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test 3: With cp1251 encoding (Windows Cyrillic)
    print("\nTest 3: With cp1251 encoding...")
    try:
        df = pd.read_csv(csv_file, encoding='cp1251')
        print(f"SUCCESS! Shape: {df.shape}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test 4: Detect encoding
    print("\nTest 4: Auto-detect encoding...")
    try:
        import chardet
        
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            confidence = result['confidence']
        
        print(f"Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
        
        df = pd.read_csv(csv_file, encoding=detected_encoding)
        print(f"SUCCESS with detected encoding! Shape: {df.shape}")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test 5: Manual inspection of first few lines
    print("\nTest 5: Manual file inspection...")
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:5]
            print("First 5 lines of file:")
            for i, line in enumerate(lines):
                print(f"  {i+1}: {repr(line)}")
        return True
    except Exception as e:
        print(f"ERROR reading file: {e}")
        
        # Try with different encoding
        try:
            with open(csv_file, 'r', encoding='cp1251') as f:
                lines = f.readlines()[:5]
                print("First 5 lines of file (cp1251):")
                for i, line in enumerate(lines):
                    print(f"  {i+1}: {repr(line)}")
            return True
        except Exception as e2:
            print(f"ERROR with cp1251: {e2}")
    
    return False

if __name__ == "__main__":
    print("CSV File Diagnostic Tool")
    print("=" * 40)
    
    success = test_csv_file()
    
    print("\n" + "=" * 40)
    if not success:
        print("All tests failed. The file may have encoding or format issues.")
        print("Try using the wells_data_fixed.csv file instead.")
    else:
        print("At least one test succeeded.")