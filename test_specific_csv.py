#!/usr/bin/env python3
"""
Test loading the specific CSV file that causes errors.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_specific_csv():
    """Test loading the specific CSV file."""
    
    csv_file = Path(__file__).parent / "Скважины устья.csv"
    
    print("🔍 Testing specific CSV file...")
    print(f"File: {csv_file}")
    print(f"Exists: {csv_file.exists()}")
    
    if not csv_file.exists():
        print("❌ File not found!")
        return False
    
    print(f"File size: {csv_file.stat().st_size} bytes")
    
    # Test 1: Basic pandas read
    print("\n📥 Test 1: Basic pandas read_csv...")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Success! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First 3 rows:")
        print(df.head(3))
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 2: With explicit encoding
    print("\n📥 Test 2: With UTF-8 encoding...")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"✅ Success! Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: With cp1251 encoding (Windows Cyrillic)
    print("\n📥 Test 3: With cp1251 encoding...")
    try:
        df = pd.read_csv(csv_file, encoding='cp1251')
        print(f"✅ Success! Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Detect encoding
    print("\n📥 Test 4: Auto-detect encoding...")
    try:
        import chardet
        
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            confidence = result['confidence']
        
        print(f"Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
        
        df = pd.read_csv(csv_file, encoding=detected_encoding)
        print(f"✅ Success with detected encoding! Shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Test using our CSVReader
    print("\n📥 Test 5: Using CSVReader class...")
    try:
        from src.io.readers.csv_reader import CSVReader
        
        reader = CSVReader()
        df = reader.read(str(csv_file))
        print(f"✅ CSVReader success! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ CSVReader error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Manual inspection of first few lines
    print("\n📥 Test 6: Manual file inspection...")
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:5]
            print("First 5 lines of file:")
            for i, line in enumerate(lines):
                print(f"  {i+1}: {repr(line)}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        
        # Try with different encoding
        try:
            with open(csv_file, 'r', encoding='cp1251') as f:
                lines = f.readlines()[:5]
                print("First 5 lines of file (cp1251):")
                for i, line in enumerate(lines):
                    print(f"  {i+1}: {repr(line)}")
        except Exception as e2:
            print(f"❌ Error with cp1251: {e2}")
    
    return True

def create_sample_data():
    """Create a sample CSV file with the same structure."""
    
    sample_file = Path(__file__).parent / "sample_wells.csv"
    
    sample_data = """Dhid,East,North,Elev,TD,Exp_Line,Year
1000,6815.933,66937.941,852.650,71.000,19_s,1958-1961
1001,6473.343,67077.553,841.880,100.100,20_s,1958-1961
1002,3470.251,65592.562,814.790,96.600,30-31_s,1958-1961
"""
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    print(f"\n✅ Created sample file: {sample_file}")
    
    # Test sample file
    try:
        df = pd.read_csv(sample_file)
        print(f"✅ Sample file loads correctly! Shape: {df.shape}")
        return sample_file
    except Exception as e:
        print(f"❌ Sample file error: {e}")
        return None

if __name__ == "__main__":
    print("🧪 CSV File Diagnostic Tool")
    print("=" * 50)
    
    success = test_specific_csv()
    
    print("\n" + "=" * 50)
    print("🏗️ Creating sample data for comparison...")
    
    sample_file = create_sample_data()
    
    print("\n" + "=" * 50)
    print("📝 Summary:")
    print("If the original file has issues, try:")
    print("1. Converting to UTF-8 encoding")
    print("2. Using the sample file for testing")
    print("3. Checking for special characters in data")
    
    if sample_file:
        print(f"\n🎯 You can test with: {sample_file}")