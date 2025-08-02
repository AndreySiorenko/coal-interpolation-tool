#!/usr/bin/env python3
"""
Debug CSV loading issue.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_csv_loading():
    """Test CSV loading with better error handling."""
    
    print("🔍 Testing CSV loading...")
    
    try:
        from src.io.readers.csv_reader import CSVReader
        
        # Look for the test file
        test_file = Path(__file__).parent / "Скважины устья.csv"
        
        if test_file.exists():
            print(f"✅ Found test file: {test_file}")
            
            reader = CSVReader()
            
            print("📥 Attempting to load CSV...")
            data = reader.read(str(test_file))
            
            print(f"✅ Successfully loaded CSV!")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   First 3 rows:")
            print(data.head(3))
            
        else:
            print(f"❌ Test file not found: {test_file}")
            
            # List available CSV files
            csv_files = list(Path(__file__).parent.glob("*.csv"))
            if csv_files:
                print("Available CSV files:")
                for f in csv_files:
                    print(f"  - {f.name}")
            else:
                print("No CSV files found in directory")
    
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        import traceback
        traceback.print_exc()

def create_test_csv():
    """Create a simple test CSV file."""
    
    test_file = Path(__file__).parent / "test_data.csv"
    
    content = """X,Y,Z,Ash_Content
1.0,1.0,10.0,15.5
2.0,2.0,12.0,18.2
3.0,3.0,8.0,12.8
4.0,4.0,15.0,22.1
5.0,5.0,11.0,16.9
"""
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Created test CSV: {test_file}")
    return test_file

if __name__ == "__main__":
    print("🧪 CSV Loading Debug Tool")
    print("=" * 40)
    
    # First try existing file
    test_csv_loading()
    
    print("\n" + "=" * 40)
    print("🏗️ Creating test CSV...")
    
    # Create and test simple file
    test_file = create_test_csv()
    
    print("📥 Testing with created file...")
    try:
        from src.io.readers.csv_reader import CSVReader
        reader = CSVReader()
        data = reader.read(str(test_file))
        print(f"✅ Test successful! Shape: {data.shape}")
        print(data.head())
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()