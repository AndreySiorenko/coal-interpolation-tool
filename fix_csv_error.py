#!/usr/bin/env python3
"""
Fix CSV loading error by improving error handling.
"""

from pathlib import Path
import shutil

def fix_application_controller():
    """Fix issues in ApplicationController."""
    
    file_path = Path(__file__).parent / "src" / "gui" / "controllers" / "application_controller.py"
    
    if not file_path.exists():
        print("❌ application_controller.py not found")
        return False
    
    # Create backup
    backup_path = file_path.with_suffix('.py.backup3')
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix potential issues
    fixes = [
        # Fix old fillna syntax
        ("mapped_data.fillna(method='ffill')", "mapped_data.ffill()"),
        ("mapped_data.fillna(method='bfill')", "mapped_data.bfill()"),
        
        # Add proper error handling for CSV reading
        (
            "df = pd.read_csv(",
            "try:\n            df = pd.read_csv("
        ),
        
        # Add better validation
        (
            "self.current_data = mapped_data",
            """# Validate DataFrame before assignment
            if mapped_data is None or len(mapped_data) == 0:
                raise ValueError("No valid data remaining after processing")
            
            self.current_data = mapped_data"""
        )
    ]
    
    # Apply fixes
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ Applied fix: {old[:30]}...")
    
    # Write the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed application_controller.py")
    return True

def add_csv_error_handling():
    """Add better error handling to CSV loading."""
    
    # Create an improved CSV loading wrapper
    wrapper_code = '''
def safe_load_csv_data(self, settings):
    """Safe CSV loading with better error handling."""
    try:
        file_path = settings['file_path']
        
        print(f"Loading CSV file: {file_path}")
        
        # Check if file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load data with custom settings
        import pandas as pd
        
        # Build read_csv parameters safely
        read_params = {
            'filepath_or_buffer': file_path,
            'delimiter': settings.get('delimiter', ','),
            'encoding': settings.get('encoding', 'utf-8'),
        }
        
        # Handle header parameter safely
        header_row = settings.get('header_row', 0)
        if header_row is not None and header_row > 0:
            read_params['header'] = header_row
        else:
            read_params['header'] = 0
        
        print(f"CSV parameters: {read_params}")
        
        # Read CSV
        df = pd.read_csv(**read_params)
        
        print(f"Loaded CSV shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Validate DataFrame
        if df is None:
            raise ValueError("Failed to read CSV file")
        
        if len(df) == 0:
            raise ValueError("CSV file is empty")
        
        # Apply column mapping safely
        mapped_data = pd.DataFrame()
        
        # Map X and Y columns
        x_col = settings.get('x_column')
        y_col = settings.get('y_column')
        
        if x_col not in df.columns:
            raise ValueError(f"X column '{x_col}' not found in data")
        if y_col not in df.columns:
            raise ValueError(f"Y column '{y_col}' not found in data")
        
        mapped_data['X'] = df[x_col]
        mapped_data['Y'] = df[y_col]
        
        # Add value columns
        value_columns = settings.get('value_columns', [])
        for col in value_columns:
            if col in df.columns:
                mapped_data[col] = df[col]
        
        # Apply validation options
        if settings.get('skip_invalid_rows', False):
            original_len = len(mapped_data)
            mapped_data = mapped_data.dropna(subset=['X', 'Y'])
            print(f"Removed {original_len - len(mapped_data)} invalid rows")
            
        if settings.get('remove_duplicates', False):
            original_len = len(mapped_data)
            mapped_data = mapped_data.drop_duplicates(subset=['X', 'Y'])
            print(f"Removed {original_len - len(mapped_data)} duplicate rows")
            
        if settings.get('fill_missing_values', False):
            mapped_data = mapped_data.ffill()
        
        # Final validation
        if len(mapped_data) == 0:
            raise ValueError("No valid data remaining after processing")
        
        print(f"Final data shape: {mapped_data.shape}")
        
        return mapped_data
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise
'''
    
    # Save to a separate file for reference
    wrapper_file = Path(__file__).parent / "csv_loading_fix.py"
    with open(wrapper_file, 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    
    print(f"✅ Created CSV loading fix: {wrapper_file}")
    return True

if __name__ == "__main__":
    print("🔧 Fixing CSV loading errors...")
    
    success1 = fix_application_controller()
    success2 = add_csv_error_handling()
    
    if success1 and success2:
        print("✅ All fixes applied!")
        print("🚀 Try loading CSV files again")
    else:
        print("❌ Some fixes failed")