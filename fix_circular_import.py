#!/usr/bin/env python3
"""
Fix circular import issue in the interpolation modules.
"""

import os
import shutil
from pathlib import Path

def fix_circular_import():
    """Fix the circular import between kriging.py and variogram_analysis.py"""
    
    print("🔧 Fixing circular import issue...")
    
    # Get the path to variogram_analysis.py
    base_path = Path(__file__).parent
    variogram_path = base_path / "src" / "core" / "interpolation" / "variogram_analysis.py"
    
    if not variogram_path.exists():
        print("❌ variogram_analysis.py not found!")
        return False
    
    # Backup the original file
    backup_path = variogram_path.with_suffix('.py.backup')
    shutil.copy2(variogram_path, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # Read the file
    with open(variogram_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the circular import - remove the import from kriging
    # Replace the problematic line
    old_line = "from .kriging import VariogramModel, VariogramModels"
    new_lines = """# Moved to avoid circular import - these are defined below
# from .kriging import VariogramModel, VariogramModels

# Local copy of VariogramModel to avoid circular import
from enum import Enum

class VariogramModel(Enum):
    \"\"\"Available variogram model types.\"\"\"
    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"
    POWER = "power"
    NUGGET = "nugget"

# VariogramModels will be imported where needed"""
    
    # Replace the import
    if old_line in content:
        content = content.replace(old_line, new_lines)
        print("✅ Fixed circular import")
    else:
        print("⚠️  Import line not found, trying alternative fix...")
        # Alternative: comment out the line if it exists in a different format
        content = content.replace("from .kriging import", "# from .kriging import")
    
    # Write the fixed content
    with open(variogram_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ File updated successfully")
    print("📝 You can restore the original with the .backup file if needed")
    
    return True

if __name__ == "__main__":
    success = fix_circular_import()
    if success:
        print("\n✅ Circular import fixed!")
        print("🚀 Now try running the program again")
    else:
        print("\n❌ Failed to fix circular import")
        print("Please check the file paths and structure")