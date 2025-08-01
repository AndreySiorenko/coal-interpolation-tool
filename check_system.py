#!/usr/bin/env python3
"""
System check script to diagnose the environment and dependencies.
"""

import sys
import os
import platform
from pathlib import Path

def check_python():
    """Check Python installation."""
    print("🐍 Python Environment:")
    print(f"   Version: {sys.version}")
    print(f"   Executable: {sys.executable}")
    print(f"   Platform: {platform.platform()}")
    print(f"   Architecture: {platform.architecture()}")
    print()

def check_dependencies():
    """Check for required and optional dependencies."""
    print("📦 Dependencies Check:")
    
    required_deps = [
        ('sys', 'Python Standard Library'),
        ('os', 'Python Standard Library'),
        ('pathlib', 'Python Standard Library'),
        ('tkinter', 'GUI Framework (usually included with Python)')
    ]
    
    optional_deps = [
        ('numpy', 'Numerical computing'),
        ('pandas', 'Data manipulation'),
        ('scipy', 'Scientific computing'),
        ('matplotlib', '2D plotting'),
        ('plotly', 'Interactive visualization'),
        ('vtk', '3D visualization'),
        ('rasterio', 'GeoTIFF support'),
        ('ezdxf', 'DXF export')
    ]
    
    print("   Required Dependencies:")
    for dep, description in required_deps:
        try:
            __import__(dep)
            print(f"   ✅ {dep} - {description}")
        except ImportError:
            print(f"   ❌ {dep} - {description} - MISSING")
    
    print("\n   Optional Dependencies:")
    for dep, description in optional_deps:
        try:
            __import__(dep)
            print(f"   ✅ {dep} - {description}")
        except ImportError:
            print(f"   ⚠️  {dep} - {description} - Not installed (will use fallbacks)")
    
    print()

def check_project_structure():
    """Check project file structure."""
    print("📁 Project Structure:")
    
    current_dir = Path(__file__).parent
    
    expected_files = [
        'main.py',
        'demo.py',
        'run_demo.py',
        'requirements.txt',
        'README.md',
        'src/gui/main_window.py',
        'src/core/interpolation/idw.py',
        'src/core/interpolation/rbf.py',
        'src/core/interpolation/kriging.py',
        'src/visualization/plot2d.py',
        'src/visualization/plot3d.py',
        'tests/unit/test_idw_interpolator.py'
    ]
    
    for file_path in expected_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
    
    print()

def check_demo_capability():
    """Check if demo can run."""
    print("🎮 Demo Capability Check:")
    
    try:
        import tkinter as tk
        print("   ✅ Tkinter available - GUI demo can run")
        
        # Test basic tkinter functionality
        root = tk.Tk()
        root.withdraw()  # Hide the window
        root.destroy()
        print("   ✅ Tkinter test successful")
        
        return True
        
    except ImportError:
        print("   ❌ Tkinter not available - GUI demo cannot run")
        return False
    except Exception as e:
        print(f"   ⚠️  Tkinter test failed: {e}")
        return False

def provide_recommendations():
    """Provide recommendations based on the system check."""
    print("💡 Recommendations:")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("   🔄 Upgrade Python to 3.8+ for full compatibility")
    else:
        print("   ✅ Python version is compatible")
    
    # Check tkinter
    try:
        import tkinter
        print("   ✅ GUI components available")
    except ImportError:
        print("   🔧 Install tkinter: sudo apt-get install python3-tk (Linux)")
        print("      or reinstall Python with tkinter support")
    
    # Check optional dependencies
    missing_optional = []
    for dep in ['numpy', 'pandas', 'scipy', 'matplotlib']:
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append(dep)
    
    if missing_optional:
        print(f"   📦 For full functionality, install: pip install {' '.join(missing_optional)}")
    else:
        print("   ✅ All core dependencies available")
    
    print("\n🚀 Quick Start Options:")
    print("   1. Run demo: python demo.py (minimal dependencies)")
    print("   2. Run main app: python main.py (requires all dependencies)")
    print("   3. Install dependencies: pip install -r requirements.txt")
    print("   4. Use Windows batch file: run.bat")
    print()

def main():
    """Main diagnostic function."""
    print("=" * 60)
    print("🔍 COAL INTERPOLATION TOOL - SYSTEM DIAGNOSTICS")
    print("=" * 60)
    print()
    
    check_python()
    check_dependencies()
    check_project_structure()
    demo_capable = check_demo_capability()
    
    print()
    provide_recommendations()
    
    if demo_capable:
        print("✅ System is ready to run the application!")
        print("   Try: python run_demo.py")
    else:
        print("⚠️  System has some limitations, but may still work in limited mode")
    
    print("\n" + "=" * 60)
    return demo_capable

if __name__ == "__main__":
    main()