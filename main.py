#!/usr/bin/env python3
"""
Main entry point for the Coal Deposit Interpolation Application.

This is the primary executable file that launches the GUI application.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Check and setup dependencies
def setup_dependencies():
    """Setup dependencies, using mocks if real libraries are unavailable."""
    missing_deps = []
    mock_mode = False
    
    # Check for required external libraries
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import scipy
    except ImportError:
        missing_deps.append('scipy')
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append('matplotlib')
    
    if missing_deps:
        print(f"Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("Running in mock mode with limited functionality...")
        
        # Setup mock environment
        from src.utils.mock_dependencies import setup_mock_environment
        mock_modules = setup_mock_environment()
        
        # Inject mock modules into sys.modules
        for name, mock_module in mock_modules.items():
            sys.modules[name] = mock_module
            if name == 'matplotlib':
                sys.modules['matplotlib.pyplot'] = mock_module.pyplot
                sys.modules['matplotlib.figure'] = type('MockModule', (), {'Figure': mock_module.pyplot.figure})
            elif name == 'scipy':
                sys.modules['scipy.spatial'] = mock_module.spatial
        
        mock_mode = True
    
    return mock_mode

try:
    # Setup dependencies (real or mock)
    mock_mode = setup_dependencies()
    
    from src.gui import MainWindow
    
    def main():
        """Main application entry point."""
        print("Starting Coal Deposit Interpolation Application...")
        print("Version: 1.0.0-rc1")
        if mock_mode:
            print("Mode: DEMO (using mock dependencies)")
        else:
            print("Mode: FULL (all dependencies available)")
        print("-" * 50)
        
        try:
            # Create and run the main application
            app = MainWindow()
            if mock_mode:
                print("Note: Running in demo mode. Some features may be limited.")
            app.run()
            
        except KeyboardInterrupt:
            print("\nApplication interrupted by user.")
            sys.exit(0)
            
        except Exception as e:
            print(f"Application error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
        print("Application closed successfully.")
        
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Critical import error: {e}")
    print("\nFailed to start application. Please check:")
    print("- Python version (3.8+ required)")
    print("- tkinter availability (usually included with Python)")
    print("- Project structure integrity")
    sys.exit(1)