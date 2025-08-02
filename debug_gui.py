#!/usr/bin/env python3
"""
Debug GUI to test the data loader dialog specifically.
"""

import sys
import tkinter as tk
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_data_loader_dialog():
    """Test the data loader dialog that's causing the error."""
    
    try:
        from src.gui.controllers.application_controller import ApplicationController
        from src.gui.dialogs.data_loader_dialog import show_data_loader_dialog
        
        # Create a simple root window
        root = tk.Tk()
        root.title("Debug Data Loader")
        root.geometry("300x200")
        
        # Create controller
        controller = ApplicationController()
        
        def test_dialog():
            """Test function to open the data loader dialog."""
            try:
                print("Opening data loader dialog...")
                result = show_data_loader_dialog(root, controller)
                print(f"Dialog result: {result}")
            except Exception as e:
                print(f"ERROR in dialog: {e}")
                import traceback
                traceback.print_exc()
        
        # Add a button to test the dialog
        test_button = tk.Button(
            root,
            text="Test Data Loader Dialog",
            command=test_dialog,
            width=20,
            height=2
        )
        test_button.pack(expand=True)
        
        print("Debug GUI created. Click the button to test the data loader dialog.")
        print("The error should occur when you try to load a file.")
        
        root.mainloop()
        
    except Exception as e:
        print(f"ERROR creating debug GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Data Loader Dialog Debug Tool")
    print("=" * 40)
    
    test_data_loader_dialog()