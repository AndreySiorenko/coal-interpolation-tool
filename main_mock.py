#!/usr/bin/env python3
"""
Mock version of the Coal Deposit Interpolation Application.
Runs with mock dependencies for testing GUI functionality.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Setup mock modules BEFORE any other imports
def setup_mock_modules():
    """Setup mock modules in sys.modules before importing application modules."""
    from src.utils.mock_dependencies import setup_mock_environment
    
    mock_modules = setup_mock_environment()
    
    # Inject mock modules into sys.modules
    for name, mock_module in mock_modules.items():
        sys.modules[name] = mock_module
        
        # Setup submodules for matplotlib
        if name == 'matplotlib':
            sys.modules['matplotlib.pyplot'] = mock_module.pyplot
            sys.modules['matplotlib.figure'] = type('MockFigureModule', (), {
                'Figure': lambda *args, **kwargs: mock_module.pyplot.figure()
            })()
            
        # Setup submodules for scipy
        elif name == 'scipy':
            sys.modules['scipy.spatial'] = mock_module.spatial

# Setup mocks before importing GUI
setup_mock_modules()

# Now import GUI components
import tkinter as tk
from tkinter import ttk, messagebox


class MockMainWindow:
    """Simplified main window for testing without full dependencies."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Coal Deposit Interpolation - Mock Mode")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup basic UI for testing."""
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.mock_load_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for panels
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Data panel
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data")
        self.setup_data_panel(data_frame)
        
        # Parameters panel
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")
        self.setup_parameters_panel(params_frame)
        
        # Visualization panel
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Visualization")
        self.setup_visualization_panel(viz_frame)
        
        # Results panel
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        self.setup_results_panel(results_frame)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="Ready - Mock Mode Active")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Mock mode warning
        warning_frame = ttk.Frame(self.root)
        warning_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        warning_label = ttk.Label(
            warning_frame, 
            text="⚠️ Running in Mock Mode - Limited functionality without pandas/numpy/scipy/matplotlib",
            foreground="orange"
        )
        warning_label.pack(padx=5, pady=2)
    
    def setup_data_panel(self, parent):
        """Setup mock data panel."""
        ttk.Label(parent, text="Data Loading", font=("Arial", 12, "bold")).pack(pady=10)
        
        load_btn = ttk.Button(parent, text="Load CSV Data", command=self.mock_load_data)
        load_btn.pack(pady=5)
        
        # Mock data display
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.data_text = tk.Text(text_frame, height=15, width=50)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.data_text.yview)
        self.data_text.config(yscrollcommand=scrollbar.set)
        
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert mock data
        mock_data = """Mock Sample Data:
X       Y       Z
1.0     1.0     10.0
2.0     2.0     15.0
3.0     3.0     20.0
4.0     4.0     25.0
5.0     5.0     30.0
1.5     1.5     12.5
2.5     2.5     17.5
3.5     3.5     22.5
4.5     4.5     27.5

Total points: 9
Data range: X(1.0-5.0), Y(1.0-5.0), Z(10.0-30.0)"""
        
        self.data_text.insert(tk.END, mock_data)
        self.data_text.config(state=tk.DISABLED)
    
    def setup_parameters_panel(self, parent):
        """Setup mock parameters panel."""
        ttk.Label(parent, text="Interpolation Parameters", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Method selection
        method_frame = ttk.Frame(parent)
        method_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        method_var = tk.StringVar(value="IDW")
        method_combo = ttk.Combobox(method_frame, textvariable=method_var, 
                                  values=["IDW", "Kriging", "RBF"], state="readonly")
        method_combo.pack(side=tk.LEFT, padx=10)
        
        # Power parameter
        power_frame = ttk.Frame(parent)
        power_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(power_frame, text="Power:").pack(side=tk.LEFT)
        power_var = tk.DoubleVar(value=2.0)
        power_spin = ttk.Spinbox(power_frame, from_=0.5, to=5.0, increment=0.1, 
                               textvariable=power_var, width=10)
        power_spin.pack(side=tk.LEFT, padx=10)
        
        # Neighbors
        neighbors_frame = ttk.Frame(parent)
        neighbors_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(neighbors_frame, text="Max neighbors:").pack(side=tk.LEFT)
        neighbors_var = tk.IntVar(value=12)
        neighbors_spin = ttk.Spinbox(neighbors_frame, from_=1, to=50, 
                                   textvariable=neighbors_var, width=10)
        neighbors_spin.pack(side=tk.LEFT, padx=10)
        
        # Apply button
        apply_btn = ttk.Button(parent, text="Apply Parameters", command=self.mock_apply_params)
        apply_btn.pack(pady=20)
    
    def setup_visualization_panel(self, parent):
        """Setup mock visualization panel."""
        ttk.Label(parent, text="Visualization", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Visualization type
        viz_frame = ttk.Frame(parent)
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(viz_frame, text="Display:").pack(side=tk.LEFT)
        viz_var = tk.StringVar(value="Points + Contours")
        viz_combo = ttk.Combobox(viz_frame, textvariable=viz_var,
                               values=["Points Only", "Contours Only", "Points + Contours"],
                               state="readonly")
        viz_combo.pack(side=tk.LEFT, padx=10)
        
        # Mock plot area
        plot_frame = ttk.LabelFrame(parent, text="Plot Area")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        mock_plot_text = tk.Text(plot_frame, height=20, bg="lightgray")
        mock_plot_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        plot_info = """🗺️ Mock Visualization Area

In full mode, this would show:
- Interactive matplotlib plots
- Scatter plot of sample points
- Contour maps of interpolation results
- Colorbar and legends
- Zoom/pan tools

Current mock data:
- 9 sample points
- X range: 1.0 - 5.0
- Y range: 1.0 - 5.0
- Z range: 10.0 - 30.0

Note: Install matplotlib, numpy, pandas, scipy
to enable full visualization functionality."""
        
        mock_plot_text.insert(tk.END, plot_info)
        mock_plot_text.config(state=tk.DISABLED)
        
        # Plot controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="Update Plot", command=self.mock_update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export Plot", command=self.mock_export_plot).pack(side=tk.LEFT, padx=5)
    
    def setup_results_panel(self, parent):
        """Setup mock results panel."""
        ttk.Label(parent, text="Interpolation Results", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Results summary
        summary_frame = ttk.LabelFrame(parent, text="Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        summary_text = """Grid size: 50 x 50
Total points: 2500
Method: IDW (power=2.0)
Processing time: ~0.5s (mock)
RMSE: 1.23 (mock)
Max error: 2.45 (mock)"""
        
        ttk.Label(summary_frame, text=summary_text, justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Export options
        export_frame = ttk.LabelFrame(parent, text="Export")
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(export_frame, text="Export CSV", command=self.mock_export_csv).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(export_frame, text="Export Image", command=self.mock_export_image).pack(side=tk.LEFT, padx=5, pady=5)
    
    def mock_load_data(self):
        """Mock data loading."""
        messagebox.showinfo("Mock Mode", "In full mode, this would open a file dialog to load CSV data.\n\nMock data is already loaded for testing.")
        self.status_label.config(text="Mock data loaded - 9 sample points")
    
    def mock_apply_params(self):
        """Mock parameter application."""
        messagebox.showinfo("Mock Mode", "Parameters applied in mock mode.\n\nIn full mode, this would update the interpolation with new parameters.")
        self.status_label.config(text="Parameters applied")
    
    def mock_update_plot(self):
        """Mock plot update."""
        messagebox.showinfo("Mock Mode", "In full mode, this would update the matplotlib visualization with current data and parameters.")
        self.status_label.config(text="Plot updated (mock)")
    
    def mock_export_plot(self):
        """Mock plot export."""
        messagebox.showinfo("Mock Mode", "In full mode, this would export the current plot as PNG/PDF/SVG.")
        self.status_label.config(text="Plot exported (mock)")
    
    def mock_export_csv(self):
        """Mock CSV export."""
        messagebox.showinfo("Mock Mode", "In full mode, this would export interpolation results as CSV file.")
        self.status_label.config(text="Results exported (mock)")
    
    def mock_export_image(self):
        """Mock image export."""
        messagebox.showinfo("Mock Mode", "In full mode, this would export the visualization as an image file.")
        self.status_label.config(text="Image exported (mock)")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Coal Deposit Interpolation Application
Version: 0.7.0 (Mock Mode)

A tool for geological data interpolation using:
- IDW (Inverse Distance Weighting)
- Kriging methods
- RBF (Radial Basis Functions)

Current Mode: MOCK (for testing)
Install pandas, numpy, scipy, matplotlib for full functionality.

© 2024 Coal Interpolation Project"""
        
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point for mock application."""
    print("Starting Coal Deposit Interpolation Application...")
    print("Version: 0.7.0")
    print("Mode: MOCK (testing without dependencies)")
    print("-" * 50)
    
    try:
        app = MockMainWindow()
        print("Mock GUI loaded successfully!")
        print("Note: This is a limited demo version for testing the interface.")
        print("Install pandas, numpy, scipy, matplotlib for full functionality.")
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