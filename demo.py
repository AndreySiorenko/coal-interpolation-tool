#!/usr/bin/env python3
"""
Demo version of the Coal Deposit Interpolation Application.
Standalone version that doesn't require external dependencies.
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox


class DemoApp:
    """Standalone demo application for interface testing."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Coal Deposit Interpolation - Demo Version")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Demo data
        self.sample_data = [
            (1.0, 1.0, 10.0), (2.0, 2.0, 15.0), (3.0, 3.0, 20.0),
            (4.0, 4.0, 25.0), (5.0, 5.0, 30.0), (1.5, 1.5, 12.5),
            (2.5, 2.5, 17.5), (3.5, 3.5, 22.5), (4.5, 4.5, 27.5)
        ]
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load CSV Data...", command=self.load_data)
        file_menu.add_command(label="Load Excel Data...", command=self.load_excel)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_command(label="Export Plot...", command=self.export_plot)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self.reset_view)
        view_menu.add_command(label="Full Screen", command=self.toggle_fullscreen)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Data Statistics", command=self.show_statistics)
        tools_menu.add_command(label="Interpolation Settings", command=self.show_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Create toolbar
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(toolbar_frame, text="Load", command=self.load_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Save", command=self.export_results).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(toolbar_frame, text="Refresh", command=self.refresh_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Settings", command=self.show_settings).pack(side=tk.LEFT, padx=2)
        
        # Create main container with paned windows
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # Left panel - Data and Parameters
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Visualization and Results  
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Ready - Demo Mode")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, length=200, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Demo mode warning
        warning_frame = ttk.Frame(self.root, style="Warning.TFrame")
        warning_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        warning_text = "WARNING: DEMO MODE - Install pandas, numpy, scipy, matplotlib for full functionality"
        ttk.Label(warning_frame, text=warning_text, foreground="orange", background="lightyellow").pack(pady=2)
    
    def setup_left_panel(self, parent):
        """Setup the left panel with data and parameters."""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Data tab
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data")
        
        # Data loading section
        data_load_frame = ttk.LabelFrame(data_frame, text="Data Loading")
        data_load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(data_load_frame, text="Load CSV File", command=self.load_data).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(data_load_frame, text="Load Excel File", command=self.load_excel).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Data preview
        preview_frame = ttk.LabelFrame(data_frame, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for data display
        columns = ("X", "Y", "Z")
        self.data_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=80, anchor=tk.CENTER)
        
        # Add sample data
        for i, (x, y, z) in enumerate(self.sample_data):
            self.data_tree.insert("", tk.END, values=(f"{x:.1f}", f"{y:.1f}", f"{z:.1f}"))
        
        # Scrollbars for treeview
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.data_tree.pack(in_=tree_frame, side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=v_scrollbar.set)
        
        # Data statistics
        stats_frame = ttk.LabelFrame(data_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        stats_text = f"""Points: {len(self.sample_data)}
X range: {min(x for x,y,z in self.sample_data):.1f} - {max(x for x,y,z in self.sample_data):.1f}
Y range: {min(y for x,y,z in self.sample_data):.1f} - {max(y for x,y,z in self.sample_data):.1f}
Z range: {min(z for x,y,z in self.sample_data):.1f} - {max(z for x,y,z in self.sample_data):.1f}"""
        
        ttk.Label(stats_frame, text=stats_text, font=("Consolas", 9)).pack(padx=10, pady=5)
        
        # Parameters tab
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")
        
        # Interpolation method
        method_frame = ttk.LabelFrame(params_frame, text="Interpolation Method")
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.method_var = tk.StringVar(value="IDW")
        methods = [("IDW (Inverse Distance Weighting)", "IDW"), 
                  ("Ordinary Kriging", "Kriging"), 
                  ("RBF (Radial Basis Functions)", "RBF")]
        
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var, 
                          value=value, command=self.on_method_change).pack(anchor=tk.W, padx=10, pady=2)
        
        # IDW parameters
        self.idw_frame = ttk.LabelFrame(params_frame, text="IDW Parameters")
        self.idw_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Power parameter
        power_frame = ttk.Frame(self.idw_frame)
        power_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(power_frame, text="Power:").pack(side=tk.LEFT)
        self.power_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(power_frame, from_=0.5, to=5.0, increment=0.1, 
                   textvariable=self.power_var, width=8).pack(side=tk.RIGHT)
        
        # Max neighbors
        neighbors_frame = ttk.Frame(self.idw_frame)
        neighbors_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(neighbors_frame, text="Max neighbors:").pack(side=tk.LEFT)
        self.neighbors_var = tk.IntVar(value=12)
        ttk.Spinbox(neighbors_frame, from_=1, to=50, 
                   textvariable=self.neighbors_var, width=8).pack(side=tk.RIGHT)
        
        # Search radius
        radius_frame = ttk.Frame(self.idw_frame)
        radius_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(radius_frame, text="Search radius:").pack(side=tk.LEFT)
        self.radius_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(radius_frame, from_=1.0, to=100.0, increment=1.0, 
                   textvariable=self.radius_var, width=8).pack(side=tk.RIGHT)
        
        # Grid parameters
        grid_frame = ttk.LabelFrame(params_frame, text="Output Grid")
        grid_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Grid size
        size_frame = ttk.Frame(grid_frame)
        size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(size_frame, text="Grid size:").pack(side=tk.LEFT)
        self.grid_size_var = tk.IntVar(value=50)
        ttk.Spinbox(size_frame, from_=10, to=200, increment=10, 
                   textvariable=self.grid_size_var, width=8).pack(side=tk.RIGHT)
        
        # Apply button
        ttk.Button(params_frame, text="Apply Parameters", 
                  command=self.apply_parameters).pack(pady=10)
    
    def setup_right_panel(self, parent):
        """Setup the right panel with visualization and results."""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Visualization tab
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Visualization")
        
        # Visualization controls
        controls_frame = ttk.Frame(viz_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Display:").pack(side=tk.LEFT)
        self.display_var = tk.StringVar(value="Both")
        display_combo = ttk.Combobox(controls_frame, textvariable=self.display_var,
                                   values=["Points Only", "Contours Only", "Both"], 
                                   state="readonly", width=12)
        display_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="Update", command=self.update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export", command=self.export_plot).pack(side=tk.LEFT, padx=5)
        
        # Mock plot area
        plot_frame = ttk.LabelFrame(viz_frame, text="Interpolation Plot")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for mock plot
        self.plot_canvas = tk.Canvas(plot_frame, bg="white", relief=tk.SUNKEN, bd=2)
        self.plot_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.draw_mock_plot()
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        
        # Interpolation summary
        summary_frame = ttk.LabelFrame(results_frame, text="Interpolation Summary")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        summary_text = """Method: IDW (Power = 2.0)
Grid size: 50 × 50 (2500 points)
Processing time: ~0.5 seconds (demo)
Memory usage: ~5 MB (demo)

Quality metrics (demo values):
RMSE: 1.23
MAE: 0.87
Max error: 2.45
R²: 0.94"""
        
        ttk.Label(summary_frame, text=summary_text, font=("Consolas", 9), 
                 justify=tk.LEFT).pack(padx=10, pady=5)
        
        # Export options
        export_frame = ttk.LabelFrame(results_frame, text="Export Results")
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(export_frame, text="Export Grid (CSV)", 
                  command=self.export_csv).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(export_frame, text="Export Plot (PNG)", 
                  command=self.export_plot).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(export_frame, text="Export Report", 
                  command=self.export_report).pack(side=tk.LEFT, padx=5, pady=5)
    
    def draw_mock_plot(self):
        """Draw a mock interpolation plot on canvas."""
        canvas = self.plot_canvas
        canvas.delete("all")
        
        # Get canvas dimensions
        canvas.update()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w <= 1 or h <= 1:  # Canvas not ready
            canvas.after(100, self.draw_mock_plot)
            return
        
        # Draw coordinate system
        margin = 50
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        
        # Axes
        canvas.create_line(margin, h - margin, w - margin, h - margin, width=2)  # X axis
        canvas.create_line(margin, margin, margin, h - margin, width=2)  # Y axis
        
        # Axis labels
        canvas.create_text(w // 2, h - 20, text="X Coordinate", font=("Arial", 10))
        canvas.create_text(20, h // 2, text="Y Coordinate", font=("Arial", 10), angle=90)
        canvas.create_text(w // 2, 20, text="Coal Deposit Interpolation (Demo)", font=("Arial", 12, "bold"))
        
        # Draw sample points
        for x, y, z in self.sample_data:
            # Map coordinates to canvas
            cx = margin + (x - 1) / 4 * plot_w
            cy = h - margin - (y - 1) / 4 * plot_h
            
            # Color based on Z value
            color_intensity = int(255 * (z - 10) / 20)
            color = f"#{color_intensity:02x}{50:02x}{255-color_intensity:02x}"
            
            # Draw point
            canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, 
                             fill=color, outline="black", width=1)
            canvas.create_text(cx + 10, cy - 10, text=f"{z:.1f}", font=("Arial", 8))
        
        # Draw mock contour lines
        for i in range(3, 8):
            level = i * 5
            color_val = max(0, min(255, int(255 * (level - 15) / 15)))
            blue_val = max(0, min(255, 255 - color_val))
            line_color = f"#{color_val:02x}64{blue_val:02x}"
            
            # Draw curved contour line
            points = []
            for j in range(20):
                angle = j * 3.14159 * 2 / 20
                radius = 30 + i * 20
                cx = w // 2 + radius * 0.7 * (j / 20 - 0.5)
                cy = h // 2 + radius * 0.5 * (j / 20 - 0.5)
                points.extend([cx, cy])
            
            if len(points) >= 4:
                canvas.create_line(points, fill=line_color, width=2, smooth=True)
        
        # Legend
        legend_x = w - 120
        legend_y = 60
        canvas.create_rectangle(legend_x - 10, legend_y - 10, 
                              legend_x + 100, legend_y + 80, 
                              fill="white", outline="black")
        canvas.create_text(legend_x + 45, legend_y, text="Z Values", font=("Arial", 10, "bold"))
        
        for i, level in enumerate([10, 15, 20, 25, 30]):
            y = legend_y + 15 + i * 12
            color_val = int(255 * (level - 10) / 20)
            color = f"#{color_val:02x}{50:02x}{255-color_val:02x}"
            canvas.create_oval(legend_x, y - 4, legend_x + 8, y + 4, fill=color, outline="black")
            canvas.create_text(legend_x + 15, y, text=f"{level}", font=("Arial", 9), anchor="w")
    
    def on_method_change(self):
        """Handle interpolation method change."""
        method = self.method_var.get()
        # Show/hide parameter frames based on method
        if method == "IDW":
            self.idw_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.idw_frame.pack_forget()
        
        self.status_var.set(f"Method changed to {method}")
    
    def load_data(self):
        """Mock data loading."""
        messagebox.showinfo("Demo Mode", 
                           "In full mode, this would open a file dialog to load CSV data.\n\n"
                           "Current demo data:\n"
                           f"• {len(self.sample_data)} sample points\n"
                           "• X range: 1.0 - 5.0\n"
                           "• Y range: 1.0 - 5.0\n"
                           "• Z range: 10.0 - 30.0")
        self.status_var.set("Demo data loaded")
    
    def load_excel(self):
        messagebox.showinfo("Demo Mode", "Excel loading will be available in full version.")
    
    def apply_parameters(self):
        """Apply interpolation parameters."""
        method = self.method_var.get()
        
        if method == "IDW":
            power = self.power_var.get()
            neighbors = self.neighbors_var.get()
            radius = self.radius_var.get()
            message = f"IDW parameters applied:\n• Power: {power:.1f}\n• Neighbors: {neighbors}\n• Radius: {radius:.1f}"
        else:
            message = f"{method} parameters would be applied in full version."
        
        messagebox.showinfo("Parameters Applied", message)
        self.status_var.set(f"Parameters applied - {method}")
        
        # Simulate progress
        self.progress['value'] = 0
        for i in range(101):
            self.progress['value'] = i
            self.root.update_idletasks()
            self.root.after(10)
        self.progress['value'] = 0
    
    def update_plot(self):
        """Update the visualization."""
        self.draw_mock_plot()
        messagebox.showinfo("Plot Updated", "Visualization updated with current parameters.")
        self.status_var.set("Plot updated")
    
    def export_plot(self):
        messagebox.showinfo("Demo Mode", "Plot export will save as PNG/PDF/SVG in full version.")
        self.status_var.set("Plot exported (demo)")
    
    def export_results(self):
        messagebox.showinfo("Demo Mode", "Results export will save interpolated grid as CSV in full version.")
        self.status_var.set("Results exported (demo)")
    
    def export_csv(self):
        messagebox.showinfo("Demo Mode", "CSV export will save the interpolated grid data in full version.")
    
    def export_report(self):
        messagebox.showinfo("Demo Mode", "Report export will generate a detailed analysis report in full version.")
    
    def show_statistics(self):
        stats_text = f"""Data Statistics (Demo):

Sample Points: {len(self.sample_data)}

X Coordinate:
  Min: {min(x for x,y,z in self.sample_data):.2f}
  Max: {max(x for x,y,z in self.sample_data):.2f}
  Mean: {sum(x for x,y,z in self.sample_data)/len(self.sample_data):.2f}

Y Coordinate:
  Min: {min(y for x,y,z in self.sample_data):.2f}
  Max: {max(y for x,y,z in self.sample_data):.2f}
  Mean: {sum(y for x,y,z in self.sample_data)/len(self.sample_data):.2f}

Z Values:
  Min: {min(z for x,y,z in self.sample_data):.2f}
  Max: {max(z for x,y,z in self.sample_data):.2f}
  Mean: {sum(z for x,y,z in self.sample_data)/len(self.sample_data):.2f}"""
        
        messagebox.showinfo("Data Statistics", stats_text)
    
    def show_settings(self):
        messagebox.showinfo("Demo Mode", "Advanced settings dialog will be available in full version.")
    
    def reset_view(self):
        self.draw_mock_plot()
        self.status_var.set("View reset")
    
    def toggle_fullscreen(self):
        current = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current)
    
    def refresh_view(self):
        self.draw_mock_plot()
        self.status_var.set("View refreshed")
    
    def show_help(self):
        help_text = """Coal Deposit Interpolation - User Guide

DEMO VERSION - Limited Functionality

Available Features:
• Interface demonstration
• Parameter adjustment
• Mock visualization
• Basic navigation

Full Version Features:
• CSV/Excel data loading
• Real interpolation calculations
• Interactive matplotlib plots
• Result export (CSV, images)
• Multiple interpolation methods
• Statistical analysis

To enable full functionality:
pip install pandas numpy scipy matplotlib

Then run: python main.py"""
        
        messagebox.showinfo("User Guide", help_text)
    
    def show_about(self):
        about_text = """Coal Deposit Interpolation Application
Version: 0.7.0 (Demo Mode)

A professional tool for geological data interpolation using:
• IDW (Inverse Distance Weighting) ✓
• Ordinary Kriging (planned)
• RBF (Radial Basis Functions) (planned)

Features:
• Interactive visualization
• Multiple export formats
• Statistical analysis
• Professional reports

Current Mode: DEMO
For full functionality, install required packages:
pandas, numpy, scipy, matplotlib

© 2024 Coal Interpolation Project
Developed with Python + Tkinter"""
        
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run the demo application."""
        self.root.mainloop()


def main():
    """Main entry point for demo application."""
    print("Starting Coal Deposit Interpolation Application...")
    print("Version: 0.7.0")
    print("Mode: DEMO (standalone version)")
    print("-" * 50)
    
    try:
        app = DemoApp()
        print("Demo GUI loaded successfully!")
        print("\nDemo features:")
        print("+ Complete GUI interface")
        print("+ Parameter controls")
        print("+ Mock data visualization")
        print("+ Menu system and dialogs")
        print("\nTo enable full functionality:")
        print("pip install pandas numpy scipy matplotlib")
        print("Then run: python main.py")
        print("\nStarting demo interface...")
        
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        sys.exit(0)
        
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("Demo application closed successfully.")


if __name__ == "__main__":
    main()