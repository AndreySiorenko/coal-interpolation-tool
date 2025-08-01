"""
Results panel widget for displaying interpolation results and statistics.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional


class ResultsPanel:
    """
    Panel for displaying interpolation results and statistics.
    
    This panel shows:
    - Interpolation statistics (min, max, mean values)
    - Grid information (dimensions, cell size)
    - Processing time and performance metrics
    - Quality metrics (if available)
    - Export options for results
    """
    
    def __init__(self, parent, controller):
        """
        Initialize the results panel.
        
        Args:
            parent: Parent widget
            controller: Application controller instance
        """
        self.controller = controller
        self.current_results = None
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text="Results", padding="5")
        
        self.create_widgets()
        self.setup_bindings()
        
    def create_widgets(self):
        """Create the panel widgets."""
        # Statistics section
        stats_frame = ttk.LabelFrame(self.frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Create variables for statistics display
        self.min_value_var = tk.StringVar(value="-")
        self.max_value_var = tk.StringVar(value="-")
        self.mean_value_var = tk.StringVar(value="-")
        self.method_var = tk.StringVar(value="-")
        
        ttk.Label(stats_frame, text="Method:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(stats_frame, textvariable=self.method_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Min Value:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(stats_frame, textvariable=self.min_value_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Max Value:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(stats_frame, textvariable=self.max_value_var).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Mean Value:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(stats_frame, textvariable=self.mean_value_var).grid(row=3, column=1, sticky=tk.W)
        
        # Grid information section
        grid_frame = ttk.LabelFrame(self.frame, text="Grid Information", padding="5")
        grid_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.grid_points_var = tk.StringVar(value="-")
        self.cell_size_var = tk.StringVar(value="-")
        self.grid_extent_var = tk.StringVar(value="-")
        
        ttk.Label(grid_frame, text="Grid Points:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(grid_frame, textvariable=self.grid_points_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(grid_frame, text="Cell Size:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(grid_frame, textvariable=self.cell_size_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(grid_frame, text="Grid Extent:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(grid_frame, textvariable=self.grid_extent_var).grid(row=2, column=1, sticky=tk.W)
        
        # Results table
        table_frame = ttk.LabelFrame(self.frame, text="Sample Results", padding="5")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Create treeview for results preview
        columns = ("X", "Y", "Value")
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=80)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        actions_frame = ttk.Frame(self.frame)
        actions_frame.pack(fill=tk.X)
        
        ttk.Button(
            actions_frame,
            text="View All Results",
            command=self.view_all_results,
            state=tk.DISABLED
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            actions_frame,
            text="Export Results",
            command=self.export_results,
            state=tk.DISABLED
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            actions_frame,
            text="Quality Report",
            command=self.show_quality_report,
            state=tk.DISABLED
        ).pack(side=tk.LEFT)
        
    def setup_bindings(self):
        """Setup event bindings."""
        self.controller.bind_event("interpolation_completed", self.on_interpolation_completed)
        
    def on_interpolation_completed(self, results: Dict[str, Any]):
        """
        Handle interpolation completed event.
        
        Args:
            results: Interpolation results
        """
        self.current_results = results
        self.update_display()
        
    def update_display(self):
        """Update the display with current results."""
        if not self.current_results:
            return
            
        stats = self.current_results['statistics']
        parameters = self.current_results['parameters']
        grid_data = self.current_results['grid']
        
        # Update statistics
        self.method_var.set(stats['method'])
        self.min_value_var.set(f"{stats['min_value']:.3f}")
        self.max_value_var.set(f"{stats['max_value']:.3f}")
        self.mean_value_var.set(f"{stats['mean_value']:.3f}")
        
        # Update grid information
        self.grid_points_var.set(f"{stats['grid_points']:,}")
        self.cell_size_var.set(f"{parameters['cell_size']:.1f} m")
        
        # Calculate grid extent
        min_x, max_x = grid_data['X'].min(), grid_data['X'].max()
        min_y, max_y = grid_data['Y'].min(), grid_data['Y'].max()
        extent_x = max_x - min_x
        extent_y = max_y - min_y
        self.grid_extent_var.set(f"{extent_x:.0f} × {extent_y:.0f} m")
        
        # Update results table with sample data
        self.update_results_table()
        
        # Enable action buttons
        for child in self.frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for button in child.winfo_children():
                    if isinstance(button, ttk.Button):
                        button.config(state=tk.NORMAL)
                        
    def update_results_table(self):
        """Update the results table with sample data."""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        if not self.current_results:
            return
            
        grid_data = self.current_results['grid']
        
        # Show first 20 rows as sample
        sample_size = min(20, len(grid_data))
        sample_data = grid_data.head(sample_size)
        
        # Get interpolated column name
        value_col = None
        for col in grid_data.columns:
            if col.endswith('_interpolated'):
                value_col = col
                break
                
        if value_col is None:
            return
            
        # Add sample data to table
        for _, row in sample_data.iterrows():
            self.results_tree.insert(
                "", tk.END,
                values=(
                    f"{row['X']:.2f}",
                    f"{row['Y']:.2f}",
                    f"{row[value_col]:.3f}"
                )
            )
            
    def view_all_results(self):
        """Show all interpolation results in a new window."""
        if not self.current_results:
            return
            
        self.create_results_window()
        
    def create_results_window(self):
        """Create a window showing all interpolation results."""
        results_window = tk.Toplevel(self.frame)
        results_window.title("All Interpolation Results")
        results_window.geometry("800x600")
        results_window.transient(self.frame.winfo_toplevel())
        results_window.grab_set()
        
        # Create frame for table
        table_frame = ttk.Frame(results_window)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview for all results
        columns = ("X", "Y", "Interpolated Value")
        full_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        for col in columns:
            full_tree.heading(col, text=col)
            full_tree.column(col, width=150)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=full_tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=full_tree.xview)
        full_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and treeview
        full_tree.grid(row=0, column=0, sticky=tk.NSEW)
        v_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        h_scrollbar.grid(row=1, column=0, sticky=tk.EW)
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Populate with all data
        grid_data = self.current_results['grid']
        value_col = None
        for col in grid_data.columns:
            if col.endswith('_interpolated'):
                value_col = col
                break
                
        if value_col:
            for _, row in grid_data.iterrows():
                full_tree.insert(
                    "", tk.END,
                    values=(
                        f"{row['X']:.2f}",
                        f"{row['Y']:.2f}",
                        f"{row[value_col]:.4f}"
                    )
                )
        
        # Close button
        ttk.Button(
            results_window,
            text="Close",
            command=results_window.destroy
        ).pack(pady=10)
        
    def export_results(self):
        """Export interpolation results."""
        if not self.current_results:
            return
            
        # This would open a file dialog and export results
        tk.messagebox.showinfo("Info", "Export results functionality - To be implemented")
        
    def show_quality_report(self):
        """Show interpolation quality report."""
        if not self.current_results:
            return
            
        # This would show quality metrics like cross-validation results
        quality_text = """
INTERPOLATION QUALITY REPORT

Method: IDW (Inverse Distance Weighted)
Parameters used:
- Power: 2.0
- Search radius: 1000m
- Max points: 12

Quality Metrics:
- Cross-validation RMSE: [To be calculated]
- Mean absolute error: [To be calculated]
- R-squared: [To be calculated]

Recommendations:
- Consider adjusting power parameter for smoother/sharper results
- Evaluate search radius based on data density
- Compare with other interpolation methods

[Full quality analysis to be implemented]
        """
        
        tk.messagebox.showinfo("Quality Report", quality_text.strip())