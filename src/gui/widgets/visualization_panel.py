"""
Visualization panel widget for displaying interpolation results.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# Matplotlib imports for visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


class VisualizationPanel:
    """
    Panel for displaying interpolation results and visualizations.
    
    This panel shows:
    - Scatter plot of original data points
    - Contour map of interpolation results
    - Color bar and legends
    - Interactive zoom and pan capabilities
    - Multiple visualization modes
    """
    
    def __init__(self, parent, controller):
        """
        Initialize the visualization panel.
        
        Args:
            parent: Parent widget
            controller: Application controller instance
        """
        self.controller = controller
        self.current_results = None
        self.current_data = None
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.visualization_mode = tk.StringVar(value="both")  # both, data_only, results_only
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text="Visualization", padding="5")
        
        self.create_widgets()
        self.setup_bindings()
        
    def create_widgets(self):
        """Create the panel widgets."""
        # Control frame at the top
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Visualization options
        options_frame = ttk.LabelFrame(control_frame, text="Display Options", padding="5")
        options_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Mode selection
        ttk.Label(options_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        mode_frame = ttk.Frame(options_frame)
        mode_frame.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Radiobutton(
            mode_frame, text="Data + Results", 
            variable=self.visualization_mode, value="both",
            command=self.update_visualization
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            mode_frame, text="Data Only", 
            variable=self.visualization_mode, value="data_only",
            command=self.update_visualization
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            mode_frame, text="Results Only", 
            variable=self.visualization_mode, value="results_only",
            command=self.update_visualization
        ).pack(side=tk.LEFT)
        
        # Control buttons
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(side=tk.RIGHT)
        
        self.reset_btn = ttk.Button(
            buttons_frame,
            text="Reset View",
            command=self.reset_view,
            state=tk.DISABLED
        )
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.export_btn = ttk.Button(
            buttons_frame,
            text="Export Image",
            command=self.export_image,
            state=tk.DISABLED
        )
        self.export_btn.pack(side=tk.LEFT)
        
        # Main visualization area
        self.viz_frame = ttk.Frame(self.frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.create_matplotlib_components()
        
    def create_matplotlib_components(self):
        """Create matplotlib figure and canvas."""
        # Create figure with subplots
        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor='white')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.viz_frame)
        self.canvas.draw()
        
        # Pack canvas
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create navigation toolbar
        toolbar_frame = ttk.Frame(self.viz_frame)
        toolbar_frame.pack(fill=tk.X)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Initial empty plot
        self.show_placeholder()
        
    def show_placeholder(self):
        """Show placeholder when no data is available."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.text(0.5, 0.5, 
                'Visualization Panel\n\n'
                '• Load data to see scatter plot\n'
                '• Run interpolation to see results\n'
                '• Use toolbar for zoom and pan\n\n'
                'Ready for data...',
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        self.canvas.draw()
        
    def setup_bindings(self):
        """Setup event bindings."""
        self.controller.bind_event("data_loaded", self.on_data_loaded)
        self.controller.bind_event("interpolation_completed", self.on_interpolation_completed)
        
    def on_data_loaded(self, data_info: Dict[str, Any]):
        """
        Handle data loaded event.
        
        Args:
            data_info: Information about the loaded data
        """
        # Get the actual data from controller
        self.current_data = self.controller.current_data
        
        # Enable controls
        self.reset_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        
        # Update visualization
        self.update_visualization()
        
    def on_interpolation_completed(self, results: Dict[str, Any]):
        """
        Handle interpolation completed event.
        
        Args:
            results: Interpolation results
        """
        self.current_results = results
        
        # Update visualization with results
        self.update_visualization()
        
    def update_visualization(self):
        """Update the visualization based on current mode and data."""
        if self.current_data is None or len(self.current_data) == 0:
            self.show_placeholder()
            return
            
        try:
            self.figure.clear()
            
            mode = self.visualization_mode.get()
            
            if mode == "both" and self.current_results is not None:
                self.create_combined_plot()
            elif mode == "data_only":
                self.create_data_plot()
            elif mode == "results_only" and self.current_results is not None:
                self.create_results_plot()
            else:
                self.create_data_plot()  # Default to data only
                
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Visualization error: {str(e)}")
            
    def create_data_plot(self):
        """Create scatter plot of original data points."""
        ax = self.figure.add_subplot(111)
        
        # Extract coordinates
        x_coords = self.current_data['X'].values
        y_coords = self.current_data['Y'].values
        
        # Get first value column for coloring points
        value_columns = [col for col in self.current_data.columns if col not in ['X', 'Y']]
        if value_columns:
            values = self.current_data[value_columns[0]].values
            scatter = ax.scatter(x_coords, y_coords, c=values, cmap='viridis', 
                               s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add colorbar
            cbar = self.figure.colorbar(scatter, ax=ax)
            cbar.set_label(f'{value_columns[0]} Values', rotation=270, labelpad=20)
        else:
            ax.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.7, 
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Data Points ({len(self.current_data)} points)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
    def create_results_plot(self):
        """Create contour plot of interpolation results."""
        if not self.current_results:
            return
            
        ax = self.figure.add_subplot(111)
        
        grid_data = self.current_results['grid']
        
        # Find interpolated column
        interpolated_col = None
        for col in grid_data.columns:
            if col.endswith('_interpolated'):
                interpolated_col = col
                break
                
        if not interpolated_col:
            ax.text(0.5, 0.5, 'No interpolated data found', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Prepare data for contouring
        x_coords = grid_data['X'].values
        y_coords = grid_data['Y'].values  
        z_values = grid_data[interpolated_col].values
        
        # Create regular grid for contouring
        x_unique = np.sort(np.unique(x_coords))
        y_unique = np.sort(np.unique(y_coords))
        
        if len(x_unique) > 1 and len(y_unique) > 1:
            # Reshape data to grid
            X, Y = np.meshgrid(x_unique, y_unique)
            Z = np.full_like(X, np.nan)
            
            # Fill grid with values
            for i, row in grid_data.iterrows():
                x_idx = np.where(x_unique == row['X'])[0]
                y_idx = np.where(y_unique == row['Y'])[0]
                if len(x_idx) > 0 and len(y_idx) > 0:
                    Z[y_idx[0], x_idx[0]] = row[interpolated_col]
            
            # Create contour plot
            levels = 20
            contour_filled = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
            contour_lines = ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.4, linewidths=0.5)
            
            # Add colorbar
            cbar = self.figure.colorbar(contour_filled, ax=ax)
            cbar.set_label('Interpolated Values', rotation=270, labelpad=20)
            
            # Add contour labels
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        else:
            # Fallback to scatter plot if grid is not regular
            scatter = ax.scatter(x_coords, y_coords, c=z_values, cmap='viridis', s=30)
            cbar = self.figure.colorbar(scatter, ax=ax)
            cbar.set_label('Interpolated Values', rotation=270, labelpad=20)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Interpolation Results')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
    def create_combined_plot(self):
        """Create combined plot showing both data points and interpolation results."""
        if not self.current_results:
            self.create_data_plot()
            return
            
        ax = self.figure.add_subplot(111)
        
        # First, create the contour plot (background)
        grid_data = self.current_results['grid']
        
        # Find interpolated column
        interpolated_col = None
        for col in grid_data.columns:
            if col.endswith('_interpolated'):
                interpolated_col = col
                break
                
        if interpolated_col:
            # Prepare data for contouring
            x_coords = grid_data['X'].values
            y_coords = grid_data['Y'].values  
            z_values = grid_data[interpolated_col].values
            
            # Create regular grid for contouring
            x_unique = np.sort(np.unique(x_coords))
            y_unique = np.sort(np.unique(y_coords))
            
            if len(x_unique) > 1 and len(y_unique) > 1:
                # Reshape data to grid
                X, Y = np.meshgrid(x_unique, y_unique)
                Z = np.full_like(X, np.nan)
                
                # Fill grid with values
                for i, row in grid_data.iterrows():
                    x_idx = np.where(x_unique == row['X'])[0]
                    y_idx = np.where(y_unique == row['Y'])[0]
                    if len(x_idx) > 0 and len(y_idx) > 0:
                        Z[y_idx[0], x_idx[0]] = row[interpolated_col]
                
                # Create contour plot with reduced alpha
                levels = 15
                contour_filled = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
                
                # Add colorbar for interpolation
                cbar = self.figure.colorbar(contour_filled, ax=ax)
                cbar.set_label('Interpolated Values', rotation=270, labelpad=20)
        
        # Overlay original data points
        data_x = self.current_data['X'].values
        data_y = self.current_data['Y'].values
        
        # Get first value column for coloring points
        value_columns = [col for col in self.current_data.columns if col not in ['X', 'Y']]
        if value_columns:
            data_values = self.current_data[value_columns[0]].values
            ax.scatter(data_x, data_y, c=data_values, cmap='plasma', 
                      s=80, alpha=0.9, edgecolors='white', linewidth=1.5,
                      label='Original Data')
        else:
            ax.scatter(data_x, data_y, c='red', s=80, alpha=0.9, 
                      edgecolors='white', linewidth=1.5, label='Original Data')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Interpolation Results with Data Points ({len(self.current_data)} points)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        
    def show_error(self, error_message: str):
        """Show error message in the plot area."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.text(0.5, 0.5, f'Error:\n{error_message}', 
               ha='center', va='center', fontsize=12, color='red',
               transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        self.canvas.draw()
        
    def reset_view(self):
        """Reset the view to show all data."""
        if self.toolbar:
            self.toolbar.home()
        
    def export_image(self):
        """Export the current visualization as an image."""
        if not self.figure:
            messagebox.showwarning("Warning", "No visualization to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Visualization",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"), 
                ("SVG files", "*.svg"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Visualization exported to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export image:\n{str(e)}")