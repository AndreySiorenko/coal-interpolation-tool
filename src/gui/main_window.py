"""
Main window of the coal interpolation application.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from typing import Optional, Dict, Any
from pathlib import Path

from .widgets.data_panel import DataPanel
from .widgets.parameters_panel import ParametersPanel
from .widgets.visualization_panel import VisualizationPanel
from .widgets.results_panel import ResultsPanel
from .controllers.application_controller import ApplicationController
from .dialogs.data_loader_dialog import show_data_loader_dialog
from .dialogs.export_dialog import show_export_dialog


class MainWindow:
    """
    Main application window for coal deposit interpolation.
    
    This class creates and manages the main GUI interface, including:
    - Menu bar with File, View, Tools, Help menus
    - Toolbar with common actions
    - Central area with docked panels for data, parameters, visualization, and results
    - Status bar for progress and messages
    - Integration with backend components
    """
    
    def __init__(self):
        """Initialize the main window."""
        self.root = tk.Tk()
        self.controller = ApplicationController()
        
        # Window configuration
        self.setup_window()
        
        # Create UI components
        self.create_menu_bar()
        self.create_toolbar()
        self.create_main_panels()
        self.create_status_bar()
        
        # Setup event bindings
        self.setup_event_bindings()
        
        # Initialize state
        self.current_file = None
        self.interpolation_results = None
        
    def setup_window(self):
        """Configure the main window properties."""
        self.root.title("Coal Deposit Interpolation - v0.4.0")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Center window on screen
        self.center_window()
        
        # Configure window closing behavior
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure style for ttk widgets
        self.setup_styles()
        
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
    def setup_styles(self):
        """Configure ttk styles for consistent appearance."""
        style = ttk.Style()
        
        # Use modern theme if available
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")
        elif "alt" in available_themes:
            style.theme_use("alt")
            
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save Project...", command=self.save_project, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Project...", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Ctrl+Q")
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset Layout", command=self.reset_layout)
        view_menu.add_separator()
        view_menu.add_checkbutton(label="Show Data Panel", variable=tk.BooleanVar(value=True))
        view_menu.add_checkbutton(label="Show Parameters Panel", variable=tk.BooleanVar(value=True))
        view_menu.add_checkbutton(label="Show Visualization Panel", variable=tk.BooleanVar(value=True))
        view_menu.add_checkbutton(label="Show Results Panel", variable=tk.BooleanVar(value=True))
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Data Statistics", command=self.show_data_statistics)
        tools_menu.add_command(label="Grid Info", command=self.show_grid_info)
        tools_menu.add_separator()
        tools_menu.add_command(label="Preferences", command=self.show_preferences)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Bind keyboard shortcuts
        self.root.bind_all("<Control-o>", lambda e: self.open_file())
        self.root.bind_all("<Control-s>", lambda e: self.save_project())
        self.root.bind_all("<Control-e>", lambda e: self.export_results())
        self.root.bind_all("<Control-q>", lambda e: self.on_closing())
        
    def create_toolbar(self):
        """Create the main toolbar."""
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # File operations
        ttk.Button(toolbar_frame, text="Open", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Save", command=self.save_project).pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Interpolation operations
        self.interpolate_btn = ttk.Button(
            toolbar_frame, 
            text="Interpolate", 
            command=self.run_interpolation,
            state=tk.DISABLED
        )
        self.interpolate_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar_frame, text="Export", command=self.export_results).pack(side=tk.LEFT, padx=2)
        
        # Separator  
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # View operations
        ttk.Button(toolbar_frame, text="Statistics", command=self.show_data_statistics).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Grid Info", command=self.show_grid_info).pack(side=tk.LEFT, padx=2)
        
    def create_main_panels(self):
        """Create the main panels in the central area."""
        # Create main paned window for horizontal split
        main_paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for data and parameters
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Create vertical paned window for left panels
        left_paned = ttk.Panedwindow(left_frame, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True)
        
        # Data panel
        self.data_panel = DataPanel(left_paned, self.controller)
        left_paned.add(self.data_panel.frame, weight=1)
        
        # Parameters panel
        self.parameters_panel = ParametersPanel(left_paned, self.controller)
        left_paned.add(self.parameters_panel.frame, weight=1)
        
        # Right panel for visualization and results
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Create vertical paned window for right panels
        right_paned = ttk.Panedwindow(right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # Visualization panel
        self.visualization_panel = VisualizationPanel(right_paned, self.controller)
        right_paned.add(self.visualization_panel.frame, weight=2)
        
        # Results panel
        self.results_panel = ResultsPanel(right_paned, self.controller)
        right_paned.add(self.results_panel.frame, weight=1)
        
    def create_status_bar(self):
        """Create the status bar at the bottom."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, 
            variable=self.progress_var, 
            maximum=100, 
            length=200
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_event_bindings(self):
        """Setup event bindings between components."""
        # Bind controller events to UI updates
        self.controller.bind_event("data_loaded", self.on_data_loaded)
        self.controller.bind_event("interpolation_started", self.on_interpolation_started)
        self.controller.bind_event("interpolation_progress", self.on_interpolation_progress)
        self.controller.bind_event("interpolation_completed", self.on_interpolation_completed)
        self.controller.bind_event("error_occurred", self.on_error_occurred)
        
    # Event handlers
    def on_data_loaded(self, data_info: Dict[str, Any]):
        """Handle data loaded event."""
        self.status_var.set(f"Loaded data: {data_info['filename']} ({data_info['rows']} rows)")
        self.interpolate_btn.config(state=tk.NORMAL)
        
    def on_interpolation_started(self):
        """Handle interpolation started event."""
        self.status_var.set("Running interpolation...")
        self.interpolate_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
    def on_interpolation_progress(self, progress: float):
        """Handle interpolation progress event."""
        self.progress_var.set(progress)
        self.root.update_idletasks()
        
    def on_interpolation_completed(self, results: Dict[str, Any]):
        """Handle interpolation completed event."""
        self.status_var.set("Interpolation completed successfully")
        self.interpolate_btn.config(state=tk.NORMAL)
        self.progress_var.set(100)
        self.interpolation_results = results
        
    def on_error_occurred(self, error_msg: str):
        """Handle error occurred event."""
        self.status_var.set("Error occurred")
        self.interpolate_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        messagebox.showerror("Error", error_msg)
        
    # Menu action handlers
    def open_file(self):
        """Open a data file using the advanced data loader dialog."""
        try:
            # Show the advanced data loader dialog
            result = show_data_loader_dialog(self.root, self.controller)
            
            if result:
                # Load data with the selected settings
                self.controller.load_data_with_settings(result)
                self.current_file = result['file_path']
                
        except Exception as e:
            self.on_error_occurred(f"Failed to load data file: {str(e)}")
                
    def save_project(self):
        """Save current project."""
        if not self.controller.has_data():
            messagebox.showwarning("Warning", "No data to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".json",
            filetypes=[("Project files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.controller.save_project(file_path)
                self.status_var.set(f"Project saved: {Path(file_path).name}")
            except Exception as e:
                self.on_error_occurred(f"Failed to save project: {str(e)}")
                
    def load_project(self):
        """Load a project file."""
        file_path = filedialog.askopenfilename(
            title="Load Project",
            filetypes=[("Project files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.controller.load_project(file_path)
                self.status_var.set(f"Project loaded: {Path(file_path).name}")
            except Exception as e:
                self.on_error_occurred(f"Failed to load project: {str(e)}")
                
    def export_results(self):
        """Export interpolation results using the advanced export dialog."""
        if not self.interpolation_results:
            messagebox.showwarning("Warning", "No results to export")
            return
            
        try:
            # Show the advanced export dialog
            result = show_export_dialog(self.root, self.interpolation_results)
            
            if result:
                # Export data with the selected settings
                self.controller.export_results_with_settings(result)
                self.status_var.set(f"Results exported: {Path(result['file_path']).name}")
                
        except Exception as e:
            self.on_error_occurred(f"Failed to export results: {str(e)}")
                
    def run_interpolation(self):
        """Run the interpolation process."""
        if not self.controller.has_data():
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        try:
            # Get parameters from parameters panel
            params = self.parameters_panel.get_parameters()
            
            # Run interpolation
            self.controller.run_interpolation(params)
            
        except Exception as e:
            self.on_error_occurred(f"Interpolation failed: {str(e)}")
            
    def reset_layout(self):
        """Reset the panel layout to default."""
        # This would restore default panel sizes and positions
        self.status_var.set("Layout reset to default")
        
    def show_data_statistics(self):
        """Show data statistics dialog."""
        if not self.controller.has_data():
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        # This would show a dialog with data statistics
        messagebox.showinfo("Info", "Data statistics dialog - To be implemented")
        
    def show_grid_info(self):
        """Show grid information dialog."""
        if not self.controller.has_data():
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        # This would show grid information based on current parameters
        messagebox.showinfo("Info", "Grid info dialog - To be implemented")
        
    def show_preferences(self):
        """Show preferences dialog."""
        messagebox.showinfo("Info", "Preferences dialog - To be implemented")
        
    def show_help(self):
        """Show help dialog."""
        help_text = """
Coal Deposit Interpolation Application

Basic Usage:
1. Open a CSV file with your borehole data
2. Configure interpolation parameters
3. Click 'Interpolate' to run the analysis  
4. View results in the visualization panel
5. Export results to various formats

For detailed documentation, see the project README.md file.
        """
        messagebox.showinfo("Help", help_text)
        
    def show_about(self):
        """Show about dialog."""
        about_text = """
Coal Deposit Interpolation v0.4.0

A professional tool for spatial interpolation of coal deposit data.

Features:
• IDW (Inverse Distance Weighted) interpolation
• Flexible grid generation
• CSV and Excel data import
• Multiple export formats

Developed for geologists and mining engineers.
        """
        messagebox.showinfo("About", about_text)
        
    def on_closing(self):
        """Handle window closing event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()
            self.root.destroy()
            
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()


def main():
    """Main entry point for the GUI application."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()