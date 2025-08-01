"""
Data panel widget for displaying and managing loaded data.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional


class DataPanel:
    """
    Panel for displaying information about loaded data.
    
    This panel shows:
    - Currently loaded file information
    - Data preview (first few rows)
    - Basic statistics
    - Column information
    """
    
    def __init__(self, parent, controller):
        """
        Initialize the data panel.
        
        Args:
            parent: Parent widget
            controller: Application controller instance
        """
        self.controller = controller
        self.data_info = None
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text="Data", padding="5")
        
        self.create_widgets()
        self.setup_bindings()
        
    def create_widgets(self):
        """Create the panel widgets."""
        # File information section
        file_frame = ttk.LabelFrame(self.frame, text="File Information", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.filename_var = tk.StringVar(value="No file loaded")
        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(file_frame, textvariable=self.filename_var).grid(row=0, column=1, sticky=tk.W)
        
        self.rows_var = tk.StringVar(value="0")
        ttk.Label(file_frame, text="Rows:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(file_frame, textvariable=self.rows_var).grid(row=1, column=1, sticky=tk.W)
        
        self.columns_var = tk.StringVar(value="0")
        ttk.Label(file_frame, text="Columns:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(file_frame, textvariable=self.columns_var).grid(row=2, column=1, sticky=tk.W)
        
        # Data bounds section
        bounds_frame = ttk.LabelFrame(self.frame, text="Data Bounds", padding="5")
        bounds_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.min_x_var = tk.StringVar(value="-")
        self.max_x_var = tk.StringVar(value="-")
        self.min_y_var = tk.StringVar(value="-")
        self.max_y_var = tk.StringVar(value="-")
        
        ttk.Label(bounds_frame, text="X Range:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(bounds_frame, textvariable=self.min_x_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(bounds_frame, text="to").grid(row=0, column=2, padx=5)
        ttk.Label(bounds_frame, textvariable=self.max_x_var).grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(bounds_frame, text="Y Range:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(bounds_frame, textvariable=self.min_y_var).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(bounds_frame, text="to").grid(row=1, column=2, padx=5)
        ttk.Label(bounds_frame, textvariable=self.max_y_var).grid(row=1, column=3, sticky=tk.W)
        
        # Columns list section
        columns_frame = ttk.LabelFrame(self.frame, text="Columns", padding="5")
        columns_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for columns
        columns = ("Name", "Type", "Missing")
        self.columns_tree = ttk.Treeview(columns_frame, columns=columns, show="headings", height=6)
        
        for col in columns:
            self.columns_tree.heading(col, text=col)
            self.columns_tree.column(col, width=80)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(columns_frame, orient=tk.VERTICAL, command=self.columns_tree.yview)
        self.columns_tree.configure(yscrollcommand=scrollbar.set)
        
        self.columns_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Actions frame
        actions_frame = ttk.Frame(self.frame)
        actions_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(
            actions_frame, 
            text="Reload Data", 
            command=self.reload_data,
            state=tk.DISABLED
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            actions_frame, 
            text="View Statistics", 
            command=self.show_statistics,
            state=tk.DISABLED
        ).pack(side=tk.LEFT)
        
    def setup_bindings(self):
        """Setup event bindings."""
        self.controller.bind_event("data_loaded", self.on_data_loaded)
        
    def on_data_loaded(self, data_info: Dict[str, Any]):
        """
        Handle data loaded event.
        
        Args:
            data_info: Information about the loaded data
        """
        self.data_info = data_info
        self.update_display()
        
    def update_display(self):
        """Update the display with current data information."""
        if not self.data_info:
            return
            
        # Update file information
        self.filename_var.set(self.data_info['filename'])
        self.rows_var.set(str(self.data_info['rows']))
        self.columns_var.set(str(len(self.data_info['columns'])))
        
        # Update bounds
        bounds = self.data_info['bounds']
        self.min_x_var.set(f"{bounds['min_x']:.2f}")
        self.max_x_var.set(f"{bounds['max_x']:.2f}")
        self.min_y_var.set(f"{bounds['min_y']:.2f}")
        self.max_y_var.set(f"{bounds['max_y']:.2f}")
        
        # Update columns list
        self.update_columns_list()
        
        # Enable buttons
        for child in self.frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for button in child.winfo_children():
                    if isinstance(button, ttk.Button):
                        button.config(state=tk.NORMAL)
                        
    def update_columns_list(self):
        """Update the columns list display."""
        # Clear existing items
        for item in self.columns_tree.get_children():
            self.columns_tree.delete(item)
            
        if not self.data_info:
            return
            
        # Get additional data info from controller
        detailed_info = self.controller.get_data_info()
        if not detailed_info:
            return
            
        data_types = detailed_info.get('data_types', {})
        missing_values = detailed_info.get('missing_values', {})
        
        # Add column information
        for column in self.data_info['columns']:
            dtype = str(data_types.get(column, 'unknown'))
            missing = missing_values.get(column, 0)
            
            self.columns_tree.insert(
                "", tk.END, 
                values=(column, dtype, missing)
            )
            
    def reload_data(self):
        """Reload the current data file."""
        # This would trigger a reload of the current file
        # For now, just show a message
        tk.messagebox.showinfo("Info", "Reload functionality - To be implemented")
        
    def show_statistics(self):
        """Show detailed data statistics."""
        if not self.data_info:
            return
            
        # Create a new window with detailed statistics
        self.create_statistics_window()
        
    def create_statistics_window(self):
        """Create a window showing detailed data statistics."""
        stats_window = tk.Toplevel(self.frame)
        stats_window.title("Data Statistics")
        stats_window.geometry("500x400")
        stats_window.transient(self.frame.winfo_toplevel())
        stats_window.grab_set()
        
        # Create notebook for different statistics tabs
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Basic statistics tab
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Statistics")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(basic_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add statistics content
        stats_content = self.generate_statistics_text()
        text_widget.insert(tk.END, stats_content)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(
            stats_window, 
            text="Close", 
            command=stats_window.destroy
        ).pack(pady=10)
        
    def generate_statistics_text(self) -> str:
        """
        Generate text content for the statistics window.
        
        Returns:
            Formatted string with data statistics
        """
        if not self.data_info:
            return "No data available"
            
        detailed_info = self.controller.get_data_info()
        if not detailed_info:
            return "Could not retrieve detailed data information"
            
        content = []
        content.append("DATA STATISTICS")
        content.append("=" * 50)
        content.append("")
        
        # File information
        content.append(f"File: {detailed_info['filename']}")
        content.append(f"Total rows: {detailed_info['rows']:,}")
        content.append(f"Total columns: {len(detailed_info['columns'])}")
        content.append("")
        
        # Spatial bounds
        bounds = detailed_info['bounds']
        content.append("SPATIAL BOUNDS")
        content.append("-" * 20)
        content.append(f"X coordinate range: {bounds['min_x']:.2f} to {bounds['max_x']:.2f}")
        content.append(f"Y coordinate range: {bounds['min_y']:.2f} to {bounds['max_y']:.2f}")
        content.append(f"Area coverage: {(bounds['max_x'] - bounds['min_x']) * (bounds['max_y'] - bounds['min_y']):.2f} square units")
        content.append("")
        
        # Column information
        content.append("COLUMN INFORMATION")
        content.append("-" * 20)
        
        data_types = detailed_info.get('data_types', {})
        missing_values = detailed_info.get('missing_values', {})
        
        for column in detailed_info['columns']:
            dtype = data_types.get(column, 'unknown')
            missing = missing_values.get(column, 0)
            missing_pct = (missing / detailed_info['rows']) * 100 if detailed_info['rows'] > 0 else 0
            
            content.append(f"{column}:")
            content.append(f"  Type: {dtype}")
            content.append(f"  Missing values: {missing} ({missing_pct:.1f}%)")
            content.append("")
            
        return "\n".join(content)