"""
Data panel widget for displaying and managing loaded data.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Optional

# Import i18n system
from ...i18n import _


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
        self.frame = ttk.LabelFrame(parent, text=_("Data"), padding="5")
        
        # Store labels for language updates
        self.ui_labels = {}
        
        self.create_widgets()
        self.setup_bindings()
        
    def create_widgets(self):
        """Create the panel widgets."""
        # File information section
        self.file_frame = ttk.LabelFrame(self.frame, text=_("File Information"), padding="5")
        self.file_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.filename_var = tk.StringVar(value=_("No file loaded"))
        self.ui_labels['file_label'] = ttk.Label(self.file_frame, text=_("File:"))
        self.ui_labels['file_label'].grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(self.file_frame, textvariable=self.filename_var).grid(row=0, column=1, sticky=tk.W)
        
        self.rows_var = tk.StringVar(value="0")
        self.ui_labels['rows_label'] = ttk.Label(self.file_frame, text=_("Rows:"))
        self.ui_labels['rows_label'].grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(self.file_frame, textvariable=self.rows_var).grid(row=1, column=1, sticky=tk.W)
        
        self.columns_var = tk.StringVar(value="0")
        self.ui_labels['columns_label'] = ttk.Label(self.file_frame, text=_("Columns:"))
        self.ui_labels['columns_label'].grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(self.file_frame, textvariable=self.columns_var).grid(row=2, column=1, sticky=tk.W)
        
        # Data bounds section
        self.bounds_frame = ttk.LabelFrame(self.frame, text=_("Data Bounds"), padding="5")
        self.bounds_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.min_x_var = tk.StringVar(value="-")
        self.max_x_var = tk.StringVar(value="-")
        self.min_y_var = tk.StringVar(value="-")
        self.max_y_var = tk.StringVar(value="-")
        
        self.ui_labels['x_range_label'] = ttk.Label(self.bounds_frame, text=_("X Range:"))
        self.ui_labels['x_range_label'].grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(self.bounds_frame, textvariable=self.min_x_var).grid(row=0, column=1, sticky=tk.W)
        self.ui_labels['to_label1'] = ttk.Label(self.bounds_frame, text=_("to"))
        self.ui_labels['to_label1'].grid(row=0, column=2, padx=5)
        ttk.Label(self.bounds_frame, textvariable=self.max_x_var).grid(row=0, column=3, sticky=tk.W)
        
        self.ui_labels['y_range_label'] = ttk.Label(self.bounds_frame, text=_("Y Range:"))
        self.ui_labels['y_range_label'].grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(self.bounds_frame, textvariable=self.min_y_var).grid(row=1, column=1, sticky=tk.W)
        self.ui_labels['to_label2'] = ttk.Label(self.bounds_frame, text=_("to"))
        self.ui_labels['to_label2'].grid(row=1, column=2, padx=5)
        ttk.Label(self.bounds_frame, textvariable=self.max_y_var).grid(row=1, column=3, sticky=tk.W)
        
        # Columns list section
        self.columns_frame = ttk.LabelFrame(self.frame, text=_("Columns"), padding="5")
        self.columns_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for columns
        self.column_headers = [_("Name"), _("Type"), _("Missing")]
        self.columns_tree = ttk.Treeview(self.columns_frame, columns=self.column_headers, show="headings", height=6)
        
        for i, col in enumerate(self.column_headers):
            self.columns_tree.heading(f"#{i+1}", text=col)
            self.columns_tree.column(f"#{i+1}", width=80)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(self.columns_frame, orient=tk.VERTICAL, command=self.columns_tree.yview)
        self.columns_tree.configure(yscrollcommand=scrollbar.set)
        
        self.columns_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Actions frame
        actions_frame = ttk.Frame(self.frame)
        actions_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.reload_btn = ttk.Button(
            actions_frame, 
            text=_("Reload Data"), 
            command=self.reload_data,
            state=tk.DISABLED
        )
        self.reload_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.statistics_btn = ttk.Button(
            actions_frame, 
            text=_("View Statistics"), 
            command=self.show_statistics,
            state=tk.DISABLED
        )
        self.statistics_btn.pack(side=tk.LEFT)
        
        # Store buttons for language updates
        self.ui_labels['reload_btn'] = self.reload_btn
        self.ui_labels['statistics_btn'] = self.statistics_btn
        
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
        if not self.data_info:
            tk.messagebox.showwarning(_("Warning"), _("No data loaded to reload"))
            return
        
        file_path = self.data_info.get('file_path')
        if not file_path:
            tk.messagebox.showwarning(_("Warning"), _("Original file path not available"))
            return
        
        # Show confirmation dialog
        if tk.messagebox.askyesno(_("Confirm Reload"), _("Reload data from file?\nAny unsaved changes will be lost.")):
            try:
                # Notify controller to reload the data
                self.controller.reload_current_data()
                tk.messagebox.showinfo(_("Success"), _("Data reloaded successfully"))
            except Exception as e:
                tk.messagebox.showerror(_("Error"), f"{_('Failed to reload data')}: {str(e)}")
        
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
    
    def update_language(self):
        """Update all text elements with new translations."""
        try:
            # Update main frame title
            self.frame.config(text=_("Data"))
            
            # Update frame titles
            self.file_frame.config(text=_("File Information"))
            self.bounds_frame.config(text=_("Data Bounds"))
            self.columns_frame.config(text=_("Columns"))
            
            # Update all labels
            label_updates = {
                'file_label': _("File:"),
                'rows_label': _("Rows:"),
                'columns_label': _("Columns:"),
                'x_range_label': _("X Range:"),
                'y_range_label': _("Y Range:"),
                'to_label1': _("to"),
                'to_label2': _("to")
            }
            
            for label_key, text in label_updates.items():
                if label_key in self.ui_labels:
                    self.ui_labels[label_key].config(text=text)
            
            # Update buttons
            if 'reload_btn' in self.ui_labels:
                self.ui_labels['reload_btn'].config(text=_("Reload Data"))
            if 'statistics_btn' in self.ui_labels:
                self.ui_labels['statistics_btn'].config(text=_("View Statistics"))
            
            # Update column headers
            headers = [_("Name"), _("Type"), _("Missing")]
            for i, header in enumerate(headers):
                self.columns_tree.heading(f"#{i+1}", text=header)
            
            # Update no file loaded message if it's currently shown
            if self.filename_var.get() in ["No file loaded", "Файл не загружен"]:
                self.filename_var.set(_("No file loaded"))
                
            print("DataPanel language updated successfully")
            
        except Exception as e:
            print(f"Error updating DataPanel language: {e}")
            import traceback
            traceback.print_exc()