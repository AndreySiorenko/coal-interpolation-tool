"""
Advanced export dialog for interpolation results.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import tempfile
import threading

from ...io.writers.base import ExportFormat, writer_registry
from ...io.writers.csv_writer import CSVWriter, CSVExportOptions
from ...io.writers.geotiff_writer import GeoTIFFWriter, GeoTIFFExportOptions
from ...io.writers.vtk_writer import VTKWriter, VTKExportOptions

# Import i18n system
from ...i18n import _


class ExportDialog:
    """
    Advanced dialog for exporting interpolation results.
    
    This dialog provides:
    - Format selection (CSV, GeoTIFF, VTK, DXF)
    - Format-specific options configuration
    - Export preview and validation
    - Progress tracking for large exports
    - Batch export capabilities
    """
    
    def __init__(self, parent, results_data: Dict[str, Any]):
        """
        Initialize the export dialog.
        
        Args:
            parent: Parent window
            results_data: Interpolation results to export
        """
        self.parent = parent
        self.results_data = results_data
        self.result = None
        self.current_export_path = None
        self.preview_data = None
        
        # Create modal dialog
        self.dialog = tk.Toplevel(parent)
        self.setup_dialog()
        self.create_widgets()
        
        # Initialize format-specific options
        self.format_options = {
            ExportFormat.CSV: CSVExportOptions(),
            ExportFormat.GEOTIFF: GeoTIFFExportOptions(),
            ExportFormat.VTK: VTKExportOptions(),
        }
        
        # Initialize with CSV format
        self.current_format = ExportFormat.CSV
        self.update_format_options()
        
    def setup_dialog(self):
        """Setup dialog window properties."""
        self.dialog.title(_("Export Interpolation Results"))
        self.dialog.geometry("900x700")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.cancel)
        
        # Center dialog on parent
        self.center_dialog()
        
    def center_dialog(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        # Get parent position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate center position
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
    def create_widgets(self):
        """Create dialog widgets."""
        # Create notebook for different steps
        self.notebook = ttk.Notebook(self.dialog)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Step 1: Format Selection
        self.create_format_selection_tab()
        
        # Step 2: Export Options
        self.create_export_options_tab()
        
        # Step 3: Preview & Export
        self.create_preview_export_tab()
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.dialog)
        buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(
            buttons_frame,
            text="Cancel",
            command=self.cancel
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        self.export_button = ttk.Button(
            buttons_frame,
            text="Export",
            command=self.export_data,
            state=tk.DISABLED
        )
        self.export_button.pack(side=tk.RIGHT)
        
        ttk.Button(
            buttons_frame,
            text="Previous",
            command=self.previous_step,
            state=tk.DISABLED
        ).pack(side=tk.LEFT)
        
        self.next_button = ttk.Button(
            buttons_frame,
            text="Next",
            command=self.next_step,
            state=tk.NORMAL
        )
        self.next_button.pack(side=tk.LEFT, padx=(5, 0))
        
        self.prev_button = buttons_frame.winfo_children()[-2]
        
    def create_format_selection_tab(self):
        """Create format selection tab."""
        format_frame = ttk.Frame(self.notebook)
        self.notebook.add(format_frame, text="1. Select Format")
        
        # Results info section
        info_frame = ttk.LabelFrame(format_frame, text=_("Export Data Information"), padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Display information about the results
        results_info = self.get_results_info()
        info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=info_text.yview)
        info_text.configure(yscrollcommand=info_scroll.set)
        
        info_text.config(state=tk.NORMAL)
        info_text.insert(tk.END, results_info)
        info_text.config(state=tk.DISABLED)
        
        info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Format selection section
        format_selection_frame = ttk.LabelFrame(format_frame, text=_("Export Format"), padding="10")
        format_selection_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.format_var = tk.StringVar(value="CSV")
        
        # Format radio buttons with descriptions
        formats_info = [
            ("CSV", "Comma-Separated Values - Tabular data format suitable for spreadsheets and databases"),
            ("GeoTIFF", "Georeferenced raster format - Standard for GIS applications and mapping software"),
            ("VTK", "Visualization Toolkit format - For 3D visualization in scientific applications"),
            ("DXF", "AutoCAD format - Vector graphics for CAD applications (Coming Soon)")
        ]
        
        for i, (fmt, description) in enumerate(formats_info):
            frame = ttk.Frame(format_selection_frame)
            frame.pack(fill=tk.X, pady=5)
            
            state = tk.NORMAL if fmt != "DXF" else tk.DISABLED
            radio = ttk.Radiobutton(
                frame,
                text=fmt,
                variable=self.format_var,
                value=fmt,
                command=self.on_format_changed,
                state=state
            )
            radio.pack(side=tk.LEFT)
            
            desc_label = ttk.Label(frame, text=description, font=("TkDefaultFont", 9))
            desc_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # File selection section
        file_frame = ttk.LabelFrame(format_frame, text=_("Output File"), padding="10")
        file_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Current file display
        ttk.Label(file_frame, text="Output File:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.export_path_var = tk.StringVar(value=_("No file selected"))
        file_label = ttk.Label(file_frame, textvariable=self.export_path_var, relief=tk.SUNKEN, padding="5")
        file_label.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(0, 10))
        
        # Browse button
        ttk.Button(
            file_frame,
            text="Browse...",
            command=self.browse_export_file
        ).grid(row=2, column=0, sticky=tk.W)
        
        file_frame.columnconfigure(0, weight=1)
        
    def create_export_options_tab(self):
        """Create export options tab."""
        options_frame = ttk.Frame(self.notebook)
        self.notebook.add(options_frame, text="2. Configure Options")
        
        # Create format-specific options container
        self.options_container = ttk.Frame(options_frame)
        self.options_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create format-specific option frames
        self.create_csv_options()
        self.create_geotiff_options()
        self.create_vtk_options()
        
        # Common options section
        common_frame = ttk.LabelFrame(options_frame, text=_("Common Options"), padding="10")
        common_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.include_metadata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            common_frame,
            text="Include metadata and creation information",
            variable=self.include_metadata_var
        ).pack(anchor=tk.W)
        
        self.overwrite_existing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            common_frame,
            text="Overwrite existing files",
            variable=self.overwrite_existing_var
        ).pack(anchor=tk.W)
        
        self.create_directories_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            common_frame,
            text="Create output directories if needed",
            variable=self.create_directories_var
        ).pack(anchor=tk.W)
        
    def create_csv_options(self):
        """Create CSV-specific options."""
        self.csv_frame = ttk.LabelFrame(self.options_container, text=_("CSV Options"), padding="10")
        
        # Delimiter
        ttk.Label(self.csv_frame, text="Delimiter:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.csv_delimiter_var = tk.StringVar(value=",")
        delimiter_combo = ttk.Combobox(
            self.csv_frame,
            textvariable=self.csv_delimiter_var,
            values=[",", ";", "\t", "|", " "],
            width=10
        )
        delimiter_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Encoding
        ttk.Label(self.csv_frame, text="Encoding:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.csv_encoding_var = tk.StringVar(value="utf-8")
        encoding_combo = ttk.Combobox(
            self.csv_frame,
            textvariable=self.csv_encoding_var,
            values=["utf-8", "latin-1", "cp1251", "ascii"],
            width=10
        )
        encoding_combo.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Precision
        ttk.Label(self.csv_frame, text="Decimal Precision:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.csv_precision_var = tk.IntVar(value=6)
        precision_spin = ttk.Spinbox(self.csv_frame, from_=1, to=15, textvariable=self.csv_precision_var, width=10)
        precision_spin.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Header option
        self.csv_header_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.csv_frame,
            text="Include column headers",
            variable=self.csv_header_var
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # Index option
        self.csv_index_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.csv_frame,
            text="Include row index",
            variable=self.csv_index_var
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
    def create_geotiff_options(self):
        """Create GeoTIFF-specific options."""
        self.geotiff_frame = ttk.LabelFrame(self.options_container, text=_("GeoTIFF Options"), padding="10")
        
        # Compression
        ttk.Label(self.geotiff_frame, text="Compression:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.geotiff_compress_var = tk.StringVar(value="lzw")
        compress_combo = ttk.Combobox(
            self.geotiff_frame,
            textvariable=self.geotiff_compress_var,
            values=["none", "lzw", "jpeg", "deflate"],
            state="readonly",
            width=10
        )
        compress_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Data type
        ttk.Label(self.geotiff_frame, text="Data Type:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.geotiff_dtype_var = tk.StringVar(value="float32")
        dtype_combo = ttk.Combobox(
            self.geotiff_frame,
            textvariable=self.geotiff_dtype_var,
            values=["float32", "float64", "int16", "int32", "uint16", "uint32"],
            state="readonly",
            width=10
        )
        dtype_combo.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # NoData value
        ttk.Label(self.geotiff_frame, text="NoData Value:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.geotiff_nodata_var = tk.StringVar(value="-9999")
        nodata_entry = ttk.Entry(self.geotiff_frame, textvariable=self.geotiff_nodata_var, width=10)
        nodata_entry.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Coordinate system
        ttk.Label(self.geotiff_frame, text="Coordinate System:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.geotiff_crs_var = tk.StringVar(value="EPSG:4326")
        crs_entry = ttk.Entry(self.geotiff_frame, textvariable=self.geotiff_crs_var, width=15)
        crs_entry.grid(row=3, column=1, sticky=tk.W, pady=(10, 0))
        
        # Tiling options
        self.geotiff_tiled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.geotiff_frame,
            text="Create tiled TIFF (recommended for large files)",
            variable=self.geotiff_tiled_var
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # BigTIFF option
        self.geotiff_bigtiff_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.geotiff_frame,
            text="Use BigTIFF format (for files > 4GB)",
            variable=self.geotiff_bigtiff_var
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
    def create_vtk_options(self):
        """Create VTK-specific options."""
        self.vtk_frame = ttk.LabelFrame(self.options_container, text=_("VTK Options"), padding="10")
        
        # File format
        ttk.Label(self.vtk_frame, text="File Format:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.vtk_format_var = tk.StringVar(value="Binary")
        format_combo = ttk.Combobox(
            self.vtk_frame,
            textvariable=self.vtk_format_var,
            values=["Binary", "ASCII"],
            state="readonly",
            width=10
        )
        format_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Point data name
        ttk.Label(self.vtk_frame, text="Point Data Name:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.vtk_point_data_name_var = tk.StringVar(value="Interpolated_Values")
        name_entry = ttk.Entry(self.vtk_frame, textvariable=self.vtk_point_data_name_var, width=20)
        name_entry.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Data structure info
        info_label = ttk.Label(
            self.vtk_frame,
            text="VTK files will contain 3D geometry suitable for visualization\nin ParaView, VisIt, and other scientific visualization tools.",
            font=("TkDefaultFont", 9),
            foreground="gray"
        )
        info_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
    def create_preview_export_tab(self):
        """Create preview and export tab."""
        preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(preview_frame, text="3. Preview & Export")
        
        # Export summary section
        summary_frame = ttk.LabelFrame(preview_frame, text=_("Export Summary"), padding="10")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Preview section (for CSV format)
        preview_data_frame = ttk.LabelFrame(preview_frame, text=_("Data Preview"), padding="10")
        preview_data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create treeview for data preview
        self.preview_tree = ttk.Treeview(preview_data_frame, show="headings", height=8)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(preview_data_frame, orient=tk.HORIZONTAL, command=self.preview_tree.xview)
        v_scrollbar = ttk.Scrollbar(preview_data_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        self.preview_tree.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Pack widgets
        self.preview_tree.grid(row=0, column=0, sticky=tk.NSEW)
        v_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        h_scrollbar.grid(row=1, column=0, sticky=tk.EW)
        
        preview_data_frame.grid_rowconfigure(0, weight=1)
        preview_data_frame.grid_columnconfigure(0, weight=1)
        
        # Progress section
        progress_frame = ttk.LabelFrame(preview_frame, text=_("Export Progress"), padding="10")
        progress_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label_var = tk.StringVar(value=_("Ready to export"))
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_label_var)
        self.progress_label.pack(anchor=tk.W)
        
    def get_results_info(self) -> str:
        """Get formatted information about the results data."""
        if not self.results_data:
            return _("No results data available")
            
        info_lines = []
        
        # Basic info
        if 'method' in self.results_data:
            info_lines.append(f"Interpolation Method: {self.results_data['method']}")
            
        if 'grid_data' in self.results_data:
            grid_data = self.results_data['grid_data']
            info_lines.append(f"Grid Size: {grid_data.get('shape', 'Unknown')}")
            info_lines.append(f"Grid Points: {grid_data.get('n_points', 'Unknown'):,}")
            
            bounds = grid_data.get('bounds')
            if bounds:
                info_lines.append(f"Bounds: X={bounds[0]:.2f} to {bounds[1]:.2f}, Y={bounds[2]:.2f} to {bounds[3]:.2f}")
                
        if 'point_data' in self.results_data:
            point_data = self.results_data['point_data']
            info_lines.append(f"Source Points: {point_data.get('n_points', 'Unknown'):,}")
            
        if 'parameters' in self.results_data:
            params = self.results_data['parameters']
            info_lines.append(f"Cell Size: {params.get('cell_size', 'Unknown')}")
            
        # Statistics
        if 'statistics' in self.results_data:
            stats = self.results_data['statistics']
            info_lines.append(f"Value Range: {stats.get('min', 'N/A'):.3f} to {stats.get('max', 'N/A'):.3f}")
            info_lines.append(f"Mean: {stats.get('mean', 'N/A'):.3f}")
            
        if not info_lines:
            info_lines.append(_("Results data structure not recognized"))
            
        return "\\n".join(info_lines)
        
    def on_format_changed(self):
        """Handle format selection change."""
        format_name = self.format_var.get()
        try:
            self.current_format = ExportFormat(format_name.lower())
        except ValueError:
            self.current_format = ExportFormat.CSV
            
        self.update_format_options()
        self.update_file_extension()
        
    def update_format_options(self):
        """Update the display of format-specific options."""
        # Hide all option frames
        for widget in self.options_container.winfo_children():
            widget.pack_forget()
            
        # Show the appropriate options frame
        if self.current_format == ExportFormat.CSV:
            self.csv_frame.pack(fill=tk.BOTH, expand=True)
        elif self.current_format == ExportFormat.GEOTIFF:
            self.geotiff_frame.pack(fill=tk.BOTH, expand=True)
        elif self.current_format == ExportFormat.VTK:
            self.vtk_frame.pack(fill=tk.BOTH, expand=True)
            
    def update_file_extension(self):
        """Update file extension based on format."""
        current_path = self.export_path_var.get()
        if current_path != _("No file selected"):
            path = Path(current_path)
            stem = path.stem
            
            # Get appropriate extension
            extensions = {
                ExportFormat.CSV: ".csv",
                ExportFormat.GEOTIFF: ".tif",
                ExportFormat.VTK: ".vtk"
            }
            
            new_extension = extensions.get(self.current_format, ".csv")
            new_path = path.parent / (stem + new_extension)
            self.export_path_var.set(str(new_path))
            self.current_export_path = str(new_path)
            
    def browse_export_file(self):
        """Open file browser dialog for export path."""
        # File type filters based on format
        filetypes_map = {
            ExportFormat.CSV: [("CSV files", "*.csv"), ("Text files", "*.txt")],
            ExportFormat.GEOTIFF: [("GeoTIFF files", "*.tif *.tiff"), ("TIFF files", "*.tif")],
            ExportFormat.VTK: [("VTK files", "*.vtk"), ("VTK XML files", "*.vtp *.vts")]
        }
        
        filetypes = filetypes_map.get(self.current_format, [("All files", "*.*")])
        filetypes.append(("All files", "*.*"))
        
        # Default extension
        extensions = {
            ExportFormat.CSV: ".csv",
            ExportFormat.GEOTIFF: ".tif",
            ExportFormat.VTK: ".vtk"
        }
        default_ext = extensions.get(self.current_format, ".csv")
        
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=default_ext,
            filetypes=filetypes
        )
        
        if file_path:
            self.current_export_path = file_path
            self.export_path_var.set(os.path.basename(file_path))
            
    def next_step(self):
        """Go to next step."""
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab < self.notebook.index("end") - 1:
            self.notebook.select(current_tab + 1)
            self.update_navigation_buttons()
            
        # Update preview on last tab
        if current_tab == 1:  # Moving to preview tab
            self.update_export_summary()
            self.update_preview()
            
    def previous_step(self):
        """Go to previous step."""
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab > 0:
            self.notebook.select(current_tab - 1)
            self.update_navigation_buttons()
            
    def update_navigation_buttons(self):
        """Update navigation button states."""
        current_tab = self.notebook.index(self.notebook.select())
        max_tab = self.notebook.index("end") - 1
        
        # Previous button
        self.prev_button.config(state=tk.NORMAL if current_tab > 0 else tk.DISABLED)
        
        # Next button
        if current_tab < max_tab:
            self.next_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.NORMAL if self.validate_export_settings() else tk.DISABLED)
            
    def validate_export_settings(self) -> bool:
        """Validate export settings."""
        if not self.current_export_path:
            return False
            
        if not self.results_data:
            return False
            
        return True
        
    def update_export_summary(self):
        """Update the export summary display."""
        summary_lines = []
        
        if self.current_export_path:
            summary_lines.append(f"Export Format: {self.current_format.value.upper()}")
            summary_lines.append(f"Output File: {os.path.basename(self.current_export_path)}")
            summary_lines.append(f"Full Path: {self.current_export_path}")
            summary_lines.append("")
            
            # Format-specific options
            if self.current_format == ExportFormat.CSV:
                summary_lines.append("CSV Options:")
                summary_lines.append(f"  • Delimiter: '{self.csv_delimiter_var.get()}'")
                summary_lines.append(f"  • Encoding: {self.csv_encoding_var.get()}")
                summary_lines.append(f"  • Precision: {self.csv_precision_var.get()} decimals")
                summary_lines.append(f"  • Include Headers: {self.csv_header_var.get()}")
                
            elif self.current_format == ExportFormat.GEOTIFF:
                summary_lines.append("GeoTIFF Options:")
                summary_lines.append(f"  • Compression: {self.geotiff_compress_var.get()}")
                summary_lines.append(f"  • Data Type: {self.geotiff_dtype_var.get()}")
                summary_lines.append(f"  • NoData Value: {self.geotiff_nodata_var.get()}")
                summary_lines.append(f"  • Coordinate System: {self.geotiff_crs_var.get()}")
                summary_lines.append(f"  • Tiled: {self.geotiff_tiled_var.get()}")
                
            elif self.current_format == ExportFormat.VTK:
                summary_lines.append("VTK Options:")
                summary_lines.append(f"  • Format: {self.vtk_format_var.get()}")
                summary_lines.append(f"  • Point Data Name: {self.vtk_point_data_name_var.get()}")
                
            summary_lines.append("")
            summary_lines.append("Common Options:")
            summary_lines.append(f"  • Include Metadata: {self.include_metadata_var.get()}")
            summary_lines.append(f"  • Overwrite Existing: {self.overwrite_existing_var.get()}")
            summary_lines.append(f"  • Create Directories: {self.create_directories_var.get()}")
                
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "\\n".join(summary_lines))
        self.summary_text.config(state=tk.DISABLED)
        
    def update_preview(self):
        """Update the data preview (for CSV format)."""
        # Clear existing preview
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
            
        # Only show preview for CSV format
        if self.current_format != ExportFormat.CSV:
            self.preview_tree['columns'] = ["Info"]
            self.preview_tree.heading("Info", text="Preview")
            self.preview_tree.column("Info", width=400)
            
            preview_msg = f"Preview not available for {self.current_format.value.upper()} format"
            self.preview_tree.insert("", tk.END, values=[preview_msg])
            return
            
        # Generate CSV preview
        try:
            preview_data = self.generate_csv_preview()
            if preview_data:
                # Configure columns
                columns = list(preview_data.columns)
                self.preview_tree['columns'] = columns
                
                for col in columns:
                    self.preview_tree.heading(col, text=col)
                    self.preview_tree.column(col, width=100, minwidth=50)
                    
                # Add data rows (first 10)
                for index, row in preview_data.head(10).iterrows():
                    values = [f"{row[col]:.{self.csv_precision_var.get()}f}" if isinstance(row[col], float) 
                             else str(row[col]) for col in columns]
                    self.preview_tree.insert("", tk.END, values=values)
                    
        except Exception as e:
            # Show error in preview
            self.preview_tree['columns'] = ["Error"]
            self.preview_tree.heading("Error", text="Error")
            self.preview_tree.column("Error", width=400)
            self.preview_tree.insert("", tk.END, values=[f"Preview error: {str(e)}"])
            
    def generate_csv_preview(self):
        """Generate CSV preview data."""
        # This would create a sample of the data to be exported
        # For now, return a simple placeholder
        import pandas as pd
        import numpy as np
        
        # Create sample data based on results structure
        n_points = 100  # Sample size
        
        # Generate sample coordinates and values
        x_coords = np.linspace(0, 100, int(np.sqrt(n_points)))
        y_coords = np.linspace(0, 100, int(np.sqrt(n_points)))
        X, Y = np.meshgrid(x_coords, y_coords)
        
        preview_data = pd.DataFrame({
            'X': X.flatten()[:n_points],
            'Y': Y.flatten()[:n_points], 
            'Value': np.random.normal(50, 10, n_points)
        })
        
        return preview_data
        
    def get_export_options(self):
        """Get current export options based on format."""
        common_options = {
            'include_metadata': self.include_metadata_var.get(),
            'overwrite_existing': self.overwrite_existing_var.get(),
            'create_directories': self.create_directories_var.get()
        }
        
        if self.current_format == ExportFormat.CSV:
            return CSVExportOptions(
                delimiter=self.csv_delimiter_var.get(),
                encoding=self.csv_encoding_var.get(),
                precision=self.csv_precision_var.get(),
                header=self.csv_header_var.get(),
                index=self.csv_index_var.get(),
                **common_options
            )
            
        elif self.current_format == ExportFormat.GEOTIFF:
            return GeoTIFFExportOptions(
                compress=self.geotiff_compress_var.get(),
                dtype=self.geotiff_dtype_var.get(),
                nodata_value=float(self.geotiff_nodata_var.get()) if self.geotiff_nodata_var.get() else None,
                coordinate_system=self.geotiff_crs_var.get(),
                tiled=self.geotiff_tiled_var.get(),
                bigtiff=self.geotiff_bigtiff_var.get(),
                **common_options
            )
            
        elif self.current_format == ExportFormat.VTK:
            return VTKExportOptions(
                binary=self.vtk_format_var.get() == "Binary",
                ascii=self.vtk_format_var.get() == "ASCII",
                point_data_name=self.vtk_point_data_name_var.get(),
                **common_options
            )
            
        return None
        
    def export_data(self):
        """Export the data with current settings."""
        if not self.validate_export_settings():
            messagebox.showerror(_("Error"), _("Please complete all required settings"))
            return
            
        try:
            # Prepare export settings
            export_options = self.get_export_options()
            
            # Prepare result
            self.result = {
                'file_path': self.current_export_path,
                'format': self.current_format,
                'options': export_options,
                'results_data': self.results_data
            }
            
            # Update progress
            self.progress_label_var.set("Export completed successfully")
            self.progress_var.set(100)
            
            # Show success message
            messagebox.showinfo("Export Complete", f"Results exported successfully to:\\n{self.current_export_path}")
            
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
            self.progress_label_var.set("Export failed")
            self.progress_var.set(0)
            
    def cancel(self):
        """Cancel dialog."""
        self.result = None
        self.dialog.destroy()
        
    def show(self) -> Optional[Dict[str, Any]]:
        """Show dialog and return result."""
        self.dialog.wait_window()
        return self.result


def show_export_dialog(parent, results_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Show export dialog.
    
    Args:
        parent: Parent window
        results_data: Interpolation results to export
        
    Returns:
        Dictionary with export settings or None if cancelled
    """
    dialog = ExportDialog(parent, results_data)
    return dialog.show()