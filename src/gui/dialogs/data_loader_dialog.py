"""
Advanced data loader dialog with preview and validation.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# Import i18n system
from ...i18n import _, add_language_change_listener, remove_language_change_listener


class DataLoaderDialog:
    """
    Advanced dialog for loading and previewing data files.
    
    This dialog provides:
    - File selection with preview
    - Column mapping and validation
    - Data type detection
    - Import settings configuration
    - Data quality assessment
    """
    
    def __init__(self, parent, controller):
        """
        Initialize the data loader dialog.
        
        Args:
            parent: Parent window
            controller: Application controller instance
        """
        self.parent = parent
        self.controller = controller
        self.result = None
        self.current_file_path = None
        self.preview_data = None
        
        # Store UI elements for language updates
        self.ui_elements = {}
        
        # Create modal dialog
        self.dialog = tk.Toplevel(parent)
        self.setup_dialog()
        self.create_widgets()
        
        # Initialize navigation button states
        self.update_navigation_buttons()
        
        # Register for language change events
        from ...i18n import add_language_change_listener
        add_language_change_listener(self.update_language)
        
    def setup_dialog(self):
        """Setup dialog window properties."""
        self.dialog.title(_("Load Data File"))
        self.dialog.geometry("800x600")
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
        
        # Step 1: File Selection
        self.create_file_selection_tab()
        
        # Step 2: Data Preview
        self.create_data_preview_tab()
        
        # Step 3: Column Mapping
        self.create_column_mapping_tab()
        
        # Step 4: Import Settings
        self.create_import_settings_tab()
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.dialog)
        buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(
            buttons_frame,
            text=_("Cancel"),
            command=self.cancel
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(
            buttons_frame,
            text=_("Load Data"),
            command=self.load_data,
            state=tk.DISABLED
        ).pack(side=tk.RIGHT)
        
        self.load_button = buttons_frame.winfo_children()[-1]
        
        ttk.Button(
            buttons_frame,
            text=_("Previous"),
            command=self.previous_step,
            state=tk.DISABLED
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            buttons_frame,
            text=_("Next"),
            command=self.next_step,
            state=tk.DISABLED
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        self.prev_button = buttons_frame.winfo_children()[-2]
        self.next_button = buttons_frame.winfo_children()[-1]
        
    def create_file_selection_tab(self):
        """Create file selection tab."""
        file_frame = ttk.Frame(self.notebook)
        self.notebook.add(file_frame, text=_("1. Select File"))
        
        # File selection section
        select_frame = ttk.LabelFrame(file_frame, text=_("File Selection"), padding="10")
        select_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Current file display
        ttk.Label(select_frame, text=_("Selected File:")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.file_path_var = tk.StringVar(value=_("No file selected"))
        file_label = ttk.Label(select_frame, textvariable=self.file_path_var, relief=tk.SUNKEN, padding="5")
        file_label.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(0, 10))
        
        # Browse button
        ttk.Button(
            select_frame,
            text=_("Browse..."),
            command=self.browse_file
        ).grid(row=2, column=0, sticky=tk.W)
        
        # Recent files
        recent_frame = ttk.LabelFrame(file_frame, text=_("Recent Files"), padding="10")
        recent_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Recent files listbox
        self.recent_listbox = tk.Listbox(recent_frame, height=8)
        recent_scrollbar = ttk.Scrollbar(recent_frame, orient=tk.VERTICAL, command=self.recent_listbox.yview)
        self.recent_listbox.configure(yscrollcommand=recent_scrollbar.set)
        
        self.recent_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        recent_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection
        self.recent_listbox.bind('<<ListboxSelect>>', self.on_recent_select)
        
        # Load recent files
        self.load_recent_files()
        
        select_frame.columnconfigure(0, weight=1)
        
    def create_data_preview_tab(self):
        """Create data preview tab."""
        preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(preview_frame, text=_("2. Preview Data"))
        
        # File info section
        info_frame = ttk.LabelFrame(preview_frame, text=_("File Information"), padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Info labels
        self.file_size_var = tk.StringVar(value="-")
        self.file_type_var = tk.StringVar(value="-")
        self.rows_count_var = tk.StringVar(value="-")
        self.cols_count_var = tk.StringVar(value="-")
        
        ttk.Label(info_frame, text=_("File Size:")).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(info_frame, textvariable=self.file_size_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(info_frame, text=_("File Type:")).grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Label(info_frame, textvariable=self.file_type_var).grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(info_frame, text=_("Rows:")).grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(info_frame, textvariable=self.rows_count_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(info_frame, text=_("Columns:")).grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        ttk.Label(info_frame, textvariable=self.cols_count_var).grid(row=1, column=3, sticky=tk.W)
        
        # Data preview section
        preview_data_frame = ttk.LabelFrame(preview_frame, text=_("Data Preview (First 10 rows)"), padding="10")
        preview_data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create treeview for data preview
        self.preview_tree = ttk.Treeview(preview_data_frame, show="headings", height=10)
        
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
        
    def create_column_mapping_tab(self):
        """Create column mapping tab."""
        mapping_frame = ttk.Frame(self.notebook)
        self.notebook.add(mapping_frame, text=_("3. Map Columns"))
        
        # Instructions
        instructions = ttk.Label(
            mapping_frame,
            text=_("Map data columns to required coordinate and value fields:"),
            font=("TkDefaultFont", 10, "bold")
        )
        instructions.pack(anchor=tk.W, padx=10, pady=10)
        
        # Mapping section
        map_frame = ttk.LabelFrame(mapping_frame, text=_("Column Mapping"), padding="10")
        map_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # X coordinate mapping
        ttk.Label(map_frame, text=_("X Coordinate:")).grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.x_column_var = tk.StringVar()
        self.x_column_combo = ttk.Combobox(map_frame, textvariable=self.x_column_var, state="readonly", width=20)
        self.x_column_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.x_column_combo.bind('<<ComboboxSelected>>', lambda e: self.update_navigation_buttons())
        
        # Y coordinate mapping  
        ttk.Label(map_frame, text=_("Y Coordinate:")).grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.y_column_var = tk.StringVar()
        self.y_column_combo = ttk.Combobox(map_frame, textvariable=self.y_column_var, state="readonly", width=20)
        self.y_column_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.y_column_combo.bind('<<ComboboxSelected>>', lambda e: self.update_navigation_buttons())
        
        # Value columns section
        value_frame = ttk.LabelFrame(mapping_frame, text=_("Value Columns"), padding="10")
        value_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Available columns
        ttk.Label(value_frame, text=_("Available Columns:")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Label(value_frame, text=_("Selected for Interpolation:")).grid(row=0, column=2, sticky=tk.W, pady=(0, 5))
        
        # Listboxes for column selection
        self.available_listbox = tk.Listbox(value_frame, height=8, selectmode=tk.MULTIPLE)
        self.available_listbox.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 10))
        
        self.selected_listbox = tk.Listbox(value_frame, height=8)
        self.selected_listbox.grid(row=1, column=2, sticky=tk.NSEW, padx=(10, 0))
        
        # Buttons for moving columns
        buttons_frame = ttk.Frame(value_frame)
        buttons_frame.grid(row=1, column=1, padx=5)
        
        ttk.Button(buttons_frame, text=_("Add →"), command=self.add_value_column).pack(pady=2)
        ttk.Button(buttons_frame, text=_("← Remove"), command=self.remove_value_column).pack(pady=2)
        ttk.Button(buttons_frame, text=_("Add All →"), command=self.add_all_value_columns).pack(pady=2)
        ttk.Button(buttons_frame, text=_("← Remove All"), command=self.remove_all_value_columns).pack(pady=2)
        
        value_frame.grid_rowconfigure(1, weight=1)
        value_frame.grid_columnconfigure(0, weight=1)
        value_frame.grid_columnconfigure(2, weight=1)
        
    def create_import_settings_tab(self):
        """Create import settings tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text=_("4. Settings"))
        
        # CSV settings
        csv_frame = ttk.LabelFrame(settings_frame, text=_("CSV Settings"), padding="10")
        csv_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Delimiter
        ttk.Label(csv_frame, text=_("Delimiter:")).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.delimiter_var = tk.StringVar(value=",")
        delimiter_combo = ttk.Combobox(
            csv_frame, 
            textvariable=self.delimiter_var,
            values=[",", ";", "\t", "|", " "],
            width=10
        )
        delimiter_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Header row
        ttk.Label(csv_frame, text=_("Header Row:")).grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.header_row_var = tk.IntVar(value=0)
        header_spin = ttk.Spinbox(csv_frame, from_=0, to=10, textvariable=self.header_row_var, width=10)
        header_spin.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Encoding
        ttk.Label(csv_frame, text=_("Encoding:")).grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.encoding_var = tk.StringVar(value="utf-8")
        encoding_combo = ttk.Combobox(
            csv_frame,
            textvariable=self.encoding_var,
            values=["utf-8", "latin-1", "cp1251", "ascii"],
            width=10
        )
        encoding_combo.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Data validation settings
        validation_frame = ttk.LabelFrame(settings_frame, text=_("Data Validation"), padding="10")
        validation_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.skip_invalid_rows_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            validation_frame,
            text=_("Skip rows with invalid coordinates"),
            variable=self.skip_invalid_rows_var
        ).pack(anchor=tk.W)
        
        self.fill_missing_values_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            validation_frame,
            text=_("Fill missing values with interpolation"),
            variable=self.fill_missing_values_var
        ).pack(anchor=tk.W)
        
        self.remove_duplicates_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            validation_frame,
            text=_("Remove duplicate coordinates"),
            variable=self.remove_duplicates_var
        ).pack(anchor=tk.W)
        
        # Summary section
        summary_frame = ttk.LabelFrame(settings_frame, text=_("Import Summary"), padding="10")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.summary_text = tk.Text(summary_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
    def browse_file(self):
        """Open file browser dialog."""
        file_path = filedialog.askopenfilename(
            title=_("Select Data File"),
            filetypes=[
                (_("CSV files"), "*.csv"),
                (_("Excel files"), "*.xlsx *.xls"),
                (_("Text files"), "*.txt"),
                (_("All files"), "*.*")
            ]
        )
        
        if file_path:
            self.select_file(file_path)
            
    def select_file(self, file_path: str):
        """Select and preview a file."""
        try:
            self.current_file_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
            
            # Update file info
            self.update_file_info()
            
            # Load preview
            self.load_file_preview()
            
            # Update column mappings
            self.update_column_mappings()
            
            # Update navigation buttons
            self.update_navigation_buttons()
            
            # Add to recent files
            self.add_to_recent_files(file_path)
            
        except Exception as e:
            messagebox.showerror(_("Error"), f"{_('Failed to load file')}: {str(e)}")
            
    def update_file_info(self):
        """Update file information display."""
        if not self.current_file_path:
            return
            
        file_path = Path(self.current_file_path)
        
        # File size
        size_bytes = file_path.stat().st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            
        self.file_size_var.set(size_str)
        self.file_type_var.set(file_path.suffix.upper())
        
    def load_file_preview(self):
        """Load and display file preview."""
        if not self.current_file_path:
            return
            
        try:
            # For now, assume CSV format
            import pandas as pd
            
            # Prepare read parameters
            read_params = {
                'filepath_or_buffer': self.current_file_path,
                'nrows': 10,
                'delimiter': self.delimiter_var.get(),
                'encoding': self.encoding_var.get()
            }
            
            # Handle header parameter
            header_row = self.header_row_var.get()
            if header_row is not None and header_row > 0:
                read_params['header'] = header_row
            else:
                read_params['header'] = 0
            
            # Read first few rows for preview
            preview_df = pd.read_csv(**read_params)
            
            # Validate the result
            if preview_df is None:
                raise ValueError("Failed to read CSV file - no data returned")
            
            if len(preview_df) == 0:
                raise ValueError("CSV file appears to be empty")
            
            self.preview_data = preview_df
            
            # Get full file info (just count, don't load all data)
            try:
                # Count lines efficiently
                with open(self.current_file_path, 'r', encoding=self.encoding_var.get()) as f:
                    line_count = sum(1 for _ in f)
                    # Subtract 1 for header if present
                    if read_params.get('header', 0) == 0:
                        data_rows = line_count - 1
                    else:
                        data_rows = line_count - read_params['header'] - 1
                
                self.rows_count_var.set(f"{max(0, data_rows):,}")
                self.cols_count_var.set(str(len(preview_df.columns)))
                
            except Exception:
                # Fallback: read small sample to get column count
                self.rows_count_var.set("Unknown")
                self.cols_count_var.set(str(len(preview_df.columns)))
            
            # Update preview tree
            self.update_preview_tree(preview_df)
            
            # Show success message
            print(f"Successfully loaded preview: {preview_df.shape}")
            
        except UnicodeDecodeError as e:
            error_msg = f"{_('Encoding error: Try different encoding (UTF-8, cp1251, etc.)')}\nDetails: {str(e)}"
            messagebox.showerror(_("Encoding Error"), error_msg)
            
        except pd.errors.EmptyDataError:
            messagebox.showerror(_("Error"), _("The CSV file appears to be empty"))
            
        except pd.errors.ParserError as e:
            error_msg = f"{_('CSV parsing error: Check delimiter and file format')}\nDetails: {str(e)}"
            messagebox.showerror(_("Parsing Error"), error_msg)
            
        except FileNotFoundError:
            messagebox.showerror(_("Error"), f"{_('File not found')}: {self.current_file_path}")
            
        except Exception as e:
            error_msg = f"{_('Failed to preview file: {error}').format(error=str(e))}\n\n{_('Troubleshooting tips:')}\n{_('• Check file encoding (try UTF-8 or cp1251)')}\n{_('• Verify delimiter (comma, semicolon, tab)')}\n{_('• Ensure file is not corrupted')}"
            messagebox.showerror(_("Error"), error_msg)
            print(f"Preview error: {e}")
            import traceback
            traceback.print_exc()
            
    def update_preview_tree(self, df):
        """Update the preview treeview with data."""
        # Clear existing data
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
            
        if df is None or df.empty:
            return
            
        # Configure columns
        columns = list(df.columns)
        self.preview_tree['columns'] = columns
        
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=100, minwidth=50)
            
        # Add data rows
        for index, row in df.iterrows():
            values = [str(row[col]) for col in columns]
            self.preview_tree.insert("", tk.END, values=values)
            
    def update_column_mappings(self):
        """Update column mapping options."""
        if self.preview_data is None or self.preview_data.empty:
            return
            
        columns = list(self.preview_data.columns)
        
        # Update coordinate combo boxes
        self.x_column_combo['values'] = columns
        self.y_column_combo['values'] = columns
        
        # Auto-detect coordinate columns
        x_candidates = []
        y_candidates = []
        
        for col in columns:
            col_lower = col.lower()
            # Check for X/East coordinate
            if any(keyword in col_lower for keyword in ['x', 'east', 'easting', 'долгота', 'восток']):
                x_candidates.append(col)
            # Check for Y/North coordinate
            elif any(keyword in col_lower for keyword in ['y', 'north', 'northing', 'широта', 'север']):
                y_candidates.append(col)
        
        # Set the first match for each coordinate
        if x_candidates and not self.x_column_var.get():
            self.x_column_var.set(x_candidates[0])
        if y_candidates and not self.y_column_var.get():
            self.y_column_var.set(y_candidates[0])
                
        # Update available columns list
        self.available_listbox.delete(0, tk.END)
        self.selected_listbox.delete(0, tk.END)
        
        # Add all non-coordinate columns to available list first
        available_cols = []
        for col in columns:
            if col not in [self.x_column_var.get(), self.y_column_var.get()]:
                available_cols.append(col)
                self.available_listbox.insert(tk.END, col)
        
        # Auto-select all available columns for interpolation (user can remove if needed)
        if available_cols:
            self.add_all_value_columns()
                
    def add_value_column(self):
        """Add selected column to value columns."""
        selection = self.available_listbox.curselection()
        for index in reversed(selection):
            item = self.available_listbox.get(index)
            self.selected_listbox.insert(tk.END, item)
            self.available_listbox.delete(index)
        self.update_navigation_buttons()
            
    def remove_value_column(self):
        """Remove selected column from value columns."""
        selection = self.selected_listbox.curselection()
        for index in reversed(selection):
            item = self.selected_listbox.get(index)
            self.available_listbox.insert(tk.END, item)
            self.selected_listbox.delete(index)
        self.update_navigation_buttons()
            
    def add_all_value_columns(self):
        """Add all available columns to value columns."""
        items = list(self.available_listbox.get(0, tk.END))
        self.available_listbox.delete(0, tk.END)
        for item in items:
            self.selected_listbox.insert(tk.END, item)
        self.update_navigation_buttons()
            
    def remove_all_value_columns(self):
        """Remove all value columns."""
        items = list(self.selected_listbox.get(0, tk.END))
        self.selected_listbox.delete(0, tk.END)
        for item in items:
            self.available_listbox.insert(tk.END, item)
        self.update_navigation_buttons()
            
    def next_step(self):
        """Go to next step."""
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab < self.notebook.index("end") - 1:
            self.notebook.select(current_tab + 1)
            self.update_navigation_buttons()
            
        # Update summary on last tab
        if current_tab == 2:  # Moving to settings tab
            self.update_import_summary()
            
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
        
        # Next button logic based on current tab
        if current_tab < max_tab:
            # Check if current step can proceed
            can_proceed = self.can_proceed_from_current_tab(current_tab)
            self.next_button.config(state=tk.NORMAL if can_proceed else tk.DISABLED)
            self.load_button.config(state=tk.DISABLED)
        else:
            # On final tab - check complete validation
            self.next_button.config(state=tk.DISABLED)
            self.load_button.config(state=tk.NORMAL if self.validate_settings() else tk.DISABLED)
            
    def can_proceed_from_current_tab(self, current_tab: int) -> bool:
        """Check if user can proceed from current tab."""
        if current_tab == 0:  # File selection tab
            return bool(self.current_file_path)
        elif current_tab == 1:  # Preview tab
            return bool(self.preview_data is not None)
        elif current_tab == 2:  # Column mapping tab
            # Allow proceeding even without value columns selected - user might want to select them later
            return bool(self.x_column_var.get() and self.y_column_var.get())
        else:
            return True
            
    def validate_settings(self) -> bool:
        """Validate import settings."""
        if not self.current_file_path:
            return False
            
        if not self.x_column_var.get() or not self.y_column_var.get():
            return False
            
        if self.selected_listbox.size() == 0:
            return False
            
        return True
        
    def update_import_summary(self):
        """Update the import summary display."""
        summary_lines = []
        
        if self.current_file_path:
            summary_lines.append(f"File: {os.path.basename(self.current_file_path)}")
            summary_lines.append(f"X Column: {self.x_column_var.get()}")
            summary_lines.append(f"Y Column: {self.y_column_var.get()}")
            
            value_cols = list(self.selected_listbox.get(0, tk.END))
            summary_lines.append(f"Value Columns ({len(value_cols)}): {', '.join(value_cols)}")
            
            summary_lines.append(f"Rows: {self.rows_count_var.get()}")
            summary_lines.append(f"Delimiter: '{self.delimiter_var.get()}'")
            summary_lines.append(f"Encoding: {self.encoding_var.get()}")
            
            # Validation options
            validations = []
            if self.skip_invalid_rows_var.get():
                validations.append("Skip invalid rows")
            if self.fill_missing_values_var.get():
                validations.append("Fill missing values")
            if self.remove_duplicates_var.get():
                validations.append("Remove duplicates")
                
            if validations:
                summary_lines.append(f"Validation: {', '.join(validations)}")
                
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "\n".join(summary_lines))
        self.summary_text.config(state=tk.DISABLED)
        
    def load_recent_files(self):
        """Load recent files list."""
        # This would load from configuration file
        recent_files = [
            "sample_data.csv",
            "coal_analysis_2024.csv", 
            "borehole_survey.csv",
            "geological_data.xlsx"
        ]
        
        for file in recent_files:
            self.recent_listbox.insert(tk.END, file)
            
    def on_recent_select(self, event):
        """Handle recent file selection."""
        selection = event.widget.curselection()
        if selection:
            # This would load the actual file path
            messagebox.showinfo(_("Info"), _("Recent file selection - To be implemented"))
            
    def add_to_recent_files(self, file_path: str):
        """Add file to recent files list."""
        # This would save to configuration
        pass
        
    def load_data(self):
        """Load the data with current settings."""
        if not self.validate_settings():
            messagebox.showerror(_("Error"), _("Please complete all required settings"))
            return
            
        try:
            # Prepare result
            self.result = {
                'file_path': self.current_file_path,
                'x_column': self.x_column_var.get(),
                'y_column': self.y_column_var.get(),
                'value_columns': list(self.selected_listbox.get(0, tk.END)),
                'delimiter': self.delimiter_var.get(),
                'encoding': self.encoding_var.get(),
                'header_row': self.header_row_var.get(),
                'skip_invalid_rows': self.skip_invalid_rows_var.get(),
                'fill_missing_values': self.fill_missing_values_var.get(),
                'remove_duplicates': self.remove_duplicates_var.get()
            }
            
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror(_("Error"), f"{_('Failed to load data')}: {str(e)}")
            
    def cancel(self):
        """Cancel dialog."""
        self.result = None
        # Unregister from language change events
        try:
            from ...i18n import remove_language_change_listener
            remove_language_change_listener(self.update_language)
        except:
            pass
        self.dialog.destroy()
        
    def update_language(self):
        """Update all text elements with new translations."""
        try:
            # Update dialog title
            self.dialog.title(_("Load Data File"))
            
            # Update notebook tab titles
            for i, tab_id in enumerate(['1. Select File', '2. Preview Data', '3. Map Columns', '4. Settings']):
                self.notebook.tab(i, text=_(tab_id))
            
            # Update "No file selected" text if currently displayed
            if self.file_path_var.get() in ["No file selected", "Файл не выбран"]:
                self.file_path_var.set(_("No file selected"))
                
            print("DataLoaderDialog language updated successfully")
            
        except Exception as e:
            print(f"Error updating DataLoaderDialog language: {e}")
            import traceback
            traceback.print_exc()
        
    def show(self) -> Optional[Dict[str, Any]]:
        """Show dialog and return result."""
        self.dialog.wait_window()
        return self.result


def show_data_loader_dialog(parent, controller) -> Optional[Dict[str, Any]]:
    """
    Show data loader dialog.
    
    Args:
        parent: Parent window
        controller: Application controller
        
    Returns:
        Dictionary with load settings or None if cancelled
    """
    dialog = DataLoaderDialog(parent, controller)
    return dialog.show()