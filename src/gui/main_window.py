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
from .dialogs.language_settings_dialog import show_language_settings_dialog

# Import i18n system
from ..i18n import _, add_language_change_listener, remove_language_change_listener


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
        
        # Initialize state and UI element storage first
        self.current_file = None
        self.interpolation_results = None
        self.ui_elements = {}
        
        # Window configuration
        self.setup_window()
        
        # Create UI components
        self.create_menu_bar()
        self.create_toolbar()
        self.create_main_panels()
        self.create_status_bar()
        
        # Setup event bindings
        self.setup_event_bindings()
        
        # Register for language change events
        add_language_change_listener(self.on_language_changed)
        
    def setup_window(self):
        """Configure the main window properties."""
        self.root.title(_("Coal Deposit Interpolation Tool") + " - v0.4.0")
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
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        # File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("File"), menu=self.file_menu)
        self.file_menu.add_command(label=_("Open") + "...", command=self.open_file, accelerator="Ctrl+O")
        self.file_menu.add_separator()
        self.file_menu.add_command(label=_("Save") + " " + _("Project") + "...", command=self.save_project, accelerator="Ctrl+S")
        self.file_menu.add_command(label=_("Load Data") + "...", command=self.load_project)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=_("Export Results") + "...", command=self.export_results, accelerator="Ctrl+E")
        self.file_menu.add_separator()
        self.file_menu.add_command(label=_("Exit"), command=self.on_closing, accelerator="Ctrl+Q")
        
        # View menu
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("View"), menu=self.view_menu)
        self.view_menu.add_command(label=_("Reset") + " " + _("Layout"), command=self.reset_layout)
        self.view_menu.add_separator()
        self.view_menu.add_checkbutton(label=_("Show") + " " + _("Data") + " " + _("Panel"), variable=tk.BooleanVar(value=True))
        self.view_menu.add_checkbutton(label=_("Show") + " " + _("Parameters") + " " + _("Panel"), variable=tk.BooleanVar(value=True))
        self.view_menu.add_checkbutton(label=_("Show") + " " + _("Visualization") + " " + _("Panel"), variable=tk.BooleanVar(value=True))
        self.view_menu.add_checkbutton(label=_("Show") + " " + _("Results") + " " + _("Panel"), variable=tk.BooleanVar(value=True))
        
        # Tools menu
        self.tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("Tools"), menu=self.tools_menu)
        self.tools_menu.add_command(label=_("Data") + " " + _("Statistics"), command=self.show_data_statistics)
        self.tools_menu.add_command(label=_("Grid") + " " + _("Information"), command=self.show_grid_info)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label=_("Language Settings"), command=self.show_language_settings)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label=_("Preferences"), command=self.show_preferences)
        
        # Help menu
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("Help"), menu=self.help_menu)
        self.help_menu.add_command(label=_("User Guide"), command=self.show_help)
        self.help_menu.add_command(label=_("About"), command=self.show_about)
        
        # Store menu references for language updates
        self.ui_elements['menus'] = {
            'file': self.file_menu,
            'view': self.view_menu, 
            'tools': self.tools_menu,
            'help': self.help_menu
        }
        
        # Bind keyboard shortcuts
        self.root.bind_all("<Control-o>", lambda e: self.open_file())
        self.root.bind_all("<Control-s>", lambda e: self.save_project())
        self.root.bind_all("<Control-e>", lambda e: self.export_results())
        self.root.bind_all("<Control-q>", lambda e: self.on_closing())
        
    def create_toolbar(self):
        """Create the main toolbar."""
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # Create a frame for the right side (language controls)
        right_frame = ttk.Frame(toolbar_frame)
        right_frame.pack(side=tk.RIGHT, padx=5)
        
        # Language toggle button
        self.create_language_toggle_button(right_frame)
        
        # File operations
        self.open_btn = ttk.Button(toolbar_frame, text=_("Open"), command=self.open_file)
        self.open_btn.pack(side=tk.LEFT, padx=2)
        
        self.save_btn = ttk.Button(toolbar_frame, text=_("Save"), command=self.save_project)
        self.save_btn.pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Interpolation operations
        self.interpolate_btn = ttk.Button(
            toolbar_frame, 
            text=_("Interpolate"), 
            command=self.run_interpolation,
            state=tk.DISABLED
        )
        self.interpolate_btn.pack(side=tk.LEFT, padx=2)
        
        self.export_btn = ttk.Button(toolbar_frame, text=_("Export"), command=self.export_results)
        self.export_btn.pack(side=tk.LEFT, padx=2)
        
        # Separator  
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # View operations - create frame for side-by-side buttons
        view_frame = ttk.Frame(toolbar_frame)
        view_frame.pack(side=tk.LEFT, padx=5)
        
        self.statistics_btn = ttk.Button(view_frame, text="Статистика", command=self.show_data_statistics, width=10)
        self.statistics_btn.pack(side=tk.LEFT, padx=1)
        
        self.grid_info_btn = ttk.Button(view_frame, text="Сетка", command=self.show_grid_info, width=8)
        self.grid_info_btn.pack(side=tk.LEFT, padx=1)
        
        # Store toolbar button references for language updates
        self.ui_elements['toolbar_buttons'] = {
            'open': self.open_btn,
            'save': self.save_btn,
            'interpolate': self.interpolate_btn,
            'export': self.export_btn,
            'statistics': self.statistics_btn,
            'grid_info': self.grid_info_btn
        }
        
        # Store panels for language updates
        self.ui_elements['panels'] = {}
    
    def create_language_toggle_button(self, parent_frame):
        """Create language toggle button in the top-right corner."""
        from ..i18n import get_current_language, set_language
        
        # Current language indicator and toggle button
        lang_frame = ttk.Frame(parent_frame)
        lang_frame.pack(side=tk.RIGHT, padx=5)
        
        # Current language label
        current_lang = get_current_language()
        lang_text = "🇷🇺 RU" if current_lang == 'ru' else "🇺🇸 EN"
        
        self.language_button = ttk.Button(
            lang_frame,
            text=lang_text,
            width=8,
            command=self.toggle_language
        )
        self.language_button.pack(side=tk.RIGHT, padx=2)
        
        # Store for updates
        self.ui_elements['language_button'] = self.language_button
    
    def toggle_language(self):
        """Toggle between Russian and English."""
        from ..i18n import get_current_language, set_language, _
        
        current_lang = get_current_language()
        new_lang = 'en' if current_lang == 'ru' else 'ru'
        
        print(f"TOGGLE: Current language: {current_lang}")
        print(f"TOGGLE: Switching to: {new_lang}")
        
        # Test translation before change
        before_test = _("File")
        print(f"TOGGLE: Before change - 'File' translates to: '{before_test}'")
        
        set_language(new_lang)
        
        # Test translation after change
        after_test = _("File")
        print(f"TOGGLE: After change - 'File' translates to: '{after_test}'")
        
        # Verify current language
        verify_lang = get_current_language()
        print(f"TOGGLE: Verified current language: {verify_lang}")
        
        # Save the preference
        try:
            from .dialogs.language_settings_dialog import save_language_preference
            save_language_preference(new_lang)
            print(f"TOGGLE: Saved language preference: {new_lang}")
        except Exception as e:
            print(f"Error saving language preference: {e}")
        
        # Update button text immediately
        self.update_language_button()
        print("TOGGLE: Button update completed")
        
    def create_main_panels(self):
        """Create the main panels in the central area."""
        # Create main paned window for horizontal split (20% - 60% - 20%)
        main_paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for data and parameters (20%)
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
        
        # Center panel for visualization (60%)
        center_frame = ttk.Frame(main_paned)
        main_paned.add(center_frame, weight=3)
        
        # Visualization panel (takes full center area)
        self.visualization_panel = VisualizationPanel(center_frame, self.controller)
        self.visualization_panel.frame.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for results (20%)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # Results panel (takes full right area)
        self.results_panel = ResultsPanel(right_frame, self.controller)
        self.results_panel.frame.pack(fill=tk.BOTH, expand=True)
        
        # Store panels for language updates
        self.ui_elements['panels'] = {
            'data': self.data_panel,
            'parameters': self.parameters_panel,
            'visualization': self.visualization_panel,
            'results': self.results_panel
        }
        
    def create_status_bar(self):
        """Create the status bar at the bottom."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status label
        self.status_var = tk.StringVar(value=_("Ready"))
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
        self.status_var.set(_("Loaded data") + f": {data_info['filename']} ({data_info['rows']} " + _("rows") + ")")
        self.interpolate_btn.config(state=tk.NORMAL)
        
        # Update value columns in parameters panel
        value_columns = self.controller.get_value_columns()
        self.parameters_panel.update_value_columns(value_columns)
        
    def on_interpolation_started(self):
        """Handle interpolation started event."""
        self.status_var.set(_("Running interpolation..."))
        self.interpolate_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
    def on_interpolation_progress(self, progress: float):
        """Handle interpolation progress event."""
        self.progress_var.set(progress)
        self.root.update_idletasks()
        
    def on_interpolation_completed(self, results: Dict[str, Any]):
        """Handle interpolation completed event."""
        self.status_var.set(_("Interpolation completed successfully"))
        self.interpolate_btn.config(state=tk.NORMAL)
        self.progress_var.set(100)
        self.interpolation_results = results
        
    def on_error_occurred(self, error_msg: str):
        """Handle error occurred event."""
        self.status_var.set(_("Error occurred"))
        self.interpolate_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        messagebox.showerror(_("Error"), error_msg)
        
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
            messagebox.showwarning(_("Warning"), _("No data to save"))
            return
            
        file_path = filedialog.asksaveasfilename(
            title=_("Save Project"),
            defaultextension=".json",
            filetypes=[(_("Project files"), "*.json"), (_("All files"), "*.*")]
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
            title=_("Load Project"),
            filetypes=[(_("Project files"), "*.json"), (_("All files"), "*.*")]
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
            messagebox.showwarning(_("Warning"), _("No results to export"))
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
            messagebox.showwarning(_("Warning"), _("Please load data first"))
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
        self.status_var.set(_("Layout reset to default"))
        
    def show_data_statistics(self):
        """Show data statistics dialog."""
        if not self.controller.has_data():
            messagebox.showwarning(_("Warning"), _("Please load data first"))
            return
            
        # Get data information from controller
        data_info = self.controller.get_data_info()
        if not data_info:
            messagebox.showwarning(_("Warning"), _("Could not retrieve data information"))
            return
        
        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title(_("Data Statistics"))
        stats_window.geometry("600x500")
        stats_window.transient(self.root)
        stats_window.grab_set()
        
        # Create notebook for different statistics tabs
        notebook = ttk.Notebook(stats_window, padding="10")
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Basic statistics tab
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text=_("Basic Statistics"))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(basic_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add statistics content
        stats_content = self._generate_statistics_text(data_info)
        text_widget.insert(tk.END, stats_content)
        text_widget.config(state=tk.DISABLED)
        
        # Spatial statistics tab
        spatial_frame = ttk.Frame(notebook)
        notebook.add(spatial_frame, text=_("Spatial Statistics"))
        
        spatial_text = tk.Text(spatial_frame, wrap=tk.WORD, font=("Courier", 10))
        spatial_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        spatial_content = self._generate_spatial_statistics(data_info)
        spatial_text.insert(tk.END, spatial_content)
        spatial_text.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(
            stats_window, 
            text=_("Close"), 
            command=stats_window.destroy
        ).pack(pady=10)
        
    def show_grid_info(self):
        """Show grid information dialog."""
        if not self.controller.has_data():
            messagebox.showwarning(_("Warning"), _("Please load data first"))
            return
            
        # Get current parameters from parameters panel
        try:
            params = self.parameters_panel.get_parameters()
        except Exception as e:
            messagebox.showerror(_("Error"), f"{_('Failed to get parameters')}: {str(e)}")
            return
        
        # Create grid info window
        info_window = tk.Toplevel(self.root)
        info_window.title(_("Grid Information"))
        info_window.geometry("500x400")
        info_window.transient(self.root)
        info_window.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(info_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame,
            text=_("Grid Information"),
            font=("TkDefaultFont", 14, "bold")
        ).pack(pady=(0, 20))
        
        # Grid details frame
        details_frame = ttk.LabelFrame(main_frame, text=_("Grid Details"), padding="15")
        details_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Get grid resolution
        grid_res = params.get('grid_resolution', 50)
        
        # Get data bounds
        data_info = self.controller.get_data_info()
        bounds = data_info.get('bounds', {})
        
        # Calculate grid dimensions
        x_range = bounds.get('max_x', 0) - bounds.get('min_x', 0)
        y_range = bounds.get('max_y', 0) - bounds.get('min_y', 0)
        
        grid_cols = int(x_range / grid_res) + 1
        grid_rows = int(y_range / grid_res) + 1
        total_points = grid_cols * grid_rows
        
        # Display grid information
        info_text = [
            f"{_('Grid Resolution')}: {grid_res}",
            f"{_('Grid Columns')}: {grid_cols}",
            f"{_('Grid Rows')}: {grid_rows}",
            f"{_('Total Grid Points')}: {total_points:,}",
            "",
            f"{_('X Range')}: {bounds.get('min_x', 0):.2f} to {bounds.get('max_x', 0):.2f}",
            f"{_('Y Range')}: {bounds.get('min_y', 0):.2f} to {bounds.get('max_y', 0):.2f}",
            "",
            f"{_('Cell Width')}: {grid_res}",
            f"{_('Cell Height')}: {grid_res}",
            f"{_('Cell Area')}: {grid_res * grid_res}"
        ]
        
        for info in info_text:
            ttk.Label(details_frame, text=info).pack(anchor=tk.W, pady=2)
        
        # Memory estimate
        memory_frame = ttk.LabelFrame(main_frame, text=_("Memory Estimate"), padding="15")
        memory_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Estimate memory usage (rough estimate: 8 bytes per float value)
        memory_mb = (total_points * 8) / (1024 * 1024)
        
        ttk.Label(
            memory_frame,
            text=f"{_('Estimated memory usage')}: {memory_mb:.1f} MB"
        ).pack(anchor=tk.W)
        
        # Close button
        ttk.Button(
            info_window,
            text=_("Close"),
            command=info_window.destroy
        ).pack(pady=10)
        
    def show_preferences(self):
        """Show preferences dialog."""
        messagebox.showinfo(_("Info"), _("Preferences dialog - To be implemented"))
        
    def show_language_settings(self):
        """Show language settings dialog."""
        try:
            selected_language = show_language_settings_dialog(self.root)
            if selected_language:
                # Language will be changed on next application restart
                pass
        except Exception as e:
            messagebox.showerror(_("Error"), f"Failed to open language settings: {str(e)}")
        
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
        
    def on_language_changed(self, old_language: str, new_language: str):
        """Handle language change event and update UI immediately."""
        try:
            print(f"MainWindow: Language changed from {old_language} to {new_language}")
            
            # Test translation
            test_translation = _("File")
            print(f"Test translation 'File': {test_translation}")
            
            # Update window title
            title_text = _("Coal Deposit Interpolation Tool") + " - v0.4.0"
            print(f"New title: {title_text}")
            self.root.title(title_text)
            
            # Update menu bar
            self.update_menu_bar()
            
            # Update toolbar buttons
            self.update_toolbar_buttons()
            
            # Update panels
            self.update_panels()
            
            # Update status bar if needed
            if hasattr(self, 'status_var'):
                current_status = self.status_var.get()
                if current_status == "Ready":
                    ready_text = _("Ready")
                    print(f"Status Ready translation: {ready_text}")
                    self.status_var.set(ready_text)
            
            # Refresh the UI
            self.root.update_idletasks()
            print("UI refresh completed")
            
        except Exception as e:
            import logging
            import traceback
            logging.error(f"Error updating UI for language change: {e}")
            traceback.print_exc()
    
    def update_menu_bar(self):
        """Update menu bar with new translations."""
        try:
            # Find menu positions by iterating through them
            file_label = _("File")
            view_label = _("View") 
            tools_label = _("Tools")
            help_label = _("Help")
            
            print(f"Updating menus: File='{file_label}', View='{view_label}', Tools='{tools_label}', Help='{help_label}'")
            
            # Update main menu labels
            try:
                # Get menu count
                menu_count = self.menubar.index("end")
                if menu_count is not None:
                    # Update each menu label
                    self.menubar.entryconfig(0, label=file_label)
                    self.menubar.entryconfig(1, label=view_label)
                    self.menubar.entryconfig(2, label=tools_label)
                    self.menubar.entryconfig(3, label=help_label)
            except tk.TclError as e:
                print(f"Error updating menu labels: {e}")
            
            # Recreate menus with new translations
            self.recreate_file_menu()
            self.recreate_view_menu()
            self.recreate_tools_menu()
            self.recreate_help_menu()
            
            print("Menu bar updated successfully")
            
        except Exception as e:
            print(f"Error updating menu bar: {e}")
            # If we can't update menus individually, recreate the entire menu bar
            try:
                self.create_menu_bar()
            except Exception as e2:
                print(f"Error recreating menu bar: {e2}")
    
    def recreate_file_menu(self):
        """Recreate file menu with new translations."""
        print("Recreating file menu...")
        open_label = _("Open") + "..."
        save_label = _("Save") + " " + _("Project") + "..."
        load_label = _("Load Data") + "..."
        export_label = _("Export Results") + "..."
        exit_label = _("Exit")
        
        print(f"File menu labels: Open='{open_label}', Save='{save_label}', Exit='{exit_label}'")
        
        self.file_menu.delete(0, 'end')
        self.file_menu.add_command(label=open_label, command=self.open_file, accelerator="Ctrl+O")
        self.file_menu.add_separator()
        self.file_menu.add_command(label=save_label, command=self.save_project, accelerator="Ctrl+S")
        self.file_menu.add_command(label=load_label, command=self.load_project)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=export_label, command=self.export_results, accelerator="Ctrl+E")
        self.file_menu.add_separator()
        self.file_menu.add_command(label=exit_label, command=self.on_closing, accelerator="Ctrl+Q")
    
    def recreate_view_menu(self):
        """Recreate view menu with new translations."""
        self.view_menu.delete(0, 'end')
        self.view_menu.add_command(label=_("Reset") + " " + _("Layout"), command=self.reset_layout)
        self.view_menu.add_separator()
        self.view_menu.add_checkbutton(label=_("Show") + " " + _("Data") + " " + _("Panel"), variable=tk.BooleanVar(value=True))
        self.view_menu.add_checkbutton(label=_("Show") + " " + _("Parameters") + " " + _("Panel"), variable=tk.BooleanVar(value=True))
        self.view_menu.add_checkbutton(label=_("Show") + " " + _("Visualization") + " " + _("Panel"), variable=tk.BooleanVar(value=True))
        self.view_menu.add_checkbutton(label=_("Show") + " " + _("Results") + " " + _("Panel"), variable=tk.BooleanVar(value=True))
    
    def recreate_tools_menu(self):
        """Recreate tools menu with new translations."""
        self.tools_menu.delete(0, 'end')
        self.tools_menu.add_command(label=_("Data") + " " + _("Statistics"), command=self.show_data_statistics)
        self.tools_menu.add_command(label=_("Grid") + " " + _("Information"), command=self.show_grid_info)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label=_("Language Settings"), command=self.show_language_settings)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label=_("Preferences"), command=self.show_preferences)
    
    def recreate_help_menu(self):
        """Recreate help menu with new translations."""
        self.help_menu.delete(0, 'end')
        self.help_menu.add_command(label=_("User Guide"), command=self.show_help)
        self.help_menu.add_command(label=_("About"), command=self.show_about)
    
    def update_toolbar_buttons(self):
        """Update toolbar button texts with new translations."""
        if 'toolbar_buttons' in self.ui_elements:
            buttons = self.ui_elements['toolbar_buttons']
            
            if 'open' in buttons:
                buttons['open'].config(text=_("Open"))
            if 'save' in buttons:
                buttons['save'].config(text=_("Save"))
            if 'interpolate' in buttons:
                buttons['interpolate'].config(text=_("Interpolate"))
            if 'export' in buttons:
                buttons['export'].config(text=_("Export"))
            if 'statistics' in buttons:
                buttons['statistics'].config(text=_("Statistics"))
            if 'grid_info' in buttons:
                buttons['grid_info'].config(text=_("Grid Info"))
        
        # Also update language button
        self.update_language_button()
    
    def update_language_button(self):
        """Update language toggle button text."""
        from ..i18n import get_current_language
        current_lang = get_current_language()
        lang_text = "🇷🇺 RU" if current_lang == 'ru' else "🇺🇸 EN"
        
        # Update toolbar language button if it exists
        if 'language_button' in self.ui_elements:
            self.ui_elements['language_button'].config(text=lang_text)
            
        print(f"Language button updated to: {lang_text}")
    
    def update_panels(self):
        """Update all panels with new translations."""
        if 'panels' in self.ui_elements:
            panels = self.ui_elements['panels']
            
            for panel_name, panel in panels.items():
                try:
                    # Check if panel has update_language method
                    if hasattr(panel, 'update_language'):
                        panel.update_language()
                        print(f"Updated {panel_name} panel language")
                    else:
                        print(f"{panel_name} panel doesn't have update_language method")
                except Exception as e:
                    print(f"Error updating {panel_name} panel: {e}")

    def on_closing(self):
        """Handle window closing event."""
        # Remove language change listener
        remove_language_change_listener(self.on_language_changed)
        
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()
            self.root.destroy()
            
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()
    
    def _generate_statistics_text(self, data_info: Dict[str, Any]) -> str:
        """Generate basic statistics text."""
        content = []
        content.append(_("DATA STATISTICS"))
        content.append("=" * 50)
        content.append("")
        
        # File information
        content.append(f"{_('File')}: {data_info.get('filename', 'N/A')}")
        content.append(f"{_('Total rows')}: {data_info.get('rows', 0):,}")
        content.append(f"{_('Total columns')}: {len(data_info.get('columns', []))}")
        content.append("")
        
        # Column information with statistics
        content.append(_("COLUMN INFORMATION"))
        content.append("-" * 30)
        
        data_types = data_info.get('data_types', {})
        missing_values = data_info.get('missing_values', {})
        
        # Get actual data for statistical calculations
        if hasattr(self.controller, 'current_data') and self.controller.current_data is not None:
            df = self.controller.current_data
            
            for column in data_info.get('columns', []):
                dtype = str(data_types.get(column, 'unknown'))
                missing = missing_values.get(column, 0)
                missing_pct = (missing / data_info.get('rows', 1)) * 100 if data_info.get('rows', 0) > 0 else 0
                
                content.append(f"\n{column}:")
                content.append(f"  {_('Type')}: {dtype}")
                content.append(f"  {_('Missing values')}: {missing} ({missing_pct:.1f}%)")
                
                # Add statistics for numeric columns
                if column in df.columns and df[column].dtype in ['float64', 'int64', 'float32', 'int32']:
                    try:
                        col_data = df[column].dropna()
                        if len(col_data) > 0:
                            content.append(f"  {_('Min Value')}: {col_data.min():.3f}")
                            content.append(f"  {_('Max Value')}: {col_data.max():.3f}")
                            content.append(f"  {_('Mean Value')}: {col_data.mean():.3f}")
                            content.append(f"  {_('Std Dev')}: {col_data.std():.3f}")
                    except Exception as e:
                        content.append(f"  {_('Statistics')}: {_('Error calculating')} - {str(e)}")
        else:
            # Fallback when no data available
            for column in data_info.get('columns', []):
                dtype = str(data_types.get(column, 'unknown'))
                missing = missing_values.get(column, 0)
                missing_pct = (missing / data_info.get('rows', 1)) * 100 if data_info.get('rows', 0) > 0 else 0
                
                content.append(f"\n{column}:")
                content.append(f"  {_('Type')}: {dtype}")
                content.append(f"  {_('Missing values')}: {missing} ({missing_pct:.1f}%)")
            
        return "\n".join(content)
    
    def _generate_spatial_statistics(self, data_info: Dict[str, Any]) -> str:
        """Generate spatial statistics text."""
        content = []
        content.append(_("SPATIAL STATISTICS"))
        content.append("=" * 50)
        content.append("")
        
        bounds = data_info.get('bounds', {})
        
        try:
            min_x = float(bounds.get('min_x', 0))
            max_x = float(bounds.get('max_x', 0))
            min_y = float(bounds.get('min_y', 0))
            max_y = float(bounds.get('max_y', 0))
            
            content.append(f"{_('X Range')}: {min_x:.2f} - {max_x:.2f}")
            content.append(f"{_('Y Range')}: {min_y:.2f} - {max_y:.2f}")
            
            # Calculate area
            width = max_x - min_x
            height = max_y - min_y
            area = width * height
            
            content.append(f"\n{_('Coverage')}:")
            content.append(f"  {_('Width')}: {width:.2f}")
            content.append(f"  {_('Height')}: {height:.2f}")
            content.append(f"  {_('Area')}: {area:.2f}")
            
            # Point density
            num_points = data_info.get('rows', 0)
            if area > 0 and num_points > 0:
                density = num_points / area
                content.append(f"\n{_('Point density')}: {density:.4f} points/unit²")
            else:
                content.append(f"\n{_('Point density')}: Cannot calculate (area = {area:.2f})")
                
        except (ValueError, TypeError) as e:
            content.append(f"Error calculating spatial statistics: {str(e)}")
            content.append(f"Bounds data: {bounds}")
        
        return "\n".join(content)


def main():
    """Main entry point for the GUI application."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()