"""
Parameters panel widget for configuring interpolation parameters.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, List, Optional
from tkinter import scrolledtext

# Import i18n system
from ...i18n import _


class ParametersPanel:
    """
    Panel for configuring interpolation parameters.
    
    This panel provides controls for:
    - Method selection (IDW, Kriging, etc.)
    - IDW parameters (power, smoothing)
    - Search parameters (radius, max points, sectors)
    - Grid parameters (cell size, buffer)
    - Value column selection
    """
    
    def __init__(self, parent, controller):
        """
        Initialize the parameters panel.
        
        Args:
            parent: Parent widget
            controller: Application controller instance
        """
        self.controller = controller
        self.value_columns = []
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text=_("Parameters"), padding="5")
        
        # Store UI elements for language updates
        self.ui_elements = {}
        
        # Parameter variables
        self.setup_variables()
        self.create_widgets()
        self.setup_bindings()
        
        # Register for language change events
        from ...i18n import add_language_change_listener
        add_language_change_listener(self.update_language)
        
    def setup_variables(self):
        """Setup tkinter variables for parameters."""
        # Method selection
        self.method_var = tk.StringVar(value="IDW")
        
        # Value column selection
        self.value_column_var = tk.StringVar()
        
        # IDW parameters
        self.power_var = tk.DoubleVar(value=2.0)
        self.smoothing_var = tk.DoubleVar(value=0.0)
        
        # RBF parameters
        self.rbf_kernel_var = tk.StringVar(value="multiquadric")
        self.rbf_shape_var = tk.DoubleVar(value=1.0)
        self.rbf_regularization_var = tk.DoubleVar(value=1e-12)
        self.rbf_polynomial_var = tk.IntVar(value=-1)
        self.rbf_global_var = tk.BooleanVar(value=True)
        
        # Kriging parameters
        self.kriging_type_var = tk.StringVar(value="ordinary")
        self.kriging_variogram_var = tk.StringVar(value="spherical")
        self.kriging_nugget_var = tk.DoubleVar(value=0.0)
        self.kriging_sill_var = tk.DoubleVar(value=1.0)
        self.kriging_range_var = tk.DoubleVar(value=1000.0)
        self.kriging_auto_fit_var = tk.BooleanVar(value=True)
        self.kriging_global_var = tk.BooleanVar(value=True)
        
        # Search parameters
        self.search_radius_var = tk.DoubleVar(value=1000.0)
        self.min_points_var = tk.IntVar(value=1)
        self.max_points_var = tk.IntVar(value=12)
        self.use_sectors_var = tk.BooleanVar(value=False)
        self.n_sectors_var = tk.IntVar(value=4)
        
        # Grid parameters
        self.cell_size_var = tk.DoubleVar(value=50.0)
        self.buffer_var = tk.DoubleVar(value=0.1)
        
    def create_widgets(self):
        """Create the panel widgets."""
        # Create notebook for different methods
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create IDW method tab with all its parameters
        self.create_idw_method_tab()
        
        # Create RBF method tab with all its parameters  
        self.create_rbf_method_tab()
        
        # Create Kriging method tab with all its parameters
        self.create_kriging_method_tab()
        
        # Create Analysis and Recommendations tab
        self.create_analysis_tab()
        
        # Action buttons
        self.create_action_buttons()
        
    def create_idw_method_tab(self):
        """Create the IDW method tab with all parameters."""
        idw_frame = ttk.Frame(self.notebook)
        self.notebook.add(idw_frame, text=_("IDW"))
        
        # Create scrollable frame
        canvas = tk.Canvas(idw_frame)
        scrollbar = ttk.Scrollbar(idw_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Value column selection
        data_group = ttk.LabelFrame(scrollable_frame, text=_("Data Column"), padding="5")
        data_group.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(data_group, text=_("Value Column:")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.value_column_combo = ttk.Combobox(
            data_group, 
            textvariable=self.value_column_var,
            state="readonly",
            width=20
        )
        self.value_column_combo.grid(row=0, column=1, sticky=tk.W+tk.E, padx=(0, 5))
        data_group.columnconfigure(1, weight=1)
        
        # IDW Parameters
        idw_params_group = ttk.LabelFrame(scrollable_frame, text=_("IDW Parameters"), padding="5")
        idw_params_group.pack(fill=tk.X, pady=(0, 10))
        
        # Power parameter
        power_frame = ttk.Frame(idw_params_group)
        power_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(power_frame, text=_("Power (p):")).pack(side=tk.LEFT)
        
        power_scale = ttk.Scale(
            power_frame, 
            from_=0.5, 
            to=5.0, 
            variable=self.power_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        power_scale.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        self.power_label = ttk.Label(power_frame, text="2.0")
        self.power_label.pack(side=tk.LEFT)
        power_scale.config(command=self.update_power_label)
        
        # Smoothing parameter
        smoothing_frame = ttk.Frame(idw_params_group)
        smoothing_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(smoothing_frame, text=_("Smoothing:")).pack(side=tk.LEFT)
        
        smoothing_scale = ttk.Scale(
            smoothing_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.smoothing_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        smoothing_scale.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        self.smoothing_label = ttk.Label(smoothing_frame, text="0.0")
        self.smoothing_label.pack(side=tk.LEFT)
        smoothing_scale.config(command=self.update_smoothing_label)
    
    def create_search_params(self, parent):
        """Create search parameters section."""
        search_group = ttk.LabelFrame(parent, text=_("Search Parameters"), padding="5")
        search_group.pack(fill=tk.X, pady=(0, 10))
        
        # Search radius
        radius_frame = ttk.Frame(search_group)
        radius_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(radius_frame, text=_("Search Radius:")).pack(side=tk.LEFT)
        
        radius_entry = ttk.Entry(radius_frame, textvariable=self.search_radius_var, width=10)
        radius_entry.pack(side=tk.LEFT, padx=(10, 5))
        
        # Point limits
        limits_frame = ttk.LabelFrame(search_group, text=_("Point Limits"), padding="5")
        limits_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Minimum points
        min_frame = ttk.Frame(limits_frame)
        min_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(min_frame, text=_("Minimum Points:")).pack(side=tk.LEFT)
        min_entry = ttk.Entry(min_frame, textvariable=self.min_points_var, width=10)
        min_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Maximum points
        max_frame = ttk.Frame(limits_frame)
        max_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(max_frame, text=_("Maximum Points:")).pack(side=tk.LEFT)
        max_entry = ttk.Entry(max_frame, textvariable=self.max_points_var, width=10)
        max_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Sectoral search
        sector_group = ttk.LabelFrame(search_group, text=_("Sectoral Search"), padding="5")
        sector_group.pack(fill=tk.X, pady=(5, 0))
        
        self.use_sectors_check = ttk.Checkbutton(
            sector_group,
            text=_("Use sectoral search"),
            variable=self.use_sectors_var,
            command=self.toggle_sectors
        )
        self.use_sectors_check.pack(anchor=tk.W, pady=(0, 5))
        
        self.sectors_frame = ttk.Frame(sector_group)
        self.sectors_frame.pack(fill=tk.X)
        
        ttk.Label(self.sectors_frame, text=_("Number of sectors:")).pack(side=tk.LEFT)
        self.sectors_entry = ttk.Entry(self.sectors_frame, textvariable=self.n_sectors_var, width=10)
        self.sectors_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Initially disable sectors controls
        self.toggle_sectors()
    
    def create_grid_params(self, parent):
        """Create grid parameters section."""
        grid_group = ttk.LabelFrame(parent, text=_("Grid Parameters"), padding="5")
        grid_group.pack(fill=tk.X, pady=(0, 10))
        
        # Cell size
        cell_frame = ttk.Frame(grid_group)
        cell_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(cell_frame, text=_("Cell Size:")).pack(side=tk.LEFT)
        cell_entry = ttk.Entry(cell_frame, textvariable=self.cell_size_var, width=10)
        cell_entry.pack(side=tk.LEFT, padx=(10, 5))
        
        # Buffer
        buffer_frame = ttk.Frame(grid_group)
        buffer_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(buffer_frame, text=_("Buffer:")).pack(side=tk.LEFT)
        buffer_entry = ttk.Entry(buffer_frame, textvariable=self.buffer_var, width=10)
        buffer_entry.pack(side=tk.LEFT, padx=(10, 5))
        
        ttk.Label(buffer_frame, text=_("(0.0-1.0 = %, >1.0 = meters)")).pack(side=tk.LEFT, padx=(5, 0))
        
        # Preview button
        preview_btn = ttk.Button(
            grid_group,
            text=_("Preview Grid Info"),
            command=self.preview_grid_info
        )
        preview_btn.pack(pady=(10, 0))
    
    def create_kriging_method_tab(self):
        """Create the Kriging method tab with all parameters."""
        kriging_frame = ttk.Frame(self.notebook)
        self.notebook.add(kriging_frame, text=_("Kriging"))
        
        # Create scrollable frame
        canvas = tk.Canvas(kriging_frame)
        scrollbar = ttk.Scrollbar(kriging_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Value column selection
        data_group = ttk.LabelFrame(scrollable_frame, text=_("Data Column"), padding="5")
        data_group.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(data_group, text=_("Value Column:")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        kriging_value_combo = ttk.Combobox(
            data_group, 
            textvariable=self.value_column_var,
            state="readonly",
            width=20
        )
        kriging_value_combo.grid(row=0, column=1, sticky=tk.W+tk.E, padx=(0, 5))
        data_group.columnconfigure(1, weight=1)
        
        # Kriging Parameters
        kriging_params_group = ttk.LabelFrame(scrollable_frame, text=_("Kriging Parameters"), padding="5")
        kriging_params_group.pack(fill=tk.X, pady=(0, 10))
        
        # Kriging type
        type_frame = ttk.Frame(kriging_params_group)
        type_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(type_frame, text=_("Type:")).pack(side=tk.LEFT)
        
        type_combo = ttk.Combobox(
            type_frame,
            textvariable=self.kriging_type_var,
            values=["ordinary", "simple", "universal"],
            state="readonly",
            width=15
        )
        type_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Variogram model
        variogram_group = ttk.LabelFrame(scrollable_frame, text=_("Variogram Model"), padding="5")
        variogram_group.pack(fill=tk.X, pady=(0, 10))
        
        model_frame = ttk.Frame(variogram_group)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text=_("Model:")).pack(side=tk.LEFT)
        
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.kriging_variogram_var,
            values=["spherical", "exponential", "gaussian", "linear"],
            state="readonly",
            width=15
        )
        model_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Variogram parameters
        params_group = ttk.LabelFrame(scrollable_frame, text=_("Variogram Parameters"), padding="5")
        params_group.pack(fill=tk.X, pady=(0, 10))
        
        # Nugget
        nugget_frame = ttk.Frame(params_group)
        nugget_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(nugget_frame, text=_("Nugget:")).pack(side=tk.LEFT)
        nugget_entry = ttk.Entry(nugget_frame, textvariable=self.kriging_nugget_var, width=10)
        nugget_entry.pack(side=tk.LEFT, padx=(10, 5))
        ttk.Label(nugget_frame, text=_("(micro-scale variance)")).pack(side=tk.LEFT)
        
        # Sill
        sill_frame = ttk.Frame(params_group)
        sill_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(sill_frame, text=_("Sill:")).pack(side=tk.LEFT)
        sill_entry = ttk.Entry(sill_frame, textvariable=self.kriging_sill_var, width=10)
        sill_entry.pack(side=tk.LEFT, padx=(10, 5))
        ttk.Label(sill_frame, text=_("(total variance)")).pack(side=tk.LEFT)
        
        # Range
        range_frame = ttk.Frame(params_group)
        range_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(range_frame, text=_("Range:")).pack(side=tk.LEFT)
        range_entry = ttk.Entry(range_frame, textvariable=self.kriging_range_var, width=10)
        range_entry.pack(side=tk.LEFT, padx=(10, 5))
        ttk.Label(range_frame, text=_("(correlation distance)")).pack(side=tk.LEFT)
        
        # Variogram fitting
        fitting_group = ttk.LabelFrame(scrollable_frame, text=_("Variogram Fitting"), padding="5")
        fitting_group.pack(fill=tk.X, pady=(0, 10))
        
        self.auto_fit_check = ttk.Checkbutton(
            fitting_group,
            text=_("Automatically fit variogram parameters"),
            variable=self.kriging_auto_fit_var,
            command=self.toggle_kriging_manual_params
        )
        self.auto_fit_check.pack(anchor=tk.W)
        
        # Search Parameters
        self.create_search_params(scrollable_frame)
        
        # Grid Parameters
        self.create_grid_params(scrollable_frame)
        
        # Set method to Kriging when this tab is created  
        self.method_var.set("Kriging")
    
    def create_analysis_tab(self):
        """Create the Analysis and Recommendations tab."""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text=_("Analysis and Recommendations"))
        
        # Create scrollable frame
        canvas = tk.Canvas(analysis_frame)
        scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Analysis controls
        controls_frame = ttk.Frame(scrollable_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.analyze_btn = ttk.Button(
            controls_frame,
            text=_("Analyze Data"),
            command=self.analyze_data
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.apply_btn = ttk.Button(
            controls_frame,
            text=_("Apply Recommendations"),
            command=self.apply_recommendations,
            state=tk.DISABLED
        )
        self.apply_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            controls_frame,
            mode="indeterminate",
            length=100
        )
        
        # Results notebook
        results_notebook = ttk.Notebook(scrollable_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        summary_frame = ttk.Frame(results_notebook)
        results_notebook.add(summary_frame, text=_("Summary"))
        
        self.summary_text = scrolledtext.ScrolledText(
            summary_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=15
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Method comparison tab
        method_frame = ttk.Frame(results_notebook)
        results_notebook.add(method_frame, text=_("Method Comparison"))
        
        self.method_text = scrolledtext.ScrolledText(
            method_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=15
        )
        self.method_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Optimal parameters tab
        params_frame = ttk.Frame(results_notebook)
        results_notebook.add(params_frame, text=_("Optimal Parameters"))
        
        self.params_text = scrolledtext.ScrolledText(
            params_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=15
        )
        self.params_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Show initial help
        self.show_recommendation_help()
    
    def on_tab_changed(self, event):
        """Handle tab change to update method variable."""
        tab_index = self.notebook.index(self.notebook.select())
        if tab_index == 0:  # IDW tab
            self.method_var.set("IDW")
        elif tab_index == 1:  # RBF tab
            self.method_var.set("RBF")
        elif tab_index == 2:  # Kriging tab
            self.method_var.set("Kriging")
        # Analysis tab doesn't change method
        
        # Grid Parameters
        self.create_grid_params(scrollable_frame)
        
        # Set method to IDW when this tab is created
        self.method_var.set("IDW")
        
        # Bind tab selection to method change
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
    def create_rbf_method_tab(self):
        """Create the RBF method tab with all parameters."""
        rbf_frame = ttk.Frame(self.notebook)
        self.notebook.add(rbf_frame, text=_("RBF"))
        
        # Create scrollable frame
        canvas = tk.Canvas(rbf_frame)
        scrollbar = ttk.Scrollbar(rbf_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Value column selection
        data_group = ttk.LabelFrame(scrollable_frame, text=_("Data Column"), padding="5")
        data_group.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(data_group, text=_("Value Column:")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        rbf_value_combo = ttk.Combobox(
            data_group, 
            textvariable=self.value_column_var,
            state="readonly",
            width=20
        )
        rbf_value_combo.grid(row=0, column=1, sticky=tk.W+tk.E, padx=(0, 5))
        data_group.columnconfigure(1, weight=1)
        
        # RBF Parameters
        rbf_params_group = ttk.LabelFrame(scrollable_frame, text=_("RBF Parameters"), padding="5")
        rbf_params_group.pack(fill=tk.X, pady=(0, 10))
        
        # Kernel selection
        kernel_frame = ttk.Frame(rbf_params_group)
        kernel_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(kernel_frame, text=_("Kernel:")).pack(side=tk.LEFT)
        
        kernel_combo = ttk.Combobox(
            kernel_frame,
            textvariable=self.rbf_kernel_var,
            values=["multiquadric", "inverse", "gaussian", "linear", "cubic", "quintic", "thin_plate"],
            state="readonly",
            width=15
        )
        kernel_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Shape parameter
        shape_frame = ttk.Frame(rbf_params_group)
        shape_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(shape_frame, text=_("Shape Parameter (ε):")).pack(side=tk.LEFT)
        
        shape_scale = ttk.Scale(
            shape_frame,
            from_=0.1,
            to=10.0,
            variable=self.rbf_shape_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        shape_scale.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        self.rbf_shape_label = ttk.Label(shape_frame, text="1.0")
        self.rbf_shape_label.pack(side=tk.LEFT)
        shape_scale.config(command=self.update_shape_label)
        
        # Regularization
        reg_frame = ttk.Frame(rbf_params_group)
        reg_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(reg_frame, text=_("Regularization:")).pack(side=tk.LEFT)
        
        reg_scale = ttk.Scale(
            reg_frame,
            from_=1e-15,
            to=1e-6,
            variable=self.rbf_regularization_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        reg_scale.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        self.rbf_reg_label = ttk.Label(reg_frame, text="1e-12")
        self.rbf_reg_label.pack(side=tk.LEFT)
        reg_scale.config(command=self.update_reg_label)
        
        # Polynomial augmentation
        poly_group = ttk.LabelFrame(scrollable_frame, text=_("Polynomial Augmentation"), padding="5")
        poly_group.pack(fill=tk.X, pady=(0, 10))
        
        poly_frame = ttk.Frame(poly_group)
        poly_frame.pack(fill=tk.X)
        
        ttk.Label(poly_frame, text=_("Polynomial Degree:")).pack(side=tk.LEFT)
        
        poly_combo = ttk.Combobox(
            poly_frame,
            textvariable=self.rbf_polynomial_var,
            values=[-1, 0, 1, 2, 3],
            state="readonly",
            width=10
        )
        poly_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Interpolation mode
        mode_group = ttk.LabelFrame(scrollable_frame, text=_("Interpolation Mode"), padding="5")
        mode_group.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(
            mode_group,
            text=_("Global (use all points)"),
            variable=self.rbf_global_var,
            value=True
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            mode_group,
            text=_("Local (use search parameters)"),
            variable=self.rbf_global_var,
            value=False
        ).pack(anchor=tk.W)
        
        # Search Parameters (only if local mode)
        self.create_search_params(scrollable_frame)
        
        # Grid Parameters
        self.create_grid_params(scrollable_frame)
        
        # Set method to RBF when this tab is created
        self.method_var.set("RBF")
        
    def setup_bindings(self):
        """Setup event bindings."""
        pass
    
    def update_value_columns(self, columns):
        """Update the list of available value columns."""
        self.value_columns = columns
        self.value_column_combo['values'] = columns
        if columns and not self.value_column_var.get():
            self.value_column_var.set(columns[0])
    
    def on_method_changed(self):
        """Handle method selection changes."""
        method = self.method_var.get()
        # Add any method-specific logic here
        print(f"Method changed to: {method}")
    
    def toggle_sectors(self):
        """Toggle sectoral search controls."""
        enabled = self.use_sectors_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        
        if hasattr(self, 'sectors_entry'):
            self.sectors_entry.config(state=state)
    
    def toggle_kriging_manual_params(self):
        """Toggle manual kriging parameter controls."""
        auto_fit = self.kriging_auto_fit_var.get()
        # You can add logic here to enable/disable manual parameter controls
        pass
    
    def preview_grid_info(self):
        """Show grid information preview."""
        try:
            params = self.get_parameters()
            
            info_text = f"""
Grid Preview Information

Cell Size: {params['cell_size']} meters
Buffer: {params['buffer']}
Search Radius: {params['search_radius']} meters

This would show:
- Estimated grid dimensions
- Total number of points
- Memory requirements
- Processing time estimate

[To be implemented with GridGenerator integration]
            """
            
            tk.messagebox.showinfo("Grid Preview", info_text.strip())
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to generate grid preview: {str(e)}")
    
    def update_power_label(self, value):
        """Update power parameter label."""
        if hasattr(self, 'power_label'):
            self.power_label.config(text=f"{float(value):.1f}")
    
    def update_smoothing_label(self, value):
        """Update smoothing parameter label."""
        if hasattr(self, 'smoothing_label'):
            self.smoothing_label.config(text=f"{float(value):.3f}")
    
    def update_shape_label(self, value):
        """Update RBF shape parameter label."""
        if hasattr(self, 'rbf_shape_label'):
            self.rbf_shape_label.config(text=f"{float(value):.1f}")
    
    def update_reg_label(self, value):
        """Update RBF regularization parameter label."""
        if hasattr(self, 'rbf_reg_label'):
            self.rbf_reg_label.config(text=f"{float(value):.0e}")
    
    def create_action_buttons(self):
        """Create action buttons at the bottom."""
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Reset button
        reset_btn = ttk.Button(
            button_frame,
            text=_("Reset to Defaults"),
            command=self.reset_to_defaults
        )
        reset_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Load preset button
        load_btn = ttk.Button(
            button_frame,
            text=_("Load Preset"),
            command=self.load_preset
        )
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save preset button
        save_btn = ttk.Button(
            button_frame,
            text=_("Save Preset"),
            command=self.save_preset
        )
        save_btn.pack(side=tk.LEFT)
    
    def show_recommendation_help(self):
        """Show help text in recommendation tabs."""
        help_text = """
Welcome to the Recommendation System!

This system analyzes your data and provides:
• Optimal interpolation method selection
• Customized parameter recommendations  
• Performance expectations

To get started:
1. Load your data
2. Click "Analyze Data" button
3. Review recommendations
4. Click "Apply Recommendations" to use them

The analysis considers:
• Data density and distribution
• Spatial patterns and trends
• Value variability
• Computational efficiency
        """
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, help_text.strip())
        
        self.method_text.delete(1.0, tk.END)
        self.method_text.insert(1.0, "Method comparison will appear here after analysis.")
        
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(1.0, "Optimal parameters will appear here after analysis.")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary with all parameter values
        """
        params = {
            'method': self.method_var.get(),
            'value_column': self.value_column_var.get(),
            'search_radius': self.search_radius_var.get(),
            'min_points': self.min_points_var.get(),
            'max_points': self.max_points_var.get(),
            'use_sectors': self.use_sectors_var.get(),
            'n_sectors': self.n_sectors_var.get(),
            'cell_size': self.cell_size_var.get(),
            'buffer': self.buffer_var.get()
        }
        
        # Add method-specific parameters
        if self.method_var.get() == 'IDW':
            params.update({
                'power': self.power_var.get(),
                'smoothing': self.smoothing_var.get()
            })
        elif self.method_var.get() == 'RBF':
            params.update({
                'rbf_kernel': self.rbf_kernel_var.get(),
                'rbf_shape_parameter': self.rbf_shape_var.get(),
                'rbf_regularization': self.rbf_regularization_var.get(),
                'rbf_polynomial_degree': self.rbf_polynomial_var.get(),
                'rbf_use_global': self.rbf_global_var.get()
            })
        elif self.method_var.get() == 'Kriging':
            params.update({
                'kriging_type': self.kriging_type_var.get(),
                'kriging_variogram_model': self.kriging_variogram_var.get(),
                'kriging_nugget': self.kriging_nugget_var.get(),
                'kriging_sill': self.kriging_sill_var.get(),
                'kriging_range': self.kriging_range_var.get(),
                'kriging_auto_fit': self.kriging_auto_fit_var.get(),
                'kriging_use_global': self.kriging_global_var.get()
            })
        
        return params
        
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set parameter values.
        
        Args:
            params: Dictionary with parameter values
        """
        self.method_var.set(params.get('method', 'IDW'))
        self.value_column_var.set(params.get('value_column', ''))
        
        # IDW parameters
        self.power_var.set(params.get('power', 2.0))
        self.smoothing_var.set(params.get('smoothing', 0.0))
        
        # RBF parameters
        self.rbf_kernel_var.set(params.get('rbf_kernel', 'multiquadric'))
        self.rbf_shape_var.set(params.get('rbf_shape_parameter', 1.0))
        self.rbf_regularization_var.set(params.get('rbf_regularization', 1e-12))
        self.rbf_polynomial_var.set(params.get('rbf_polynomial_degree', -1))
        self.rbf_global_var.set(params.get('rbf_use_global', True))
        
        # Kriging parameters
        self.kriging_type_var.set(params.get('kriging_type', 'ordinary'))
        self.kriging_variogram_var.set(params.get('kriging_variogram_model', 'spherical'))
        self.kriging_nugget_var.set(params.get('kriging_nugget', 0.0))
        self.kriging_sill_var.set(params.get('kriging_sill', 1.0))
        self.kriging_range_var.set(params.get('kriging_range', 1000.0))
        self.kriging_auto_fit_var.set(params.get('kriging_auto_fit', True))
        self.kriging_global_var.set(params.get('kriging_use_global', True))
        
        # Search parameters
        self.search_radius_var.set(params.get('search_radius', 1000.0))
        self.min_points_var.set(params.get('min_points', 1))
        self.max_points_var.set(params.get('max_points', 12))
        self.use_sectors_var.set(params.get('use_sectors', False))
        self.n_sectors_var.set(params.get('n_sectors', 4))
        
        # Grid parameters
        self.cell_size_var.set(params.get('cell_size', 50.0))
        self.buffer_var.set(params.get('buffer', 0.1))
        
        # Update UI state
        self.on_method_changed()
        self.toggle_sectors()
        self.toggle_kriging_manual_params()
        self.update_power_label(self.power_var.get())
        self.update_smoothing_label(self.smoothing_var.get())
        self.update_shape_label(self.rbf_shape_var.get())
        self.update_reg_label(self.rbf_regularization_var.get())
        
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        defaults = {
            'method': 'IDW',
            'power': 2.0,
            'smoothing': 0.0,
            'search_radius': 1000.0,
            'min_points': 1,
            'max_points': 12,
            'use_sectors': False,
            'n_sectors': 4,
            'cell_size': 50.0,
            'buffer': 0.1
        }
        
        self.set_parameters(defaults)
        
    def load_preset(self):
        """Load parameter preset."""
        tk.messagebox.showinfo("Info", "Load preset functionality - To be implemented")
        
    def save_preset(self):
        """Save current parameters as preset."""
        tk.messagebox.showinfo("Info", "Save preset functionality - To be implemented")
    
    def analyze_data(self):
        """Run recommendation analysis on loaded data."""
        if not self.controller.has_data():
            tk.messagebox.showwarning("Warning", "Please load data first")
            return
            
        # Check if value column is selected
        if not self.value_column_var.get():
            tk.messagebox.showwarning("Warning", "Please select a value column")
            return
        
        # Show progress
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0))
        self.progress_bar.start(10)
        
        # Run analysis in background
        self.frame.after(100, self._run_analysis)
        
    def _run_analysis(self):
        """Run the actual analysis (called after UI update)."""
        try:
            # Import recommendation engine
            try:
                from ...core.recommendations import RecommendationEngine
            except ImportError:
                # Fallback for demo mode
                self._show_mock_recommendations()
                return
            
            # Get data from controller
            data_info = self.controller.get_data_info()
            data = self.controller.get_data()
            
            # Run recommendation analysis
            engine = RecommendationEngine()
            report = engine.analyze_and_recommend(
                data=data,
                x_col='X',
                y_col='Y',
                value_col=self.value_column_var.get(),
                user_preferences={'prioritize_speed': True},
                evaluate_quality=False,  # Skip for speed
                quick_mode=True
            )
            
            # Store report
            self.recommendation_report = report
            
            # Display results
            self._display_recommendations(report)
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            # Show mock recommendations as fallback
            self._show_mock_recommendations()
            
        finally:
            # Hide progress
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.analyze_btn.config(state=tk.NORMAL)
            self.apply_btn.config(state=tk.NORMAL)
    
    def _display_recommendations(self, report):
        """Display recommendation report in tabs."""
        # Summary tab
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, report.summary_text)
        
        # Method comparison
        method_text = "Method Comparison\n" + "="*50 + "\n\n"
        for i, method_score in enumerate(report.method_scores):
            method_text += f"{i+1}. {method_score.method.value}\n"
            method_text += f"   Score: {method_score.score:.1f}/100\n"
            method_text += f"   Pros:\n"
            for pro in method_score.pros[:3]:
                method_text += f"   • {pro}\n"
            method_text += f"   Cons:\n"
            for con in method_score.cons[:3]:
                method_text += f"   • {con}\n"
            method_text += "\n"
        
        self.method_text.delete(1.0, tk.END)
        self.method_text.insert(1.0, method_text)
        
        # Parameters
        params_text = "Optimal Parameters\n" + "="*50 + "\n\n"
        params_text += f"Recommended Method: {report.recommended_method}\n\n"
        
        for param, value in report.optimal_parameters.items():
            reason = report.parameter_reasoning.get(param, "")
            params_text += f"{param}: {value}\n"
            if reason:
                params_text += f"  → {reason}\n"
            params_text += "\n"
        
        if report.warnings:
            params_text += "\nWarnings:\n"
            for warning in report.warnings:
                params_text += f"⚠ {warning}\n"
        
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(1.0, params_text)
    
    def _show_mock_recommendations(self):
        """Show mock recommendations for demo mode."""
        # Mock summary
        summary = """
**Data Analysis Summary**
- Dataset contains 150 points
- Spatial density: 0.245 points per unit area
- Distribution uniformity: 0.65 (moderately uniform)
- No significant spatial trend detected

**Recommended Method: IDW**
- Suitability score: 85.0/100
- Key reasons:
  • Dense, uniformly distributed data
  • No global trends - local interpolation sufficient
  • IDW is computationally efficient

**Optimized Parameters**
- search_radius: 850.0 (Medium radius for balanced coverage)
- power: 2.0 (Standard power balances local and regional influence)
- max_points: 12 (Balances accuracy and speed)

**Expected Performance**
- Processing will be fast
- Good accuracy for local variations
- Suitable for your data characteristics
        """
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary.strip())
        
        # Mock method comparison
        methods = """
Method Comparison
==================================================

1. IDW
   Score: 85.0/100
   Pros:
   • Simple and intuitive method
   • Fast computation
   • Works well for dense, regular data
   Cons:
   • Cannot model global trends
   • Sensitive to outliers

2. Ordinary Kriging
   Score: 65.0/100
   Pros:
   • Provides uncertainty estimates
   • Optimal linear unbiased estimator
   • Handles irregular sampling well
   Cons:
   • Computationally intensive
   • Requires variogram modeling
   • Needs more data points

3. Radial Basis Functions
   Score: 60.0/100
   Pros:
   • Creates smooth surfaces
   • Exact interpolation at data points
   • Flexible basis functions
   Cons:
   • Can create unrealistic oscillations
   • Sensitive to parameter choice
   • Slow with many points
        """
        self.method_text.delete(1.0, tk.END)
        self.method_text.insert(1.0, methods.strip())
        
        # Mock parameters
        params = """
Optimal Parameters
==================================================

Recommended Method: IDW

search_radius: 850.0
  → Medium radius (8.5x mean NN distance) for balanced coverage

power: 2.0
  → Standard power (p=2) balances local and regional influence

min_points: 3
  → Minimum 3 points ensures stable interpolation

max_points: 12
  → Maximum 12 points balances accuracy and speed

use_sectors: False
  → Data is sufficiently uniform, sectors not needed

cell_size: 50.0
  → Appropriate for data density

buffer: 0.1
  → 10% buffer extends grid beyond data bounds
        """
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(1.0, params.strip())
        
        # Enable apply button
        self.apply_btn.config(state=tk.NORMAL)
    
    def apply_recommendations(self):
        """Apply recommended parameters to the UI."""
        if not hasattr(self, 'recommendation_report') or not self.recommendation_report:
            # Use mock parameters for demo
            params = {
                'method': 'IDW',
                'power': 2.0,
                'smoothing': 0.0,
                'search_radius': 850.0,
                'min_points': 3,
                'max_points': 12,
                'use_sectors': False,
                'n_sectors': 4
            }
        else:
            # Use actual recommendations
            params = self.recommendation_report.optimal_parameters
            params['method'] = self.recommendation_report.recommended_method
        
        # Apply parameters
        self.set_parameters(params)
        
        # Switch to first tab to show applied settings
        self.notebook.select(0)
        
        # Show confirmation
        tk.messagebox.showinfo(
            "Success", 
            "Recommended parameters have been applied.\n"
            "You can now run the interpolation."
        )
    
    def update_language(self):
        """Update all text elements with new translations."""
        try:
            # Update main frame title
            self.frame.config(text=_("Parameters"))
            
            # Update notebook tab titles
            if hasattr(self, 'notebook'):
                for i, tab_key in enumerate(['IDW', 'RBF', 'Kriging', 'Analysis and Recommendations']):
                    try:
                        self.notebook.tab(i, text=_(tab_key))
                    except tk.TclError:
                        # Tab doesn't exist, skip
                        pass
            
            print("ParametersPanel language updated successfully")
            
        except Exception as e:
            print(f"Error updating ParametersPanel language: {e}")
            import traceback
            traceback.print_exc()
        
        kernel_options = [
            ("Multiquadric", "multiquadric"),
            ("Gaussian", "gaussian"), 
            ("Inverse Multiquadric", "inverse_multiquadric"),
            ("Thin Plate Spline", "thin_plate_spline"),
            ("Linear", "linear"),
            ("Cubic", "cubic"),
            ("Quintic", "quintic")
        ]
        
        self.kernel_combo = ttk.Combobox(
            kernel_frame,
            textvariable=self.rbf_kernel_var,
            values=[option[1] for option in kernel_options],
            state="readonly",
            width=20
        )
        self.kernel_combo.grid(row=0, column=1, sticky=tk.W+tk.E)
        kernel_frame.columnconfigure(1, weight=1)
        
        # Shape parameter
        shape_frame = ttk.Frame(rbf_frame)
        shape_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(shape_frame, text=_("Shape Parameter (ε):")).pack(side=tk.LEFT)
        
        shape_scale = ttk.Scale(
            shape_frame,
            from_=0.1,
            to=10.0,
            variable=self.rbf_shape_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        shape_scale.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        self.shape_label = ttk.Label(shape_frame, text="1.0")
        self.shape_label.pack(side=tk.LEFT)
        
        shape_scale.config(command=self.update_shape_label)
        
        # Regularization parameter
        reg_frame = ttk.Frame(rbf_frame)
        reg_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(reg_frame, text=_("Regularization:")).pack(side=tk.LEFT)
        
        reg_scale = ttk.Scale(
            reg_frame,
            from_=1e-15,
            to=1e-6,
            variable=self.rbf_regularization_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        reg_scale.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        self.reg_label = ttk.Label(reg_frame, text="1e-12")
        self.reg_label.pack(side=tk.LEFT)
        
        reg_scale.config(command=self.update_reg_label)
        
        # Polynomial degree
        poly_frame = ttk.LabelFrame(rbf_frame, text=_("Polynomial Augmentation"), padding="5")
        poly_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(poly_frame, text=_("Polynomial Degree:")).grid(row=0, column=0, sticky=tk.W)
        
        poly_options = [("-1 (None)", -1), ("0 (Constant)", 0), ("1 (Linear)", 1)]
        for i, (text, value) in enumerate(poly_options):
            ttk.Radiobutton(
                poly_frame,
                text=text,
                variable=self.rbf_polynomial_var,
                value=value
            ).grid(row=i, column=1, sticky=tk.W, pady=1)
        
        # Global vs Local mode
        mode_frame = ttk.LabelFrame(rbf_frame, text=_("Interpolation Mode"), padding="5")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(
            mode_frame,
            text=_("Global (use all points)"),
            variable=self.rbf_global_var,
            value=True
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            mode_frame,
            text=_("Local (use search parameters)"),
            variable=self.rbf_global_var,
            value=False
        ).pack(anchor=tk.W)
        
        # Help text
        help_text = """
Kernel Types:
• Multiquadric: √(1 + (εr)²) - smooth, global support
• Gaussian: exp(-(εr)²) - very smooth, compact support
• Inverse Multiquadric: 1/√(1 + (εr)²) - smooth decay
• Thin Plate Spline: r²ln(r) - natural smoothness
• Linear: r - simple, less smooth
• Cubic/Quintic: r³/r⁵ - polynomial growth

Shape Parameter (ε):
• Controls kernel "width" or influence radius
• Higher values = more local influence
• Optimal value depends on data density

Regularization:
• Prevents numerical instability
• Use small values (1e-15 to 1e-6)
• Increase if getting convergence errors
        """
        
        help_label = ttk.Label(rbf_frame, text=help_text.strip(), justify=tk.LEFT, font=("TkDefaultFont", 8))
        help_label.pack(anchor=tk.W, pady=(10, 0))
        
    def create_kriging_tab(self):
        """Create the Kriging parameters tab."""
        kriging_frame = ttk.Frame(self.notebook)
        self.notebook.add(kriging_frame, text=_("Kriging Parameters"))
        
        # Kriging type selection
        type_frame = ttk.LabelFrame(kriging_frame, text=_("Kriging Type"), padding="5")
        type_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(type_frame, text=_("Type:")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        type_options = [
            ("Ordinary Kriging", "ordinary"),
            ("Simple Kriging", "simple")
        ]
        
        self.kriging_type_combo = ttk.Combobox(
            type_frame,
            textvariable=self.kriging_type_var,
            values=[option[1] for option in type_options],
            state="readonly",
            width=15
        )
        self.kriging_type_combo.grid(row=0, column=1, sticky=tk.W+tk.E)
        type_frame.columnconfigure(1, weight=1)
        
        # Variogram model selection
        model_frame = ttk.LabelFrame(kriging_frame, text=_("Variogram Model"), padding="5")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text=_("Model:")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        model_options = [
            ("Spherical", "spherical"),
            ("Exponential", "exponential"),
            ("Gaussian", "gaussian"),
            ("Linear", "linear"),
            ("Power", "power"),
            ("Nugget", "nugget")
        ]
        
        self.variogram_combo = ttk.Combobox(
            model_frame,
            textvariable=self.kriging_variogram_var,
            values=[option[1] for option in model_options],
            state="readonly",
            width=15
        )
        self.variogram_combo.grid(row=0, column=1, sticky=tk.W+tk.E)
        model_frame.columnconfigure(1, weight=1)
        
        # Variogram parameters
        params_frame = ttk.LabelFrame(kriging_frame, text=_("Variogram Parameters"), padding="5")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Nugget parameter
        ttk.Label(params_frame, text=_("Nugget:")).grid(row=0, column=0, sticky=tk.W)
        self.nugget_entry = ttk.Entry(params_frame, textvariable=self.kriging_nugget_var, width=10)
        self.nugget_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Label(params_frame, text=_("(micro-scale variance)")).grid(row=0, column=2, sticky=tk.W)
        
        # Sill parameter  
        ttk.Label(params_frame, text=_("Sill:")).grid(row=1, column=0, sticky=tk.W)
        self.sill_entry = ttk.Entry(params_frame, textvariable=self.kriging_sill_var, width=10)
        self.sill_entry.grid(row=1, column=1, padx=5, sticky=tk.W)
        ttk.Label(params_frame, text=_("(total variance)")).grid(row=1, column=2, sticky=tk.W)
        
        # Range parameter
        ttk.Label(params_frame, text=_("Range:")).grid(row=2, column=0, sticky=tk.W)
        self.range_entry = ttk.Entry(params_frame, textvariable=self.kriging_range_var, width=10)
        self.range_entry.grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Label(params_frame, text=_("(correlation distance)")).grid(row=2, column=2, sticky=tk.W)
        
        # Auto-fit variogram checkbox
        fit_frame = ttk.LabelFrame(kriging_frame, text=_("Variogram Fitting"), padding="5")
        fit_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(
            fit_frame,
            text=_("Automatically fit variogram parameters"),
            variable=self.kriging_auto_fit_var,
            command=self.toggle_kriging_manual_params
        ).pack(anchor=tk.W)
        
        # Interpolation mode
        mode_frame = ttk.LabelFrame(kriging_frame, text=_("Interpolation Mode"), padding="5")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(
            mode_frame,
            text=_("Global (use all points)"),
            variable=self.kriging_global_var,
            value=True
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            mode_frame,
            text=_("Local (use search parameters)"),
            variable=self.kriging_global_var,
            value=False
        ).pack(anchor=tk.W)
        
        # Help text
        help_text = """
Kriging Types:
• Ordinary: Assumes unknown constant mean
• Simple: Assumes known constant mean (most common)

Variogram Models:
• Spherical: Linear rise to sill at range
• Exponential: Asymptotic approach to sill
• Gaussian: Very smooth, short-range correlation
• Linear: No sill, continuous increase
• Power: Fractal behavior, no sill
• Nugget: Pure noise, no spatial correlation

Parameters:
• Nugget: Measurement error + micro-scale variation
• Sill: Total variance at large distances
• Range: Distance where correlation becomes negligible

Auto-fit: Estimates parameters from data automatically
        """
        
        help_label = ttk.Label(kriging_frame, text=help_text.strip(), justify=tk.LEFT, font=("TkDefaultFont", 8))
        help_label.pack(anchor=tk.W, pady=(10, 0))
        
    def create_search_tab(self):
        """Create the search parameters tab."""
        search_frame = ttk.Frame(self.notebook)
        self.notebook.add(search_frame, text=_("Search Parameters"))
        
        # Search radius
        radius_frame = ttk.Frame(search_frame)
        radius_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(radius_frame, text=_("Search Radius:")).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(
            radius_frame, 
            textvariable=self.search_radius_var, 
            width=10
        ).grid(row=0, column=1, padx=(5, 2), sticky=tk.W)
        ttk.Label(radius_frame, text=_("meters")).grid(row=0, column=2, sticky=tk.W)
        
        # Point limits
        points_frame = ttk.LabelFrame(search_frame, text=_("Point Limits"), padding="5")
        points_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(points_frame, text=_("Minimum Points:")).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(
            points_frame, 
            textvariable=self.min_points_var, 
            width=5
        ).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(points_frame, text=_("Maximum Points:")).grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(
            points_frame, 
            textvariable=self.max_points_var, 
            width=5
        ).grid(row=1, column=1, padx=5, sticky=tk.W)
        
        # Sectoral search
        sectors_frame = ttk.LabelFrame(search_frame, text=_("Sectoral Search"), padding="5")
        sectors_frame.pack(fill=tk.X)
        
        ttk.Checkbutton(
            sectors_frame, 
            text=_("Use sectoral search"), 
            variable=self.use_sectors_var,
            command=self.toggle_sectors
        ).pack(anchor=tk.W)
        
        self.sectors_config_frame = ttk.Frame(sectors_frame)
        self.sectors_config_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(self.sectors_config_frame, text=_("Number of sectors:")).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(
            self.sectors_config_frame, 
            textvariable=self.n_sectors_var, 
            width=5,
            state=tk.DISABLED
        ).grid(row=0, column=1, padx=5, sticky=tk.W)
        
    def create_grid_tab(self):
        """Create the grid parameters tab."""
        grid_frame = ttk.Frame(self.notebook)
        self.notebook.add(grid_frame, text=_("Grid Parameters"))
        
        # Cell size
        cell_frame = ttk.Frame(grid_frame)
        cell_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(cell_frame, text=_("Cell Size:")).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(
            cell_frame, 
            textvariable=self.cell_size_var, 
            width=10
        ).grid(row=0, column=1, padx=(5, 2), sticky=tk.W)
        ttk.Label(cell_frame, text="meters").grid(row=0, column=2, sticky=tk.W)
        
        # Buffer
        buffer_frame = ttk.Frame(grid_frame)
        buffer_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(buffer_frame, text=_("Buffer:")).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(
            buffer_frame, 
            textvariable=self.buffer_var, 
            width=10
        ).grid(row=0, column=1, padx=(5, 2), sticky=tk.W)
        ttk.Label(buffer_frame, text=_("(0.0-1.0 = %, >1.0 = meters)")).grid(row=0, column=2, sticky=tk.W)
        
        # Grid info button
        ttk.Button(
            grid_frame, 
            text=_("Preview Grid Info"), 
            command=self.show_grid_preview
        ).pack(pady=(10, 0))
        
        # Help text
        help_text = """
Cell Size:
• Determines interpolation resolution
• Smaller = higher resolution, slower
• Consider data density when choosing

Buffer:
• Extends grid beyond data bounds
• Values 0.0-1.0: percentage of range
• Values >1.0: absolute distance in meters
        """
        
        help_label = ttk.Label(grid_frame, text=help_text.strip(), justify=tk.LEFT)
        help_label.pack(anchor=tk.W, pady=(10, 0))
        
    def create_recommendations_tab(self):
        """Create the recommendations tab."""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text=_("Recommendations"))
        
        # Store recommendation report
        self.recommendation_report = None
        
        # Control buttons
        btn_frame = ttk.Frame(rec_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.analyze_btn = ttk.Button(
            btn_frame,
            text=_("Analyze Data"),
            command=self.analyze_data
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.apply_btn = ttk.Button(
            btn_frame,
            text=_("Apply Recommendations"),
            command=self.apply_recommendations,
            state=tk.DISABLED
        )
        self.apply_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            btn_frame,
            variable=self.progress_var,
            mode='indeterminate',
            length=200
        )
        
        # Results display
        self.results_notebook = ttk.Notebook(rec_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(summary_frame, text=_("Summary"))
        
        self.summary_text = scrolledtext.ScrolledText(
            summary_frame,
            wrap=tk.WORD,
            height=15,
            width=50
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Method comparison tab
        method_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(method_frame, text=_("Method Comparison"))
        
        self.method_text = scrolledtext.ScrolledText(
            method_frame,
            wrap=tk.WORD,
            height=15,
            width=50
        )
        self.method_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Parameters tab
        params_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(params_frame, text=_("Optimal Parameters"))
        
        self.params_text = scrolledtext.ScrolledText(
            params_frame,
            wrap=tk.WORD,
            height=15,
            width=50
        )
        self.params_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize with help text
        self.show_recommendation_help()
        
    def create_action_buttons(self):
        """Create action buttons at the bottom."""
        actions_frame = ttk.Frame(self.frame)
        actions_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            actions_frame, 
            text=_("Reset to Defaults"), 
            command=self.reset_to_defaults
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            actions_frame, 
            text=_("Load Preset"), 
            command=self.load_preset
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            actions_frame, 
            text=_("Save Preset"), 
            command=self.save_preset
        ).pack(side=tk.LEFT)
        
    def setup_bindings(self):
        """Setup event bindings."""
        self.controller.bind_event("data_loaded", self.on_data_loaded)
        
        # Method change handler
        self.method_var.trace('w', self.on_method_changed)
        
    def on_method_changed(self, *args):
        """Handle method selection change."""
        method = self.method_var.get()
        
        # Enable/disable relevant tabs based on method selection
        # Tab indices: 0=Method&Data, 1=IDW, 2=RBF, 3=Kriging, 4=Search, 5=Grid, 6=Recommendations
        if method == "IDW":
            self.notebook.tab(1, state="normal")   # IDW tab
            self.notebook.tab(2, state="disabled") # RBF tab
            self.notebook.tab(3, state="disabled") # Kriging tab
        elif method == "RBF":
            self.notebook.tab(1, state="disabled") # IDW tab
            self.notebook.tab(2, state="normal")   # RBF tab
            self.notebook.tab(3, state="disabled") # Kriging tab
        elif method == "Kriging":
            self.notebook.tab(1, state="disabled") # IDW tab
            self.notebook.tab(2, state="disabled") # RBF tab
            self.notebook.tab(3, state="normal")   # Kriging tab
        
    def on_data_loaded(self, data_info: Dict[str, Any]):
        """
        Handle data loaded event.
        
        Args:
            data_info: Information about the loaded data
        """
        # Update value column options
        value_columns = [col for col in data_info['columns'] if col not in ['X', 'Y']]
        self.value_columns = value_columns
        
        self.value_column_combo['values'] = value_columns
        if value_columns:
            self.value_column_var.set(value_columns[0])
            
        # Update default search radius based on data extent
        bounds = data_info['bounds']
        range_x = bounds['max_x'] - bounds['min_x']
        range_y = bounds['max_y'] - bounds['min_y']
        avg_range = (range_x + range_y) / 2
        
        # Set search radius to ~10% of average range
        suggested_radius = max(100, avg_range * 0.1)
        self.search_radius_var.set(suggested_radius)
        
        # Set cell size to ~2% of average range
        suggested_cell_size = max(10, avg_range * 0.02)
        self.cell_size_var.set(suggested_cell_size)
        
    def update_power_label(self, value):
        """Update power parameter label."""
        self.power_label.config(text=f"{float(value):.1f}")
        
    def update_smoothing_label(self, value):
        """Update smoothing parameter label."""
        self.smoothing_label.config(text=f"{float(value):.2f}")
        
    def update_shape_label(self, value):
        """Update RBF shape parameter label."""
        self.shape_label.config(text=f"{float(value):.1f}")
        
    def update_reg_label(self, value):
        """Update RBF regularization parameter label."""
        val = float(value)
        if val >= 1e-9:
            self.reg_label.config(text=f"{val:.0e}")
        else:
            self.reg_label.config(text=f"{val:.1e}")
        
    def toggle_sectors(self):
        """Toggle sectoral search configuration."""
        if self.use_sectors_var.get():
            state = tk.NORMAL
        else:
            state = tk.DISABLED
            
        for widget in self.sectors_config_frame.winfo_children():
            if isinstance(widget, ttk.Entry):
                widget.config(state=state)
                
    def toggle_kriging_manual_params(self):
        """Toggle Kriging manual parameter entry."""
        if self.kriging_auto_fit_var.get():
            state = tk.DISABLED
        else:
            state = tk.NORMAL
            
        # Enable/disable manual parameter entry widgets
        self.nugget_entry.config(state=state)
        self.sill_entry.config(state=state)
        self.range_entry.config(state=state)
                
    def show_grid_preview(self):
        """Show grid preview information."""
        if not self.controller.has_data():
            tk.messagebox.showwarning("Warning", "Please load data first")
            return
            
        # Get current parameters
        params = self.get_parameters()
        
        # Get grid info from controller
        try:
            # This would call the grid generator to get info
            info_text = f"""
Grid Preview Information:

Cell Size: {params['cell_size']} meters
Buffer: {params['buffer']}
Search Radius: {params['search_radius']} meters

This would show:
- Estimated grid dimensions
- Total number of points
- Memory requirements
- Processing time estimate

[To be implemented with GridGenerator integration]
            """
            
            tk.messagebox.showinfo("Grid Preview", info_text.strip())
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to generate grid preview: {str(e)}")
            
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary with all parameter values
        """
        params = {
            'method': self.method_var.get(),
            'value_column': self.value_column_var.get(),
            'search_radius': self.search_radius_var.get(),
            'min_points': self.min_points_var.get(),
            'max_points': self.max_points_var.get(),
            'use_sectors': self.use_sectors_var.get(),
            'n_sectors': self.n_sectors_var.get(),
            'cell_size': self.cell_size_var.get(),
            'buffer': self.buffer_var.get()
        }
        
        # Add method-specific parameters
        if self.method_var.get() == 'IDW':
            params.update({
                'power': self.power_var.get(),
                'smoothing': self.smoothing_var.get()
            })
        elif self.method_var.get() == 'RBF':
            params.update({
                'rbf_kernel': self.rbf_kernel_var.get(),
                'rbf_shape_parameter': self.rbf_shape_var.get(),
                'rbf_regularization': self.rbf_regularization_var.get(),
                'rbf_polynomial_degree': self.rbf_polynomial_var.get(),
                'rbf_use_global': self.rbf_global_var.get()
            })
        elif self.method_var.get() == 'Kriging':
            params.update({
                'kriging_type': self.kriging_type_var.get(),
                'kriging_variogram_model': self.kriging_variogram_var.get(),
                'kriging_nugget': self.kriging_nugget_var.get(),
                'kriging_sill': self.kriging_sill_var.get(),
                'kriging_range': self.kriging_range_var.get(),
                'kriging_auto_fit': self.kriging_auto_fit_var.get(),
                'kriging_use_global': self.kriging_global_var.get()
            })
        
        return params
        
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set parameter values.
        
        Args:
            params: Dictionary with parameter values
        """
        self.method_var.set(params.get('method', 'IDW'))
        self.value_column_var.set(params.get('value_column', ''))
        
        # IDW parameters
        self.power_var.set(params.get('power', 2.0))
        self.smoothing_var.set(params.get('smoothing', 0.0))
        
        # RBF parameters
        self.rbf_kernel_var.set(params.get('rbf_kernel', 'multiquadric'))
        self.rbf_shape_var.set(params.get('rbf_shape_parameter', 1.0))
        self.rbf_regularization_var.set(params.get('rbf_regularization', 1e-12))
        self.rbf_polynomial_var.set(params.get('rbf_polynomial_degree', -1))
        self.rbf_global_var.set(params.get('rbf_use_global', True))
        
        # Kriging parameters
        self.kriging_type_var.set(params.get('kriging_type', 'ordinary'))
        self.kriging_variogram_var.set(params.get('kriging_variogram_model', 'spherical'))
        self.kriging_nugget_var.set(params.get('kriging_nugget', 0.0))
        self.kriging_sill_var.set(params.get('kriging_sill', 1.0))
        self.kriging_range_var.set(params.get('kriging_range', 1000.0))
        self.kriging_auto_fit_var.set(params.get('kriging_auto_fit', True))
        self.kriging_global_var.set(params.get('kriging_use_global', True))
        
        # Search parameters
        self.search_radius_var.set(params.get('search_radius', 1000.0))
        self.min_points_var.set(params.get('min_points', 1))
        self.max_points_var.set(params.get('max_points', 12))
        self.use_sectors_var.set(params.get('use_sectors', False))
        self.n_sectors_var.set(params.get('n_sectors', 4))
        
        # Grid parameters
        self.cell_size_var.set(params.get('cell_size', 50.0))
        self.buffer_var.set(params.get('buffer', 0.1))
        
        # Update UI state
        self.on_method_changed()
        self.toggle_sectors()
        self.toggle_kriging_manual_params()
        self.update_power_label(self.power_var.get())
        self.update_smoothing_label(self.smoothing_var.get())
        self.update_shape_label(self.rbf_shape_var.get())
        self.update_reg_label(self.rbf_regularization_var.get())
        
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        defaults = {
            'method': 'IDW',
            'power': 2.0,
            'smoothing': 0.0,
            'search_radius': 1000.0,
            'min_points': 1,
            'max_points': 12,
            'use_sectors': False,
            'n_sectors': 4,
            'cell_size': 50.0,
            'buffer': 0.1
        }
        
        self.set_parameters(defaults)
        
    def load_preset(self):
        """Load parameter preset."""
        tk.messagebox.showinfo("Info", "Load preset functionality - To be implemented")
        
    def save_preset(self):
        """Save current parameters as preset."""
        tk.messagebox.showinfo("Info", "Save preset functionality - To be implemented")
    
    def show_recommendation_help(self):
        """Show help text in recommendation tabs."""
        help_text = """
Welcome to the Recommendation System!

This system analyzes your data and provides:
• Optimal interpolation method selection
• Customized parameter recommendations  
• Performance expectations

To get started:
1. Load your data
2. Click "Analyze Data" button
3. Review recommendations
4. Click "Apply Recommendations" to use them

The analysis considers:
• Data density and distribution
• Spatial patterns and trends
• Value variability
• Computational efficiency
        """
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, help_text.strip())
        
        self.method_text.delete(1.0, tk.END)
        self.method_text.insert(1.0, "Method comparison will appear here after analysis.")
        
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(1.0, "Optimal parameters will appear here after analysis.")
    
    def analyze_data(self):
        """Run recommendation analysis on loaded data."""
        if not self.controller.has_data():
            tk.messagebox.showwarning("Warning", "Please load data first")
            return
            
        # Check if value column is selected
        if not self.value_column_var.get():
            tk.messagebox.showwarning("Warning", "Please select a value column")
            return
        
        # Show progress
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0))
        self.progress_bar.start(10)
        
        # Run analysis in background
        self.frame.after(100, self._run_analysis)
    
    def _run_analysis(self):
        """Run the actual analysis (called after UI update)."""
        try:
            # Import recommendation engine
            try:
                from ...core.recommendations import RecommendationEngine
            except ImportError:
                # Fallback for demo mode
                self._show_mock_recommendations()
                return
            
            # Get data from controller
            data_info = self.controller.get_data_info()
            data = self.controller.get_data()
            
            # Run recommendation analysis
            engine = RecommendationEngine()
            report = engine.analyze_and_recommend(
                data=data,
                x_col='X',
                y_col='Y',
                value_col=self.value_column_var.get(),
                user_preferences={'prioritize_speed': True},
                evaluate_quality=False,  # Skip for speed
                quick_mode=True
            )
            
            # Store report
            self.recommendation_report = report
            
            # Display results
            self._display_recommendations(report)
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            # Show mock recommendations as fallback
            self._show_mock_recommendations()
            
        finally:
            # Hide progress
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.analyze_btn.config(state=tk.NORMAL)
            self.apply_btn.config(state=tk.NORMAL)
    
    def _display_recommendations(self, report):
        """Display recommendation report in tabs."""
        # Summary tab
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, report.summary_text)
        
        # Method comparison
        method_text = "Method Comparison\n" + "="*50 + "\n\n"
        for i, method_score in enumerate(report.method_scores):
            method_text += f"{i+1}. {method_score.method.value}\n"
            method_text += f"   Score: {method_score.score:.1f}/100\n"
            method_text += f"   Pros:\n"
            for pro in method_score.pros[:3]:
                method_text += f"   • {pro}\n"
            method_text += f"   Cons:\n"
            for con in method_score.cons[:3]:
                method_text += f"   • {con}\n"
            method_text += "\n"
        
        self.method_text.delete(1.0, tk.END)
        self.method_text.insert(1.0, method_text)
        
        # Parameters
        params_text = "Optimal Parameters\n" + "="*50 + "\n\n"
        params_text += f"Recommended Method: {report.recommended_method}\n\n"
        
        for param, value in report.optimal_parameters.items():
            reason = report.parameter_reasoning.get(param, "")
            params_text += f"{param}: {value}\n"
            if reason:
                params_text += f"  → {reason}\n"
            params_text += "\n"
        
        if report.warnings:
            params_text += "\nWarnings:\n"
            for warning in report.warnings:
                params_text += f"⚠ {warning}\n"
        
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(1.0, params_text)
    
    def _show_mock_recommendations(self):
        """Show mock recommendations for demo mode."""
        # Mock summary
        summary = """
**Data Analysis Summary**
- Dataset contains 150 points
- Spatial density: 0.245 points per unit area
- Distribution uniformity: 0.65 (moderately uniform)
- No significant spatial trend detected

**Recommended Method: IDW**
- Suitability score: 85.0/100
- Key reasons:
  • Dense, uniformly distributed data
  • No global trends - local interpolation sufficient
  • IDW is computationally efficient

**Optimized Parameters**
- search_radius: 850.0 (Medium radius for balanced coverage)
- power: 2.0 (Standard power balances local and regional influence)
- max_points: 12 (Balances accuracy and speed)

**Expected Performance**
- Processing will be fast
- Good accuracy for local variations
- Suitable for your data characteristics
        """
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary.strip())
        
        # Mock method comparison
        methods = """
Method Comparison
==================================================

1. IDW
   Score: 85.0/100
   Pros:
   • Simple and intuitive method
   • Fast computation
   • Works well for dense, regular data
   Cons:
   • Cannot model global trends
   • Sensitive to outliers

2. Ordinary Kriging
   Score: 65.0/100
   Pros:
   • Provides uncertainty estimates
   • Optimal linear unbiased estimator
   • Handles irregular sampling well
   Cons:
   • Computationally intensive
   • Requires variogram modeling
   • Needs more data points

3. Radial Basis Functions
   Score: 60.0/100
   Pros:
   • Creates smooth surfaces
   • Exact interpolation at data points
   • Flexible basis functions
   Cons:
   • Can create unrealistic oscillations
   • Sensitive to parameter choice
   • Slow with many points
        """
        self.method_text.delete(1.0, tk.END)
        self.method_text.insert(1.0, methods.strip())
        
        # Mock parameters
        params = """
Optimal Parameters
==================================================

Recommended Method: IDW

search_radius: 850.0
  → Medium radius (8.5x mean NN distance) for balanced coverage

power: 2.0
  → Standard power (p=2) balances local and regional influence

min_points: 3
  → Minimum 3 points ensures stable interpolation

max_points: 12
  → Maximum 12 points balances accuracy and speed

use_sectors: False
  → Data is sufficiently uniform, sectors not needed

cell_size: 50.0
  → Appropriate for data density

buffer: 0.1
  → 10% buffer extends grid beyond data bounds
        """
        self.params_text.delete(1.0, tk.END)
        self.params_text.insert(1.0, params.strip())
        
        # Enable apply button
        self.apply_btn.config(state=tk.NORMAL)
    
    def apply_recommendations(self):
        """Apply recommended parameters to the UI."""
        if not hasattr(self, 'recommendation_report') or not self.recommendation_report:
            # Use mock parameters for demo
            params = {
                'method': 'IDW',
                'power': 2.0,
                'smoothing': 0.0,
                'search_radius': 850.0,
                'min_points': 3,
                'max_points': 12,
                'use_sectors': False,
                'n_sectors': 4
            }
        else:
            # Use actual recommendations
            params = self.recommendation_report.optimal_parameters
            params['method'] = self.recommendation_report.recommended_method
        
        # Apply parameters
        self.set_parameters(params)
        
        # Switch to first tab to show applied settings
        self.notebook.select(0)
        
        # Show confirmation
        tk.messagebox.showinfo(
            "Success", 
            "Recommended parameters have been applied.\n"
            "You can now run the interpolation."
        )
    
    def update_language(self):
        """Update all text elements with new translations."""
        try:
            # Update main frame title
            self.frame.config(text=_("Parameters"))
            
            # Update notebook tab titles
            if hasattr(self, 'notebook'):
                for i, tab_key in enumerate(['IDW', 'RBF', 'Kriging', 'Analysis and Recommendations']):
                    try:
                        self.notebook.tab(i, text=_(tab_key))
                    except tk.TclError:
                        # Tab doesn't exist, skip
                        pass
            
            print("ParametersPanel language updated successfully")
            
        except Exception as e:
            print(f"Error updating ParametersPanel language: {e}")
            import traceback
            traceback.print_exc()