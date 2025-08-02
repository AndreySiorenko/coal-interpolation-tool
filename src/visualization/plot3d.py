"""
3D visualization utilities using VTK for advanced scientific visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import warnings

# Try to import VTK - it's optional but recommended for 3D visualization
try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
    
    # Define mock vtk types for type hints when VTK is available
    vtkActor = vtk.vtkActor
    vtkVolume = vtk.vtkVolume
except ImportError:
    vtk = None
    VTK_AVAILABLE = False
    
    # Create mock types for type hints when VTK is not available
    class MockVTKType:
        pass
    vtkActor = MockVTKType
    vtkVolume = MockVTKType

# Fallback to matplotlib for basic 3D plotting
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


class VTKRenderer:
    """
    VTK-based 3D renderer for high-quality scientific visualization.
    
    Provides advanced 3D rendering capabilities including:
    - Point clouds with scalar coloring
    - Surface meshes from interpolation results
    - Isosurfaces and contour surfaces
    - Volume rendering for 3D data
    - Interactive navigation and selection
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize VTK renderer.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK is required for 3D visualization. Install with: pip install vtk")
        
        self.width = width
        self.height = height
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.actors = []
        
        self._setup_renderer()
    
    def _setup_renderer(self):
        """Setup VTK renderer, window, and interactor."""
        # Create renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.2)  # Dark blue background
        
        # Create render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(self.width, self.height)
        self.render_window.SetWindowName("Coal Interpolation 3D Viewer")
        
        # Create interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        # Set up camera
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 0, 1000)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        
        # Add axes
        self._add_axes()
    
    def _add_axes(self):
        """Add coordinate axes to the scene."""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(100, 100, 100)
        axes.SetShaftType(0)  # Cylinder shaft
        axes.SetAxisLabels(1)
        axes.SetCylinderRadius(0.02)
        axes.SetConeRadius(0.4)
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        
        # Position axes at origin
        axes.SetPosition(0, 0, 0)
        
        self.renderer.AddActor(axes)
    
    def render_points(self,
                     data: pd.DataFrame,
                     x_col: str,
                     y_col: str,
                     z_col: str,
                     value_col: Optional[str] = None,
                     point_size: float = 5.0,
                     colormap: str = 'viridis') -> vtkActor:
        """
        Render 3D point cloud from well data.
        
        Args:
            data: DataFrame with well data
            x_col: X coordinate column name
            y_col: Y coordinate column name
            z_col: Z coordinate column name
            value_col: Value column name for coloring (optional)
            point_size: Size of rendered points
            colormap: Colormap name
            
        Returns:
            VTK actor for the points
        """
        # Extract coordinates
        points = vtk.vtkPoints()
        n_points = len(data)
        
        for i in range(n_points):
            x = data.iloc[i][x_col]
            y = data.iloc[i][y_col]
            z = data.iloc[i][z_col]
            points.InsertNextPoint(x, y, z)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Add scalar values if provided
        if value_col is not None:
            values = numpy_support.numpy_to_vtk(data[value_col].values)
            values.SetName(value_col)
            polydata.GetPointData().SetScalars(values)
        
        # Create vertices for rendering points
        vertices = vtk.vtkVertexGlyphFilter()
        vertices.SetInputData(polydata)
        vertices.Update()
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertices.GetOutputPort())
        
        if value_col is not None:
            # Set color mapping
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToMapScalars()
            
            # Apply colormap
            lut = self._create_lookup_table(data[value_col].values, colormap)
            mapper.SetLookupTable(lut)
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        
        # Add to renderer
        self.renderer.AddActor(actor)
        self.actors.append(actor)
        
        return actor
    
    def render_surface(self,
                      x_grid: np.ndarray,
                      y_grid: np.ndarray,
                      z_grid: np.ndarray,
                      colormap: str = 'viridis',
                      opacity: float = 1.0) -> vtkActor:
        """
        Render interpolated surface from grid data.
        
        Args:
            x_grid: X coordinate grid
            y_grid: Y coordinate grid
            z_grid: Interpolated values as heights
            colormap: Colormap name
            opacity: Surface opacity (0-1)
            
        Returns:
            VTK actor for the surface
        """
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        dims = x_grid.shape
        grid.SetDimensions(dims[1], dims[0], 1)
        
        # Create points
        points = vtk.vtkPoints()
        scalars = vtk.vtkFloatArray()
        scalars.SetName("Interpolated Values")
        
        for i in range(dims[0]):
            for j in range(dims[1]):
                x = x_grid[i, j]
                y = y_grid[i, j]
                z = z_grid[i, j] if not np.isnan(z_grid[i, j]) else 0
                
                points.InsertNextPoint(x, y, z)
                scalars.InsertNextValue(z)
        
        grid.SetPoints(points)
        grid.GetPointData().SetScalars(scalars)
        
        # Create surface mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(grid)
        
        # Apply colormap
        values = z_grid[~np.isnan(z_grid)]
        if len(values) > 0:
            lut = self._create_lookup_table(values, colormap)
            mapper.SetLookupTable(lut)
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetInterpolationToGouraud()
        
        # Add to renderer
        self.renderer.AddActor(actor)
        self.actors.append(actor)
        
        return actor
    
    def render_isosurface(self,
                         x_grid: np.ndarray,
                         y_grid: np.ndarray,
                         z_grid: np.ndarray,
                         isovalue: float,
                         color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                         opacity: float = 0.7) -> vtkActor:
        """
        Render isosurface at specified value.
        
        Args:
            x_grid: X coordinate grid
            y_grid: Y coordinate grid
            z_grid: Value grid
            isovalue: Value for isosurface
            color: RGB color tuple (0-1)
            opacity: Surface opacity (0-1)
            
        Returns:
            VTK actor for the isosurface
        """
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        dims = x_grid.shape
        grid.SetDimensions(dims[1], dims[0], 1)
        
        # Create points and scalars
        points = vtk.vtkPoints()
        scalars = vtk.vtkFloatArray()
        scalars.SetName("Values")
        
        for i in range(dims[0]):
            for j in range(dims[1]):
                x = x_grid[i, j]
                y = y_grid[i, j]
                z = 0  # Flat surface for 2D data
                value = z_grid[i, j] if not np.isnan(z_grid[i, j]) else 0
                
                points.InsertNextPoint(x, y, z)
                scalars.InsertNextValue(value)
        
        grid.SetPoints(points)
        grid.GetPointData().SetScalars(scalars)
        
        # Create contour filter
        contour = vtk.vtkContourFilter()
        contour.SetInputData(grid)
        contour.SetValue(0, isovalue)
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLineWidth(2)
        
        # Add to renderer
        self.renderer.AddActor(actor)
        self.actors.append(actor)
        
        return actor
    
    def render_volume(self,
                     x_grid: np.ndarray,
                     y_grid: np.ndarray,
                     z_grid: np.ndarray,
                     colormap: str = 'viridis',
                     opacity_function: Optional[Callable] = None) -> vtkVolume:
        """
        Render volume visualization of 3D data.
        
        Args:
            x_grid: X coordinate grid
            y_grid: Y coordinate grid
            z_grid: Z coordinate grid (3D data)
            colormap: Colormap name
            opacity_function: Custom opacity transfer function
            
        Returns:
            VTK volume for the data
        """
        if z_grid.ndim != 3:
            raise ValueError("Volume rendering requires 3D data")
        
        # Create image data
        image_data = vtk.vtkImageData()
        dims = z_grid.shape
        image_data.SetDimensions(dims[2], dims[1], dims[0])
        
        # Set spacing based on grid
        x_spacing = (x_grid.max() - x_grid.min()) / dims[2]
        y_spacing = (y_grid.max() - y_grid.min()) / dims[1]
        z_spacing = 1.0  # Assume unit spacing in Z
        image_data.SetSpacing(x_spacing, y_spacing, z_spacing)
        
        # Set origin
        image_data.SetOrigin(x_grid.min(), y_grid.min(), 0)
        
        # Convert data to VTK array
        flat_data = z_grid.flatten(order='F')  # Fortran order for VTK
        vtk_array = numpy_support.numpy_to_vtk(flat_data)
        vtk_array.SetName("Volume Data")
        
        image_data.GetPointData().SetScalars(vtk_array)
        
        # Create volume mapper
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(image_data)
        
        # Create color transfer function
        color_function = vtk.vtkColorTransferFunction()
        value_range = [flat_data.min(), flat_data.max()]
        
        # Apply colormap
        n_colors = 256
        for i in range(n_colors):
            val = value_range[0] + i * (value_range[1] - value_range[0]) / (n_colors - 1)
            # Simple grayscale for now - could implement full colormaps
            gray = i / (n_colors - 1)
            color_function.AddRGBPoint(val, gray, gray, gray)
        
        # Create opacity transfer function
        opacity_function_vtk = vtk.vtkPiecewiseFunction()
        if opacity_function is None:
            # Default linear opacity
            opacity_function_vtk.AddPoint(value_range[0], 0.0)
            opacity_function_vtk.AddPoint(value_range[1], 1.0)
        else:
            # Custom opacity function
            for i in range(n_colors):
                val = value_range[0] + i * (value_range[1] - value_range[0]) / (n_colors - 1)
                opacity = opacity_function(val)
                opacity_function_vtk.AddPoint(val, opacity)
        
        # Create volume property
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_function)
        volume_property.SetScalarOpacity(opacity_function_vtk)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        
        # Create volume
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        # Add to renderer
        self.renderer.AddVolume(volume)
        
        return volume
    
    def _create_lookup_table(self, values: np.ndarray, colormap: str) -> vtk.vtkLookupTable:
        """
        Create VTK lookup table for colormap.
        
        Args:
            values: Data values for range
            colormap: Colormap name
            
        Returns:
            VTK lookup table
        """
        lut = vtk.vtkLookupTable()
        n_colors = 256
        lut.SetNumberOfTableValues(n_colors)
        
        value_range = [values.min(), values.max()]
        lut.SetRange(value_range)
        
        # Simple colormaps - could be extended with matplotlib colormaps
        if colormap == 'viridis':
            # Viridis-like colormap
            for i in range(n_colors):
                t = i / (n_colors - 1)
                r = max(0, min(1, 0.267004 + t * (0.993248 - 0.267004)))
                g = max(0, min(1, 0.004874 + t * (0.906157 - 0.004874)))
                b = max(0, min(1, 0.329415 + t * (0.143936 - 0.329415)))
                lut.SetTableValue(i, r, g, b, 1.0)
        elif colormap == 'plasma':
            # Plasma-like colormap
            for i in range(n_colors):
                t = i / (n_colors - 1)
                r = max(0, min(1, 0.050383 + t * (0.940015 - 0.050383)))
                g = max(0, min(1, 0.029803 + t * (0.975158 - 0.029803)))
                b = max(0, min(1, 0.527975 + t * (0.131326 - 0.527975)))
                lut.SetTableValue(i, r, g, b, 1.0)
        else:
            # Default rainbow colormap
            for i in range(n_colors):
                t = i / (n_colors - 1)
                r = max(0, min(1, 1.0 - t))
                g = max(0, min(1, 2 * t * (1 - t)))
                b = max(0, min(1, t))
                lut.SetTableValue(i, r, g, b, 1.0)
        
        lut.Build()
        return lut
    
    def add_colorbar(self, actor: vtkActor, title: str = "Values"):
        """
        Add colorbar to the renderer.
        
        Args:
            actor: Actor to create colorbar for
            title: Colorbar title
        """
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(actor.GetMapper().GetLookupTable())
        scalar_bar.SetTitle(title)
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        
        # Position colorbar
        scalar_bar.SetPosition(0.85, 0.1)
        scalar_bar.SetWidth(0.1)
        scalar_bar.SetHeight(0.8)
        
        self.renderer.AddActor2D(scalar_bar)
    
    def set_camera_position(self, position: Tuple[float, float, float],
                           focal_point: Tuple[float, float, float] = (0, 0, 0),
                           view_up: Tuple[float, float, float] = (0, 0, 1)):
        """
        Set camera position and orientation.
        
        Args:
            position: Camera position (x, y, z)
            focal_point: Point camera looks at
            view_up: Up direction vector
        """
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(position)
        camera.SetFocalPoint(focal_point)
        camera.SetViewUp(view_up)
        self.renderer.ResetCameraClippingRange()
    
    def start_interaction(self):
        """Start interactive visualization."""
        # Set up interaction style
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Start interaction
        self.render_window.Render()
        self.interactor.Start()
    
    def render_to_image(self, filepath: str, magnification: int = 1):
        """
        Render scene to image file.
        
        Args:
            filepath: Output image path
            magnification: Image magnification factor
        """
        # Create window to image filter
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.render_window)
        window_to_image.SetMagnification(magnification)
        window_to_image.Update()
        
        # Determine file format and create writer
        ext = filepath.lower().split('.')[-1]
        
        if ext == 'png':
            writer = vtk.vtkPNGWriter()
        elif ext == 'jpg' or ext == 'jpeg':
            writer = vtk.vtkJPEGWriter()
        elif ext == 'tif' or ext == 'tiff':
            writer = vtk.vtkTIFFWriter()
        elif ext == 'bmp':
            writer = vtk.vtkBMPWriter()
        else:
            writer = vtk.vtkPNGWriter()  # Default to PNG
        
        writer.SetFileName(filepath)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
    
    def clear_scene(self):
        """Clear all actors from the scene."""
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
        self.actors.clear()
    
        def close(self):
            """Close the render window and clean up."""
            if self.render_window:
                self.render_window.Finalize()
            if self.interactor:
                self.interactor.TerminateApp()

class Plot3D:
    """
    3D plotting utilities with fallback to matplotlib if VTK unavailable.
    """
    
    def __init__(self, use_vtk: bool = True, figure_size: Tuple[float, float] = (12, 9)):
        """
        Initialize 3D plotter.
        
        Args:
            use_vtk: Whether to use VTK for rendering (falls back to matplotlib)
            figure_size: Figure size for matplotlib backend
        """
        self.use_vtk = use_vtk and VTK_AVAILABLE
        self.figure_size = figure_size
        self.vtk_renderer = None
        self.mpl_figure = None
        
        if self.use_vtk:
            self.vtk_renderer = VTKRenderer()
        elif not MATPLOTLIB_AVAILABLE:
            raise ImportError("Neither VTK nor matplotlib available for 3D plotting")
    
    def render_points_3d(self, 
                        data: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        z_col: str,
                        value_col: Optional[str] = None,
                        **kwargs):
        """
        Render 3D point cloud.
        
        Args:
            data: DataFrame with data
            x_col: X coordinate column
            y_col: Y coordinate column  
            z_col: Z coordinate column
            value_col: Value column for coloring
            **kwargs: Additional rendering arguments
        """
        if self.use_vtk:
            return self.vtk_renderer.render_points(data, x_col, y_col, z_col, value_col, **kwargs)
        else:
            return self._render_points_matplotlib(data, x_col, y_col, z_col, value_col, **kwargs)
    
    def _render_points_matplotlib(self, data, x_col, y_col, z_col, value_col, **kwargs):
        """Fallback matplotlib 3D point rendering."""
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        x = data[x_col].values
        y = data[y_col].values
        z = data[z_col].values
        
        if value_col is not None:
            colors = data[value_col].values
            scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', **kwargs)
            plt.colorbar(scatter, label=value_col)
        else:
            ax.scatter(x, y, z, **kwargs)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        
        self.mpl_figure = fig
        return ax
    
    def render_surface_3d(self, x_grid, y_grid, z_grid, **kwargs):
        """
        Render 3D surface.
        
        Args:
            x_grid: X coordinate grid
            y_grid: Y coordinate grid
            z_grid: Z value grid
            **kwargs: Additional rendering arguments
        """
        if self.use_vtk:
            return self.vtk_renderer.render_surface(x_grid, y_grid, z_grid, **kwargs)
        else:
            return self._render_surface_matplotlib(x_grid, y_grid, z_grid, **kwargs)
    
    def _render_surface_matplotlib(self, x_grid, y_grid, z_grid, **kwargs):
        """Fallback matplotlib 3D surface rendering."""
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        surface = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', **kwargs)
        plt.colorbar(surface, label='Interpolated Values')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Interpolated Value')
        
        self.mpl_figure = fig
        return ax
    
    def show(self):
        """Display the 3D visualization."""
        if self.use_vtk and self.vtk_renderer:
            self.vtk_renderer.start_interaction()
        elif self.mpl_figure:
            plt.show()
    
    def save_image(self, filepath: str, **kwargs):
        """
        Save visualization to image file.
        
        Args:
            filepath: Output file path
            **kwargs: Additional save arguments
        """
        if self.use_vtk and self.vtk_renderer:
            self.vtk_renderer.render_to_image(filepath, **kwargs)
        elif self.mpl_figure:
            self.mpl_figure.savefig(filepath, **kwargs)
    
    def close(self):
        """Close visualization and clean up."""
        if self.use_vtk and self.vtk_renderer:
            self.vtk_renderer.close()
        elif self.mpl_figure:
            plt.close(self.mpl_figure)