"""
2D visualization utilities using matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, Any, Optional, Tuple, List, Union
import warnings


class Plot2D:
    """
    2D plotting utilities for interpolation data and results.
    
    Provides methods for:
    - Scatter plots of well data
    - Contour plots of interpolation results
    - Combined data and results visualization
    - Statistical plots (histograms, box plots)
    """
    
    def __init__(self, figure_size: Tuple[float, float] = (12, 8), dpi: int = 100):
        """
        Initialize 2D plotter.
        
        Args:
            figure_size: Figure size in inches (width, height)
            dpi: Figure resolution
        """
        self.figure_size = figure_size
        self.dpi = dpi
        self.current_figure = None
        self.current_axes = None
        
    def create_figure(self, nrows: int = 1, ncols: int = 1) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create a new figure with subplots.
        
        Args:
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(nrows, ncols, figsize=self.figure_size, dpi=self.dpi)
        self.current_figure = fig
        self.current_axes = axes
        return fig, axes
    
    def plot_scatter(self, 
                    data: pd.DataFrame,
                    x_col: str,
                    y_col: str,
                    value_col: str,
                    ax: Optional[plt.Axes] = None,
                    title: str = "Well Data",
                    colormap: str = 'viridis',
                    point_size: float = 50,
                    show_colorbar: bool = True,
                    **kwargs) -> plt.Axes:
        """
        Create scatter plot of well data points.
        
        Args:
            data: DataFrame with well data
            x_col: X coordinate column name
            y_col: Y coordinate column name
            value_col: Value column name for coloring
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            colormap: Colormap name
            point_size: Size of scatter points
            show_colorbar: Whether to show colorbar
            **kwargs: Additional scatter plot arguments
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = self.create_figure()
        
        # Extract coordinates and values
        x = data[x_col].values
        y = data[y_col].values
        values = data[value_col].values
        
        # Create scatter plot
        scatter = ax.scatter(x, y, c=values, s=point_size, cmap=colormap, 
                           alpha=0.8, edgecolors='black', linewidth=0.5, **kwargs)
        
        # Add colorbar
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label(value_col, rotation=270, labelpad=15)
        
        # Set labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        return ax
    
    def plot_contour(self,
                    x_grid: np.ndarray,
                    y_grid: np.ndarray,
                    z_grid: np.ndarray,
                    ax: Optional[plt.Axes] = None,
                    title: str = "Interpolation Results",
                    colormap: str = 'viridis',
                    levels: Optional[Union[int, List[float]]] = None,
                    show_contour_lines: bool = True,
                    show_colorbar: bool = True,
                    contour_line_color: str = 'black',
                    contour_line_width: float = 0.5,
                    **kwargs) -> plt.Axes:
        """
        Create contour plot of interpolation results.
        
        Args:
            x_grid: X coordinate grid
            y_grid: Y coordinate grid  
            z_grid: Interpolated values grid
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            colormap: Colormap name
            levels: Contour levels (int for number of levels, list for specific values)
            show_contour_lines: Whether to show contour lines
            show_colorbar: Whether to show colorbar
            contour_line_color: Color of contour lines
            contour_line_width: Width of contour lines
            **kwargs: Additional contour plot arguments
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = self.create_figure()
        
        # Handle NaN values
        mask = ~np.isnan(z_grid)
        if not np.any(mask):
            raise ValueError("All interpolated values are NaN")
        
        # Create filled contour plot
        if levels is None:
            levels = 20
        
        contourf = ax.contourf(x_grid, y_grid, z_grid, levels=levels, 
                              cmap=colormap, extend='both', **kwargs)
        
        # Add contour lines
        if show_contour_lines:
            contour_lines = ax.contour(x_grid, y_grid, z_grid, levels=levels,
                                      colors=contour_line_color, linewidths=contour_line_width)
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%g')
        
        # Add colorbar
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(contourf, cax=cax)
            cbar.set_label('Interpolated Value', rotation=270, labelpad=15)
        
        # Set labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        
        return ax
    
    def plot_combined(self,
                     data: pd.DataFrame,
                     x_col: str,
                     y_col: str,
                     value_col: str,
                     x_grid: np.ndarray,
                     y_grid: np.ndarray,
                     z_grid: np.ndarray,
                     ax: Optional[plt.Axes] = None,
                     title: str = "Data and Interpolation Results",
                     colormap: str = 'viridis',
                     point_size: float = 30,
                     show_colorbar: bool = True,
                     **kwargs) -> plt.Axes:
        """
        Create combined plot showing both data points and interpolation results.
        
        Args:
            data: DataFrame with well data
            x_col: X coordinate column name
            y_col: Y coordinate column name
            value_col: Value column name
            x_grid: X coordinate grid
            y_grid: Y coordinate grid
            z_grid: Interpolated values grid
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            colormap: Colormap name
            point_size: Size of data points
            show_colorbar: Whether to show colorbar
            **kwargs: Additional plot arguments
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = self.create_figure()
        
        # Plot interpolation results as contour
        contourf = ax.contourf(x_grid, y_grid, z_grid, levels=20, 
                              cmap=colormap, alpha=0.8, extend='both')
        
        # Overlay data points
        x = data[x_col].values
        y = data[y_col].values
        values = data[value_col].values
        
        scatter = ax.scatter(x, y, c=values, s=point_size, cmap=colormap,
                           edgecolors='white', linewidth=1, zorder=5)
        
        # Add colorbar
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(contourf, cax=cax)
            cbar.set_label(value_col, rotation=270, labelpad=15)
        
        # Set labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        
        return ax
    
    def plot_histogram(self,
                      data: pd.DataFrame,
                      value_col: str,
                      ax: Optional[plt.Axes] = None,
                      title: Optional[str] = None,
                      bins: Union[int, str] = 'auto',
                      density: bool = False,
                      show_stats: bool = True,
                      **kwargs) -> plt.Axes:
        """
        Create histogram of data values.
        
        Args:
            data: DataFrame with data
            value_col: Value column name
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            bins: Number of bins or binning strategy
            density: Whether to normalize to density
            show_stats: Whether to show statistics on plot
            **kwargs: Additional histogram arguments
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = self.create_figure()
        
        values = data[value_col].dropna().values
        
        # Create histogram
        n, bins, patches = ax.hist(values, bins=bins, density=density, 
                                  alpha=0.7, edgecolor='black', **kwargs)
        
        # Add statistics text
        if show_stats:
            mean_val = np.mean(values)
            std_val = np.std(values)
            median_val = np.median(values)
            
            stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMedian: {median_val:.3f}\nN: {len(values)}'
            ax.text(0.75, 0.75, stats_text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel(value_col)
        ax.set_ylabel('Density' if density else 'Frequency')
        if title is None:
            title = f'Distribution of {value_col}'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_boxplot(self,
                    data: pd.DataFrame,
                    value_cols: List[str],
                    ax: Optional[plt.Axes] = None,
                    title: str = "Box Plot Comparison",
                    **kwargs) -> plt.Axes:
        """
        Create box plot for comparing multiple variables.
        
        Args:
            data: DataFrame with data
            value_cols: List of column names to plot
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            **kwargs: Additional boxplot arguments
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = self.create_figure()
        
        # Prepare data for boxplot
        plot_data = [data[col].dropna().values for col in value_cols]
        
        # Create box plot
        bp = ax.boxplot(plot_data, labels=value_cols, patch_artist=True, **kwargs)
        
        # Customize appearance
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_cols)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set labels and title
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        if len(max(value_cols, key=len)) > 8:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        return ax
    
    def plot_correlation_matrix(self,
                               data: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               ax: Optional[plt.Axes] = None,
                               title: str = "Correlation Matrix",
                               colormap: str = 'RdBu_r',
                               annot: bool = True,
                               **kwargs) -> plt.Axes:
        """
        Create correlation matrix heatmap.
        
        Args:
            data: DataFrame with data
            columns: Columns to include (all numeric if None)
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            colormap: Colormap name
            annot: Whether to annotate cells with values
            **kwargs: Additional imshow arguments
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = self.create_figure()
        
        # Select columns
        if columns is None:
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data[columns]
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap=colormap, aspect='auto',
                      vmin=-1, vmax=1, **kwargs)
        
        # Set ticks and labels
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add correlation values as text
        if annot:
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    value = corr_matrix.iloc[i, j]
                    color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=color, fontweight='bold')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        ax.set_title(title)
        
        return ax
    
    def save_figure(self, 
                   filepath: str,
                   dpi: Optional[int] = None,
                   bbox_inches: str = 'tight',
                   **kwargs):
        """
        Save current figure to file.
        
        Args:
            filepath: Output file path
            dpi: Resolution (uses default if None)
            bbox_inches: Bounding box settings
            **kwargs: Additional savefig arguments
        """
        if self.current_figure is None:
            raise ValueError("No figure to save")
        
        if dpi is None:
            dpi = self.dpi
        
        self.current_figure.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    
    def close_figure(self):
        """Close current figure and clear references."""
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None
            self.current_axes = None
    
    def show(self):
        """Display current figure."""
        if self.current_figure is not None:
            plt.show()


class MatplotlibRenderer:
    """
    Enhanced matplotlib renderer with advanced features.
    """
    
    def __init__(self):
        """Initialize renderer."""
        self.plot2d = Plot2D()
        
    def render_interpolation_comparison(self,
                                      data: pd.DataFrame,
                                      results: Dict[str, Dict[str, Any]],
                                      x_col: str,
                                      y_col: str,
                                      value_col: str,
                                      output_path: Optional[str] = None) -> plt.Figure:
        """
        Render comparison of multiple interpolation methods.
        
        Args:
            data: Original data
            results: Dictionary of interpolation results by method name
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            output_path: Output file path (optional)
            
        Returns:
            Figure object
        """
        n_methods = len(results)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (method_name, result) in enumerate(results.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot interpolation result
            x_grid = result['x_grid']
            y_grid = result['y_grid'] 
            z_grid = result['z_grid']
            
            self.plot2d.plot_combined(data, x_col, y_col, value_col,
                                    x_grid, y_grid, z_grid, ax=ax,
                                    title=f'{method_name} Interpolation')
        
        # Hide unused subplots
        total_subplots = n_rows * n_cols
        for idx in range(n_methods, total_subplots):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig