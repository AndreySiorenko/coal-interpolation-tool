"""
Interactive visualization utilities using plotly for web-based visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
import warnings

# Try to import plotly - it's optional
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    PLOTLY_AVAILABLE = False


class InteractivePlot:
    """
    Interactive plotting utilities using Plotly for web-based visualizations.
    
    Provides interactive features including:
    - Zoomable and pannable plots
    - Hover information
    - Brushing and linking
    - Animation controls
    - Web export capabilities
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize interactive plotter.
        
        Args:
            theme: Plotly theme name
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualization. Install with: pip install plotly")
        
        self.theme = theme
        self.figures = []
    
    def create_scatter(self,
                      data: pd.DataFrame,
                      x_col: str,
                      y_col: str,
                      value_col: Optional[str] = None,
                      size_col: Optional[str] = None,
                      title: str = "Interactive Scatter Plot",
                      colorscale: str = 'Viridis',
                      **kwargs) -> go.Figure:
        """
        Create interactive scatter plot.
        
        Args:
            data: DataFrame with data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Column for color mapping
            size_col: Column for size mapping
            title: Plot title
            colorscale: Plotly colorscale name
            **kwargs: Additional scatter arguments
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Prepare hover data
        hover_data = [x_col, y_col]
        if value_col:
            hover_data.append(value_col)
        if size_col and size_col != value_col:
            hover_data.append(size_col)
        
        # Create scatter trace  
        scatter_args = {
            'x': data[x_col],
            'y': data[y_col],
            'mode': 'markers',
            'name': 'Wells',
            'hovertemplate': '<br>'.join([
                f'{col}: %{{customdata[{i}]}}' for i, col in enumerate(hover_data)
            ]) + '<extra></extra>',
            'customdata': data[hover_data].values
        }
        
        if value_col:
            scatter_args.update({
                'marker': {
                    'color': data[value_col],
                    'colorscale': colorscale,
                    'colorbar': {'title': value_col},
                    'size': data[size_col] if size_col else 8,
                    'sizemode': 'diameter',
                    'sizeref': 2 * max(data[size_col]) / (40**2) if size_col else None,
                    'line': {'width': 0.5, 'color': 'DarkSlateGrey'}
                }
            })
        else:
            scatter_args.update({
                'marker': {
                    'size': data[size_col] if size_col else 8,
                    'color': 'blue',
                    'line': {'width': 0.5, 'color': 'DarkSlateGrey'}
                }
            })
        
        fig.add_trace(go.Scatter(**scatter_args))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template=self.theme,
            hovermode='closest'
        )
        
        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        self.figures.append(fig)
        return fig
    
    def create_contour(self,
                      x_grid: np.ndarray,
                      y_grid: np.ndarray,
                      z_grid: np.ndarray,
                      title: str = "Interactive Contour Plot",
                      colorscale: str = 'Viridis',
                      show_scale: bool = True,
                      contour_lines: bool = True,
                      **kwargs) -> go.Figure:
        """
        Create interactive contour plot.
        
        Args:
            x_grid: X coordinate grid
            y_grid: Y coordinate grid
            z_grid: Value grid
            title: Plot title
            colorscale: Plotly colorscale name
            show_scale: Whether to show colorbar
            contour_lines: Whether to show contour lines
            **kwargs: Additional contour arguments
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Create contour trace
        contour_args = {
            'x': x_grid[0, :],  # First row for x values
            'y': y_grid[:, 0],  # First column for y values
            'z': z_grid,
            'type': 'contour',
            'colorscale': colorscale,
            'showscale': show_scale,
            'line': {'width': 1, 'color': 'white'} if contour_lines else {'width': 0},
            'hovertemplate': 'X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>'
        }
        contour_args.update(kwargs)
        
        fig.add_trace(go.Contour(**contour_args))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            template=self.theme
        )
        
        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        self.figures.append(fig)
        return fig
    
    def create_surface_3d(self,
                         x_grid: np.ndarray,
                         y_grid: np.ndarray,
                         z_grid: np.ndarray,
                         title: str = "Interactive 3D Surface",
                         colorscale: str = 'Viridis',
                         **kwargs) -> go.Figure:
        """
        Create interactive 3D surface plot.
        
        Args:
            x_grid: X coordinate grid
            y_grid: Y coordinate grid
            z_grid: Value grid
            title: Plot title
            colorscale: Plotly colorscale name
            **kwargs: Additional surface arguments
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Create surface trace
        surface_args = {
            'x': x_grid,
            'y': y_grid,
            'z': z_grid,
            'colorscale': colorscale,
            'hovertemplate': 'X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
        }
        surface_args.update(kwargs)
        
        fig.add_trace(go.Surface(**surface_args))
        
        # Update layout for 3D
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Value',
                aspectmode='cube'
            ),
            template=self.theme
        )
        
        self.figures.append(fig)
        return fig
    
    def create_scatter_3d(self,
                         data: pd.DataFrame,
                         x_col: str,
                         y_col: str,
                         z_col: str,
                         value_col: Optional[str] = None,
                         size_col: Optional[str] = None,
                         title: str = "Interactive 3D Scatter",
                         colorscale: str = 'Viridis',
                         **kwargs) -> go.Figure:
        """
        Create interactive 3D scatter plot.
        
        Args:
            data: DataFrame with data
            x_col: X coordinate column
            y_col: Y coordinate column
            z_col: Z coordinate column
            value_col: Column for color mapping
            size_col: Column for size mapping
            title: Plot title
            colorscale: Plotly colorscale name
            **kwargs: Additional scatter arguments
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Prepare hover data
        hover_data = [x_col, y_col, z_col]
        if value_col:
            hover_data.append(value_col)
        if size_col and size_col != value_col:
            hover_data.append(size_col)
        
        # Create 3D scatter trace
        scatter_args = {
            'x': data[x_col],
            'y': data[y_col],
            'z': data[z_col],
            'mode': 'markers',
            'name': 'Wells',
            'hovertemplate': '<br>'.join([
                f'{col}: %{{customdata[{i}]}}' for i, col in enumerate(hover_data)
            ]) + '<extra></extra>',
            'customdata': data[hover_data].values
        }
        
        if value_col:
            scatter_args.update({
                'marker': {
                    'color': data[value_col],
                    'colorscale': colorscale,
                    'colorbar': {'title': value_col},
                    'size': data[size_col] if size_col else 5,
                    'line': {'width': 0.5, 'color': 'DarkSlateGrey'}
                }
            })
        else:
            scatter_args.update({
                'marker': {
                    'size': data[size_col] if size_col else 5,
                    'color': 'blue',
                    'line': {'width': 0.5, 'color': 'DarkSlateGrey'}
                }
            })
        
        fig.add_trace(go.Scatter3d(**scatter_args))
        
        # Update layout for 3D
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                aspectmode='cube'
            ),
            template=self.theme
        )
        
        self.figures.append(fig)
        return fig
    
    def create_histogram(self,
                        data: pd.DataFrame,
                        value_col: str,
                        bins: Optional[int] = None,
                        title: Optional[str] = None,
                        show_rug: bool = True,
                        **kwargs) -> go.Figure:
        """
        Create interactive histogram.
        
        Args:
            data: DataFrame with data
            value_col: Column to histogram
            bins: Number of bins
            title: Plot title
            show_rug: Whether to show rug plot
            **kwargs: Additional histogram arguments
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        values = data[value_col].dropna()
        
        # Create histogram trace
        hist_args = {
            'x': values,
            'nbinsx': bins,
            'name': value_col,
            'hovertemplate': 'Bin: %{x}<br>Count: %{y}<extra></extra>'
        }
        hist_args.update(kwargs)
        
        fig.add_trace(go.Histogram(**hist_args))
        
        # Add rug plot
        if show_rug:
            fig.add_trace(go.Scatter(
                x=values,
                y=np.zeros(len(values)),
                mode='markers',
                marker=dict(
                    symbol='line-ns-open',
                    size=8,
                    color='rgba(0,0,0,0.4)'
                ),
                name='Data Points',
                yaxis='y2',
                hoverinfo='x'
            ))
            
            # Update layout for rug plot
            fig.update_layout(
                yaxis2=dict(
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    showticklabels=False,
                    range=[0, 1]
                )
            )
        
        # Update layout
        if title is None:
            title = f'Distribution of {value_col}'
        
        fig.update_layout(
            title=title,
            xaxis_title=value_col,
            yaxis_title='Frequency',
            template=self.theme
        )
        
        self.figures.append(fig)
        return fig
    
    def create_comparison_subplots(self,
                                  data: pd.DataFrame,
                                  results: Dict[str, Dict[str, Any]],
                                  x_col: str,
                                  y_col: str,
                                  value_col: str,
                                  title: str = "Method Comparison") -> go.Figure:
        """
        Create subplot comparison of multiple interpolation methods.
        
        Args:
            data: Original data
            results: Dictionary of interpolation results by method
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            title: Overall title
            
        Returns:
            Plotly figure with subplots
        """
        n_methods = len(results)
        n_cols = min(2, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(results.keys()),
            specs=[[{'type': 'scatter'} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        # Plot each method
        for idx, (method_name, result) in enumerate(results.items()):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            # Add contour plot
            x_grid = result['x_grid']
            y_grid = result['y_grid']
            z_grid = result['z_grid']
            
            fig.add_trace(
                go.Contour(
                    x=x_grid[0, :],
                    y=y_grid[:, 0],
                    z=z_grid,
                    colorscale='Viridis',
                    showscale=False,
                    line={'width': 0.5},
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add data points
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=data[value_col],
                        colorscale='Viridis',
                        line=dict(width=0.5, color='white')
                    ),
                    name=f'{method_name} Data',
                    showlegend=False,
                    hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{value_col}: %{{marker.color}}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            height=400 * n_rows
        )
        
        # Update axes
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(title_text=x_col, row=i, col=j)
                fig.update_yaxes(title_text=y_col, scaleanchor=f"x{i if i > 1 or j > 1 else ''}", 
                               scaleratio=1, row=i, col=j)
        
        self.figures.append(fig)
        return fig
    
    def create_animation(self,
                        data_frames: List[Dict[str, Any]],
                        x_col: str,
                        y_col: str,
                        value_col: str,
                        frame_col: str,
                        title: str = "Animated Visualization") -> go.Figure:
        """
        Create animated visualization showing changes over time/parameter.
        
        Args:
            data_frames: List of data dictionaries for each frame
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            frame_col: Column indicating frame/time
            title: Plot title
            
        Returns:
            Plotly figure with animation
        """
        fig = go.Figure()
        
        # Add initial frame
        first_frame = data_frames[0]
        fig.add_trace(
            go.Scatter(
                x=first_frame[x_col],
                y=first_frame[y_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=first_frame[value_col],
                    colorscale='Viridis',
                    colorbar={'title': value_col}
                ),
                name='Data'
            )
        )
        
        # Create animation frames
        frames = []
        for frame_data in data_frames:
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=frame_data[x_col],
                        y=frame_data[y_col],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=frame_data[value_col],
                            colorscale='Viridis'
                        )
                    )
                ],
                name=str(frame_data[frame_col])
            )
            frames.append(frame)
        
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template=self.theme,
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play", method="animate", 
                             args=[None, {"frame": {"duration": 500, "redraw": True}}]),
                        dict(label="Pause", method="animate", 
                             args=[[None], {"frame": {"duration": 0, "redraw": False}, 
                                           "mode": "immediate", 
                                           "transition": {"duration": 0}}])
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue={"prefix": f"{frame_col}: "},
                    pad={"b": 20, "t": 60},
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[[frame.name], 
                                  {"frame": {"duration": 300, "redraw": True},
                                   "mode": "immediate",
                                   "transition": {"duration": 300}}],
                            label=frame.name,
                            method="animate"
                        ) for frame in frames
                    ]
                )
            ]
        )
        
        self.figures.append(fig)
        return fig
    
    def show(self, figure: Optional[go.Figure] = None, **kwargs):
        """
        Display interactive plot.
        
        Args:
            figure: Specific figure to show (last created if None)
            **kwargs: Additional show arguments
        """
        if figure is None and self.figures:
            figure = self.figures[-1]
        
        if figure:
            figure.show(**kwargs)
    
    def save_html(self, 
                  figure: Optional[go.Figure] = None,
                  filepath: str = "plot.html",
                  **kwargs):
        """
        Save interactive plot as HTML file.
        
        Args:
            figure: Figure to save (last created if None)
            filepath: Output HTML file path
            **kwargs: Additional save arguments
        """
        if figure is None and self.figures:
            figure = self.figures[-1]
        
        if figure:
            figure.write_html(filepath, **kwargs)
    
    def save_image(self,
                   figure: Optional[go.Figure] = None,
                   filepath: str = "plot.png",
                   width: int = 1200,
                   height: int = 800,
                   **kwargs):
        """
        Save plot as static image (requires kaleido package).
        
        Args:
            figure: Figure to save (last created if None)
            filepath: Output image file path
            width: Image width in pixels
            height: Image height in pixels
            **kwargs: Additional save arguments
        """
        if figure is None and self.figures:
            figure = self.figures[-1]
        
        if figure:
            try:
                figure.write_image(filepath, width=width, height=height, **kwargs)
            except Exception as e:
                warnings.warn(f"Could not save image: {e}. Install kaleido: pip install kaleido")
    
    def clear_figures(self):
        """Clear all stored figures."""
        self.figures.clear()