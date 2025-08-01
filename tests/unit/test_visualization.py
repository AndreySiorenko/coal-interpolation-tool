"""
Tests for visualization modules.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from visualization.plot2d import Plot2D, MatplotlibRenderer


class TestPlot2D:
    """Test cases for Plot2D class."""
    
    def setup_method(self):
        """Set up test data."""
        self.plotter = Plot2D()
        
        # Create sample data
        np.random.seed(42)
        n_points = 20
        self.data = pd.DataFrame({
            'X': np.random.uniform(0, 100, n_points),
            'Y': np.random.uniform(0, 100, n_points),
            'Value': np.random.uniform(10, 50, n_points),
            'Size': np.random.uniform(5, 20, n_points)
        })
        
        # Create sample grid
        x = np.linspace(0, 100, 10)
        y = np.linspace(0, 100, 10)
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        self.z_grid = np.sin(self.x_grid/20) * np.cos(self.y_grid/20) * 20 + 30
    
    @patch('matplotlib.pyplot.subplots')
    def test_create_figure(self, mock_subplots):
        """Test figure creation."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        fig, ax = self.plotter.create_figure()
        
        mock_subplots.assert_called_once_with(1, 1, figsize=(12, 8), dpi=100)
        assert fig == mock_fig
        assert ax == mock_ax
        assert self.plotter.current_figure == mock_fig
        assert self.plotter.current_axes == mock_ax
    
    @patch('matplotlib.pyplot.subplots')
    @patch('mpl_toolkits.axes_grid1.make_axes_locatable')
    @patch('matplotlib.pyplot.colorbar')
    def test_plot_scatter(self, mock_colorbar, mock_make_axes, mock_subplots):
        """Test scatter plot creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        mock_divider = Mock()
        mock_cax = Mock()
        mock_make_axes.return_value = mock_divider
        mock_divider.append_axes.return_value = mock_cax
        
        mock_scatter = Mock()
        mock_ax.scatter.return_value = mock_scatter
        
        # Test scatter plot
        result_ax = self.plotter.plot_scatter(
            self.data, 'X', 'Y', 'Value',
            title="Test Scatter",
            colormap='plasma'
        )
        
        # Verify calls
        mock_ax.scatter.assert_called_once()
        call_args = mock_ax.scatter.call_args
        assert len(call_args[0]) == 2  # x, y
        assert 'c' in call_args[1]
        assert call_args[1]['cmap'] == 'plasma'
        
        mock_ax.set_xlabel.assert_called_with('X')
        mock_ax.set_ylabel.assert_called_with('Y')
        mock_ax.set_title.assert_called_with("Test Scatter")
        mock_ax.grid.assert_called_with(True, alpha=0.3)
        mock_ax.set_aspect.assert_called_with('equal', adjustable='box')
        
        assert result_ax == mock_ax
    
    @patch('matplotlib.pyplot.subplots')
    @patch('mpl_toolkits.axes_grid1.make_axes_locatable')
    @patch('matplotlib.pyplot.colorbar')
    def test_plot_contour(self, mock_colorbar, mock_make_axes, mock_subplots):
        """Test contour plot creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        mock_divider = Mock()
        mock_cax = Mock()
        mock_make_axes.return_value = mock_divider
        mock_divider.append_axes.return_value = mock_cax
        
        mock_contourf = Mock()
        mock_contour = Mock()
        mock_ax.contourf.return_value = mock_contourf
        mock_ax.contour.return_value = mock_contour
        
        # Test contour plot
        result_ax = self.plotter.plot_contour(
            self.x_grid, self.y_grid, self.z_grid,
            title="Test Contour",
            levels=15,
            colormap='viridis'
        )
        
        # Verify calls
        mock_ax.contourf.assert_called_once()
        mock_ax.contour.assert_called_once()
        mock_ax.clabel.assert_called_once()
        
        mock_ax.set_xlabel.assert_called_with('X Coordinate')
        mock_ax.set_ylabel.assert_called_with('Y Coordinate')
        mock_ax.set_title.assert_called_with("Test Contour")
        mock_ax.set_aspect.assert_called_with('equal', adjustable='box')
        
        assert result_ax == mock_ax
    
    @patch('matplotlib.pyplot.subplots')
    @patch('mpl_toolkits.axes_grid1.make_axes_locatable')
    @patch('matplotlib.pyplot.colorbar')
    def test_plot_combined(self, mock_colorbar, mock_make_axes, mock_subplots):
        """Test combined plot creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        mock_divider = Mock()
        mock_cax = Mock()
        mock_make_axes.return_value = mock_divider
        mock_divider.append_axes.return_value = mock_cax
        
        mock_contourf = Mock()
        mock_scatter = Mock()
        mock_ax.contourf.return_value = mock_contourf
        mock_ax.scatter.return_value = mock_scatter
        
        # Test combined plot
        result_ax = self.plotter.plot_combined(
            self.data, 'X', 'Y', 'Value',
            self.x_grid, self.y_grid, self.z_grid,
            title="Test Combined"
        )
        
        # Verify both contour and scatter were called
        mock_ax.contourf.assert_called_once()
        mock_ax.scatter.assert_called_once()
        
        mock_ax.set_xlabel.assert_called_with('X')
        mock_ax.set_ylabel.assert_called_with('Y')
        mock_ax.set_title.assert_called_with("Test Combined")
        mock_ax.set_aspect.assert_called_with('equal', adjustable='box')
        
        assert result_ax == mock_ax
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_histogram(self, mock_subplots):
        """Test histogram creation."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.hist.return_value = (None, None, None)
        
        # Test histogram
        result_ax = self.plotter.plot_histogram(
            self.data, 'Value',
            title="Test Histogram",
            bins=20,
            show_stats=True
        )
        
        # Verify calls
        mock_ax.hist.assert_called_once()
        call_args = mock_ax.hist.call_args
        assert call_args[1]['bins'] == 20
        
        mock_ax.text.assert_called_once()  # Statistics text
        mock_ax.set_xlabel.assert_called_with('Value')
        mock_ax.set_title.assert_called_with("Test Histogram")
        mock_ax.grid.assert_called_with(True, alpha=0.3)
        
        assert result_ax == mock_ax
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.cm')
    def test_plot_boxplot(self, mock_cm, mock_subplots):
        """Test box plot creation."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock colormap
        mock_colors = np.array([[1, 0, 0], [0, 1, 0]])
        mock_cm.Set3.return_value = mock_colors
        
        # Mock boxplot return
        mock_bp = {
            'boxes': [Mock(), Mock()]
        }
        mock_ax.boxplot.return_value = mock_bp
        
        # Test boxplot
        value_cols = ['Value', 'Size']
        result_ax = self.plotter.plot_boxplot(
            self.data, value_cols,
            title="Test Boxplot"
        )
        
        # Verify calls
        mock_ax.boxplot.assert_called_once()
        call_args = mock_ax.boxplot.call_args
        assert call_args[1]['labels'] == value_cols
        assert call_args[1]['patch_artist'] == True
        
        mock_ax.set_ylabel.assert_called_with('Value')
        mock_ax.set_title.assert_called_with("Test Boxplot")
        mock_ax.grid.assert_called_with(True, alpha=0.3)
        
        assert result_ax == mock_ax
    
    @patch('matplotlib.pyplot.subplots')
    @patch('mpl_toolkits.axes_grid1.make_axes_locatable')
    @patch('matplotlib.pyplot.colorbar')
    def test_plot_correlation_matrix(self, mock_colorbar, mock_make_axes, mock_subplots):
        """Test correlation matrix creation."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        mock_divider = Mock()
        mock_cax = Mock()
        mock_make_axes.return_value = mock_divider
        mock_divider.append_axes.return_value = mock_cax
        
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im
        
        # Test correlation matrix
        result_ax = self.plotter.plot_correlation_matrix(
            self.data,
            columns=['X', 'Y', 'Value'],
            title="Test Correlation",
            colormap='RdBu_r'
        )
        
        # Verify calls
        mock_ax.imshow.assert_called_once()
        call_args = mock_ax.imshow.call_args
        assert call_args[1]['cmap'] == 'RdBu_r'
        assert call_args[1]['vmin'] == -1
        assert call_args[1]['vmax'] == 1
        
        mock_ax.set_xticks.assert_called_once()
        mock_ax.set_yticks.assert_called_once()
        mock_ax.set_xticklabels.assert_called_once()
        mock_ax.set_yticklabels.assert_called_once()
        mock_ax.set_title.assert_called_with("Test Correlation")
        
        assert result_ax == mock_ax
    
    def test_plot_contour_all_nan(self):
        """Test contour plot with all NaN values."""
        z_grid_nan = np.full_like(self.z_grid, np.nan)
        
        with pytest.raises(ValueError, match="All interpolated values are NaN"):
            self.plotter.plot_contour(self.x_grid, self.y_grid, z_grid_nan)
    
    @patch('matplotlib.pyplot.close')
    def test_close_figure(self, mock_close):
        """Test figure closing."""
        mock_fig = Mock()
        self.plotter.current_figure = mock_fig
        self.plotter.current_axes = Mock()
        
        self.plotter.close_figure()
        
        mock_close.assert_called_once_with(mock_fig)
        assert self.plotter.current_figure is None
        assert self.plotter.current_axes is None
    
    def test_save_figure_no_figure(self):
        """Test save with no current figure."""
        with pytest.raises(ValueError, match="No figure to save"):
            self.plotter.save_figure("test.png")


class TestMatplotlibRenderer:
    """Test cases for MatplotlibRenderer class."""
    
    def setup_method(self):
        """Set up test data."""
        self.renderer = MatplotlibRenderer()
        
        # Create sample data
        np.random.seed(42)
        n_points = 20
        self.data = pd.DataFrame({
            'X': np.random.uniform(0, 100, n_points),
            'Y': np.random.uniform(0, 100, n_points),
            'Value': np.random.uniform(10, 50, n_points)
        })
        
        # Create sample results
        x = np.linspace(0, 100, 10)
        y = np.linspace(0, 100, 10)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid1 = np.sin(x_grid/20) * np.cos(y_grid/20) * 20 + 30
        z_grid2 = np.cos(x_grid/15) * np.sin(y_grid/15) * 15 + 35
        
        self.results = {
            'IDW': {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'z_grid': z_grid1
            },
            'Kriging': {
                'x_grid': x_grid,
                'y_grid': y_grid,
                'z_grid': z_grid2
            }
        }
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_render_interpolation_comparison(self, mock_tight_layout, mock_subplots):
        """Test interpolation comparison rendering."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = np.array([Mock(), Mock()]).reshape(1, 2)
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Configure plot2d mock
        with patch.object(self.renderer.plot2d, 'plot_combined') as mock_plot_combined:
            result_fig = self.renderer.render_interpolation_comparison(
                self.data, self.results, 'X', 'Y', 'Value'
            )
            
            # Verify subplots creation
            mock_subplots.assert_called_once_with(1, 2, figsize=(10, 4))
            
            # Verify plot_combined called for each method
            assert mock_plot_combined.call_count == 2
            
            mock_tight_layout.assert_called_once()
            assert result_fig == mock_fig
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_render_comparison_single_method(self, mock_tight_layout, mock_subplots):
        """Test comparison with single method."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, [mock_ax])
        
        single_result = {'IDW': self.results['IDW']}
        
        with patch.object(self.renderer.plot2d, 'plot_combined'):
            result_fig = self.renderer.render_interpolation_comparison(
                self.data, single_result, 'X', 'Y', 'Value'
            )
            
            mock_subplots.assert_called_once_with(1, 1, figsize=(5, 4))
            assert result_fig == mock_fig
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout') 
    def test_render_comparison_hide_unused_subplots(self, mock_tight_layout, mock_subplots):
        """Test hiding unused subplots."""
        mock_fig = Mock()
        # Create 4 subplots (2x2) but only use 2
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        with patch.object(self.renderer.plot2d, 'plot_combined'):
            self.renderer.render_interpolation_comparison(
                self.data, self.results, 'X', 'Y', 'Value'
            )
            
            # Check that unused subplots are hidden
            mock_axes[1, 0].set_visible.assert_called_with(False)
            mock_axes[1, 1].set_visible.assert_called_with(False)


# Mock tests for modules that might not be available
class TestVisualizationImports:
    """Test visualization module imports and fallbacks."""
    
    def test_plot2d_import(self):
        """Test that Plot2D can be instantiated."""
        plotter = Plot2D()
        assert plotter.figure_size == (12, 8)
        assert plotter.dpi == 100
        assert plotter.current_figure is None
        assert plotter.current_axes is None
    
    def test_matplotlib_renderer_import(self):
        """Test that MatplotlibRenderer can be instantiated."""
        renderer = MatplotlibRenderer()
        assert renderer.plot2d is not None
        assert isinstance(renderer.plot2d, Plot2D)


# Tests that require specific modules can be conditional
@pytest.mark.skipif(True, reason="VTK not available in test environment")
class TestPlot3D:
    """Test cases for Plot3D class (conditional)."""
    
    def test_plot3d_fallback(self):
        """Test Plot3D fallback behavior."""
        # This would test matplotlib fallback
        pass


@pytest.mark.skipif(True, reason="Plotly not available in test environment")
class TestInteractivePlot:
    """Test cases for InteractivePlot class (conditional)."""
    
    def test_interactive_plot_creation(self):
        """Test interactive plot creation.""" 
        # This would test plotly functionality
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])