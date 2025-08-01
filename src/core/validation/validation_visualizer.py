"""
Validation visualization module.

Provides comprehensive visualization tools for validation results including:
- Error distribution plots
- Spatial error maps
- Cross-validation plots
- Uncertainty visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from .cross_validator import CrossValidationResult
from .bootstrap_validator import BootstrapResult
from .uncertainty_quantifier import UncertaintyResult
from .quality_metrics import MetricsResult


class ValidationVisualizer:
    """
    Comprehensive visualization tools for validation results.
    
    Provides various plot types for analyzing validation results
    including error distributions, spatial patterns, and uncertainty.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[float, float] = (12, 8)):
        """
        Initialize validation visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for validation visualization")
        
        self.style = style
        self.figsize = figsize
        
        # Set style if available
        if style in plt.style.available:
            plt.style.use(style)
    
    def plot_cross_validation_results(self,
                                     results: CrossValidationResult,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive cross-validation results plot.
        
        Args:
            results: CrossValidation results
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Cross-Validation Results - {results.method.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Predicted vs Actual scatter plot
        ax = axes[0, 0]
        self._plot_predicted_vs_actual(ax, results.actual_values, results.predictions)
        ax.set_title('Predicted vs Actual Values')
        
        # 2. Residuals vs Predicted
        ax = axes[0, 1]
        self._plot_residuals_vs_predicted(ax, results.predictions, results.errors)
        ax.set_title('Residuals vs Predicted')
        
        # 3. Error distribution
        ax = axes[0, 2]
        self._plot_error_distribution(ax, results.errors)
        ax.set_title('Error Distribution')
        
        # 4. Spatial error map (if spatial data available)
        ax = axes[1, 0]
        if results.spatial_errors is not None:
            self._plot_spatial_errors(ax, results.spatial_errors)
            ax.set_title('Spatial Error Distribution')
        else:
            ax.text(0.5, 0.5, 'No spatial data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Spatial Errors (N/A)')
        
        # 5. Q-Q plot
        ax = axes[1, 1]
        self._plot_qq_plot(ax, results.errors)
        ax.set_title('Q-Q Plot (Normality Check)')
        
        # 6. Metrics summary
        ax = axes[1, 2]
        self._plot_metrics_summary(ax, results.metrics)
        ax.set_title('Validation Metrics')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_bootstrap_uncertainty(self,
                                  results: BootstrapResult,
                                  coordinates: Optional[np.ndarray] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot bootstrap uncertainty results.
        
        Args:
            results: Bootstrap results
            coordinates: Spatial coordinates for plots
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bootstrap Uncertainty Analysis', fontsize=16, fontweight='bold')
        
        # 1. Prediction uncertainty map
        ax = axes[0, 0]
        if coordinates is not None and coordinates.shape[1] >= 2:
            scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                               c=results.prediction_std, cmap='viridis',
                               s=50, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Prediction Std Dev')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Spatial Uncertainty Distribution')
        else:
            ax.plot(results.prediction_std, 'o-')
            ax.set_xlabel('Point Index')
            ax.set_ylabel('Prediction Std Dev')
            ax.set_title('Prediction Uncertainty by Point')
        
        # 2. Confidence intervals
        ax = axes[0, 1]
        n_points = len(results.prediction_mean)
        x_points = np.arange(n_points)
        
        ax.plot(x_points, results.prediction_mean, 'b-', label='Mean Prediction', linewidth=2)
        
        # Plot confidence intervals
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        alphas = [0.3, 0.4, 0.5]
        
        for i, (conf_name, conf_data) in enumerate(results.confidence_intervals.items()):
            if i < len(colors):
                ax.fill_between(x_points, conf_data[:, 0], conf_data[:, 1],
                              color=colors[i], alpha=alphas[i], 
                              label=f'{conf_name.replace("ci_", "")}% CI')
        
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Predicted Value')
        ax.set_title('Prediction Confidence Intervals')
        ax.legend()
        
        # 3. Convergence plot
        ax = axes[1, 0]
        if results.convergence_metrics:
            conv = results.convergence_metrics
            if 'running_mean' in conv:
                ax.plot(conv['n_iterations'], conv['running_mean'], 'b-', 
                       label='Running Mean')
                ax2 = ax.twinx()
                ax2.plot(conv['n_iterations'], conv['running_std'], 'r-', 
                        label='Running Std')
                ax2.set_ylabel('Running Std Dev', color='r')
                ax.set_xlabel('Bootstrap Iteration')
                ax.set_ylabel('Running Mean', color='b')
                ax.set_title('Bootstrap Convergence')
                
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # 4. Uncertainty histogram
        ax = axes[1, 1]
        ax.hist(results.prediction_std, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(results.prediction_std), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(results.prediction_std):.3f}')
        ax.axvline(np.median(results.prediction_std), color='orange', linestyle='--',
                  label=f'Median: {np.median(results.prediction_std):.3f}')
        ax.set_xlabel('Prediction Standard Deviation')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Uncertainties')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_spatial_validation_errors(self,
                                      spatial_errors: pd.DataFrame,
                                      x_col: str = 'easting',
                                      y_col: str = 'northing',
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create detailed spatial error visualization.
        
        Args:
            spatial_errors: DataFrame with spatial error information
            x_col: X coordinate column name
            y_col: Y coordinate column name
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spatial Validation Error Analysis', fontsize=16, fontweight='bold')
        
        x = spatial_errors[x_col].values
        y = spatial_errors[y_col].values
        errors = spatial_errors['error'].values
        abs_errors = spatial_errors['abs_error'].values
        
        # 1. Raw errors map
        ax = axes[0, 0]
        scatter = ax.scatter(x, y, c=errors, cmap='RdBu_r', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Error')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Spatial Distribution of Errors')
        ax.set_aspect('equal', adjustable='box')
        
        # 2. Absolute errors map
        ax = axes[0, 1]
        scatter = ax.scatter(x, y, c=abs_errors, cmap='Reds', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Absolute Error')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Spatial Distribution of Absolute Errors')
        ax.set_aspect('equal', adjustable='box')
        
        # 3. Error magnitude by location
        ax = axes[1, 0]
        # Create bubble plot with error magnitude
        sizes = (abs_errors - abs_errors.min()) / (abs_errors.max() - abs_errors.min()) * 200 + 20
        scatter = ax.scatter(x, y, s=sizes, c=errors, cmap='RdBu_r', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Error')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Error Magnitude (bubble size) and Sign (color)')
        ax.set_aspect('equal', adjustable='box')
        
        # 4. Error statistics by spatial region
        ax = axes[1, 1]
        # Divide space into grid and calculate statistics
        n_bins = 5
        x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
        y_bins = np.linspace(y.min(), y.max(), n_bins + 1)
        
        error_grid = np.zeros((n_bins, n_bins))
        count_grid = np.zeros((n_bins, n_bins))
        
        for i in range(len(x)):
            x_idx = min(np.digitize(x[i], x_bins) - 1, n_bins - 1)
            y_idx = min(np.digitize(y[i], y_bins) - 1, n_bins - 1)
            error_grid[y_idx, x_idx] += errors[i]
            count_grid[y_idx, x_idx] += 1
        
        # Calculate mean errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_error_grid = np.divide(error_grid, count_grid, 
                                      out=np.zeros_like(error_grid), 
                                      where=count_grid!=0)
        
        im = ax.imshow(mean_error_grid, cmap='RdBu_r', aspect='auto',
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      origin='lower')
        plt.colorbar(im, ax=ax, label='Mean Error')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Spatial Error Patterns (Gridded)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_quantification(self,
                                       results: UncertaintyResult,
                                       coordinates: Optional[np.ndarray] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot uncertainty quantification results.
        
        Args:
            results: Uncertainty quantification results
            coordinates: Spatial coordinates
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Uncertainty Quantification Results', fontsize=16, fontweight='bold')
        
        # 1. Prediction with uncertainty bands
        ax = axes[0, 0]
        n_points = len(results.prediction_mean)
        x_points = np.arange(n_points)
        
        ax.plot(x_points, results.prediction_mean, 'b-', linewidth=2, label='Mean Prediction')
        
        # Plot prediction intervals
        colors = ['lightblue', 'lightgreen']
        for i, (conf_name, conf_data) in enumerate(results.prediction_intervals.items()):
            if i < 2:  # Limit to 2 confidence levels for clarity
                ax.fill_between(x_points, conf_data[:, 0], conf_data[:, 1],
                              color=colors[i], alpha=0.4, 
                              label=f'{conf_name.replace("pi_", "")}% PI')
        
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Predicted Value')
        ax.set_title('Predictions with Uncertainty')
        ax.legend()
        
        # 2. Uncertainty map
        ax = axes[0, 1]
        if coordinates is not None and coordinates.shape[1] >= 2:
            scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                               c=results.prediction_std, cmap='plasma',
                               s=60, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Prediction Std Dev')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Spatial Uncertainty Map')
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.bar(range(len(results.prediction_std)), results.prediction_std, 
                  alpha=0.7, color='orange')
            ax.set_xlabel('Point Index')
            ax.set_ylabel('Prediction Std Dev')
            ax.set_title('Uncertainty by Point')
        
        # 3. Convergence (if available)
        ax = axes[1, 0]
        if 'mean_convergence' in results.convergence_info:
            conv = results.convergence_info
            ax.plot(conv['check_points'], conv['mean_convergence'], 'bo-', 
                   label='Mean Convergence')
            ax.set_xlabel('Number of Simulations')
            ax.set_ylabel('Mean Prediction')
            ax.set_title('Monte Carlo Convergence')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"Method: {results.convergence_info.get('method', 'Unknown')}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Convergence Information')
        
        # 4. Sensitivity analysis (if available)
        ax = axes[1, 1]
        if results.sensitivity_indices:
            # Plot sensitivity indices as bar chart
            params = list(results.sensitivity_indices.keys())
            if params:
                # Use mean sensitivity across all points
                sensitivities = [np.mean(results.sensitivity_indices[param]) for param in params]
                
                bars = ax.bar(params, sensitivities, alpha=0.7, color='lightcoral')
                ax.set_xlabel('Parameters')
                ax.set_ylabel('Sensitivity Index')
                ax.set_title('Parameter Sensitivity')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, sens in zip(bars, sensitivities):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{sens:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No sensitivity analysis available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter Sensitivity (N/A)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_predicted_vs_actual(self, ax, actual, predicted):
        """Plot predicted vs actual values with 1:1 line."""
        ax.scatter(actual, predicted, alpha=0.6, s=30)
        
        # 1:1 line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
               linewidth=2, label='1:1 Line')
        
        # R² calculation
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(actual, predicted)
        ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}', 
               transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_residuals_vs_predicted(self, ax, predicted, errors):
        """Plot residuals vs predicted values."""
        ax.scatter(predicted, errors, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.grid(True, alpha=0.3)
    
    def _plot_error_distribution(self, ax, errors):
        """Plot error distribution histogram with normal overlay."""
        ax.hist(errors, bins=30, density=True, alpha=0.7, color='skyblue', 
               edgecolor='black', label='Observed')
        
        # Overlay normal distribution
        mu, sigma = np.mean(errors), np.std(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        normal_dist = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, normal_dist, 'r-', linewidth=2, label='Normal')
        
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_spatial_errors(self, ax, spatial_errors):
        """Plot spatial distribution of errors."""
        if 'easting' in spatial_errors.columns and 'northing' in spatial_errors.columns:
            x_col, y_col = 'easting', 'northing'
        else:
            # Try to find coordinate columns
            coord_cols = [col for col in spatial_errors.columns if col.lower() in ['x', 'y']]
            if len(coord_cols) >= 2:
                x_col, y_col = coord_cols[0], coord_cols[1]
            else:
                ax.text(0.5, 0.5, 'No spatial coordinates found', 
                       ha='center', va='center', transform=ax.transAxes)
                return
        
        scatter = ax.scatter(spatial_errors[x_col], spatial_errors[y_col], 
                           c=spatial_errors['error'], cmap='RdBu_r', s=40, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Error')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_qq_plot(self, ax, errors):
        """Plot Q-Q plot for normality check."""
        from scipy import stats
        
        sorted_errors = np.sort(errors)
        n = len(sorted_errors)
        
        # Theoretical quantiles (standard normal)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
        
        ax.scatter(theoretical_quantiles, sorted_errors, alpha=0.6, s=30)
        
        # Reference line
        ax.plot(theoretical_quantiles, theoretical_quantiles * np.std(errors) + np.mean(errors), 
               'r--', linewidth=2, label='Normal Reference')
        
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_summary(self, ax, metrics):
        """Plot validation metrics as a table."""
        ax.axis('off')
        
        # Select key metrics
        key_metrics = ['rmse', 'mae', 'r_squared', 'nash_sutcliffe', 'willmott_d', 'bias']
        
        table_data = []
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    table_data.append([metric.upper().replace('_', ' '), f'{value:.4f}'])
        
        if table_data:
            table = ax.table(cellText=table_data,
                           colLabels=['Metric', 'Value'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(2):
                    cell = table[i, j]
                    if i == 0:  # Header
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        else:
            ax.text(0.5, 0.5, 'No metrics available', 
                   ha='center', va='center', transform=ax.transAxes)