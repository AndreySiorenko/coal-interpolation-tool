"""
Bootstrap validation module for uncertainty quantification.

Implements bootstrap methods for:
- Parameter uncertainty estimation
- Prediction confidence intervals
- Model stability assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from ..interpolation.base import BaseInterpolator


@dataclass
class BootstrapResult:
    """Container for bootstrap validation results."""
    n_bootstrap: int
    predictions: np.ndarray  # Shape: (n_bootstrap, n_points)
    prediction_mean: np.ndarray
    prediction_std: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]
    parameter_distributions: Optional[Dict[str, np.ndarray]] = None
    convergence_metrics: Optional[Dict[str, List[float]]] = None


class BootstrapValidator:
    """
    Bootstrap validation engine for interpolation uncertainty quantification.
    
    Provides bootstrap-based uncertainty estimation including:
    - Bootstrap confidence intervals
    - Parameter stability analysis
    - Prediction uncertainty bands
    """
    
    def __init__(self, 
                 n_bootstrap: int = 100,
                 confidence_levels: List[float] = None,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None):
        """
        Initialize bootstrap validator.
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            confidence_levels: Confidence levels for intervals (default: [0.95, 0.90])
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_levels = confidence_levels or [0.95, 0.90]
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def bootstrap_predictions(self,
                            interpolator: BaseInterpolator,
                            data: pd.DataFrame,
                            x_col: str,
                            y_col: str,
                            value_col: str,
                            prediction_points: np.ndarray,
                            z_col: Optional[str] = None,
                            sample_size: Optional[int] = None,
                            track_parameters: bool = False,
                            **interpolator_params) -> BootstrapResult:
        """
        Generate bootstrap predictions with confidence intervals.
        
        Args:
            interpolator: Interpolator instance
            data: Training data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            prediction_points: Points to predict at (n_points, n_dims)
            z_col: Z coordinate column (optional)
            sample_size: Bootstrap sample size (default: same as data)
            track_parameters: Whether to track parameter distributions
            **interpolator_params: Additional interpolator parameters
            
        Returns:
            BootstrapResult with predictions and confidence intervals
        """
        self.logger.info(f"Starting bootstrap validation with {self.n_bootstrap} iterations")
        
        n_data = len(data)
        n_pred = len(prediction_points)
        
        if sample_size is None:
            sample_size = n_data
        
        # Prepare coordinate columns
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        # Initialize storage
        bootstrap_predictions = np.zeros((self.n_bootstrap, n_pred))
        parameter_history = [] if track_parameters else None
        
        # Perform bootstrap iterations
        if self.n_jobs == 1:
            # Sequential processing
            for i in range(self.n_bootstrap):
                preds, params = self._single_bootstrap(
                    interpolator, data, coord_cols, value_col,
                    prediction_points, sample_size, track_parameters,
                    interpolator_params
                )
                bootstrap_predictions[i] = preds
                if track_parameters and params:
                    parameter_history.append(params)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        self._single_bootstrap,
                        interpolator, data, coord_cols, value_col,
                        prediction_points, sample_size, track_parameters,
                        interpolator_params
                    ): i for i in range(self.n_bootstrap)
                }
                
                for future in as_completed(futures):
                    idx = futures[future]
                    preds, params = future.result()
                    bootstrap_predictions[idx] = preds
                    if track_parameters and params:
                        parameter_history.append(params)
        
        # Calculate statistics
        prediction_mean = np.mean(bootstrap_predictions, axis=0)
        prediction_std = np.std(bootstrap_predictions, axis=0, ddof=1)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            ci_upper = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
            
            confidence_intervals[f'ci_{int(conf_level*100)}'] = np.column_stack([ci_lower, ci_upper])
        
        # Process parameter distributions if tracked
        parameter_distributions = None
        if parameter_history:
            parameter_distributions = self._process_parameter_history(parameter_history)
        
        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(bootstrap_predictions)
        
        return BootstrapResult(
            n_bootstrap=self.n_bootstrap,
            predictions=bootstrap_predictions,
            prediction_mean=prediction_mean,
            prediction_std=prediction_std,
            confidence_intervals=confidence_intervals,
            parameter_distributions=parameter_distributions,
            convergence_metrics=convergence_metrics
        )
    
    def bootstrap_validation_metrics(self,
                                   interpolator: BaseInterpolator,
                                   data: pd.DataFrame,
                                   x_col: str,
                                   y_col: str,
                                   value_col: str,
                                   z_col: Optional[str] = None,
                                   test_fraction: float = 0.2,
                                   metric_func: Optional[Callable] = None,
                                   **interpolator_params) -> Dict[str, Any]:
        """
        Estimate validation metric uncertainty using bootstrap.
        
        Args:
            interpolator: Interpolator instance
            data: Input data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            z_col: Z coordinate column (optional)
            test_fraction: Fraction of data for testing
            metric_func: Custom metric function (default: RMSE)
            **interpolator_params: Additional interpolator parameters
            
        Returns:
            Dictionary with metric statistics and confidence intervals
        """
        self.logger.info("Calculating bootstrap validation metrics")
        
        n_data = len(data)
        test_size = int(n_data * test_fraction)
        
        # Default metric function (RMSE)
        if metric_func is None:
            metric_func = lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Prepare coordinate columns
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        # Store metrics from each bootstrap iteration
        metrics = []
        
        for i in range(self.n_bootstrap):
            # Create bootstrap sample
            indices = np.random.choice(n_data, size=n_data, replace=True)
            bootstrap_data = data.iloc[indices].copy()
            
            # Split into train/test
            test_indices = np.random.choice(len(bootstrap_data), size=test_size, replace=False)
            train_mask = np.ones(len(bootstrap_data), dtype=bool)
            train_mask[test_indices] = False
            
            train_data = bootstrap_data.iloc[train_mask]
            test_data = bootstrap_data.iloc[test_indices]
            
            # Fit and predict
            interpolator.fit(train_data, coord_cols, value_col, **interpolator_params)
            test_coords = test_data[coord_cols].values
            predictions = interpolator.predict(test_coords)
            
            # Calculate metric
            actual_values = test_data[value_col].values
            metric_value = metric_func(actual_values, predictions)
            metrics.append(metric_value)
        
        metrics = np.array(metrics)
        
        # Calculate statistics
        results = {
            'mean': float(np.mean(metrics)),
            'std': float(np.std(metrics, ddof=1)),
            'median': float(np.median(metrics)),
            'min': float(np.min(metrics)),
            'max': float(np.max(metrics)),
            'bootstrap_metrics': metrics
        }
        
        # Add confidence intervals
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            lower = np.percentile(metrics, (alpha / 2) * 100)
            upper = np.percentile(metrics, (1 - alpha / 2) * 100)
            results[f'ci_{int(conf_level*100)}'] = (float(lower), float(upper))
        
        return results
    
    def spatial_bootstrap(self,
                         interpolator: BaseInterpolator,
                         data: pd.DataFrame,
                         x_col: str,
                         y_col: str,
                         value_col: str,
                         prediction_points: np.ndarray,
                         z_col: Optional[str] = None,
                         block_size: Optional[float] = None,
                         **interpolator_params) -> BootstrapResult:
        """
        Perform spatial bootstrap accounting for spatial correlation.
        
        Args:
            interpolator: Interpolator instance
            data: Training data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            prediction_points: Points to predict at
            z_col: Z coordinate column (optional)
            block_size: Size of spatial blocks for bootstrap
            **interpolator_params: Additional interpolator parameters
            
        Returns:
            BootstrapResult with spatial bootstrap results
        """
        self.logger.info("Performing spatial bootstrap")
        
        # Get coordinates
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        coordinates = data[coord_cols].values
        
        # Determine block size if not provided
        if block_size is None:
            # Use 10% of spatial extent as default
            ranges = np.ptp(coordinates[:, :2], axis=0)
            block_size = np.mean(ranges) * 0.1
            self.logger.info(f"Using automatic block size: {block_size:.2f}")
        
        n_pred = len(prediction_points)
        bootstrap_predictions = np.zeros((self.n_bootstrap, n_pred))
        
        for i in range(self.n_bootstrap):
            # Create spatial blocks
            bootstrap_indices = self._spatial_block_sample(
                coordinates[:, :2], block_size, len(data)
            )
            
            # Create bootstrap sample
            bootstrap_data = data.iloc[bootstrap_indices].copy()
            
            # Fit and predict
            interpolator.fit(bootstrap_data, coord_cols, value_col, **interpolator_params)
            predictions = interpolator.predict(prediction_points)
            bootstrap_predictions[i] = predictions
        
        # Calculate statistics
        prediction_mean = np.mean(bootstrap_predictions, axis=0)
        prediction_std = np.std(bootstrap_predictions, axis=0, ddof=1)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            ci_upper = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
            
            confidence_intervals[f'ci_{int(conf_level*100)}'] = np.column_stack([ci_lower, ci_upper])
        
        return BootstrapResult(
            n_bootstrap=self.n_bootstrap,
            predictions=bootstrap_predictions,
            prediction_mean=prediction_mean,
            prediction_std=prediction_std,
            confidence_intervals=confidence_intervals,
            convergence_metrics=self._calculate_convergence_metrics(bootstrap_predictions)
        )
    
    def _single_bootstrap(self,
                         interpolator: BaseInterpolator,
                         data: pd.DataFrame,
                         coord_cols: List[str],
                         value_col: str,
                         prediction_points: np.ndarray,
                         sample_size: int,
                         track_parameters: bool,
                         interpolator_params: Dict[str, Any]) -> Tuple[np.ndarray, Optional[Dict]]:
        """Perform a single bootstrap iteration."""
        # Create bootstrap sample
        indices = np.random.choice(len(data), size=sample_size, replace=True)
        bootstrap_data = data.iloc[indices].copy()
        
        # Fit interpolator
        interpolator.fit(bootstrap_data, coord_cols, value_col, **interpolator_params)
        
        # Make predictions
        predictions = interpolator.predict(prediction_points)
        
        # Extract parameters if requested
        parameters = None
        if track_parameters:
            parameters = self._extract_parameters(interpolator)
        
        return predictions, parameters
    
    def _spatial_block_sample(self,
                            coordinates: np.ndarray,
                            block_size: float,
                            target_size: int) -> np.ndarray:
        """Create spatial block bootstrap sample."""
        n_points = len(coordinates)
        selected_indices = []
        
        while len(selected_indices) < target_size:
            # Select random center point
            center_idx = np.random.randint(n_points)
            center = coordinates[center_idx]
            
            # Find all points within block
            distances = np.sqrt(np.sum((coordinates - center) ** 2, axis=1))
            block_indices = np.where(distances <= block_size)[0]
            
            # Add block to sample (with replacement)
            selected_indices.extend(block_indices)
        
        # Trim to target size
        selected_indices = selected_indices[:target_size]
        
        return np.array(selected_indices)
    
    def _extract_parameters(self, interpolator: BaseInterpolator) -> Dict[str, Any]:
        """Extract parameters from fitted interpolator."""
        parameters = {}
        
        # Try to extract common parameters
        if hasattr(interpolator, 'power') and interpolator.power is not None:
            parameters['power'] = float(interpolator.power)
        
        if hasattr(interpolator, 'epsilon') and interpolator.epsilon is not None:
            parameters['epsilon'] = float(interpolator.epsilon)
        
        if hasattr(interpolator, 'range_') and interpolator.range_ is not None:
            parameters['range'] = float(interpolator.range_)
        
        if hasattr(interpolator, 'sill_') and interpolator.sill_ is not None:
            parameters['sill'] = float(interpolator.sill_)
        
        if hasattr(interpolator, 'nugget_') and interpolator.nugget_ is not None:
            parameters['nugget'] = float(interpolator.nugget_)
        
        return parameters
    
    def _process_parameter_history(self, 
                                 parameter_history: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Process parameter history into distributions."""
        if not parameter_history:
            return {}
        
        # Get all parameter names
        all_params = set()
        for params in parameter_history:
            all_params.update(params.keys())
        
        # Create arrays for each parameter
        distributions = {}
        for param in all_params:
            values = []
            for params in parameter_history:
                if param in params:
                    values.append(params[param])
                else:
                    values.append(np.nan)
            distributions[param] = np.array(values)
        
        return distributions
    
    def _calculate_convergence_metrics(self, 
                                     bootstrap_predictions: np.ndarray) -> Dict[str, List[float]]:
        """Calculate convergence metrics for bootstrap iterations."""
        n_bootstrap, n_points = bootstrap_predictions.shape
        
        # Calculate running mean and std
        running_mean = []
        running_std = []
        
        for i in range(1, n_bootstrap + 1):
            subset = bootstrap_predictions[:i]
            running_mean.append(np.mean(np.mean(subset, axis=0)))
            running_std.append(np.mean(np.std(subset, axis=0, ddof=1)))
        
        # Calculate coefficient of variation
        cv_progression = []
        for i in range(10, n_bootstrap + 1, 10):
            subset = bootstrap_predictions[:i]
            mean_pred = np.mean(subset, axis=0)
            std_pred = np.std(subset, axis=0, ddof=1)
            cv = np.mean(std_pred / (np.abs(mean_pred) + 1e-10))
            cv_progression.append(cv)
        
        return {
            'running_mean': running_mean,
            'running_std': running_std,
            'cv_progression': cv_progression,
            'n_iterations': list(range(1, n_bootstrap + 1)),
            'cv_iterations': list(range(10, n_bootstrap + 1, 10))
        }