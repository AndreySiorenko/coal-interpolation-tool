"""
Cross-validation module for interpolation methods.

Implements various cross-validation strategies including:
- Leave-one-out (LOO) validation
- K-fold cross-validation
- Spatial k-fold cross-validation
- Stratified k-fold cross-validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from scipy.spatial import cKDTree
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from ..interpolation.base import BaseInterpolator


@dataclass
class CrossValidationResult:
    """Container for cross-validation results."""
    method: str
    n_folds: int
    predictions: np.ndarray
    actual_values: np.ndarray
    errors: np.ndarray
    fold_indices: List[np.ndarray]
    metrics: Dict[str, float]
    fold_metrics: List[Dict[str, float]]
    spatial_errors: Optional[pd.DataFrame] = None


class CrossValidator:
    """
    Cross-validation engine for interpolation methods.
    
    Provides comprehensive cross-validation capabilities with various
    strategies and parallel processing support.
    """
    
    def __init__(self, n_jobs: int = -1, random_state: Optional[int] = None):
        """
        Initialize cross-validator.
        
        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.n_jobs = n_jobs if n_jobs > 0 else None  # None = all cores
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def leave_one_out(self,
                     interpolator: BaseInterpolator,
                     data: pd.DataFrame,
                     x_col: str,
                     y_col: str,
                     value_col: str,
                     z_col: Optional[str] = None,
                     **interpolator_params) -> CrossValidationResult:
        """
        Perform Leave-One-Out Cross-Validation (LOOCV).
        
        Args:
            interpolator: Interpolator instance
            data: Input DataFrame
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column to interpolate
            z_col: Z coordinate column (optional)
            **interpolator_params: Additional parameters for interpolator
            
        Returns:
            CrossValidationResult with LOO validation results
        """
        self.logger.info(f"Starting Leave-One-Out validation for {interpolator.__class__.__name__}")
        
        n_points = len(data)
        predictions = np.zeros(n_points)
        actual_values = data[value_col].values
        
        # Prepare coordinate columns
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        # Create fold indices (each point is a fold)
        fold_indices = [np.array([i]) for i in range(n_points)]
        
        # Perform LOO validation
        if self.n_jobs == 1:
            # Sequential processing
            for i in range(n_points):
                pred = self._validate_single_fold(
                    interpolator, data, i, coord_cols, value_col, 
                    interpolator_params
                )
                predictions[i] = pred
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        self._validate_single_fold,
                        interpolator, data, i, coord_cols, value_col,
                        interpolator_params
                    ): i for i in range(n_points)
                }
                
                for future in as_completed(futures):
                    idx = futures[future]
                    predictions[idx] = future.result()
        
        # Calculate errors and metrics
        errors = predictions - actual_values
        metrics = self._calculate_metrics(actual_values, predictions)
        
        # Create spatial error DataFrame
        spatial_errors = data[[x_col, y_col]].copy()
        if z_col:
            spatial_errors[z_col] = data[z_col]
        spatial_errors['actual'] = actual_values
        spatial_errors['predicted'] = predictions
        spatial_errors['error'] = errors
        spatial_errors['abs_error'] = np.abs(errors)
        spatial_errors['squared_error'] = errors ** 2
        
        return CrossValidationResult(
            method='leave_one_out',
            n_folds=n_points,
            predictions=predictions,
            actual_values=actual_values,
            errors=errors,
            fold_indices=fold_indices,
            metrics=metrics,
            fold_metrics=[],  # Not applicable for LOO
            spatial_errors=spatial_errors
        )
    
    def k_fold(self,
               interpolator: BaseInterpolator,
               data: pd.DataFrame,
               x_col: str,
               y_col: str,
               value_col: str,
               z_col: Optional[str] = None,
               n_folds: int = 5,
               shuffle: bool = True,
               **interpolator_params) -> CrossValidationResult:
        """
        Perform K-fold cross-validation.
        
        Args:
            interpolator: Interpolator instance
            data: Input DataFrame
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column to interpolate
            z_col: Z coordinate column (optional)
            n_folds: Number of folds
            shuffle: Whether to shuffle data before splitting
            **interpolator_params: Additional parameters for interpolator
            
        Returns:
            CrossValidationResult with k-fold validation results
        """
        self.logger.info(f"Starting {n_folds}-fold validation for {interpolator.__class__.__name__}")
        
        n_points = len(data)
        indices = np.arange(n_points)
        
        if shuffle:
            np.random.shuffle(indices)
        
        # Create fold indices
        fold_indices = []
        fold_size = n_points // n_folds
        for i in range(n_folds):
            start_idx = i * fold_size
            if i == n_folds - 1:
                # Last fold gets remaining points
                fold_idx = indices[start_idx:]
            else:
                fold_idx = indices[start_idx:start_idx + fold_size]
            fold_indices.append(fold_idx)
        
        # Perform k-fold validation
        predictions = np.zeros(n_points)
        fold_metrics = []
        
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        for fold_idx, test_indices in enumerate(fold_indices):
            self.logger.debug(f"Processing fold {fold_idx + 1}/{n_folds}")
            
            # Create train/test split
            train_mask = np.ones(n_points, dtype=bool)
            train_mask[test_indices] = False
            
            train_data = data.iloc[train_mask].copy()
            test_data = data.iloc[test_indices].copy()
            
            # Fit interpolator on training data
            interpolator.fit(train_data, coord_cols, value_col, **interpolator_params)
            
            # Predict on test data
            test_coords = test_data[coord_cols].values
            fold_predictions = interpolator.predict(test_coords)
            
            # Store predictions
            predictions[test_indices] = fold_predictions
            
            # Calculate fold metrics
            fold_actual = test_data[value_col].values
            fold_errors = fold_predictions - fold_actual
            fold_metric = self._calculate_metrics(fold_actual, fold_predictions)
            fold_metrics.append(fold_metric)
        
        # Calculate overall metrics
        actual_values = data[value_col].values
        errors = predictions - actual_values
        metrics = self._calculate_metrics(actual_values, predictions)
        
        # Create spatial error DataFrame
        spatial_errors = data[[x_col, y_col]].copy()
        if z_col:
            spatial_errors[z_col] = data[z_col]
        spatial_errors['actual'] = actual_values
        spatial_errors['predicted'] = predictions
        spatial_errors['error'] = errors
        spatial_errors['abs_error'] = np.abs(errors)
        spatial_errors['squared_error'] = errors ** 2
        spatial_errors['fold'] = -1  # Initialize
        for fold_idx, indices in enumerate(fold_indices):
            spatial_errors.loc[spatial_errors.index[indices], 'fold'] = fold_idx
        
        return CrossValidationResult(
            method='k_fold',
            n_folds=n_folds,
            predictions=predictions,
            actual_values=actual_values,
            errors=errors,
            fold_indices=fold_indices,
            metrics=metrics,
            fold_metrics=fold_metrics,
            spatial_errors=spatial_errors
        )
    
    def spatial_k_fold(self,
                      interpolator: BaseInterpolator,
                      data: pd.DataFrame,
                      x_col: str,
                      y_col: str,
                      value_col: str,
                      z_col: Optional[str] = None,
                      n_folds: int = 5,
                      buffer_distance: Optional[float] = None,
                      **interpolator_params) -> CrossValidationResult:
        """
        Perform spatial k-fold cross-validation.
        
        This method creates spatially contiguous folds to better assess
        interpolation performance with spatial autocorrelation.
        
        Args:
            interpolator: Interpolator instance
            data: Input DataFrame
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column to interpolate
            z_col: Z coordinate column (optional)
            n_folds: Number of spatial folds
            buffer_distance: Buffer distance around test regions
            **interpolator_params: Additional parameters for interpolator
            
        Returns:
            CrossValidationResult with spatial k-fold validation results
        """
        self.logger.info(f"Starting spatial {n_folds}-fold validation")
        
        # Get coordinates
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        coordinates = data[coord_cols].values
        n_points = len(data)
        
        # Create spatial folds using k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_folds, random_state=self.random_state)
        fold_labels = kmeans.fit_predict(coordinates[:, :2])  # Use only X,Y for clustering
        
        # Create fold indices
        fold_indices = []
        for i in range(n_folds):
            fold_indices.append(np.where(fold_labels == i)[0])
        
        # Apply buffer if specified
        if buffer_distance is not None:
            fold_indices = self._apply_spatial_buffer(
                fold_indices, coordinates, buffer_distance
            )
        
        # Perform spatial k-fold validation
        predictions = np.full(n_points, np.nan)
        fold_metrics = []
        
        for fold_idx, test_indices in enumerate(fold_indices):
            if len(test_indices) == 0:
                continue
                
            self.logger.debug(f"Processing spatial fold {fold_idx + 1}/{n_folds}")
            
            # Create train/test split
            train_indices = np.setdiff1d(np.arange(n_points), test_indices)
            
            if len(train_indices) < 3:
                self.logger.warning(f"Skipping fold {fold_idx}: insufficient training points")
                continue
            
            train_data = data.iloc[train_indices].copy()
            test_data = data.iloc[test_indices].copy()
            
            # Fit interpolator on training data
            interpolator.fit(train_data, coord_cols, value_col, **interpolator_params)
            
            # Predict on test data
            test_coords = test_data[coord_cols].values
            fold_predictions = interpolator.predict(test_coords)
            
            # Store predictions
            predictions[test_indices] = fold_predictions
            
            # Calculate fold metrics
            fold_actual = test_data[value_col].values
            fold_errors = fold_predictions - fold_actual
            fold_metric = self._calculate_metrics(fold_actual, fold_predictions)
            fold_metrics.append(fold_metric)
        
        # Handle any remaining NaN predictions
        valid_mask = ~np.isnan(predictions)
        if not np.all(valid_mask):
            self.logger.warning(f"{np.sum(~valid_mask)} points were not validated")
        
        # Calculate overall metrics using only valid predictions
        actual_values = data[value_col].values
        errors = np.full(n_points, np.nan)
        errors[valid_mask] = predictions[valid_mask] - actual_values[valid_mask]
        
        metrics = self._calculate_metrics(
            actual_values[valid_mask], 
            predictions[valid_mask]
        )
        
        # Create spatial error DataFrame
        spatial_errors = data[[x_col, y_col]].copy()
        if z_col:
            spatial_errors[z_col] = data[z_col]
        spatial_errors['actual'] = actual_values
        spatial_errors['predicted'] = predictions
        spatial_errors['error'] = errors
        spatial_errors['abs_error'] = np.abs(errors)
        spatial_errors['squared_error'] = errors ** 2
        spatial_errors['fold'] = fold_labels
        
        return CrossValidationResult(
            method='spatial_k_fold',
            n_folds=n_folds,
            predictions=predictions,
            actual_values=actual_values,
            errors=errors,
            fold_indices=fold_indices,
            metrics=metrics,
            fold_metrics=fold_metrics,
            spatial_errors=spatial_errors
        )
    
    def stratified_k_fold(self,
                         interpolator: BaseInterpolator,
                         data: pd.DataFrame,
                         x_col: str,
                         y_col: str,
                         value_col: str,
                         z_col: Optional[str] = None,
                         n_folds: int = 5,
                         n_strata: int = 5,
                         **interpolator_params) -> CrossValidationResult:
        """
        Perform stratified k-fold cross-validation.
        
        This method ensures each fold has similar value distributions.
        
        Args:
            interpolator: Interpolator instance
            data: Input DataFrame
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column to interpolate
            z_col: Z coordinate column (optional)
            n_folds: Number of folds
            n_strata: Number of strata for value stratification
            **interpolator_params: Additional parameters for interpolator
            
        Returns:
            CrossValidationResult with stratified k-fold validation results
        """
        self.logger.info(f"Starting stratified {n_folds}-fold validation")
        
        # Stratify by value quantiles
        values = data[value_col].values
        quantiles = np.linspace(0, 1, n_strata + 1)
        value_bins = np.quantile(values, quantiles)
        value_bins[0] -= 1e-10  # Ensure minimum is included
        value_bins[-1] += 1e-10  # Ensure maximum is included
        
        # Assign strata
        strata = np.digitize(values, value_bins) - 1
        
        # Create stratified folds
        fold_indices = [[] for _ in range(n_folds)]
        
        for stratum in range(n_strata):
            stratum_indices = np.where(strata == stratum)[0]
            np.random.shuffle(stratum_indices)
            
            # Distribute stratum indices across folds
            for i, idx in enumerate(stratum_indices):
                fold_indices[i % n_folds].append(idx)
        
        # Convert to numpy arrays
        fold_indices = [np.array(fold) for fold in fold_indices]
        
        # Perform validation using k_fold logic
        n_points = len(data)
        predictions = np.zeros(n_points)
        fold_metrics = []
        
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        for fold_idx, test_indices in enumerate(fold_indices):
            self.logger.debug(f"Processing stratified fold {fold_idx + 1}/{n_folds}")
            
            # Create train/test split
            train_mask = np.ones(n_points, dtype=bool)
            train_mask[test_indices] = False
            
            train_data = data.iloc[train_mask].copy()
            test_data = data.iloc[test_indices].copy()
            
            # Fit interpolator on training data
            interpolator.fit(train_data, coord_cols, value_col, **interpolator_params)
            
            # Predict on test data
            test_coords = test_data[coord_cols].values
            fold_predictions = interpolator.predict(test_coords)
            
            # Store predictions
            predictions[test_indices] = fold_predictions
            
            # Calculate fold metrics
            fold_actual = test_data[value_col].values
            fold_errors = fold_predictions - fold_actual
            fold_metric = self._calculate_metrics(fold_actual, fold_predictions)
            fold_metrics.append(fold_metric)
        
        # Calculate overall metrics
        actual_values = data[value_col].values
        errors = predictions - actual_values
        metrics = self._calculate_metrics(actual_values, predictions)
        
        # Create spatial error DataFrame
        spatial_errors = data[[x_col, y_col]].copy()
        if z_col:
            spatial_errors[z_col] = data[z_col]
        spatial_errors['actual'] = actual_values
        spatial_errors['predicted'] = predictions
        spatial_errors['error'] = errors
        spatial_errors['abs_error'] = np.abs(errors)
        spatial_errors['squared_error'] = errors ** 2
        spatial_errors['stratum'] = strata
        spatial_errors['fold'] = -1  # Initialize
        for fold_idx, indices in enumerate(fold_indices):
            spatial_errors.loc[spatial_errors.index[indices], 'fold'] = fold_idx
        
        return CrossValidationResult(
            method='stratified_k_fold',
            n_folds=n_folds,
            predictions=predictions,
            actual_values=actual_values,
            errors=errors,
            fold_indices=fold_indices,
            metrics=metrics,
            fold_metrics=fold_metrics,
            spatial_errors=spatial_errors
        )
    
    def _validate_single_fold(self,
                            interpolator: BaseInterpolator,
                            data: pd.DataFrame,
                            test_idx: int,
                            coord_cols: List[str],
                            value_col: str,
                            interpolator_params: Dict[str, Any]) -> float:
        """Validate a single fold (for LOO)."""
        # Create train data (all except test point)
        train_mask = np.ones(len(data), dtype=bool)
        train_mask[test_idx] = False
        train_data = data.iloc[train_mask].copy()
        
        # Fit interpolator
        interpolator.fit(train_data, coord_cols, value_col, **interpolator_params)
        
        # Predict on test point
        test_coords = data.iloc[[test_idx]][coord_cols].values
        prediction = interpolator.predict(test_coords)[0]
        
        return prediction
    
    def _apply_spatial_buffer(self,
                            fold_indices: List[np.ndarray],
                            coordinates: np.ndarray,
                            buffer_distance: float) -> List[np.ndarray]:
        """Apply spatial buffer around test regions."""
        buffered_indices = []
        kdtree = cKDTree(coordinates[:, :2])  # Use only X,Y
        
        for test_indices in fold_indices:
            if len(test_indices) == 0:
                buffered_indices.append(test_indices)
                continue
            
            # Find all points within buffer distance of test points
            test_coords = coordinates[test_indices, :2]
            buffer_indices = set(test_indices)
            
            for coord in test_coords:
                nearby = kdtree.query_ball_point(coord, buffer_distance)
                buffer_indices.update(nearby)
            
            buffered_indices.append(np.array(list(buffer_indices)))
        
        return buffered_indices
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate validation metrics."""
        errors = predicted - actual
        n = len(actual)
        
        # Basic metrics
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        
        # Relative metrics
        mean_actual = np.mean(actual)
        if mean_actual != 0:
            mape = np.mean(np.abs(errors / actual)) * 100
            normalized_rmse = rmse / mean_actual * 100
        else:
            mape = np.inf
            normalized_rmse = np.inf
        
        # R-squared
        ss_total = np.sum((actual - mean_actual) ** 2)
        ss_residual = np.sum(errors ** 2)
        if ss_total > 0:
            r_squared = 1 - (ss_residual / ss_total)
        else:
            r_squared = 0.0
        
        # Nash-Sutcliffe efficiency
        if ss_total > 0:
            nash_sutcliffe = 1 - (ss_residual / ss_total)
        else:
            nash_sutcliffe = -np.inf
        
        # Willmott's index of agreement
        numerator = np.sum(errors ** 2)
        denominator = np.sum((np.abs(predicted - mean_actual) + 
                            np.abs(actual - mean_actual)) ** 2)
        if denominator > 0:
            willmott_d = 1 - (numerator / denominator)
        else:
            willmott_d = 0.0
        
        # Bias metrics
        bias = np.mean(errors)
        percent_bias = (bias / mean_actual * 100) if mean_actual != 0 else np.inf
        
        return {
            'n_samples': n,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'normalized_rmse': float(normalized_rmse),
            'r_squared': float(r_squared),
            'nash_sutcliffe': float(nash_sutcliffe),
            'willmott_d': float(willmott_d),
            'bias': float(bias),
            'percent_bias': float(percent_bias),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'std_error': float(np.std(errors))
        }