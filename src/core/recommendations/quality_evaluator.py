"""
Quality evaluator for interpolation results.

Provides cross-validation and quality metrics to assess
interpolation performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


@dataclass 
class QualityMetrics:
    """Container for interpolation quality metrics."""
    rmse: float  # Root Mean Square Error
    mae: float   # Mean Absolute Error
    r_squared: float  # Coefficient of determination
    mape: float  # Mean Absolute Percentage Error
    max_error: float
    min_error: float
    std_error: float
    bias: float  # Mean error (systematic bias)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'r_squared': self.r_squared,
            'mape': self.mape,
            'max_error': self.max_error,
            'min_error': self.min_error,
            'std_error': self.std_error,
            'bias': self.bias
        }


@dataclass
class CrossValidationResult:
    """Results from cross-validation."""
    method: str
    metrics: QualityMetrics
    fold_metrics: List[QualityMetrics]
    error_locations: np.ndarray  # Coordinates of errors
    error_values: np.ndarray     # Error magnitudes
    computation_time: float
    n_folds: int
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of cross-validation results."""
        return {
            'method': self.method,
            'overall_metrics': self.metrics.to_dict(),
            'computation_time': self.computation_time,
            'n_folds': self.n_folds,
            'fold_variability': {
                'rmse_std': np.std([m.rmse for m in self.fold_metrics]),
                'mae_std': np.std([m.mae for m in self.fold_metrics])
            }
        }


class QualityEvaluator:
    """
    Evaluates interpolation quality using cross-validation.
    
    Supports:
    - Leave-one-out cross-validation (LOOCV)
    - K-fold cross-validation
    - Spatial cross-validation
    - Multiple quality metrics
    - Error visualization data
    """
    
    def __init__(self):
        """Initialize quality evaluator."""
        self.data: Optional[pd.DataFrame] = None
        self.interpolator = None
        
    def evaluate(self,
                interpolator: Any,
                data: pd.DataFrame,
                x_col: str,
                y_col: str,
                value_col: str,
                z_col: Optional[str] = None,
                cv_method: str = 'loocv',
                n_folds: int = 5,
                n_jobs: int = 1) -> CrossValidationResult:
        """
        Evaluate interpolation quality using cross-validation.
        
        Args:
            interpolator: Interpolator instance
            data: Training data
            x_col, y_col, value_col: Column names
            z_col: Optional Z coordinate column
            cv_method: 'loocv', 'kfold', or 'spatial'
            n_folds: Number of folds for k-fold CV
            n_jobs: Number of parallel jobs
            
        Returns:
            CrossValidationResult with quality metrics
        """
        self.data = data
        self.interpolator = interpolator
        
        start_time = time.time()
        
        # Select cross-validation method
        if cv_method == 'loocv':
            result = self._loocv(data, x_col, y_col, value_col, z_col, n_jobs)
        elif cv_method == 'kfold':
            result = self._kfold_cv(data, x_col, y_col, value_col, z_col, n_folds, n_jobs)
        elif cv_method == 'spatial':
            result = self._spatial_cv(data, x_col, y_col, value_col, z_col, n_folds)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        computation_time = time.time() - start_time
        result.computation_time = computation_time
        
        return result
    
    def _loocv(self,
               data: pd.DataFrame,
               x_col: str,
               y_col: str, 
               value_col: str,
               z_col: Optional[str],
               n_jobs: int) -> CrossValidationResult:
        """
        Leave-one-out cross-validation.
        
        Most accurate but computationally expensive method.
        """
        n_points = len(data)
        errors = np.zeros(n_points)
        predictions = np.zeros(n_points)
        actuals = data[value_col].values
        
        coord_cols = [x_col, y_col]
        if z_col:
            coord_cols.append(z_col)
        coordinates = data[coord_cols].values
        
        # For small datasets, use sequential processing
        if n_points < 50 or n_jobs == 1:
            for i in range(n_points):
                # Leave out point i
                train_mask = np.ones(n_points, dtype=bool)
                train_mask[i] = False
                
                train_data = data[train_mask]
                test_point = coordinates[i].reshape(1, -1)
                
                try:
                    # Fit on training data
                    self.interpolator.fit(train_data, x_col, y_col, value_col, z_col)
                    
                    # Predict on test point
                    pred = self.interpolator.predict(test_point)[0]
                    predictions[i] = pred
                    errors[i] = pred - actuals[i]
                except Exception as e:
                    warnings.warn(f"Error in LOOCV fold {i}: {e}")
                    predictions[i] = np.nan
                    errors[i] = np.nan
        else:
            # Parallel processing for larger datasets
            def process_fold(i):
                train_mask = np.ones(n_points, dtype=bool)
                train_mask[i] = False
                train_data = data[train_mask]
                test_point = coordinates[i].reshape(1, -1)
                
                try:
                    # Create a copy of interpolator for thread safety
                    interp_copy = type(self.interpolator)(**self.interpolator.get_parameters())
                    interp_copy.fit(train_data, x_col, y_col, value_col, z_col)
                    pred = interp_copy.predict(test_point)[0]
                    return i, pred, pred - actuals[i]
                except Exception as e:
                    return i, np.nan, np.nan
            
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(process_fold, i) for i in range(n_points)]
                
                for future in as_completed(futures):
                    i, pred, error = future.result()
                    predictions[i] = pred
                    errors[i] = error
        
        # Calculate metrics
        metrics = self._calculate_metrics(actuals, predictions)
        
        # For LOOCV, each fold has only one point, so fold metrics not meaningful
        fold_metrics = []
        
        return CrossValidationResult(
            method=self.interpolator.get_method_name(),
            metrics=metrics,
            fold_metrics=fold_metrics,
            error_locations=coordinates,
            error_values=errors,
            computation_time=0.0,  # Will be set by caller
            n_folds=n_points
        )
    
    def _kfold_cv(self,
                  data: pd.DataFrame,
                  x_col: str,
                  y_col: str,
                  value_col: str,
                  z_col: Optional[str],
                  n_folds: int,
                  n_jobs: int) -> CrossValidationResult:
        """K-fold cross-validation."""
        n_points = len(data)
        
        # Shuffle indices for random folds
        indices = np.arange(n_points)
        np.random.shuffle(indices)
        
        # Create folds
        fold_size = n_points // n_folds
        fold_metrics = []
        all_errors = []
        all_coordinates = []
        
        for fold in range(n_folds):
            # Define test indices for this fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_points
            test_indices = indices[start_idx:end_idx]
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Split data
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            
            coord_cols = [x_col, y_col]
            if z_col:
                coord_cols.append(z_col)
                
            test_coords = test_data[coord_cols].values
            test_actuals = test_data[value_col].values
            
            try:
                # Fit and predict
                self.interpolator.fit(train_data, x_col, y_col, value_col, z_col)
                predictions = self.interpolator.predict(test_coords)
                
                # Calculate fold metrics
                fold_metric = self._calculate_metrics(test_actuals, predictions)
                fold_metrics.append(fold_metric)
                
                # Store errors and locations
                errors = predictions - test_actuals
                all_errors.extend(errors)
                all_coordinates.extend(test_coords)
                
            except Exception as e:
                warnings.warn(f"Error in fold {fold}: {e}")
                # Add dummy metrics for failed fold
                fold_metrics.append(QualityMetrics(
                    rmse=np.nan, mae=np.nan, r_squared=np.nan,
                    mape=np.nan, max_error=np.nan, min_error=np.nan,
                    std_error=np.nan, bias=np.nan
                ))
        
        # Calculate overall metrics
        all_errors = np.array(all_errors)
        all_coordinates = np.array(all_coordinates)
        
        # Reconstruct predictions from errors
        # Note: This is approximate since we don't have all actuals aligned
        overall_metrics = self._aggregate_fold_metrics(fold_metrics)
        
        return CrossValidationResult(
            method=self.interpolator.get_method_name(),
            metrics=overall_metrics,
            fold_metrics=fold_metrics,
            error_locations=all_coordinates,
            error_values=all_errors,
            computation_time=0.0,
            n_folds=n_folds
        )
    
    def _spatial_cv(self,
                    data: pd.DataFrame,
                    x_col: str,
                    y_col: str,
                    value_col: str,
                    z_col: Optional[str],
                    n_folds: int) -> CrossValidationResult:
        """
        Spatial cross-validation.
        
        Creates spatially coherent folds to avoid optimistic bias.
        """
        # Implement spatial blocking
        coord_cols = [x_col, y_col]
        coordinates = data[coord_cols].values
        
        # Create spatial blocks using k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_folds, random_state=42)
        fold_labels = kmeans.fit_predict(coordinates)
        
        # Continue with k-fold logic but using spatial folds
        fold_metrics = []
        all_errors = []
        all_coordinates = []
        
        for fold in range(n_folds):
            test_mask = fold_labels == fold
            train_mask = ~test_mask
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            if len(test_data) == 0:
                continue
                
            coord_cols = [x_col, y_col]
            if z_col:
                coord_cols.append(z_col)
                
            test_coords = test_data[coord_cols].values
            test_actuals = test_data[value_col].values
            
            try:
                self.interpolator.fit(train_data, x_col, y_col, value_col, z_col)
                predictions = self.interpolator.predict(test_coords)
                
                fold_metric = self._calculate_metrics(test_actuals, predictions)
                fold_metrics.append(fold_metric)
                
                errors = predictions - test_actuals
                all_errors.extend(errors)
                all_coordinates.extend(test_coords)
                
            except Exception as e:
                warnings.warn(f"Error in spatial fold {fold}: {e}")
        
        all_errors = np.array(all_errors)
        all_coordinates = np.array(all_coordinates)
        overall_metrics = self._aggregate_fold_metrics(fold_metrics)
        
        return CrossValidationResult(
            method=self.interpolator.get_method_name(),
            metrics=overall_metrics,
            fold_metrics=fold_metrics,
            error_locations=all_coordinates,
            error_values=all_errors,
            computation_time=0.0,
            n_folds=n_folds
        )
    
    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> QualityMetrics:
        """Calculate quality metrics from actual and predicted values."""
        # Remove NaN values
        mask = ~(np.isnan(actuals) | np.isnan(predictions))
        actuals = actuals[mask]
        predictions = predictions[mask]
        
        if len(actuals) == 0:
            return QualityMetrics(
                rmse=np.nan, mae=np.nan, r_squared=np.nan,
                mape=np.nan, max_error=np.nan, min_error=np.nan,
                std_error=np.nan, bias=np.nan
            )
        
        errors = predictions - actuals
        
        # Basic metrics
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        bias = np.mean(errors)
        std_error = np.std(errors)
        
        # R-squared
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((actuals - np.mean(actuals))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # MAPE (avoid division by zero)
        non_zero_mask = actuals != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(errors[non_zero_mask] / actuals[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        # Error bounds
        max_error = np.max(np.abs(errors))
        min_error = np.min(np.abs(errors))
        
        return QualityMetrics(
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
            mape=mape,
            max_error=max_error,
            min_error=min_error,
            std_error=std_error,
            bias=bias
        )
    
    def _aggregate_fold_metrics(self, fold_metrics: List[QualityMetrics]) -> QualityMetrics:
        """Aggregate metrics from multiple folds."""
        # Remove folds with NaN metrics
        valid_metrics = [m for m in fold_metrics if not np.isnan(m.rmse)]
        
        if not valid_metrics:
            return QualityMetrics(
                rmse=np.nan, mae=np.nan, r_squared=np.nan,
                mape=np.nan, max_error=np.nan, min_error=np.nan,
                std_error=np.nan, bias=np.nan
            )
        
        # Average metrics across folds
        return QualityMetrics(
            rmse=np.mean([m.rmse for m in valid_metrics]),
            mae=np.mean([m.mae for m in valid_metrics]),
            r_squared=np.mean([m.r_squared for m in valid_metrics]),
            mape=np.mean([m.mape for m in valid_metrics if not np.isnan(m.mape)]),
            max_error=np.max([m.max_error for m in valid_metrics]),
            min_error=np.min([m.min_error for m in valid_metrics]),
            std_error=np.mean([m.std_error for m in valid_metrics]),
            bias=np.mean([m.bias for m in valid_metrics])
        )
    
    def compare_methods(self,
                       methods: List[Tuple[str, Any]],
                       data: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       value_col: str,
                       z_col: Optional[str] = None,
                       cv_method: str = 'kfold',
                       n_folds: int = 5) -> Dict[str, CrossValidationResult]:
        """
        Compare multiple interpolation methods.
        
        Args:
            methods: List of (name, interpolator) tuples
            Other args same as evaluate()
            
        Returns:
            Dictionary mapping method names to results
        """
        results = {}
        
        for name, interpolator in methods:
            try:
                result = self.evaluate(
                    interpolator, data, x_col, y_col, value_col, z_col,
                    cv_method, n_folds
                )
                results[name] = result
            except Exception as e:
                warnings.warn(f"Failed to evaluate {name}: {e}")
                
        return results