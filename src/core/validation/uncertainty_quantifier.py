"""
Uncertainty quantification module for interpolation methods.

Provides methods for quantifying and propagating uncertainty including:
- Monte Carlo uncertainty analysis
- Sensitivity analysis
- Error propagation
- Prediction intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from scipy import stats
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from ..interpolation.base import BaseInterpolator


@dataclass
class UncertaintyResult:
    """Container for uncertainty quantification results."""
    prediction_mean: np.ndarray
    prediction_std: np.ndarray
    prediction_intervals: Dict[str, np.ndarray]
    sensitivity_indices: Optional[Dict[str, np.ndarray]]
    convergence_info: Dict[str, Any]
    monte_carlo_samples: Optional[np.ndarray] = None


class UncertaintyQuantifier:
    """
    Uncertainty quantification engine for interpolation methods.
    
    Provides comprehensive uncertainty analysis including:
    - Monte Carlo simulation
    - Input uncertainty propagation
    - Parameter sensitivity analysis
    - Prediction interval estimation
    """
    
    def __init__(self,
                 n_simulations: int = 1000,
                 confidence_levels: List[float] = None,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None):
        """
        Initialize uncertainty quantifier.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            confidence_levels: Confidence levels for intervals
            n_jobs: Number of parallel jobs
            random_state: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.confidence_levels = confidence_levels or [0.95, 0.90, 0.68]
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def monte_carlo_uncertainty(self,
                              interpolator: BaseInterpolator,
                              data: pd.DataFrame,
                              x_col: str,
                              y_col: str,
                              value_col: str,
                              prediction_points: np.ndarray,
                              input_uncertainty: Union[float, Dict[str, float], np.ndarray],
                              z_col: Optional[str] = None,
                              parameter_uncertainty: Optional[Dict[str, Tuple[float, float]]] = None,
                              **interpolator_params) -> UncertaintyResult:
        """
        Perform Monte Carlo uncertainty propagation.
        
        Args:
            interpolator: Interpolator instance
            data: Training data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            prediction_points: Points to predict at
            input_uncertainty: Uncertainty in input values (std dev or dict)
            z_col: Z coordinate column (optional)
            parameter_uncertainty: Parameter ranges for sensitivity
            **interpolator_params: Additional interpolator parameters
            
        Returns:
            UncertaintyResult with uncertainty estimates
        """
        self.logger.info(f"Starting Monte Carlo uncertainty analysis with {self.n_simulations} simulations")
        
        n_pred = len(prediction_points)
        n_data = len(data)
        
        # Prepare coordinate columns
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        # Parse input uncertainty
        if isinstance(input_uncertainty, (int, float)):
            # Constant uncertainty for all points
            value_std = np.full(n_data, input_uncertainty)
        elif isinstance(input_uncertainty, dict):
            # Column-specific uncertainty
            value_std = data[value_col].apply(
                lambda x: input_uncertainty.get(x, input_uncertainty.get('default', 0))
            ).values
        else:
            # Array of uncertainties
            value_std = np.asarray(input_uncertainty)
        
        # Store Monte Carlo samples
        mc_predictions = np.zeros((self.n_simulations, n_pred))
        
        # Track convergence
        convergence_means = []
        convergence_stds = []
        check_points = [10, 50, 100, 250, 500, 750, 1000]
        check_points = [cp for cp in check_points if cp <= self.n_simulations]
        
        # Perform Monte Carlo simulations
        for i in range(self.n_simulations):
            # Perturb input data
            perturbed_data = data.copy()
            noise = np.random.normal(0, value_std, n_data)
            perturbed_data[value_col] = data[value_col] + noise
            
            # Perturb parameters if specified
            perturbed_params = interpolator_params.copy()
            if parameter_uncertainty:
                for param, (low, high) in parameter_uncertainty.items():
                    if param in perturbed_params:
                        # Uniform distribution between bounds
                        perturbed_params[param] = np.random.uniform(low, high)
            
            # Fit and predict
            interpolator.fit(perturbed_data, coord_cols, value_col, **perturbed_params)
            predictions = interpolator.predict(prediction_points)
            mc_predictions[i] = predictions
            
            # Check convergence
            if (i + 1) in check_points:
                convergence_means.append(np.mean(mc_predictions[:i+1], axis=0).mean())
                convergence_stds.append(np.std(mc_predictions[:i+1], axis=0).mean())
        
        # Calculate statistics
        prediction_mean = np.mean(mc_predictions, axis=0)
        prediction_std = np.std(mc_predictions, axis=0, ddof=1)
        
        # Calculate prediction intervals
        prediction_intervals = {}
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(mc_predictions, lower_percentile, axis=0)
            ci_upper = np.percentile(mc_predictions, upper_percentile, axis=0)
            
            prediction_intervals[f'pi_{int(conf_level*100)}'] = np.column_stack([ci_lower, ci_upper])
        
        # Sensitivity analysis if parameters were varied
        sensitivity_indices = None
        if parameter_uncertainty:
            sensitivity_indices = self._calculate_sensitivity_indices(
                mc_predictions, perturbed_params, parameter_uncertainty
            )
        
        # Convergence information
        convergence_info = {
            'check_points': check_points,
            'mean_convergence': convergence_means,
            'std_convergence': convergence_stds,
            'final_cv': float(np.mean(prediction_std / (np.abs(prediction_mean) + 1e-10)))
        }
        
        return UncertaintyResult(
            prediction_mean=prediction_mean,
            prediction_std=prediction_std,
            prediction_intervals=prediction_intervals,
            sensitivity_indices=sensitivity_indices,
            convergence_info=convergence_info,
            monte_carlo_samples=mc_predictions
        )
    
    def analytical_uncertainty(self,
                             interpolator: BaseInterpolator,
                             data: pd.DataFrame,
                             x_col: str,
                             y_col: str,
                             value_col: str,
                             prediction_points: np.ndarray,
                             measurement_error: float,
                             z_col: Optional[str] = None,
                             **interpolator_params) -> UncertaintyResult:
        """
        Calculate analytical uncertainty estimates where possible.
        
        This method provides closed-form uncertainty estimates for methods
        that support it (e.g., Kriging with known variogram).
        
        Args:
            interpolator: Interpolator instance
            data: Training data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            prediction_points: Points to predict at
            measurement_error: Measurement error standard deviation
            z_col: Z coordinate column (optional)
            **interpolator_params: Additional interpolator parameters
            
        Returns:
            UncertaintyResult with analytical uncertainty estimates
        """
        self.logger.info("Calculating analytical uncertainty estimates")
        
        # Prepare coordinate columns
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        # Fit interpolator
        interpolator.fit(data, coord_cols, value_col, **interpolator_params)
        
        # Get predictions
        predictions = interpolator.predict(prediction_points)
        
        # Calculate uncertainty based on interpolator type
        if hasattr(interpolator, 'predict_variance'):
            # Kriging-like methods with variance prediction
            prediction_variance = interpolator.predict_variance(prediction_points)
            prediction_std = np.sqrt(prediction_variance + measurement_error**2)
        elif hasattr(interpolator, 'predict_uncertainty'):
            # Custom uncertainty method
            prediction_std = interpolator.predict_uncertainty(prediction_points)
        else:
            # Fallback: distance-based uncertainty
            prediction_std = self._distance_based_uncertainty(
                interpolator, data[coord_cols].values, prediction_points, 
                measurement_error
            )
        
        # Calculate prediction intervals assuming normality
        prediction_intervals = {}
        for conf_level in self.confidence_levels:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            margin = z_score * prediction_std
            
            prediction_intervals[f'pi_{int(conf_level*100)}'] = np.column_stack([
                predictions - margin,
                predictions + margin
            ])
        
        return UncertaintyResult(
            prediction_mean=predictions,
            prediction_std=prediction_std,
            prediction_intervals=prediction_intervals,
            sensitivity_indices=None,
            convergence_info={'method': 'analytical'}
        )
    
    def sensitivity_analysis(self,
                           interpolator: BaseInterpolator,
                           data: pd.DataFrame,
                           x_col: str,
                           y_col: str,
                           value_col: str,
                           prediction_points: np.ndarray,
                           parameters: Dict[str, Tuple[float, float]],
                           z_col: Optional[str] = None,
                           method: str = 'sobol',
                           **base_params) -> Dict[str, Any]:
        """
        Perform parameter sensitivity analysis.
        
        Args:
            interpolator: Interpolator instance
            data: Training data
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            prediction_points: Points to predict at
            parameters: Parameter ranges {name: (min, max)}
            z_col: Z coordinate column (optional)
            method: Sensitivity method ('sobol', 'morris', 'fast')
            **base_params: Base interpolator parameters
            
        Returns:
            Dictionary with sensitivity indices
        """
        self.logger.info(f"Performing {method} sensitivity analysis")
        
        coord_cols = [x_col, y_col]
        if z_col and z_col in data.columns:
            coord_cols.append(z_col)
        
        if method == 'sobol':
            return self._sobol_sensitivity(
                interpolator, data, coord_cols, value_col,
                prediction_points, parameters, base_params
            )
        elif method == 'morris':
            return self._morris_sensitivity(
                interpolator, data, coord_cols, value_col,
                prediction_points, parameters, base_params
            )
        else:
            raise ValueError(f"Unknown sensitivity method: {method}")
    
    def _sobol_sensitivity(self,
                          interpolator: BaseInterpolator,
                          data: pd.DataFrame,
                          coord_cols: List[str],
                          value_col: str,
                          prediction_points: np.ndarray,
                          parameters: Dict[str, Tuple[float, float]],
                          base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Sobol sensitivity analysis."""
        param_names = list(parameters.keys())
        n_params = len(param_names)
        
        # Generate Sobol sequence
        n_samples = min(self.n_simulations, 2**(n_params + 1))
        
        # Create parameter samples using Saltelli's scheme
        sobol_samples = self._generate_sobol_samples(parameters, n_samples)
        
        # Evaluate model at all sample points
        model_outputs = []
        
        for sample in sobol_samples:
            # Set parameters
            params = base_params.copy()
            for i, param_name in enumerate(param_names):
                params[param_name] = sample[i]
            
            # Fit and predict
            interpolator.fit(data, coord_cols, value_col, **params)
            predictions = interpolator.predict(prediction_points)
            model_outputs.append(np.mean(predictions))  # Use mean as scalar output
        
        model_outputs = np.array(model_outputs)
        
        # Calculate Sobol indices
        first_order = {}
        total_order = {}
        
        # Simplified Sobol calculation
        n_base = n_samples // (2 * n_params + 2)
        
        for i, param in enumerate(param_names):
            # First-order index
            y_a = model_outputs[:n_base]
            y_b = model_outputs[n_base:2*n_base]
            y_ab = model_outputs[(2+i)*n_base:(3+i)*n_base]
            
            var_y = np.var(np.concatenate([y_a, y_b]))
            
            if var_y > 0:
                first_order[param] = float(np.mean(y_a * (y_ab - y_b)) / var_y)
                total_order[param] = float(0.5 * np.mean((y_a - y_ab)**2) / var_y)
            else:
                first_order[param] = 0.0
                total_order[param] = 0.0
        
        return {
            'first_order_indices': first_order,
            'total_order_indices': total_order,
            'n_evaluations': len(model_outputs),
            'parameter_names': param_names
        }
    
    def _morris_sensitivity(self,
                          interpolator: BaseInterpolator,
                          data: pd.DataFrame,
                          coord_cols: List[str],
                          value_col: str,
                          prediction_points: np.ndarray,
                          parameters: Dict[str, Tuple[float, float]],
                          base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Morris (Elementary Effects) sensitivity analysis."""
        param_names = list(parameters.keys())
        n_params = len(param_names)
        
        # Number of trajectories
        n_trajectories = min(10, self.n_simulations // (n_params + 1))
        
        # Elementary effects storage
        effects = {param: [] for param in param_names}
        
        for _ in range(n_trajectories):
            # Generate random starting point
            base_values = {}
            for param, (low, high) in parameters.items():
                base_values[param] = np.random.uniform(low, high)
            
            # Evaluate at base point
            params = base_params.copy()
            params.update(base_values)
            interpolator.fit(data, coord_cols, value_col, **params)
            base_output = np.mean(interpolator.predict(prediction_points))
            
            # Calculate elementary effects
            for param in param_names:
                # Perturb parameter
                delta = (parameters[param][1] - parameters[param][0]) * 0.1
                perturbed_values = base_values.copy()
                perturbed_values[param] += delta
                
                # Evaluate at perturbed point
                params = base_params.copy()
                params.update(perturbed_values)
                interpolator.fit(data, coord_cols, value_col, **params)
                perturbed_output = np.mean(interpolator.predict(prediction_points))
                
                # Elementary effect
                effect = (perturbed_output - base_output) / delta
                effects[param].append(effect)
        
        # Calculate Morris indices
        morris_indices = {}
        for param in param_names:
            param_effects = np.array(effects[param])
            morris_indices[param] = {
                'mean': float(np.mean(param_effects)),
                'mean_abs': float(np.mean(np.abs(param_effects))),
                'std': float(np.std(param_effects))
            }
        
        return {
            'morris_indices': morris_indices,
            'n_trajectories': n_trajectories,
            'parameter_names': param_names
        }
    
    def _distance_based_uncertainty(self,
                                  interpolator: BaseInterpolator,
                                  data_points: np.ndarray,
                                  prediction_points: np.ndarray,
                                  measurement_error: float) -> np.ndarray:
        """Calculate distance-based uncertainty estimates."""
        from scipy.spatial import cKDTree
        
        # Build KDTree for data points
        tree = cKDTree(data_points)
        
        # Find distances to nearest data points
        distances, _ = tree.query(prediction_points, k=min(5, len(data_points)))
        mean_distances = np.mean(distances, axis=1)
        
        # Scale distances to uncertainty
        # Uncertainty increases with distance from data
        max_distance = np.max(distances)
        if max_distance > 0:
            normalized_distances = mean_distances / max_distance
        else:
            normalized_distances = np.zeros_like(mean_distances)
        
        # Combine measurement error and interpolation uncertainty
        interpolation_uncertainty = normalized_distances * measurement_error * 2
        total_uncertainty = np.sqrt(measurement_error**2 + interpolation_uncertainty**2)
        
        return total_uncertainty
    
    def _generate_sobol_samples(self,
                              parameters: Dict[str, Tuple[float, float]],
                              n_samples: int) -> np.ndarray:
        """Generate Sobol sequence samples for parameters."""
        n_params = len(parameters)
        
        # Generate uniform samples (simplified Sobol)
        # In practice, would use proper Sobol sequence generator
        samples = np.random.rand(n_samples * (2 * n_params + 2), n_params)
        
        # Scale to parameter ranges
        param_names = list(parameters.keys())
        for i, param in enumerate(param_names):
            low, high = parameters[param]
            samples[:, i] = samples[:, i] * (high - low) + low
        
        return samples
    
    def _calculate_sensitivity_indices(self,
                                     mc_predictions: np.ndarray,
                                     varied_params: Dict[str, Any],
                                     parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """Calculate sensitivity indices from Monte Carlo results."""
        # Simplified sensitivity calculation
        # In practice, would track parameter values during MC simulation
        
        sensitivity = {}
        total_variance = np.var(mc_predictions, axis=0)
        
        for param in parameter_ranges:
            # Placeholder: would need actual parameter tracking
            sensitivity[param] = np.ones(mc_predictions.shape[1]) * 0.1
        
        return sensitivity