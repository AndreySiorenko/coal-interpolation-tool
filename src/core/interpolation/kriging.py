"""
Kriging interpolation implementation.

Kriging is a geostatistical interpolation technique that provides optimal 
unbiased prediction and uncertainty quantification based on spatial covariance
structure characterized by variograms.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Union, List, Callable
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize
from scipy.linalg import solve, LinAlgError
from scipy.spatial.distance import pdist, squareform
import warnings

from .base import BaseInterpolator, InterpolationParameters, SearchParameters
from .base import FittingError, PredictionError
from .variogram_analysis import VariogramAnalyzer, VariogramAnalysisOptions, quick_variogram_fit


class VariogramModel(Enum):
    """Available variogram model types."""
    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"
    POWER = "power"
    NUGGET = "nugget"


class KrigingType(Enum):
    """Available kriging types."""
    ORDINARY = "ordinary"
    SIMPLE = "simple"
    UNIVERSAL = "universal"


@dataclass
class KrigingParameters(InterpolationParameters):
    """Parameters specific to Kriging interpolation."""
    kriging_type: KrigingType = KrigingType.ORDINARY     # Type of kriging
    variogram_model: VariogramModel = VariogramModel.SPHERICAL  # Variogram model
    nugget: float = 0.0                                  # Nugget effect
    sill: float = 1.0                                    # Sill (total variance)
    range_param: float = 1000.0                          # Range parameter
    use_global: bool = True                              # Use global kriging (vs local)
    
    # For simple kriging
    mean_value: Optional[float] = None                   # Known mean (simple kriging)
    
    # Model fitting parameters
    auto_fit_variogram: bool = True                      # Automatically fit variogram
    fit_nugget: bool = True                              # Fit nugget parameter
    fit_sill: bool = True                                # Fit sill parameter
    fit_range: bool = True                               # Fit range parameter
    
    # Numerical stability
    condition_number_threshold: float = 1e12             # Condition number threshold
    regularization: float = 1e-10                        # Small regularization for stability


class VariogramModels:
    """Collection of variogram model functions."""
    
    @staticmethod
    def spherical(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """
        Spherical variogram model.
        
        γ(h) = nugget + sill * (1.5*h/a - 0.5*(h/a)³) for h ≤ a
        γ(h) = nugget + sill for h > a
        
        Args:
            h: Distance array
            nugget: Nugget effect
            sill: Sill variance
            range_param: Range parameter
            
        Returns:
            Variogram values
        """
        gamma = np.full_like(h, nugget + sill, dtype=float)
        mask = (h > 0) & (h < range_param)
        h_normalized = h[mask] / range_param
        gamma[mask] = nugget + sill * (1.5 * h_normalized - 0.5 * h_normalized**3)
        gamma[h == 0] = 0.0  # Variogram is 0 at distance 0
        return gamma
    
    @staticmethod
    def exponential(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """
        Exponential variogram model.
        
        γ(h) = nugget + sill * (1 - exp(-h/a))
        
        Args:
            h: Distance array
            nugget: Nugget effect
            sill: Sill variance
            range_param: Range parameter
            
        Returns:
            Variogram values
        """
        gamma = nugget + sill * (1 - np.exp(-h / range_param))
        gamma[h == 0] = 0.0
        return gamma
    
    @staticmethod
    def gaussian(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """
        Gaussian variogram model.
        
        γ(h) = nugget + sill * (1 - exp(-(h/a)²))
        
        Args:
            h: Distance array
            nugget: Nugget effect
            sill: Sill variance
            range_param: Range parameter
            
        Returns:
            Variogram values
        """
        gamma = nugget + sill * (1 - np.exp(-(h / range_param)**2))
        gamma[h == 0] = 0.0
        return gamma
    
    @staticmethod
    def linear(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """
        Linear variogram model.
        
        γ(h) = nugget + slope * h
        
        Args:
            h: Distance array
            nugget: Nugget effect
            sill: Used as slope parameter
            range_param: Not used (kept for interface consistency)
            
        Returns:
            Variogram values
        """
        gamma = nugget + sill * h
        gamma[h == 0] = 0.0
        return gamma
    
    @staticmethod
    def power(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """
        Power variogram model.
        
        γ(h) = nugget + c * h^α
        
        Args:
            h: Distance array
            nugget: Nugget effect
            sill: Scaling parameter
            range_param: Power parameter (alpha)
            
        Returns:
            Variogram values
        """
        # Prevent issues with h=0 for non-integer powers
        h_safe = np.where(h == 0, 1e-10, h)
        gamma = nugget + sill * h_safe**range_param
        gamma[h == 0] = 0.0
        return gamma
    
    @staticmethod
    def nugget(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """
        Pure nugget variogram model.
        
        γ(h) = 0 for h = 0
        γ(h) = nugget for h > 0
        
        Args:
            h: Distance array
            nugget: Nugget effect
            sill: Not used
            range_param: Not used
            
        Returns:
            Variogram values
        """
        gamma = np.where(h == 0, 0.0, nugget)
        return gamma


class KrigingInterpolator(BaseInterpolator):
    """
    Kriging interpolation with automatic variogram fitting.
    
    Kriging is an optimal geostatistical interpolation method that:
    - Provides unbiased predictions with minimum variance
    - Quantifies prediction uncertainty (kriging variance)
    - Honors the spatial correlation structure via variogram modeling
    - Exact interpolator (passes through data points)
    
    The method fits a variogram model to characterize spatial correlation,
    then uses this model to compute optimal weights for interpolation.
    
    Parameters:
        search_params: Search parameters for finding neighbors
        kriging_params: Kriging-specific parameters including variogram model
    """
    
    def __init__(self,
                 search_params: Optional[SearchParameters] = None,
                 kriging_params: Optional[KrigingParameters] = None):
        """Initialize Kriging interpolator with given parameters."""
        super().__init__(search_params)
        self.kriging_params = kriging_params or KrigingParameters()
        
        # Training data storage
        self.training_points = None
        self.training_values = None
        self.variogram_function = None
        
        # Fitted variogram parameters
        self.fitted_nugget = None
        self.fitted_sill = None
        self.fitted_range = None
        
        # For global kriging
        self.kriging_matrix = None
        self.kriging_weights = None
        
        # Mean for simple kriging
        self.data_mean = None
        
        # Get variogram function
        self.variogram_function = self._get_variogram_function()
        
        # Advanced variogram analysis
        self.variogram_analyzer = None
        self.variogram_analysis_results = None
    
    def get_method_name(self) -> str:
        """Return human-readable name of the interpolation method."""
        kriging_type = self.kriging_params.kriging_type.value.title()
        model_type = self.kriging_params.variogram_model.value.title()
        return f"{kriging_type} Kriging ({model_type})"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters of the interpolator."""
        params = {
            'kriging_type': self.kriging_params.kriging_type.value,
            'variogram_model': self.kriging_params.variogram_model.value,
            'nugget': self.kriging_params.nugget,
            'sill': self.kriging_params.sill,
            'range': self.kriging_params.range_param,
            'use_global': self.kriging_params.use_global,
            'auto_fit_variogram': self.kriging_params.auto_fit_variogram,
        }
        
        # Add fitted parameters if available
        if self.is_fitted:
            if self.fitted_nugget is not None:
                params['fitted_nugget'] = self.fitted_nugget
            if self.fitted_sill is not None:
                params['fitted_sill'] = self.fitted_sill
            if self.fitted_range is not None:
                params['fitted_range'] = self.fitted_range
        
        # Add search parameters if using local kriging
        if not self.kriging_params.use_global:
            params.update({
                'search_radius': self.search_params.search_radius,
                'min_points': self.search_params.min_points,
                'max_points': self.search_params.max_points,
            })
        
        return params
    
    def set_parameters(self, **params) -> 'KrigingInterpolator':
        """Set interpolator parameters."""
        # Handle kriging-specific parameters
        kriging_param_names = [
            'kriging_type', 'variogram_model', 'nugget', 'sill', 'range_param',
            'use_global', 'auto_fit_variogram', 'mean_value', 'fit_nugget',
            'fit_sill', 'fit_range', 'condition_number_threshold', 'regularization'
        ]
        
        for key in kriging_param_names:
            if key in params:
                if key == 'kriging_type':
                    if isinstance(params[key], str):
                        try:
                            kriging_type = KrigingType(params[key].lower())
                            setattr(self.kriging_params, key, kriging_type)
                        except ValueError:
                            raise ValueError(f"Invalid kriging type: {params[key]}")
                    elif isinstance(params[key], KrigingType):
                        setattr(self.kriging_params, key, params[key])
                elif key == 'variogram_model':
                    if isinstance(params[key], str):
                        try:
                            model = VariogramModel(params[key].lower())
                            setattr(self.kriging_params, key, model)
                        except ValueError:
                            raise ValueError(f"Invalid variogram model: {params[key]}")
                    elif isinstance(params[key], VariogramModel):
                        setattr(self.kriging_params, key, params[key])
                else:
                    setattr(self.kriging_params, key, params[key])
        
        # Update variogram function
        self.variogram_function = self._get_variogram_function()
        
        # Let parent handle search parameters
        super().set_parameters(**params)
        
        return self
    
    def _get_variogram_function(self) -> Callable:
        """Get the variogram function based on current parameters."""
        model_map = {
            VariogramModel.SPHERICAL: VariogramModels.spherical,
            VariogramModel.EXPONENTIAL: VariogramModels.exponential,
            VariogramModel.GAUSSIAN: VariogramModels.gaussian,
            VariogramModel.LINEAR: VariogramModels.linear,
            VariogramModel.POWER: VariogramModels.power,
            VariogramModel.NUGGET: VariogramModels.nugget,
        }
        
        return model_map[self.kriging_params.variogram_model]
    
    def fit(self,
            data: pd.DataFrame,
            x_col: str,
            y_col: str,
            value_col: str,
            z_col: Optional[str] = None,
            **kwargs) -> 'KrigingInterpolator':
        """
        Fit the Kriging interpolator to training data.
        
        Args:
            data: Training data DataFrame
            x_col: Name of X coordinate column
            y_col: Name of Y coordinate column
            value_col: Name of value column to interpolate
            z_col: Name of Z coordinate column (for 3D interpolation)
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            FittingError: If fitting fails
        """
        try:
            # Validate input data
            clean_data = self._validate_training_data(data, x_col, y_col, value_col, z_col)
            
            # Store column mappings
            self.coordinate_columns = {'X': x_col, 'Y': y_col}
            if z_col:
                self.coordinate_columns['Z'] = z_col
            self.value_column = value_col
            
            # Extract coordinates and values
            coord_cols = [x_col, y_col]
            if z_col:
                coord_cols.append(z_col)
                
            self.training_points = clean_data[coord_cols].values
            self.training_values = clean_data[value_col].values
            self.training_data = clean_data
            
            # Calculate data mean for simple kriging
            self.data_mean = np.mean(self.training_values)
            
            # Fit variogram if requested
            if self.kriging_params.auto_fit_variogram:
                self._fit_variogram()
            else:
                # Use provided parameters
                self.fitted_nugget = self.kriging_params.nugget
                self.fitted_sill = self.kriging_params.sill
                self.fitted_range = self.kriging_params.range_param
            
            # Pre-compute kriging matrix for global kriging
            if self.kriging_params.use_global:
                self._setup_global_kriging()
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            raise FittingError(f"Kriging fitting failed: {e}")
    
    def _fit_variogram(self):
        """Fit variogram model to empirical variogram."""
        # Calculate empirical variogram
        distances = pdist(self.training_points)
        values_diff = pdist(self.training_values.reshape(-1, 1))
        gamma_exp = 0.5 * values_diff.flatten()**2
        
        # Remove zero distances to avoid issues
        nonzero_mask = distances > 1e-10
        distances = distances[nonzero_mask]
        gamma_exp = gamma_exp[nonzero_mask]
        
        if len(distances) == 0:
            warnings.warn("No valid distances for variogram fitting. Using default parameters.")
            self.fitted_nugget = self.kriging_params.nugget
            self.fitted_sill = self.kriging_params.sill
            self.fitted_range = self.kriging_params.range_param
            return
        
        # Initial parameter estimates
        max_distance = np.max(distances)
        max_gamma = np.max(gamma_exp)
        
        initial_nugget = np.min(gamma_exp) if self.kriging_params.fit_nugget else self.kriging_params.nugget
        initial_sill = max_gamma - initial_nugget if self.kriging_params.fit_sill else self.kriging_params.sill
        initial_range = max_distance / 3 if self.kriging_params.fit_range else self.kriging_params.range_param
        
        # Parameter bounds
        bounds = []
        x0 = []
        
        if self.kriging_params.fit_nugget:
            bounds.append((0, max_gamma))
            x0.append(max(0, initial_nugget))
        if self.kriging_params.fit_sill:
            bounds.append((1e-10, max_gamma * 2))
            x0.append(max(1e-10, initial_sill))
        if self.kriging_params.fit_range:
            bounds.append((max_distance / 100, max_distance * 2))
            x0.append(max(max_distance / 100, initial_range))
        
        if not bounds:
            # No parameters to fit
            self.fitted_nugget = self.kriging_params.nugget
            self.fitted_sill = self.kriging_params.sill
            self.fitted_range = self.kriging_params.range_param
            return
        
        # Objective function for fitting
        def objective(params):
            param_idx = 0
            nugget = params[param_idx] if self.kriging_params.fit_nugget else self.kriging_params.nugget
            if self.kriging_params.fit_nugget:
                param_idx += 1
                
            sill = params[param_idx] if self.kriging_params.fit_sill else self.kriging_params.sill
            if self.kriging_params.fit_sill:
                param_idx += 1
                
            range_param = params[param_idx] if self.kriging_params.fit_range else self.kriging_params.range_param
            
            # Calculate model variogram
            gamma_model = self.variogram_function(distances, nugget, sill, range_param)
            
            # Return weighted least squares error
            return np.sum((gamma_exp - gamma_model)**2)
        
        try:
            # Optimize parameters
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                param_idx = 0
                self.fitted_nugget = result.x[param_idx] if self.kriging_params.fit_nugget else self.kriging_params.nugget
                if self.kriging_params.fit_nugget:
                    param_idx += 1
                    
                self.fitted_sill = result.x[param_idx] if self.kriging_params.fit_sill else self.kriging_params.sill
                if self.kriging_params.fit_sill:
                    param_idx += 1
                    
                self.fitted_range = result.x[param_idx] if self.kriging_params.fit_range else self.kriging_params.range_param
            else:
                warnings.warn("Variogram fitting failed. Using initial estimates.")
                self.fitted_nugget = initial_nugget
                self.fitted_sill = initial_sill
                self.fitted_range = initial_range
                
        except Exception as e:
            warnings.warn(f"Variogram fitting error: {e}. Using initial estimates.")
            self.fitted_nugget = initial_nugget
            self.fitted_sill = initial_sill
            self.fitted_range = initial_range
    
    def _setup_global_kriging(self):
        """Setup global kriging system."""
        n_points = len(self.training_points)
        
        # Calculate distance matrix
        distances = squareform(pdist(self.training_points))
        
        # Calculate variogram matrix
        gamma_matrix = self.variogram_function(
            distances, self.fitted_nugget, self.fitted_sill, self.fitted_range
        )
        
        # Setup kriging system based on type
        if self.kriging_params.kriging_type == KrigingType.ORDINARY:
            # Ordinary kriging: [Γ 1; 1^T 0] [λ; μ] = [γ; 1]
            self.kriging_matrix = np.zeros((n_points + 1, n_points + 1))
            self.kriging_matrix[:n_points, :n_points] = gamma_matrix
            self.kriging_matrix[:n_points, -1] = 1.0
            self.kriging_matrix[-1, :n_points] = 1.0
            
        elif self.kriging_params.kriging_type == KrigingType.SIMPLE:
            # Simple kriging: Γ λ = γ
            self.kriging_matrix = gamma_matrix
            
        elif self.kriging_params.kriging_type == KrigingType.UNIVERSAL:
            # Universal kriging: [Γ F; F^T 0] [λ; β] = [γ; f]
            # Build trend matrix for training points
            trend_matrix = self._build_trend_matrix(self.training_points)
            n_trend = trend_matrix.shape[1]
            
            self.kriging_matrix = np.zeros((n_points + n_trend, n_points + n_trend))
            self.kriging_matrix[:n_points, :n_points] = gamma_matrix
            self.kriging_matrix[:n_points, n_points:] = trend_matrix
            self.kriging_matrix[n_points:, :n_points] = trend_matrix.T
            
        # Add regularization for numerical stability
        if self.kriging_params.kriging_type == KrigingType.ORDINARY:
            np.fill_diagonal(self.kriging_matrix[:-1, :-1], 
                            np.diag(self.kriging_matrix[:-1, :-1]) + self.kriging_params.regularization)
        else:
            np.fill_diagonal(self.kriging_matrix, 
                            np.diag(self.kriging_matrix) + self.kriging_params.regularization)
    
    def predict(self,
                points: Union[np.ndarray, pd.DataFrame, List[Tuple[float, float]]],
                return_variance: bool = False,
                **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict values at given points using Kriging interpolation.
        
        Args:
            points: Points to predict at
            return_variance: Whether to return kriging variance
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted values, optionally with kriging variance
            
        Raises:
            PredictionError: If prediction fails
        """
        if not self.is_fitted:
            raise PredictionError("Kriging interpolator must be fitted before prediction")
        
        try:
            # Parse input points
            pred_points = self._parse_prediction_points(points)
            n_pred = len(pred_points)
            
            if self.kriging_params.use_global:
                # Global kriging prediction
                predictions, variances = self._predict_global(pred_points)
            else:
                # Local kriging prediction
                predictions = np.zeros(n_pred)
                variances = np.zeros(n_pred) if return_variance else None
                
                for i, point in enumerate(pred_points):
                    pred_val, pred_var = self._predict_local(point)
                    predictions[i] = pred_val
                    if return_variance:
                        variances[i] = pred_var
            
            if return_variance:
                return predictions, variances
            else:
                return predictions
                
        except Exception as e:
            raise PredictionError(f"Kriging prediction failed: {e}")
    
    def _predict_global(self, pred_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using global kriging (all training points).
        
        Args:
            pred_points: Points to predict at
            
        Returns:
            Tuple of (predictions, variances)
        """
        n_pred = len(pred_points)
        n_train = len(self.training_points)
        predictions = np.zeros(n_pred)
        variances = np.zeros(n_pred)
        
        for i, pred_point in enumerate(pred_points):
            # Calculate distances from prediction point to all training points
            if len(pred_point) == 2:  # 2D
                distances = np.sqrt(
                    (self.training_points[:, 0] - pred_point[0])**2 + 
                    (self.training_points[:, 1] - pred_point[1])**2
                )
            else:  # 3D
                distances = np.sqrt(
                    (self.training_points[:, 0] - pred_point[0])**2 + 
                    (self.training_points[:, 1] - pred_point[1])**2 +
                    (self.training_points[:, 2] - pred_point[2])**2
                )
            
            # Calculate variogram values
            gamma_vec = self.variogram_function(
                distances, self.fitted_nugget, self.fitted_sill, self.fitted_range
            )
            
            # Setup RHS vector based on kriging type
            if self.kriging_params.kriging_type == KrigingType.ORDINARY:
                rhs = np.zeros(n_train + 1)
                rhs[:n_train] = gamma_vec
                rhs[-1] = 1.0
            elif self.kriging_params.kriging_type == KrigingType.UNIVERSAL:
                # Universal kriging: [γ; f]  
                trend_pred = self._build_trend_matrix(pred_point.reshape(1, -1))
                rhs = np.zeros(n_train + trend_pred.shape[1])
                rhs[:n_train] = gamma_vec
                rhs[n_train:] = trend_pred.flatten()
            else:  # Simple kriging
                rhs = gamma_vec
            
            try:
                # Solve kriging system
                weights = solve(self.kriging_matrix, rhs)
                
                # Calculate prediction
                if self.kriging_params.kriging_type == KrigingType.ORDINARY:
                    lambda_weights = weights[:n_train]
                    predictions[i] = np.sum(lambda_weights * self.training_values)
                    
                    # Calculate kriging variance
                    variances[i] = np.sum(lambda_weights * gamma_vec) + weights[-1]
                    
                elif self.kriging_params.kriging_type == KrigingType.UNIVERSAL:
                    lambda_weights = weights[:n_train]
                    predictions[i] = np.sum(lambda_weights * self.training_values)
                    
                    # Calculate kriging variance (similar to ordinary kriging)
                    trend_pred = self._build_trend_matrix(pred_point.reshape(1, -1))
                    variances[i] = np.sum(lambda_weights * gamma_vec) + np.sum(weights[n_train:] * trend_pred.flatten())
                    
                else:  # Simple kriging
                    mean_val = self.kriging_params.mean_value or self.data_mean
                    residuals = self.training_values - mean_val
                    predictions[i] = mean_val + np.sum(weights * residuals)
                    
                    # Calculate kriging variance
                    variances[i] = np.sum(weights * gamma_vec)
                
            except LinAlgError:
                # Fallback to nearest neighbor
                warnings.warn("Kriging system singular, using nearest neighbor")
                nearest_idx = np.argmin(distances)
                predictions[i] = self.training_values[nearest_idx]
                variances[i] = self.fitted_nugget + self.fitted_sill
        
        return predictions, variances
    
    def _predict_local(self, pred_point: np.ndarray) -> Tuple[float, float]:
        """
        Predict using local kriging (subset of training points).
        
        Args:
            pred_point: Single point to predict at
            
        Returns:
            Tuple of (prediction, variance)
        """
        # Find neighbors
        neighbor_indices, distances = self._find_neighbors(pred_point, self.training_points)
        
        if len(neighbor_indices) == 0:
            # No neighbors found, return mean
            mean_val = self.kriging_params.mean_value or self.data_mean
            return mean_val, self.fitted_sill
        
        # Get neighbor data
        neighbor_points = self.training_points[neighbor_indices]
        neighbor_values = self.training_values[neighbor_indices]
        n_neighbors = len(neighbor_indices)
        
        # Calculate distance matrix among neighbors
        neighbor_distances = squareform(pdist(neighbor_points))
        
        # Calculate variogram matrix
        gamma_matrix = self.variogram_function(
            neighbor_distances, self.fitted_nugget, self.fitted_sill, self.fitted_range
        )
        
        # Calculate distances from prediction point to neighbors
        pred_distances = distances  # Already calculated in _find_neighbors
        gamma_vec = self.variogram_function(
            pred_distances, self.fitted_nugget, self.fitted_sill, self.fitted_range
        )
        
        # Setup local kriging system
        if self.kriging_params.kriging_type == KrigingType.ORDINARY:
            # Ordinary kriging system
            A = np.zeros((n_neighbors + 1, n_neighbors + 1))
            A[:n_neighbors, :n_neighbors] = gamma_matrix
            A[:n_neighbors, -1] = 1.0
            A[-1, :n_neighbors] = 1.0
            
            # Add regularization
            np.fill_diagonal(A[:-1, :-1], np.diag(A[:-1, :-1]) + self.kriging_params.regularization)
            
            rhs = np.zeros(n_neighbors + 1)
            rhs[:n_neighbors] = gamma_vec
            rhs[-1] = 1.0
            
        else:  # Simple kriging
            A = gamma_matrix
            np.fill_diagonal(A, np.diag(A) + self.kriging_params.regularization)
            rhs = gamma_vec
        
        try:
            # Solve system
            weights = solve(A, rhs)
            
            # Calculate prediction and variance
            if self.kriging_params.kriging_type == KrigingType.ORDINARY:
                lambda_weights = weights[:n_neighbors]
                prediction = np.sum(lambda_weights * neighbor_values)
                variance = np.sum(lambda_weights * gamma_vec) + weights[-1]
                
            else:  # Simple kriging
                mean_val = self.kriging_params.mean_value or self.data_mean
                residuals = neighbor_values - mean_val
                prediction = mean_val + np.sum(weights * residuals)
                variance = np.sum(weights * gamma_vec)
            
            return prediction, max(0, variance)  # Ensure non-negative variance
            
        except LinAlgError:
            # Fallback to nearest neighbor
            nearest_idx = np.argmin(pred_distances)
            return neighbor_values[nearest_idx], self.fitted_nugget + self.fitted_sill
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training data and model state."""
        summary = super().get_training_summary()
        
        if self.is_fitted:
            summary.update({
                'fitted_nugget': self.fitted_nugget,
                'fitted_sill': self.fitted_sill,
                'fitted_range': self.fitted_range,
                'kriging_type': self.kriging_params.kriging_type.value,
                'variogram_model': self.kriging_params.variogram_model.value,
                'data_mean': self.data_mean,
                'auto_fitted': self.kriging_params.auto_fit_variogram,
            })
        
        return summary
    
    def get_variogram_values(self, max_distance: Optional[float] = None, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get variogram values for plotting.
        
        Args:
            max_distance: Maximum distance for variogram
            n_points: Number of points to calculate
            
        Returns:
            Tuple of (distances, variogram_values)
        """
        if not self.is_fitted:
            raise PredictionError("Must fit interpolator before getting variogram values")
        
        if max_distance is None:
            # Use maximum distance in training data
            max_distance = np.max(pdist(self.training_points))
        
        distances = np.linspace(0, max_distance, n_points)
        variogram_values = self.variogram_function(
            distances, self.fitted_nugget, self.fitted_sill, self.fitted_range
        )
        
        return distances, variogram_values
    
    def perform_advanced_variogram_analysis(self, 
                                          options: Optional[VariogramAnalysisOptions] = None) -> Dict[str, Any]:
        """
        Perform comprehensive variogram analysis with advanced modeling.
        
        Args:
            options: Analysis options for variogram modeling
            
        Returns:
            Dictionary with complete analysis results
            
        Raises:
            PredictionError: If no training data is available
        """
        if self.training_points is None or self.training_values is None:
            raise PredictionError("Must fit interpolator before variogram analysis")
        
        # Use default options if not provided
        if options is None:
            options = VariogramAnalysisOptions(
                cross_validate=True,
                detect_anisotropy=True,
                create_plots=False  # Don't create plots by default
            )
        
        # Create variogram analyzer
        self.variogram_analyzer = VariogramAnalyzer(options)
        
        # Perform analysis
        self.variogram_analysis_results = self.variogram_analyzer.analyze_variogram(
            self.training_points, self.training_values
        )
        
        # Update fitted parameters if better model was found
        if options.fit_multiple_models and 'best_model' in self.variogram_analysis_results:
            best_model = self.variogram_analysis_results['best_model']
            if best_model and best_model.r_squared > 0.7:  # Only use if reasonably good fit
                self.fitted_nugget = best_model.nugget
                self.fitted_sill = best_model.sill
                self.fitted_range = best_model.range_param
                
                # Update variogram model type
                self.kriging_params.variogram_model = best_model.model_type
                self.variogram_function = self._get_variogram_function()
                
                # Recompute kriging matrix if using global kriging
                if self.kriging_params.use_global:
                    self._setup_global_kriging()
        
        return self.variogram_analysis_results
    
    def get_variogram_analysis_results(self) -> Optional[Dict[str, Any]]:
        """
        Get results from advanced variogram analysis.
        
        Returns:
            Analysis results dictionary or None if analysis not performed
        """
        return self.variogram_analysis_results
    
    def optimize_variogram_automatically(self) -> bool:
        """
        Automatically optimize variogram model using advanced analysis.
        
        This method performs comprehensive variogram analysis and selects
        the best model based on multiple criteria including cross-validation.
        
        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Perform advanced analysis with comprehensive options
            options = VariogramAnalysisOptions(
                fit_multiple_models=True,
                cross_validate=True,
                detect_anisotropy=True,
                create_plots=False
            )
            
            results = self.perform_advanced_variogram_analysis(options)
            
            # Check if we got valid results
            if 'best_model' in results and results['best_model']:
                best_model = results['best_model']
                
                # Update our parameters
                self.fitted_nugget = best_model.nugget
                self.fitted_sill = best_model.sill
                self.fitted_range = best_model.range_param
                
                # Update model type
                self.kriging_params.variogram_model = best_model.model_type
                self.variogram_function = self._get_variogram_function()
                
                # Recompute kriging matrix if using global kriging
                if self.kriging_params.use_global:
                    self._setup_global_kriging()
                
                return True
            
            return False
            
        except Exception as e:
            warnings.warn(f"Automatic variogram optimization failed: {e}")
            return False
    
    def get_experimental_variogram(self, 
                                 direction: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Calculate and return experimental variogram data.
        
        Args:
            direction: Direction for directional variogram (degrees)
            
        Returns:
            Dictionary with experimental variogram data
        """
        if self.training_points is None or self.training_values is None:
            raise PredictionError("Must fit interpolator before calculating experimental variogram")
        
        # Create temporary analyzer for experimental variogram calculation
        analyzer = VariogramAnalyzer()
        exp_variogram = analyzer.calculate_experimental_variogram(
            self.training_points, self.training_values, direction
        )
        
        return {
            'distances': exp_variogram.distances,
            'semivariances': exp_variogram.semivariances,
            'n_pairs': exp_variogram.n_pairs,
            'direction': exp_variogram.direction
        }
    
    def detect_anisotropy(self) -> Optional[Dict[str, Any]]:
        """
        Detect spatial anisotropy in the data.
        
        Returns:
            Dictionary with anisotropy analysis results or None if analysis fails
        """
        try:
            # Perform basic directional analysis
            options = VariogramAnalysisOptions(
                n_directions=4,
                detect_anisotropy=True,
                fit_multiple_models=False,
                create_plots=False
            )
            
            analyzer = VariogramAnalyzer(options)
            
            # Calculate directional variograms
            directional_variograms = analyzer.calculate_directional_variograms(
                self.training_points, self.training_values
            )
            analyzer.experimental_variograms.update(directional_variograms)
            
            # Detect anisotropy
            anisotropy_result = analyzer.detect_anisotropy()
            
            if anisotropy_result:
                return {
                    'is_anisotropic': anisotropy_result.is_anisotropic,
                    'major_direction': anisotropy_result.major_direction,
                    'minor_direction': anisotropy_result.minor_direction,
                    'anisotropy_ratio': anisotropy_result.anisotropy_ratio,
                    'major_range': anisotropy_result.major_range,
                    'minor_range': anisotropy_result.minor_range
                }
            
            return None
            
        except Exception as e:
            warnings.warn(f"Anisotropy detection failed: {e}")
            return None
    
    def _build_trend_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Build trend matrix for universal kriging.
        
        This method builds a matrix of trend functions (basis functions) for 
        universal kriging. The default implementation uses polynomial trends up to 
        linear terms: [1, x, y] for 2D and [1, x, y, z] for 3D.
        
        Args:
            points: Coordinate points array (N x 2 for 2D, N x 3 for 3D)
            
        Returns:
            Trend matrix (N x n_trend_functions)
        """
        n_points, n_dims = points.shape
        
        if n_dims == 2:
            # 2D trend functions: [1, x, y]
            trend_matrix = np.column_stack([
                np.ones(n_points),      # Constant
                points[:, 0],           # Linear in x
                points[:, 1]            # Linear in y
            ])
        elif n_dims == 3:
            # 3D trend functions: [1, x, y, z]
            trend_matrix = np.column_stack([
                np.ones(n_points),      # Constant
                points[:, 0],           # Linear in x
                points[:, 1],           # Linear in y
                points[:, 2]            # Linear in z
            ])
        else:
            raise ValueError(f"Unsupported dimensionality: {n_dims}")
        
        return trend_matrix