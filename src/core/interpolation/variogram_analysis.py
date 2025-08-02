"""
Advanced variogram analysis tools for kriging interpolation.

This module provides comprehensive tools for variogram modeling including:
- Experimental variogram calculation with directional analysis
- Automatic model fitting with multiple model types
- Cross-validation and model selection
- Interactive variogram visualization and fitting
- Anisotropy detection and modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Moved to avoid circular import - these are defined below
# from .kriging import VariogramModel, VariogramModels

# Local copy of VariogramModel to avoid circular import
from enum import Enum

class VariogramModel(Enum):
    """Available variogram model types."""
    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"
    POWER = "power"
    NUGGET = "nugget"


class VariogramModels:
    """Collection of variogram model functions."""
    
    @staticmethod
    def spherical(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """Spherical variogram model."""
        gamma = np.full_like(h, nugget + sill, dtype=float)
        mask = (h > 0) & (h < range_param)
        h_normalized = h[mask] / range_param
        gamma[mask] = nugget + sill * (1.5 * h_normalized - 0.5 * h_normalized**3)
        gamma[h == 0] = 0.0
        return gamma
    
    @staticmethod
    def exponential(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """Exponential variogram model."""
        gamma = nugget + sill * (1 - np.exp(-h / range_param))
        gamma[h == 0] = 0.0
        return gamma
    
    @staticmethod
    def gaussian(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """Gaussian variogram model."""
        gamma = nugget + sill * (1 - np.exp(-(h / range_param)**2))
        gamma[h == 0] = 0.0
        return gamma
    
    @staticmethod
    def linear(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """Linear variogram model."""
        gamma = nugget + sill * (h / range_param)
        gamma[h == 0] = 0.0
        return gamma
    
    @staticmethod
    def power(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """Power variogram model."""
        gamma = nugget + sill * (h / range_param)**0.5
        gamma[h == 0] = 0.0
        return gamma
    
    @staticmethod
    def nugget(h: np.ndarray, nugget: float, sill: float, range_param: float) -> np.ndarray:
        """Pure nugget model."""
        gamma = np.full_like(h, nugget, dtype=float)
        gamma[h == 0] = 0.0
        return gamma


@dataclass
class VariogramAnalysisOptions:
    """Options for variogram analysis."""
    max_distance: Optional[float] = None      # Maximum distance for analysis
    n_lags: int = 15                          # Number of lag classes
    lag_tolerance: float = 0.5                # Tolerance for lag binning
    
    # Directional analysis
    n_directions: int = 4                     # Number of directions
    direction_tolerance: float = 22.5         # Tolerance in degrees
    
    # Anisotropy analysis
    detect_anisotropy: bool = True            # Whether to detect anisotropy
    anisotropy_angles: List[float] = None     # Angles to test (degrees)
    
    # Model fitting
    fit_multiple_models: bool = True          # Fit multiple models and select best
    use_weighted_fitting: bool = True         # Use weighted least squares
    cross_validate: bool = True               # Use cross-validation for model selection
    cv_folds: int = 5                         # Number of CV folds
    
    # Visualization
    create_plots: bool = True                 # Create visualization plots
    plot_residuals: bool = True               # Plot fitting residuals
    
    def __post_init__(self):
        """Initialize default values."""
        if self.anisotropy_angles is None:
            self.anisotropy_angles = [0, 45, 90, 135]


@dataclass
class ExperimentalVariogram:
    """Experimental variogram data."""
    distances: np.ndarray                     # Lag distances
    semivariances: np.ndarray                 # Experimental semivariances
    n_pairs: np.ndarray                       # Number of pairs per lag
    direction: Optional[float] = None         # Direction in degrees (for directional variograms)
    direction_tolerance: Optional[float] = None


@dataclass
class VariogramModelResult:
    """Result of variogram model fitting."""
    model_type: VariogramModel                # Model type
    nugget: float                             # Fitted nugget
    sill: float                               # Fitted sill
    range_param: float                        # Fitted range
    r_squared: float                          # Goodness of fit
    rmse: float                               # Root mean square error
    aic: float                                # Akaike Information Criterion
    cv_score: Optional[float] = None          # Cross-validation score
    
    @property
    def total_sill(self) -> float:
        """Total sill (nugget + sill)."""
        return self.nugget + self.sill


@dataclass
class AnisotropyResult:
    """Result of anisotropy analysis."""
    is_anisotropic: bool                      # Whether anisotropy detected
    major_direction: Optional[float] = None   # Major axis direction (degrees)
    minor_direction: Optional[float] = None   # Minor axis direction (degrees)
    anisotropy_ratio: Optional[float] = None  # Ratio of major to minor range
    major_range: Optional[float] = None       # Range in major direction
    minor_range: Optional[float] = None       # Range in minor direction


class VariogramAnalyzer:
    """
    Advanced variogram analysis and modeling.
    
    This class provides comprehensive tools for variogram analysis including
    experimental variogram calculation, automatic model fitting, anisotropy
    detection, and model validation.
    """
    
    def __init__(self, options: Optional[VariogramAnalysisOptions] = None):
        """
        Initialize variogram analyzer.
        
        Args:
            options: Analysis options
        """
        self.options = options or VariogramAnalysisOptions()
        self.experimental_variograms: Dict[str, ExperimentalVariogram] = {}
        self.fitted_models: Dict[str, List[VariogramModelResult]] = {}
        self.best_model: Optional[VariogramModelResult] = None
        self.anisotropy_result: Optional[AnisotropyResult] = None
        
    def analyze_variogram(self, 
                         coordinates: np.ndarray, 
                         values: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive variogram analysis.
        
        Args:
            coordinates: Point coordinates (N x 2 or N x 3)
            values: Values at points
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Calculate experimental variogram
        self.experimental_variograms['omnidirectional'] = self.calculate_experimental_variogram(
            coordinates, values
        )
        results['experimental_variogram'] = self.experimental_variograms['omnidirectional']
        
        # Directional analysis
        if self.options.n_directions > 1:
            directional_variograms = self.calculate_directional_variograms(
                coordinates, values
            )
            self.experimental_variograms.update(directional_variograms)
            results['directional_variograms'] = directional_variograms
        
        # Anisotropy detection
        if self.options.detect_anisotropy and self.options.n_directions > 1:
            self.anisotropy_result = self.detect_anisotropy()
            results['anisotropy'] = self.anisotropy_result
        
        # Model fitting
        if self.options.fit_multiple_models:
            models = self.fit_all_models('omnidirectional')
            self.fitted_models['omnidirectional'] = models
            self.best_model = self.select_best_model(models)
            results['fitted_models'] = models
            results['best_model'] = self.best_model
        
        # Cross-validation
        if self.options.cross_validate and self.best_model:
            cv_results = self.cross_validate_model(coordinates, values, self.best_model)
            results['cross_validation'] = cv_results
        
        # Create plots
        if self.options.create_plots:
            plots = self.create_variogram_plots()
            results['plots'] = plots
            
        return results
    
    def calculate_experimental_variogram(self, 
                                       coordinates: np.ndarray, 
                                       values: np.ndarray,
                                       direction: Optional[float] = None,
                                       direction_tolerance: Optional[float] = None) -> ExperimentalVariogram:
        """
        Calculate experimental variogram.
        
        Args:
            coordinates: Point coordinates
            values: Values at points
            direction: Direction in degrees (None for omnidirectional)
            direction_tolerance: Angular tolerance in degrees
            
        Returns:
            Experimental variogram
        """
        n_points = len(coordinates)
        
        # Calculate all pairwise distances and differences
        distances = pdist(coordinates)
        value_differences = pdist(values.reshape(-1, 1))
        semivariances = 0.5 * value_differences**2
        
        # Handle directional variogram
        if direction is not None:
            # Calculate angles between points
            angles = self._calculate_pairwise_angles(coordinates)
            direction_tolerance = direction_tolerance or self.options.direction_tolerance
            
            # Filter by direction
            angle_mask = self._angle_in_direction(angles, direction, direction_tolerance)
            distances = distances[angle_mask]
            semivariances = semivariances[angle_mask]
        
        # Determine lag parameters
        max_distance = self.options.max_distance
        if max_distance is None:
            max_distance = np.percentile(distances, 75)  # Use 75th percentile
            
        lag_size = max_distance / self.options.n_lags
        
        # Create lag bins
        lag_centers = []
        lag_semivariances = []
        lag_n_pairs = []
        
        for i in range(self.options.n_lags):
            lag_min = i * lag_size
            lag_max = (i + 1) * lag_size
            lag_center = (lag_min + lag_max) / 2
            
            # Find pairs in this lag
            tolerance = lag_size * self.options.lag_tolerance
            mask = (distances >= lag_min - tolerance) & (distances < lag_max + tolerance)
            
            if np.sum(mask) > 0:
                lag_centers.append(lag_center)
                lag_semivariances.append(np.mean(semivariances[mask]))
                lag_n_pairs.append(np.sum(mask))
        
        return ExperimentalVariogram(
            distances=np.array(lag_centers),
            semivariances=np.array(lag_semivariances),
            n_pairs=np.array(lag_n_pairs),
            direction=direction,
            direction_tolerance=direction_tolerance
        )
    
    def calculate_directional_variograms(self, 
                                       coordinates: np.ndarray, 
                                       values: np.ndarray) -> Dict[str, ExperimentalVariogram]:
        """
        Calculate directional variograms.
        
        Args:
            coordinates: Point coordinates
            values: Values at points
            
        Returns:
            Dictionary of directional variograms
        """
        directional_variograms = {}
        angles = np.linspace(0, 180, self.options.n_directions, endpoint=False)
        
        for angle in angles:
            key = f"direction_{angle:.0f}"
            directional_variograms[key] = self.calculate_experimental_variogram(
                coordinates, values, direction=angle
            )
            
        return directional_variograms
    
    def fit_variogram_model(self, 
                          experimental_variogram: ExperimentalVariogram,
                          model_type: VariogramModel,
                          initial_params: Optional[Dict[str, float]] = None) -> VariogramModelResult:
        """
        Fit a specific variogram model to experimental data.
        
        Args:
            experimental_variogram: Experimental variogram data
            model_type: Type of variogram model to fit
            initial_params: Initial parameter estimates
            
        Returns:
            Fitted model result
        """
        distances = experimental_variogram.distances
        semivariances = experimental_variogram.semivariances
        n_pairs = experimental_variogram.n_pairs
        
        # Get model function
        model_func = self._get_model_function(model_type)
        
        # Initial parameter estimates
        if initial_params is None:
            initial_params = self._estimate_initial_parameters(
                distances, semivariances, model_type
            )
        
        nugget_init = initial_params.get('nugget', 0.0)
        sill_init = initial_params.get('sill', np.var(semivariances))
        range_init = initial_params.get('range', np.max(distances) / 3)
        
        # Parameter bounds
        max_semivar = np.max(semivariances)
        max_distance = np.max(distances)
        
        bounds = [
            (0, max_semivar),                    # nugget
            (1e-10, max_semivar * 2),           # sill
            (max_distance / 100, max_distance * 2)  # range
        ]
        
        # Weights for fitting (more weight to lags with more pairs)
        if self.options.use_weighted_fitting:
            weights = np.sqrt(n_pairs) / np.sum(n_pairs)
        else:
            weights = np.ones(len(distances))
        
        # Objective function
        def objective(params):
            nugget, sill, range_param = params
            try:
                model_values = model_func(distances, nugget, sill, range_param)
                residuals = semivariances - model_values
                return np.sum(weights * residuals**2)
            except:
                return 1e10
        
        # Optimize parameters
        try:
            result = minimize(
                objective, 
                x0=[nugget_init, sill_init, range_init],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                fitted_nugget, fitted_sill, fitted_range = result.x
            else:
                # Try differential evolution as fallback
                result = differential_evolution(
                    objective, bounds, seed=42, maxiter=100
                )
                fitted_nugget, fitted_sill, fitted_range = result.x
                
        except Exception as e:
            warnings.warn(f"Model fitting failed: {e}")
            fitted_nugget, fitted_sill, fitted_range = nugget_init, sill_init, range_init
        
        # Calculate goodness of fit metrics
        model_values = model_func(distances, fitted_nugget, fitted_sill, fitted_range)
        
        # R-squared
        ss_res = np.sum((semivariances - model_values)**2)
        ss_tot = np.sum((semivariances - np.mean(semivariances))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # RMSE
        rmse = np.sqrt(np.mean((semivariances - model_values)**2))
        
        # AIC (Akaike Information Criterion)
        n = len(distances)
        aic = n * np.log(ss_res / n) + 2 * 3  # 3 parameters
        
        return VariogramModelResult(
            model_type=model_type,
            nugget=fitted_nugget,
            sill=fitted_sill,
            range_param=fitted_range,
            r_squared=r_squared,
            rmse=rmse,
            aic=aic
        )
    
    def fit_all_models(self, variogram_key: str = 'omnidirectional') -> List[VariogramModelResult]:
        """
        Fit all available variogram models.
        
        Args:
            variogram_key: Key of experimental variogram to fit
            
        Returns:
            List of fitted model results
        """
        if variogram_key not in self.experimental_variograms:
            raise ValueError(f"No experimental variogram found for key: {variogram_key}")
        
        experimental_variogram = self.experimental_variograms[variogram_key]
        models_to_fit = [
            VariogramModel.SPHERICAL,
            VariogramModel.EXPONENTIAL,
            VariogramModel.GAUSSIAN,
            VariogramModel.LINEAR
        ]
        
        fitted_models = []
        for model_type in models_to_fit:
            try:
                fitted_model = self.fit_variogram_model(experimental_variogram, model_type)
                fitted_models.append(fitted_model)
            except Exception as e:
                warnings.warn(f"Failed to fit {model_type.value} model: {e}")
                continue
        
        # Sort by AIC (lower is better)
        fitted_models.sort(key=lambda x: x.aic)
        
        return fitted_models
    
    def select_best_model(self, fitted_models: List[VariogramModelResult]) -> Optional[VariogramModelResult]:
        """
        Select the best model based on multiple criteria.
        
        Args:
            fitted_models: List of fitted models
            
        Returns:
            Best model or None if no valid models
        """
        if not fitted_models:
            return None
        
        # Filter models with reasonable fit
        valid_models = [m for m in fitted_models if m.r_squared > 0.5 and m.rmse < np.inf]
        
        if not valid_models:
            # Return best by AIC if no models meet criteria
            return fitted_models[0] if fitted_models else None
        
        # Score models using multiple criteria
        scores = []
        for model in valid_models:
            # Normalize metrics (0-1 scale, higher is better)
            r2_score = model.r_squared
            aic_score = 1 / (1 + model.aic / min(m.aic for m in valid_models))
            rmse_score = 1 / (1 + model.rmse / min(m.rmse for m in valid_models))
            
            # Combined score (equal weights)
            combined_score = (r2_score + aic_score + rmse_score) / 3
            scores.append(combined_score)
        
        # Return model with highest score
        best_idx = np.argmax(scores)
        return valid_models[best_idx]
    
    def detect_anisotropy(self) -> AnisotropyResult:
        """
        Detect anisotropy in directional variograms.
        
        Returns:
            Anisotropy analysis result
        """
        if len(self.experimental_variograms) < 2:
            return AnisotropyResult(is_anisotropic=False)
        
        # Get directional variograms
        directional_variograms = {
            k: v for k, v in self.experimental_variograms.items() 
            if k.startswith('direction_')
        }
        
        if len(directional_variograms) < 2:
            return AnisotropyResult(is_anisotropic=False)
        
        # Fit simple spherical model to each direction
        ranges = {}
        for direction, variogram in directional_variograms.items():
            try:
                fitted_model = self.fit_variogram_model(variogram, VariogramModel.SPHERICAL)
                ranges[direction] = fitted_model.range_param
            except:
                continue
        
        if len(ranges) < 2:
            return AnisotropyResult(is_anisotropic=False)
        
        # Find major and minor directions
        range_values = list(ranges.values())
        directions = [float(k.split('_')[1]) for k in ranges.keys()]
        
        max_range_idx = np.argmax(range_values)
        min_range_idx = np.argmin(range_values)
        
        major_range = range_values[max_range_idx]
        minor_range = range_values[min_range_idx]
        major_direction = directions[max_range_idx]
        minor_direction = directions[min_range_idx]
        
        # Calculate anisotropy ratio
        anisotropy_ratio = major_range / minor_range if minor_range > 0 else np.inf
        
        # Consider anisotropic if ratio > 1.5
        is_anisotropic = anisotropy_ratio > 1.5
        
        return AnisotropyResult(
            is_anisotropic=is_anisotropic,
            major_direction=major_direction,
            minor_direction=minor_direction,
            anisotropy_ratio=anisotropy_ratio,
            major_range=major_range,
            minor_range=minor_range
        )
    
    def cross_validate_model(self, 
                           coordinates: np.ndarray, 
                           values: np.ndarray,
                           model: VariogramModelResult,
                           n_folds: Optional[int] = None) -> Dict[str, float]:
        """
        Cross-validate variogram model using kriging predictions.
        
        Args:
            coordinates: Point coordinates
            values: Values at points
            model: Fitted variogram model
            n_folds: Number of CV folds
            
        Returns:
            Cross-validation metrics
        """
        from .kriging import KrigingInterpolator, KrigingParameters, KrigingType
        
        n_folds = n_folds or self.options.cv_folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_errors = []
        
        for train_idx, test_idx in kf.split(coordinates):
            # Split data
            train_coords = coordinates[train_idx]
            train_values = values[train_idx]
            test_coords = coordinates[test_idx]
            test_values = values[test_idx]
            
            # Create DataFrames for kriging
            train_df = pd.DataFrame({
                'X': train_coords[:, 0],
                'Y': train_coords[:, 1],
                'Value': train_values
            })
            
            # Setup kriging with fitted model parameters
            kriging_params = KrigingParameters(
                kriging_type=KrigingType.ORDINARY,
                variogram_model=model.model_type,
                nugget=model.nugget,
                sill=model.sill,
                range_param=model.range_param,
                auto_fit_variogram=False  # Use provided parameters
            )
            
            # Fit and predict
            try:
                kriging = KrigingInterpolator(kriging_params=kriging_params)
                kriging.fit(train_df, 'X', 'Y', 'Value')
                predictions = kriging.predict(test_coords)
                
                # Calculate errors
                errors = test_values - predictions
                cv_errors.extend(errors)
                
            except Exception as e:
                warnings.warn(f"CV fold failed: {e}")
                continue
        
        if not cv_errors:
            return {'cv_rmse': np.inf, 'cv_mae': np.inf, 'cv_r2': 0}
        
        cv_errors = np.array(cv_errors)
        
        return {
            'cv_rmse': np.sqrt(np.mean(cv_errors**2)),
            'cv_mae': np.mean(np.abs(cv_errors)),
            'cv_r2': 1 - np.var(cv_errors) / np.var(values) if np.var(values) > 0 else 0
        }
    
    def create_variogram_plots(self) -> Dict[str, Any]:
        """
        Create variogram visualization plots.
        
        Returns:
            Dictionary with plot information
        """
        plots = {}
        
        try:
            # Omnidirectional variogram plot
            if 'omnidirectional' in self.experimental_variograms:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                exp_var = self.experimental_variograms['omnidirectional']
                ax.scatter(exp_var.distances, exp_var.semivariances, 
                          s=exp_var.n_pairs/10, alpha=0.7, label='Experimental')
                
                # Plot fitted models
                if 'omnidirectional' in self.fitted_models:
                    max_distance = np.max(exp_var.distances)
                    h_range = np.linspace(0, max_distance, 100)
                    
                    for i, model in enumerate(self.fitted_models['omnidirectional'][:3]):  # Top 3 models
                        model_func = self._get_model_function(model.model_type)
                        gamma_values = model_func(h_range, model.nugget, model.sill, model.range_param)
                        
                        label = f"{model.model_type.value.title()} (R²={model.r_squared:.3f})"
                        linestyle = '-' if i == 0 else '--'
                        ax.plot(h_range, gamma_values, linestyle, label=label)
                
                ax.set_xlabel('Distance')
                ax.set_ylabel('Semivariance')
                ax.set_title('Variogram Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plots['omnidirectional'] = fig
            
            # Directional variograms plot
            directional_vars = {k: v for k, v in self.experimental_variograms.items() 
                              if k.startswith('direction_')}
            
            if len(directional_vars) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                colors = plt.cm.Set1(np.linspace(0, 1, len(directional_vars)))
                
                for (direction, exp_var), color in zip(directional_vars.items(), colors):
                    angle = direction.split('_')[1]
                    ax.scatter(exp_var.distances, exp_var.semivariances, 
                             color=color, alpha=0.7, label=f'{angle}°')
                
                ax.set_xlabel('Distance') 
                ax.set_ylabel('Semivariance')
                ax.set_title('Directional Variograms')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plots['directional'] = fig
            
            # Anisotropy rose plot
            if self.anisotropy_result and self.anisotropy_result.is_anisotropic:
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                
                # Plot anisotropy ellipse
                angles = np.linspace(0, 2*np.pi, 100)
                major_angle = np.radians(self.anisotropy_result.major_direction)
                
                # Simple ellipse representation
                ranges = []
                ratio = self.anisotropy_result.anisotropy_ratio
                for angle in angles:
                    # Simplified anisotropy calculation
                    range_val = self.anisotropy_result.minor_range * np.sqrt(
                        (np.cos(angle - major_angle)**2) + 
                        (ratio * np.sin(angle - major_angle)**2)
                    )
                    ranges.append(range_val)
                
                ax.plot(angles, ranges, 'b-', linewidth=2)
                ax.fill(angles, ranges, alpha=0.3)
                ax.set_title('Anisotropy Rose\n' + 
                           f'Ratio: {ratio:.2f}, Major: {self.anisotropy_result.major_direction:.0f}°')
                
                plots['anisotropy'] = fig
                
        except Exception as e:
            warnings.warn(f"Plot creation failed: {e}")
            
        return plots
    
    # Helper methods
    
    def _calculate_pairwise_angles(self, coordinates: np.ndarray) -> np.ndarray:
        """Calculate angles between all point pairs."""
        n_points = len(coordinates)
        angles = []
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dx = coordinates[j, 0] - coordinates[i, 0]
                dy = coordinates[j, 1] - coordinates[i, 1]
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < 0:
                    angle += 180  # Convert to 0-180 range
                angles.append(angle)
        
        return np.array(angles)
    
    def _angle_in_direction(self, angles: np.ndarray, direction: float, tolerance: float) -> np.ndarray:
        """Check if angles are within tolerance of target direction."""
        # Handle wrapping around 180 degrees
        diff1 = np.abs(angles - direction)
        diff2 = np.abs(angles - (direction + 180))
        diff3 = np.abs(angles - (direction - 180))
        
        min_diff = np.minimum(np.minimum(diff1, diff2), diff3)
        return min_diff <= tolerance
    
    def _get_model_function(self, model_type: VariogramModel) -> Callable:
        """Get variogram model function."""
        model_map = {
            VariogramModel.SPHERICAL: VariogramModels.spherical,
            VariogramModel.EXPONENTIAL: VariogramModels.exponential,
            VariogramModel.GAUSSIAN: VariogramModels.gaussian,
            VariogramModel.LINEAR: VariogramModels.linear,
            VariogramModel.POWER: VariogramModels.power,
            VariogramModel.NUGGET: VariogramModels.nugget,
        }
        return model_map[model_type]
    
    def _estimate_initial_parameters(self, 
                                   distances: np.ndarray, 
                                   semivariances: np.ndarray,
                                   model_type: VariogramModel) -> Dict[str, float]:
        """Estimate initial parameters for model fitting."""
        max_semivar = np.max(semivariances)
        max_distance = np.max(distances)
        
        # Basic estimates
        nugget = np.min(semivariances) if len(semivariances) > 0 else 0
        sill = max_semivar - nugget
        range_param = max_distance / 3
        
        # Model-specific adjustments
        if model_type == VariogramModel.LINEAR:
            range_param = 1.0  # Not used for linear model
        elif model_type == VariogramModel.EXPONENTIAL:
            range_param = max_distance / 5  # Exponential reaches sill more slowly
        elif model_type == VariogramModel.GAUSSIAN:
            range_param = max_distance / 4  # Gaussian reaches sill very quickly
        
        return {
            'nugget': max(0, nugget),
            'sill': max(1e-10, sill),
            'range': max(max_distance / 100, range_param)
        }


def analyze_spatial_structure(coordinates: np.ndarray, 
                            values: np.ndarray,
                            options: Optional[VariogramAnalysisOptions] = None) -> Dict[str, Any]:
    """
    Convenience function for comprehensive spatial structure analysis.
    
    Args:
        coordinates: Point coordinates (N x 2 or N x 3)
        values: Values at points
        options: Analysis options
        
    Returns:
        Complete analysis results
    """
    analyzer = VariogramAnalyzer(options)
    return analyzer.analyze_variogram(coordinates, values)


def quick_variogram_fit(coordinates: np.ndarray, 
                       values: np.ndarray,
                       model_type: Optional[VariogramModel] = None) -> VariogramModelResult:
    """
    Quick variogram model fitting for immediate use.
    
    Args:
        coordinates: Point coordinates
        values: Values at points
        model_type: Specific model type (None for automatic selection)
        
    Returns:
        Best fitted model
    """
    options = VariogramAnalysisOptions(
        fit_multiple_models=model_type is None,
        create_plots=False,
        cross_validate=False
    )
    
    analyzer = VariogramAnalyzer(options)
    exp_var = analyzer.calculate_experimental_variogram(coordinates, values)
    analyzer.experimental_variograms['omnidirectional'] = exp_var
    
    if model_type:
        return analyzer.fit_variogram_model(exp_var, model_type)
    else:
        models = analyzer.fit_all_models()
        return analyzer.select_best_model(models)