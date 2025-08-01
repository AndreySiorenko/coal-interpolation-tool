"""
RBF (Radial Basis Function) interpolation implementation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Union, List, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy.linalg import solve, LinAlgError, svd
from scipy.spatial.distance import cdist

from .base import BaseInterpolator, InterpolationParameters, SearchParameters
from .base import FittingError, PredictionError


class RBFKernel(Enum):
    """Available RBF kernel types."""
    GAUSSIAN = "gaussian"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    THIN_PLATE_SPLINE = "thin_plate_spline"
    LINEAR = "linear"
    CUBIC = "cubic"
    QUINTIC = "quintic"


@dataclass
class RBFParameters(InterpolationParameters):
    """Parameters specific to RBF interpolation."""
    kernel: RBFKernel = RBFKernel.MULTIQUADRIC  # RBF kernel type
    shape_parameter: float = 1.0                # Shape parameter (epsilon)
    regularization: float = 1e-12               # Regularization parameter
    polynomial_degree: int = -1                 # Polynomial degree (-1 for none, 0 for constant, 1 for linear)
    use_global: bool = True                     # Use global RBF (True) or local (False)
    condition_number_threshold: float = 1e12    # Threshold for ill-conditioned matrices


class RBFKernels:
    """Collection of RBF kernel functions."""
    
    @staticmethod
    def gaussian(r: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Gaussian RBF: exp(-(ε*r)²)
        
        Args:
            r: Distance array
            epsilon: Shape parameter
            
        Returns:
            Kernel values
        """
        return np.exp(-(epsilon * r) ** 2)
    
    @staticmethod
    def multiquadric(r: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Multiquadric RBF: √(1 + (ε*r)²)
        
        Args:
            r: Distance array
            epsilon: Shape parameter
            
        Returns:
            Kernel values
        """
        return np.sqrt(1 + (epsilon * r) ** 2)
    
    @staticmethod
    def inverse_multiquadric(r: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Inverse Multiquadric RBF: 1/√(1 + (ε*r)²)
        
        Args:
            r: Distance array
            epsilon: Shape parameter
            
        Returns:
            Kernel values
        """
        return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)
    
    @staticmethod
    def thin_plate_spline(r: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """
        Thin Plate Spline RBF: r² * ln(r)
        
        Args:
            r: Distance array
            epsilon: Not used, kept for interface consistency
            
        Returns:
            Kernel values
        """
        # Handle r = 0 case
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = r ** 2 * np.log(r + 1e-15)  # Add small value to avoid log(0)
        
        # Set exact zeros to zero
        result[r == 0] = 0
        return result
    
    @staticmethod
    def linear(r: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """
        Linear RBF: r
        
        Args:
            r: Distance array
            epsilon: Not used, kept for interface consistency
            
        Returns:
            Kernel values
        """
        return r
    
    @staticmethod
    def cubic(r: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """
        Cubic RBF: r³
        
        Args:
            r: Distance array
            epsilon: Not used, kept for interface consistency
            
        Returns:
            Kernel values
        """
        return r ** 3
    
    @staticmethod
    def quintic(r: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """
        Quintic RBF: r⁵
        
        Args:
            r: Distance array
            epsilon: Not used, kept for interface consistency
            
        Returns:
            Kernel values
        """
        return r ** 5


class RBFInterpolator(BaseInterpolator):
    """
    Radial Basis Function (RBF) interpolation.
    
    RBF interpolation constructs a function of the form:
        f(x) = Σ λᵢ φ(||x - xᵢ||) + P(x)
        
    where φ is the radial basis function, λᵢ are weights, and P(x) is an
    optional polynomial term.
    
    The method provides exact interpolation at data points and creates
    smooth surfaces between them. Different RBF kernels provide different
    characteristics in terms of smoothness and locality.
    
    Parameters:
        search_params: Search parameters for finding neighbors (used in local mode)
        rbf_params: RBF-specific parameters including kernel type and shape parameter
    """
    
    def __init__(self, 
                 search_params: Optional[SearchParameters] = None,
                 rbf_params: Optional[RBFParameters] = None):
        """Initialize RBF interpolator with given parameters."""
        super().__init__(search_params)
        self.rbf_params = rbf_params or RBFParameters()
        
        # Training data storage
        self.training_points = None
        self.training_values = None
        self.weights = None
        self.polynomial_weights = None
        
        # Kernel function
        self._kernel_func = self._get_kernel_function()
        
        # Condition number tracking
        self.condition_number = None
        
    def get_method_name(self) -> str:
        """Return human-readable name of the interpolation method."""
        return f"Radial Basis Function ({self.rbf_params.kernel.value.replace('_', ' ').title()})"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters of the interpolator."""
        params = {
            'kernel': self.rbf_params.kernel.value,
            'shape_parameter': self.rbf_params.shape_parameter,
            'regularization': self.rbf_params.regularization,
            'polynomial_degree': self.rbf_params.polynomial_degree,
            'use_global': self.rbf_params.use_global,
            'condition_number_threshold': self.rbf_params.condition_number_threshold
        }
        
        # Add search parameters if using local RBF
        if not self.rbf_params.use_global:
            params.update({
                'search_radius': self.search_params.search_radius,
                'min_points': self.search_params.min_points,
                'max_points': self.search_params.max_points,
            })
        
        return params
    
    def set_parameters(self, **params) -> 'RBFInterpolator':
        """Set interpolator parameters."""
        # Handle RBF-specific parameters
        rbf_param_names = ['kernel', 'shape_parameter', 'regularization', 
                          'polynomial_degree', 'use_global', 'condition_number_threshold']
        
        for key in rbf_param_names:
            if key in params:
                if key == 'kernel':
                    # Handle kernel parameter
                    kernel_value = params[key]
                    if isinstance(kernel_value, str):
                        try:
                            kernel_enum = RBFKernel(kernel_value.lower())
                            setattr(self.rbf_params, key, kernel_enum)
                        except ValueError:
                            raise ValueError(f"Invalid kernel: {kernel_value}")
                    elif isinstance(kernel_value, RBFKernel):
                        setattr(self.rbf_params, key, kernel_value)
                else:
                    setattr(self.rbf_params, key, params[key])
        
        # Update kernel function
        self._kernel_func = self._get_kernel_function()
        
        # Let parent handle search parameters
        super().set_parameters(**params)
        
        return self
    
    def _get_kernel_function(self) -> Callable:
        """Get the kernel function based on current parameters."""
        kernel_map = {
            RBFKernel.GAUSSIAN: RBFKernels.gaussian,
            RBFKernel.MULTIQUADRIC: RBFKernels.multiquadric,
            RBFKernel.INVERSE_MULTIQUADRIC: RBFKernels.inverse_multiquadric,
            RBFKernel.THIN_PLATE_SPLINE: RBFKernels.thin_plate_spline,
            RBFKernel.LINEAR: RBFKernels.linear,
            RBFKernel.CUBIC: RBFKernels.cubic,
            RBFKernel.QUINTIC: RBFKernels.quintic,
        }
        
        return kernel_map[self.rbf_params.kernel]
    
    def fit(self, 
            data: pd.DataFrame,
            x_col: str,
            y_col: str,
            value_col: str,
            z_col: Optional[str] = None,
            **kwargs) -> 'RBFInterpolator':
        """
        Fit the RBF interpolator to training data.
        
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
            
            # Build RBF system and solve
            self._solve_rbf_system()
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            raise FittingError(f"RBF fitting failed: {e}")
    
    def _solve_rbf_system(self):
        """
        Solve the RBF interpolation system.
        
        Constructs and solves the linear system:
        [Φ P] [λ] = [f]
        [P^T 0] [c]   [0]
        
        where Φ is the RBF matrix, P is the polynomial matrix,
        λ are RBF weights, and c are polynomial coefficients.
        """
        n_points = len(self.training_points)
        
        # Build RBF matrix
        distances = cdist(self.training_points, self.training_points)
        phi_matrix = self._kernel_func(distances, self.rbf_params.shape_parameter)
        
        # Add regularization to diagonal
        phi_matrix += self.rbf_params.regularization * np.eye(n_points)
        
        # Handle polynomial term
        if self.rbf_params.polynomial_degree >= 0:
            # Build polynomial matrix
            poly_matrix = self._build_polynomial_matrix(self.training_points)
            n_poly = poly_matrix.shape[1]
            
            # Augmented system: [Φ P; P^T 0] [λ; c] = [f; 0]
            system_size = n_points + n_poly
            A = np.zeros((system_size, system_size))
            b = np.zeros(system_size)
            
            # Fill RBF part
            A[:n_points, :n_points] = phi_matrix
            A[:n_points, n_points:] = poly_matrix
            A[n_points:, :n_points] = poly_matrix.T
            
            # Fill RHS
            b[:n_points] = self.training_values
            
        else:
            # Simple RBF system: Φ λ = f
            A = phi_matrix
            b = self.training_values
            n_poly = 0
        
        # Check condition number
        try:
            self.condition_number = np.linalg.cond(A)
            if self.condition_number > self.rbf_params.condition_number_threshold:
                warnings.warn(f"System is ill-conditioned (cond={self.condition_number:.2e}). "
                            f"Consider increasing regularization parameter.")
        except:
            self.condition_number = np.inf
        
        # Solve system
        try:
            # Try standard solve first
            solution = solve(A, b)
        except LinAlgError:
            # Fall back to SVD-based solution
            warnings.warn("Standard solver failed, using SVD-based solver.")
            try:
                U, s, Vt = svd(A)
                # Regularized pseudoinverse
                s_inv = np.where(s > 1e-12, 1.0/s, 0.0)
                A_pinv = Vt.T @ np.diag(s_inv) @ U.T
                solution = A_pinv @ b
            except Exception as e:
                raise FittingError(f"SVD solver also failed: {e}")
        
        # Extract weights
        self.weights = solution[:n_points]
        if n_poly > 0:
            self.polynomial_weights = solution[n_points:]
        else:
            self.polynomial_weights = None
    
    def _build_polynomial_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Build polynomial matrix for given points.
        
        Args:
            points: Coordinate points array
            
        Returns:
            Polynomial matrix
        """
        n_points, n_dims = points.shape
        degree = self.rbf_params.polynomial_degree
        
        if degree == 0:
            # Constant term only
            return np.ones((n_points, 1))
        elif degree == 1:
            # Linear polynomial: [1, x, y, (z)]
            if n_dims == 2:
                poly_matrix = np.column_stack([
                    np.ones(n_points),
                    points[:, 0],  # x
                    points[:, 1]   # y
                ])
            else:  # 3D
                poly_matrix = np.column_stack([
                    np.ones(n_points),
                    points[:, 0],  # x
                    points[:, 1],  # y
                    points[:, 2]   # z
                ])
        else:
            # Higher order polynomials not implemented
            raise NotImplementedError(f"Polynomial degree {degree} not implemented")
        
        return poly_matrix
    
    def predict(self, 
                points: Union[np.ndarray, pd.DataFrame, List[Tuple[float, float]]],
                **kwargs) -> np.ndarray:
        """
        Predict values at given points using RBF interpolation.
        
        Args:
            points: Points to predict at
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predicted values
            
        Raises:
            PredictionError: If prediction fails
        """
        if not self.is_fitted:
            raise PredictionError("RBF interpolator must be fitted before prediction")
        
        try:
            # Parse input points
            pred_points = self._parse_prediction_points(points)
            n_pred = len(pred_points)
            
            if self.rbf_params.use_global:
                # Global RBF prediction
                predictions = self._predict_global(pred_points)
            else:
                # Local RBF prediction
                predictions = np.zeros(n_pred)
                for i, point in enumerate(pred_points):
                    predictions[i] = self._predict_local(point)
            
            return predictions
            
        except Exception as e:
            raise PredictionError(f"RBF prediction failed: {e}")
    
    def _predict_global(self, pred_points: np.ndarray) -> np.ndarray:
        """
        Predict using global RBF (all training points).
        
        Args:
            pred_points: Points to predict at
            
        Returns:
            Predicted values
        """
        # Calculate distances from prediction points to training points
        distances = cdist(pred_points, self.training_points)
        
        # Evaluate RBF kernel
        phi_pred = self._kernel_func(distances, self.rbf_params.shape_parameter)
        
        # RBF contribution
        predictions = phi_pred @ self.weights
        
        # Add polynomial contribution if present
        if self.polynomial_weights is not None:
            poly_pred = self._build_polynomial_matrix(pred_points)
            predictions += poly_pred @ self.polynomial_weights
        
        return predictions
    
    def _predict_local(self, pred_point: np.ndarray) -> float:
        """
        Predict using local RBF (subset of training points).
        
        Args:
            pred_point: Single point to predict at
            
        Returns:
            Predicted value
        """
        # Find neighbors
        neighbor_indices, _ = self._find_neighbors(pred_point, self.training_points)
        
        if len(neighbor_indices) == 0:
            # No neighbors found, return mean
            return np.mean(self.training_values)
        
        # Get neighbor data
        neighbor_points = self.training_points[neighbor_indices]
        neighbor_values = self.training_values[neighbor_indices]
        
        # Build local RBF system for these neighbors
        n_neighbors = len(neighbor_indices)
        distances = cdist(neighbor_points, neighbor_points)
        phi_matrix = self._kernel_func(distances, self.rbf_params.shape_parameter)
        phi_matrix += self.rbf_params.regularization * np.eye(n_neighbors)
        
        # Solve local system
        try:
            local_weights = solve(phi_matrix, neighbor_values)
        except LinAlgError:
            # Fall back to pseudoinverse
            local_weights = np.linalg.pinv(phi_matrix) @ neighbor_values
        
        # Predict at target point
        pred_distances = cdist([pred_point], neighbor_points)[0]
        phi_pred = self._kernel_func(pred_distances, self.rbf_params.shape_parameter)
        
        return phi_pred @ local_weights
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training data and model state."""
        summary = super().get_training_summary()
        
        if self.is_fitted:
            summary.update({
                'condition_number': self.condition_number,
                'n_weights': len(self.weights) if self.weights is not None else 0,
                'has_polynomial': self.polynomial_weights is not None,
                'kernel_type': self.rbf_params.kernel.value,
                'regularization_used': self.rbf_params.regularization > 0,
            })
        
        return summary
    
    def estimate_shape_parameter(self, method: str = 'leave_one_out') -> float:
        """
        Estimate optimal shape parameter using cross-validation.
        
        Args:
            method: Estimation method ('leave_one_out')
            
        Returns:
            Estimated optimal shape parameter
        """
        if not self.is_fitted:
            raise PredictionError("Must fit interpolator before parameter estimation")
        
        if method == 'leave_one_out':
            return self._estimate_shape_loo()
        else:
            raise ValueError(f"Unknown estimation method: {method}")
    
    def _estimate_shape_loo(self) -> float:
        """
        Estimate shape parameter using leave-one-out cross-validation.
        
        Returns:
            Optimal shape parameter
        """
        # Test range of shape parameters
        epsilon_range = np.logspace(-2, 2, 20)  # From 0.01 to 100
        best_epsilon = self.rbf_params.shape_parameter
        best_error = np.inf
        
        original_epsilon = self.rbf_params.shape_parameter
        
        try:
            for epsilon in epsilon_range:
                # Set test epsilon
                self.rbf_params.shape_parameter = epsilon
                self._kernel_func = self._get_kernel_function()
                
                # Refit with new parameter
                self._solve_rbf_system()
                
                # Calculate LOO error
                loo_error = self._calculate_loo_error()
                
                if loo_error < best_error:
                    best_error = loo_error
                    best_epsilon = epsilon
            
            return best_epsilon
            
        finally:
            # Restore original parameter
            self.rbf_params.shape_parameter = original_epsilon
            self._kernel_func = self._get_kernel_function()
            self._solve_rbf_system()
    
    def _calculate_loo_error(self) -> float:
        """
        Calculate leave-one-out cross-validation error.
        
        Returns:
            LOO RMSE error
        """
        n_points = len(self.training_points)
        errors = []
        
        for i in range(n_points):
            # Remove point i
            train_mask = np.ones(n_points, dtype=bool)
            train_mask[i] = False
            
            train_points = self.training_points[train_mask]
            train_values = self.training_values[train_mask]
            test_point = self.training_points[i:i+1]
            test_value = self.training_values[i]
            
            # Fit local model
            distances = cdist(train_points, train_points)
            phi_matrix = self._kernel_func(distances, self.rbf_params.shape_parameter)
            phi_matrix += self.rbf_params.regularization * np.eye(len(train_points))
            
            try:
                weights = solve(phi_matrix, train_values)
            except LinAlgError:
                weights = np.linalg.pinv(phi_matrix) @ train_values
            
            # Predict at test point
            test_distances = cdist(test_point, train_points)[0]
            phi_pred = self._kernel_func(test_distances, self.rbf_params.shape_parameter)
            prediction = phi_pred @ weights
            
            errors.append((prediction - test_value)**2)
        
        return np.sqrt(np.mean(errors))
    
    def optimize_all_parameters(self, method: str = 'grid_search') -> Dict[str, Any]:
        """
        Optimize all RBF parameters using comprehensive search methods.
        
        Args:
            method: Optimization method ('grid_search', 'random_search', 'bayesian')
            
        Returns:
            Dictionary with optimization results and best parameters
        """
        if not self.is_fitted:
            raise PredictionError("Must fit interpolator before parameter optimization")
        
        if method == 'grid_search':
            return self._optimize_grid_search()
        elif method == 'random_search':
            return self._optimize_random_search()
        elif method == 'bayesian':
            return self._optimize_bayesian()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_grid_search(self) -> Dict[str, Any]:
        """
        Optimize parameters using grid search.
        
        Returns:
            Dictionary with optimization results
        """
        # Save original parameters
        original_kernel = self.rbf_params.kernel
        original_shape = self.rbf_params.shape_parameter
        original_reg = self.rbf_params.regularization
        original_poly = self.rbf_params.polynomial_degree
        
        # Define parameter grids
        kernels_to_test = [
            RBFKernel.MULTIQUADRIC,
            RBFKernel.INVERSE_MULTIQUADRIC,
            RBFKernel.GAUSSIAN,
            RBFKernel.THIN_PLATE_SPLINE
        ]
        
        shape_params = np.logspace(-2, 2, 10)  # 0.01 to 100
        regularizations = np.logspace(-15, -8, 8)  # 1e-15 to 1e-8
        poly_degrees = [-1, 0, 1]  # No polynomial, constant, linear
        
        best_error = np.inf
        best_params = {}
        results = []
        
        total_combinations = len(kernels_to_test) * len(shape_params) * len(regularizations) * len(poly_degrees)
        current_combination = 0
        
        try:
            for kernel in kernels_to_test:
                for shape_param in shape_params:
                    for regularization in regularizations:
                        for poly_degree in poly_degrees:
                            current_combination += 1
                            
                            try:
                                # Set test parameters
                                self.rbf_params.kernel = kernel
                                self.rbf_params.shape_parameter = shape_param
                                self.rbf_params.regularization = regularization
                                self.rbf_params.polynomial_degree = poly_degree
                                self._kernel_func = self._get_kernel_function()
                                
                                # Refit with new parameters
                                self._solve_rbf_system()
                                
                                # Calculate cross-validation error
                                cv_error = self._calculate_loo_error()
                                
                                result = {
                                    'kernel': kernel.value,
                                    'shape_parameter': shape_param,
                                    'regularization': regularization,
                                    'polynomial_degree': poly_degree,
                                    'cv_error': cv_error,
                                    'condition_number': self.condition_number
                                }
                                results.append(result)
                                
                                if cv_error < best_error:
                                    best_error = cv_error
                                    best_params = result.copy()
                                    
                            except Exception as e:
                                # Skip invalid parameter combinations
                                continue
                            
            return {
                'method': 'grid_search',
                'best_params': best_params,
                'best_cv_error': best_error,
                'all_results': results,
                'total_combinations': total_combinations,
                'successful_evaluations': len(results)
            }
            
        finally:
            # Restore best parameters or original if optimization failed
            if best_params:
                self.rbf_params.kernel = RBFKernel(best_params['kernel'])
                self.rbf_params.shape_parameter = best_params['shape_parameter']
                self.rbf_params.regularization = best_params['regularization']
                self.rbf_params.polynomial_degree = best_params['polynomial_degree']
            else:
                # Restore original parameters
                self.rbf_params.kernel = original_kernel
                self.rbf_params.shape_parameter = original_shape
                self.rbf_params.regularization = original_reg
                self.rbf_params.polynomial_degree = original_poly
            
            self._kernel_func = self._get_kernel_function()
            self._solve_rbf_system()
    
    def _optimize_random_search(self, n_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize parameters using random search.
        
        Args:
            n_iterations: Number of random parameter combinations to try
            
        Returns:
            Dictionary with optimization results
        """
        import random
        
        # Save original parameters
        original_kernel = self.rbf_params.kernel
        original_shape = self.rbf_params.shape_parameter
        original_reg = self.rbf_params.regularization
        original_poly = self.rbf_params.polynomial_degree
        
        kernels_to_test = [
            RBFKernel.MULTIQUADRIC,
            RBFKernel.INVERSE_MULTIQUADRIC,
            RBFKernel.GAUSSIAN,
            RBFKernel.THIN_PLATE_SPLINE
        ]
        
        best_error = np.inf
        best_params = {}
        results = []
        
        try:
            for i in range(n_iterations):
                try:
                    # Generate random parameters
                    kernel = random.choice(kernels_to_test)
                    shape_param = 10 ** random.uniform(-2, 2)  # 0.01 to 100
                    regularization = 10 ** random.uniform(-15, -8)  # 1e-15 to 1e-8
                    poly_degree = random.choice([-1, 0, 1])
                    
                    # Set test parameters
                    self.rbf_params.kernel = kernel
                    self.rbf_params.shape_parameter = shape_param
                    self.rbf_params.regularization = regularization
                    self.rbf_params.polynomial_degree = poly_degree
                    self._kernel_func = self._get_kernel_function()
                    
                    # Refit with new parameters
                    self._solve_rbf_system()
                    
                    # Calculate cross-validation error
                    cv_error = self._calculate_loo_error()
                    
                    result = {
                        'iteration': i,
                        'kernel': kernel.value,
                        'shape_parameter': shape_param,
                        'regularization': regularization,
                        'polynomial_degree': poly_degree,
                        'cv_error': cv_error,
                        'condition_number': self.condition_number
                    }
                    results.append(result)
                    
                    if cv_error < best_error:
                        best_error = cv_error
                        best_params = result.copy()
                        
                except Exception as e:
                    # Skip invalid parameter combinations
                    continue
                    
            return {
                'method': 'random_search',
                'best_params': best_params,
                'best_cv_error': best_error,
                'all_results': results,
                'total_iterations': n_iterations,
                'successful_evaluations': len(results)
            }
            
        finally:
            # Restore best parameters or original if optimization failed
            if best_params:
                self.rbf_params.kernel = RBFKernel(best_params['kernel'])
                self.rbf_params.shape_parameter = best_params['shape_parameter']
                self.rbf_params.regularization = best_params['regularization']
                self.rbf_params.polynomial_degree = best_params['polynomial_degree']
            else:
                # Restore original parameters
                self.rbf_params.kernel = original_kernel
                self.rbf_params.shape_parameter = original_shape
                self.rbf_params.regularization = original_reg
                self.rbf_params.polynomial_degree = original_poly
            
            self._kernel_func = self._get_kernel_function()
            self._solve_rbf_system()
    
    def _optimize_bayesian(self) -> Dict[str, Any]:
        """
        Optimize parameters using Bayesian optimization (simplified version).
        
        Returns:
            Dictionary with optimization results
        """
        # For now, implement a simplified version using scipy's differential evolution
        # which provides a good balance between exploration and exploitation
        from scipy.optimize import differential_evolution
        
        # Save original parameters
        original_kernel = self.rbf_params.kernel
        original_shape = self.rbf_params.shape_parameter
        original_reg = self.rbf_params.regularization
        original_poly = self.rbf_params.polynomial_degree
        
        kernels_to_test = [
            RBFKernel.MULTIQUADRIC,
            RBFKernel.INVERSE_MULTIQUADRIC,
            RBFKernel.GAUSSIAN,
            RBFKernel.THIN_PLATE_SPLINE
        ]
        
        best_error = np.inf
        best_params = {}
        all_results = []
        
        try:
            # Optimize each kernel separately
            for kernel in kernels_to_test:
                try:
                    # Set kernel
                    self.rbf_params.kernel = kernel
                    self._kernel_func = self._get_kernel_function()
                    
                    # Define objective function for this kernel
                    def objective(params):
                        try:
                            shape_param, log_reg, poly_degree = params
                            regularization = 10 ** log_reg
                            poly_degree = int(round(poly_degree))
                            
                            # Set parameters
                            self.rbf_params.shape_parameter = shape_param
                            self.rbf_params.regularization = regularization
                            self.rbf_params.polynomial_degree = poly_degree
                            
                            # Refit and evaluate
                            self._solve_rbf_system()
                            cv_error = self._calculate_loo_error()
                            
                            return cv_error
                            
                        except Exception:
                            return 1e10  # Return large error for invalid parameters
                    
                    # Parameter bounds: [shape_param, log_regularization, polynomial_degree]
                    bounds = [
                        (0.01, 100),    # shape parameter
                        (-15, -8),      # log regularization
                        (-1, 1)         # polynomial degree
                    ]
                    
                    # Optimize
                    result = differential_evolution(
                        objective, 
                        bounds, 
                        maxiter=50, 
                        popsize=10,
                        seed=42
                    )
                    
                    if result.success:
                        shape_param, log_reg, poly_degree = result.x
                        regularization = 10 ** log_reg
                        poly_degree = int(round(poly_degree))
                        
                        kernel_result = {
                            'kernel': kernel.value,
                            'shape_parameter': shape_param,
                            'regularization': regularization,
                            'polynomial_degree': poly_degree,
                            'cv_error': result.fun,
                            'optimization_success': True
                        }
                        all_results.append(kernel_result)
                        
                        if result.fun < best_error:
                            best_error = result.fun
                            best_params = kernel_result.copy()
                            
                except Exception as e:
                    # Skip kernels that fail optimization
                    continue
            
            return {
                'method': 'bayesian',
                'best_params': best_params,
                'best_cv_error': best_error,
                'all_results': all_results,
                'kernels_tested': len(all_results)
            }
            
        finally:
            # Restore best parameters or original if optimization failed
            if best_params:
                self.rbf_params.kernel = RBFKernel(best_params['kernel'])
                self.rbf_params.shape_parameter = best_params['shape_parameter']
                self.rbf_params.regularization = best_params['regularization']
                self.rbf_params.polynomial_degree = best_params['polynomial_degree']
            else:
                # Restore original parameters
                self.rbf_params.kernel = original_kernel
                self.rbf_params.shape_parameter = original_shape
                self.rbf_params.regularization = original_reg
                self.rbf_params.polynomial_degree = original_poly
            
            self._kernel_func = self._get_kernel_function()
            self._solve_rbf_system()
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for RBF parameter optimization based on data characteristics.
        
        Returns:
            Dictionary with optimization recommendations
        """
        if not self.is_fitted:
            raise PredictionError("Must fit interpolator before getting recommendations")
        
        n_points = len(self.training_points)
        data_dimension = self.training_points.shape[1]
        
        # Analyze data characteristics
        coord_ranges = np.ptp(self.training_points, axis=0)  # Range in each dimension
        value_range = np.ptp(self.training_values)
        value_std = np.std(self.training_values)
        
        # Calculate point density
        if data_dimension == 2:
            area = coord_ranges[0] * coord_ranges[1]
            density = n_points / area if area > 0 else 0
        else:
            volume = np.prod(coord_ranges)
            density = n_points / volume if volume > 0 else 0
        
        recommendations = {
            'data_characteristics': {
                'n_points': n_points,
                'dimension': data_dimension,
                'coordinate_ranges': coord_ranges.tolist(),
                'value_range': float(value_range),
                'value_std': float(value_std),
                'point_density': float(density)
            }
        }
        
        # Kernel recommendations based on data size and characteristics
        if n_points < 100:
            recommended_kernels = ['multiquadric', 'inverse_multiquadric']
            recommended_method = 'grid_search'
            recommendations['size_category'] = 'small'
        elif n_points < 1000:
            recommended_kernels = ['multiquadric', 'gaussian', 'thin_plate_spline']
            recommended_method = 'random_search'
            recommendations['size_category'] = 'medium'
        else:
            recommended_kernels = ['multiquadric', 'gaussian']
            recommended_method = 'bayesian'
            recommendations['size_category'] = 'large'
        
        # Shape parameter recommendations
        typical_distance = np.mean(coord_ranges)
        if typical_distance > 0:
            shape_param_suggestion = 1.0 / typical_distance
        else:
            shape_param_suggestion = 1.0
        
        # Regularization recommendations based on condition number
        if self.condition_number and self.condition_number > 1e10:
            reg_suggestion = 1e-8
        elif self.condition_number and self.condition_number > 1e8:
            reg_suggestion = 1e-10
        else:
            reg_suggestion = 1e-12
        
        recommendations.update({
            'recommended_kernels': recommended_kernels,
            'recommended_method': recommended_method,
            'shape_parameter_suggestion': float(shape_param_suggestion),
            'regularization_suggestion': float(reg_suggestion),
            'polynomial_recommendations': {
                'smooth_data': 'Use polynomial_degree = -1 (no polynomial)',
                'trending_data': 'Use polynomial_degree = 1 (linear trend)',
                'constant_offset': 'Use polynomial_degree = 0 (constant term)'
            }
        })
        
        return recommendations
    
    def auto_optimize_parameters(self) -> Dict[str, Any]:
        """
        Automatically optimize RBF parameters using the best method for the data size.
        
        Returns:
            Dictionary with optimization results
        """
        recommendations = self.get_optimization_recommendations()
        recommended_method = recommendations['recommended_method']
        
        # Use recommended optimization method
        return self.optimize_all_parameters(method=recommended_method)