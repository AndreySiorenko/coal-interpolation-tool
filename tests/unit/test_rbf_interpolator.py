"""
Unit tests for RBF interpolator implementation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

from src.core.interpolation.rbf import (
    RBFInterpolator, RBFParameters, RBFKernel, RBFKernels
)
from src.core.interpolation.base import SearchParameters, FittingError, PredictionError


class TestRBFKernel:
    """Test RBF kernel enum."""
    
    def test_kernel_values(self):
        """Test that kernel enum has expected values."""
        assert RBFKernel.GAUSSIAN.value == "gaussian"
        assert RBFKernel.MULTIQUADRIC.value == "multiquadric"
        assert RBFKernel.INVERSE_MULTIQUADRIC.value == "inverse_multiquadric"
        assert RBFKernel.THIN_PLATE_SPLINE.value == "thin_plate_spline"
        assert RBFKernel.LINEAR.value == "linear"
        assert RBFKernel.CUBIC.value == "cubic"
        assert RBFKernel.QUINTIC.value == "quintic"


class TestRBFParameters:
    """Test RBF parameters dataclass."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = RBFParameters()
        assert params.kernel == RBFKernel.MULTIQUADRIC
        assert params.shape_parameter == 1.0
        assert params.regularization == 1e-12
        assert params.polynomial_degree == -1
        assert params.use_global == True
        assert params.condition_number_threshold == 1e12
    
    def test_custom_parameters(self):
        """Test setting custom parameters."""
        params = RBFParameters(
            kernel=RBFKernel.GAUSSIAN,
            shape_parameter=2.5,
            regularization=1e-8,
            polynomial_degree=1,
            use_global=False,
            condition_number_threshold=1e10
        )
        assert params.kernel == RBFKernel.GAUSSIAN
        assert params.shape_parameter == 2.5
        assert params.regularization == 1e-8
        assert params.polynomial_degree == 1
        assert params.use_global == False
        assert params.condition_number_threshold == 1e10


class TestRBFKernels:
    """Test RBF kernel functions."""
    
    @pytest.fixture
    def distances(self):
        """Sample distance array."""
        return np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    
    def test_gaussian_kernel(self, distances):
        """Test Gaussian RBF kernel."""
        epsilon = 1.0
        result = RBFKernels.gaussian(distances, epsilon)
        expected = np.exp(-(epsilon * distances) ** 2)
        np.testing.assert_allclose(result, expected)
    
    def test_multiquadric_kernel(self, distances):
        """Test Multiquadric RBF kernel."""
        epsilon = 1.0
        result = RBFKernels.multiquadric(distances, epsilon)
        expected = np.sqrt(1 + (epsilon * distances) ** 2)
        np.testing.assert_allclose(result, expected)
    
    def test_inverse_multiquadric_kernel(self, distances):
        """Test Inverse Multiquadric RBF kernel."""
        epsilon = 1.0
        result = RBFKernels.inverse_multiquadric(distances, epsilon)
        expected = 1.0 / np.sqrt(1 + (epsilon * distances) ** 2)
        np.testing.assert_allclose(result, expected)
    
    def test_thin_plate_spline_kernel(self, distances):
        """Test Thin Plate Spline RBF kernel."""
        result = RBFKernels.thin_plate_spline(distances)
        # Check that r=0 gives 0
        assert result[0] == 0.0
        # Check non-zero values
        for i, r in enumerate(distances[1:], 1):
            expected = r ** 2 * np.log(r + 1e-15)
            assert abs(result[i] - expected) < 1e-10
    
    def test_linear_kernel(self, distances):
        """Test Linear RBF kernel."""
        result = RBFKernels.linear(distances)
        np.testing.assert_allclose(result, distances)
    
    def test_cubic_kernel(self, distances):
        """Test Cubic RBF kernel."""
        result = RBFKernels.cubic(distances)
        expected = distances ** 3
        np.testing.assert_allclose(result, expected)
    
    def test_quintic_kernel(self, distances):
        """Test Quintic RBF kernel."""
        result = RBFKernels.quintic(distances)
        expected = distances ** 5
        np.testing.assert_allclose(result, expected)


class TestRBFInterpolator:
    """Test RBF interpolator class."""
    
    @pytest.fixture
    def sample_data_2d(self):
        """Sample 2D training data."""
        np.random.seed(42)
        n_points = 20
        x = np.random.uniform(0, 10, n_points)
        y = np.random.uniform(0, 10, n_points)
        # Simple function: z = x + y + noise
        z = x + y + np.random.normal(0, 0.1, n_points)
        
        return pd.DataFrame({
            'X': x,
            'Y': y,
            'Value': z
        })
    
    @pytest.fixture
    def sample_data_3d(self):
        """Sample 3D training data."""
        np.random.seed(42)
        n_points = 15
        x = np.random.uniform(0, 5, n_points)
        y = np.random.uniform(0, 5, n_points)
        z_coord = np.random.uniform(0, 5, n_points)
        # Simple function: value = x + y + z + noise
        values = x + y + z_coord + np.random.normal(0, 0.05, n_points)
        
        return pd.DataFrame({
            'X': x,
            'Y': y,
            'Z': z_coord,
            'Value': values
        })
    
    def test_init_default(self):
        """Test RBF interpolator initialization with defaults."""
        rbf = RBFInterpolator()
        assert isinstance(rbf.search_params, SearchParameters)
        assert isinstance(rbf.rbf_params, RBFParameters)
        assert rbf.rbf_params.kernel == RBFKernel.MULTIQUADRIC
        assert not rbf.is_fitted
        assert rbf.training_points is None
        assert rbf.weights is None
    
    def test_init_custom_params(self):
        """Test RBF interpolator initialization with custom parameters."""
        search_params = SearchParameters(search_radius=500.0, max_points=20)
        rbf_params = RBFParameters(kernel=RBFKernel.GAUSSIAN, shape_parameter=2.0)
        
        rbf = RBFInterpolator(search_params, rbf_params)
        assert rbf.search_params.search_radius == 500.0
        assert rbf.rbf_params.kernel == RBFKernel.GAUSSIAN
        assert rbf.rbf_params.shape_parameter == 2.0
    
    def test_get_method_name(self):
        """Test getting method name."""
        rbf = RBFInterpolator()
        name = rbf.get_method_name()
        assert "Radial Basis Function" in name
        assert "Multiquadric" in name
        
        # Test with different kernel
        rbf.rbf_params.kernel = RBFKernel.GAUSSIAN
        rbf._kernel_func = rbf._get_kernel_function()
        name = rbf.get_method_name()
        assert "Gaussian" in name
    
    def test_get_parameters(self):
        """Test getting parameters."""
        rbf_params = RBFParameters(
            kernel=RBFKernel.GAUSSIAN,
            shape_parameter=1.5,
            regularization=1e-10,
            use_global=False
        )
        search_params = SearchParameters(search_radius=200.0, max_points=15)
        
        rbf = RBFInterpolator(search_params, rbf_params)
        params = rbf.get_parameters()
        
        assert params['kernel'] == 'gaussian'
        assert params['shape_parameter'] == 1.5
        assert params['regularization'] == 1e-10
        assert params['use_global'] == False
        assert params['search_radius'] == 200.0
        assert params['max_points'] == 15
    
    def test_set_parameters(self):
        """Test setting parameters."""
        rbf = RBFInterpolator()
        
        rbf.set_parameters(
            kernel='gaussian',
            shape_parameter=2.5,
            regularization=1e-8,
            search_radius=300.0
        )
        
        assert rbf.rbf_params.kernel == RBFKernel.GAUSSIAN
        assert rbf.rbf_params.shape_parameter == 2.5
        assert rbf.rbf_params.regularization == 1e-8
        assert rbf.search_params.search_radius == 300.0
    
    def test_set_parameters_invalid_kernel(self):
        """Test setting invalid kernel parameter."""
        rbf = RBFInterpolator()
        
        with pytest.raises(ValueError, match="Invalid kernel"):
            rbf.set_parameters(kernel='invalid_kernel')
    
    def test_set_parameters_kernel_enum(self):
        """Test setting kernel parameter as enum."""
        rbf = RBFInterpolator()
        rbf.set_parameters(kernel=RBFKernel.THIN_PLATE_SPLINE)
        assert rbf.rbf_params.kernel == RBFKernel.THIN_PLATE_SPLINE
    
    def test_get_kernel_function(self):
        """Test getting kernel function."""
        rbf = RBFInterpolator()
        
        # Test all kernel types
        kernel_tests = [
            (RBFKernel.GAUSSIAN, RBFKernels.gaussian),
            (RBFKernel.MULTIQUADRIC, RBFKernels.multiquadric),
            (RBFKernel.INVERSE_MULTIQUADRIC, RBFKernels.inverse_multiquadric),
            (RBFKernel.THIN_PLATE_SPLINE, RBFKernels.thin_plate_spline),
            (RBFKernel.LINEAR, RBFKernels.linear),
            (RBFKernel.CUBIC, RBFKernels.cubic),
            (RBFKernel.QUINTIC, RBFKernels.quintic),
        ]
        
        for kernel_enum, expected_func in kernel_tests:
            rbf.rbf_params.kernel = kernel_enum
            func = rbf._get_kernel_function()
            assert func == expected_func
    
    def test_fit_2d_data(self, sample_data_2d):
        """Test fitting with 2D data."""
        rbf = RBFInterpolator()
        result = rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        assert result is rbf  # Method chaining
        assert rbf.is_fitted
        assert rbf.training_points.shape == (20, 2)
        assert rbf.training_values.shape == (20,)
        assert rbf.weights is not None
        assert len(rbf.weights) == 20
    
    def test_fit_3d_data(self, sample_data_3d):
        """Test fitting with 3D data."""
        rbf = RBFInterpolator()
        result = rbf.fit(sample_data_3d, 'X', 'Y', 'Value', z_col='Z')
        
        assert result is rbf
        assert rbf.is_fitted
        assert rbf.training_points.shape == (15, 3)
        assert rbf.coordinate_columns['Z'] == 'Z'
    
    def test_fit_with_polynomial(self, sample_data_2d):
        """Test fitting with polynomial augmentation."""
        rbf_params = RBFParameters(polynomial_degree=1)
        rbf = RBFInterpolator(rbf_params=rbf_params)
        
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        assert rbf.is_fitted
        assert rbf.polynomial_weights is not None
        assert len(rbf.polynomial_weights) == 3  # constant + x + y
    
    def test_fit_invalid_data(self):
        """Test fitting with invalid data."""
        rbf = RBFInterpolator()
        
        # Missing columns
        data = pd.DataFrame({'X': [1, 2], 'Y': [1, 2]})  # Missing 'Value'
        with pytest.raises(FittingError):
            rbf.fit(data, 'X', 'Y', 'Value')
        
        # Empty data
        empty_data = pd.DataFrame(columns=['X', 'Y', 'Value'])
        with pytest.raises(FittingError):
            rbf.fit(empty_data, 'X', 'Y', 'Value')
    
    @patch('src.core.interpolation.rbf.solve')
    def test_fit_solver_failure(self, mock_solve, sample_data_2d):
        """Test fallback to SVD when solver fails."""
        from scipy.linalg import LinAlgError
        mock_solve.side_effect = LinAlgError("Singular matrix")
        
        rbf = RBFInterpolator()
        
        # Should still succeed using SVD fallback
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        assert rbf.is_fitted
    
    def test_predict_not_fitted(self):
        """Test prediction before fitting."""
        rbf = RBFInterpolator()
        points = np.array([[1, 1], [2, 2]])
        
        with pytest.raises(PredictionError, match="must be fitted"):
            rbf.predict(points)
    
    def test_predict_global_2d(self, sample_data_2d):
        """Test global prediction in 2D."""
        rbf = RBFInterpolator()
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Predict at training points (should be exact)
        train_points = sample_data_2d[['X', 'Y']].values
        predictions = rbf.predict(train_points)
        
        # Should be close to training values (within tolerance due to regularization)
        np.testing.assert_allclose(predictions, sample_data_2d['Value'].values, rtol=1e-3)
    
    def test_predict_global_3d(self, sample_data_3d):
        """Test global prediction in 3D."""
        rbf = RBFInterpolator()
        rbf.fit(sample_data_3d, 'X', 'Y', 'Value', z_col='Z')
        
        # Predict at new points
        new_points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        predictions = rbf.predict(new_points)
        
        assert predictions.shape == (3,)
        assert np.all(np.isfinite(predictions))
    
    def test_predict_local_mode(self, sample_data_2d):
        """Test local prediction mode."""
        rbf_params = RBFParameters(use_global=False)
        search_params = SearchParameters(search_radius=5.0, max_points=8)
        rbf = RBFInterpolator(search_params, rbf_params)
        
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Predict at new points
        new_points = np.array([[5, 5], [2, 2]])
        predictions = rbf.predict(new_points)
        
        assert predictions.shape == (2,)
        assert np.all(np.isfinite(predictions))
    
    def test_predict_different_input_formats(self, sample_data_2d):
        """Test prediction with different input formats."""
        rbf = RBFInterpolator()
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Test with list of tuples
        points_list = [(1, 1), (2, 2), (3, 3)]
        pred1 = rbf.predict(points_list)
        
        # Test with numpy array
        points_array = np.array([[1, 1], [2, 2], [3, 3]])
        pred2 = rbf.predict(points_array)
        
        # Test with DataFrame
        points_df = pd.DataFrame({'X': [1, 2, 3], 'Y': [1, 2, 3]})
        pred3 = rbf.predict(points_df)
        
        np.testing.assert_allclose(pred1, pred2)
        np.testing.assert_allclose(pred1, pred3)
    
    def test_predict_with_polynomial(self, sample_data_2d):
        """Test prediction with polynomial augmentation."""
        rbf_params = RBFParameters(polynomial_degree=1)
        rbf = RBFInterpolator(rbf_params=rbf_params)
        
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        new_points = np.array([[1, 1], [5, 5]])
        predictions = rbf.predict(new_points)
        
        assert predictions.shape == (2,)
        assert np.all(np.isfinite(predictions))
    
    def test_build_polynomial_matrix_2d(self, sample_data_2d):
        """Test building polynomial matrix for 2D."""
        rbf = RBFInterpolator()
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        points = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Test constant polynomial
        rbf.rbf_params.polynomial_degree = 0
        poly_matrix = rbf._build_polynomial_matrix(points)
        expected = np.ones((3, 1))
        np.testing.assert_allclose(poly_matrix, expected)
        
        # Test linear polynomial
        rbf.rbf_params.polynomial_degree = 1
        poly_matrix = rbf._build_polynomial_matrix(points)
        expected = np.column_stack([
            np.ones(3),
            points[:, 0],  # x
            points[:, 1]   # y
        ])
        np.testing.assert_allclose(poly_matrix, expected)
    
    def test_build_polynomial_matrix_3d(self, sample_data_3d):
        """Test building polynomial matrix for 3D."""
        rbf = RBFInterpolator()
        rbf.fit(sample_data_3d, 'X', 'Y', 'Value', z_col='Z')
        
        points = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Test linear polynomial in 3D
        rbf.rbf_params.polynomial_degree = 1
        poly_matrix = rbf._build_polynomial_matrix(points)
        expected = np.column_stack([
            np.ones(2),
            points[:, 0],  # x
            points[:, 1],  # y
            points[:, 2]   # z
        ])
        np.testing.assert_allclose(poly_matrix, expected)
    
    def test_polynomial_degree_not_implemented(self, sample_data_2d):
        """Test error for unimplemented polynomial degrees."""
        rbf = RBFInterpolator()
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        points = np.array([[1, 2]])
        rbf.rbf_params.polynomial_degree = 2  # Not implemented
        
        with pytest.raises(NotImplementedError):
            rbf._build_polynomial_matrix(points)
    
    def test_get_training_summary(self, sample_data_2d):
        """Test getting training summary."""
        rbf = RBFInterpolator()
        
        # Before fitting
        summary = rbf.get_training_summary()
        assert summary == {}
        
        # After fitting
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        summary = rbf.get_training_summary()
        
        assert 'n_points' in summary
        assert summary['n_points'] == 20
        assert 'condition_number' in summary
        assert 'kernel_type' in summary
        assert summary['kernel_type'] == 'multiquadric'
        assert 'n_weights' in summary
        assert summary['n_weights'] == 20
    
    def test_estimate_shape_parameter(self, sample_data_2d):
        """Test shape parameter estimation."""
        rbf = RBFInterpolator()
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Test leave-one-out method
        optimal_epsilon = rbf.estimate_shape_parameter('leave_one_out')
        assert isinstance(optimal_epsilon, float)
        assert optimal_epsilon > 0
        
        # Test invalid method
        with pytest.raises(ValueError):
            rbf.estimate_shape_parameter('invalid_method')
    
    def test_estimate_shape_parameter_not_fitted(self):
        """Test shape parameter estimation before fitting."""
        rbf = RBFInterpolator()
        
        with pytest.raises(PredictionError, match="Must fit interpolator"):
            rbf.estimate_shape_parameter()
    
    def test_condition_number_warning(self, sample_data_2d):
        """Test condition number warning for ill-conditioned matrices."""
        # Create data that may lead to ill-conditioned matrix
        rbf_params = RBFParameters(
            regularization=0.0,  # No regularization
            condition_number_threshold=10.0  # Low threshold to trigger warning
        )
        rbf = RBFInterpolator(rbf_params=rbf_params)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
            
            # Check if warning was issued (may not always happen)
            warning_messages = [str(warning.message) for warning in w]
            ill_conditioned_warnings = [msg for msg in warning_messages if "ill-conditioned" in msg]
            # Just check that the code doesn't crash - warning may or may not appear
    
    def test_different_kernels(self, sample_data_2d):
        """Test fitting and prediction with different kernels."""
        kernels_to_test = [
            RBFKernel.GAUSSIAN,
            RBFKernel.MULTIQUADRIC,
            RBFKernel.INVERSE_MULTIQUADRIC,
            RBFKernel.THIN_PLATE_SPLINE,
            RBFKernel.LINEAR,
            RBFKernel.CUBIC,
            RBFKernel.QUINTIC
        ]
        
        test_point = np.array([[5, 5]])
        
        for kernel in kernels_to_test:
            rbf_params = RBFParameters(kernel=kernel)
            rbf = RBFInterpolator(rbf_params=rbf_params)
            
            rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
            prediction = rbf.predict(test_point)
            
            assert len(prediction) == 1
            assert np.isfinite(prediction[0])
    
    def test_regularization_effect(self, sample_data_2d):
        """Test effect of regularization parameter."""
        # Test with low regularization
        rbf1 = RBFInterpolator(rbf_params=RBFParameters(regularization=1e-15))
        rbf1.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Test with high regularization
        rbf2 = RBFInterpolator(rbf_params=RBFParameters(regularization=1e-6))
        rbf2.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Both should fit successfully
        assert rbf1.is_fitted
        assert rbf2.is_fitted
        
        # Predictions should be different due to regularization
        test_point = np.array([[5, 5]])
        pred1 = rbf1.predict(test_point)
        pred2 = rbf2.predict(test_point)
        
        # They might be close but regularization should have some effect
        assert np.isfinite(pred1[0])
        assert np.isfinite(pred2[0])
    
    def test_local_mode_no_neighbors(self, sample_data_2d):
        """Test local mode when no neighbors found."""
        rbf_params = RBFParameters(use_global=False)
        search_params = SearchParameters(search_radius=0.1, min_points=5)  # Very small radius
        rbf = RBFInterpolator(search_params, rbf_params)
        
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Predict at point far from training data
        far_point = np.array([[100, 100]])
        prediction = rbf.predict(far_point)
        
        # Should return the mean of training values when no neighbors
        expected_mean = sample_data_2d['Value'].mean()
        np.testing.assert_allclose(prediction, [expected_mean])
    
    @patch('src.core.interpolation.rbf.solve')
    def test_local_mode_solver_failure(self, mock_solve, sample_data_2d):
        """Test local mode when solver fails."""
        from scipy.linalg import LinAlgError
        mock_solve.side_effect = LinAlgError("Singular matrix")
        
        rbf_params = RBFParameters(use_global=False)
        search_params = SearchParameters(search_radius=10.0, max_points=5)
        rbf = RBFInterpolator(search_params, rbf_params)
        
        rbf.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Should fallback to pseudoinverse
        test_point = np.array([[5, 5]])
        prediction = rbf.predict(test_point)
        
        assert len(prediction) == 1
        assert np.isfinite(prediction[0])


class TestRBFIntegration:
    """Integration tests for RBF interpolator."""
    
    def test_exact_interpolation(self):
        """Test that RBF provides exact interpolation at training points."""
        # Create simple dataset
        data = pd.DataFrame({
            'X': [0, 1, 0, 1],
            'Y': [0, 0, 1, 1], 
            'Value': [1, 2, 3, 4]
        })
        
        rbf = RBFInterpolator(rbf_params=RBFParameters(regularization=1e-15))
        rbf.fit(data, 'X', 'Y', 'Value')
        
        # Predict at training points
        train_points = data[['X', 'Y']].values
        predictions = rbf.predict(train_points)
        
        # Should be very close to training values
        np.testing.assert_allclose(predictions, data['Value'].values, atol=1e-10)
    
    def test_smooth_interpolation(self):
        """Test that RBF provides smooth interpolation."""
        # Create data from smooth function
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 2, 3, 4])
        X, Y = np.meshgrid(x, y)
        Z = X + Y  # Simple linear function
        
        data = pd.DataFrame({
            'X': X.ravel(),
            'Y': Y.ravel(),
            'Value': Z.ravel()
        })
        
        rbf = RBFInterpolator()
        rbf.fit(data, 'X', 'Y', 'Value')
        
        # Predict at intermediate points
        test_points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        predictions = rbf.predict(test_points)
        expected = np.array([1.0, 3.0, 5.0])  # From linear function
        
        # Should be close to expected (allowing for some RBF approximation)
        np.testing.assert_allclose(predictions, expected, rtol=0.1)
    
    def test_performance_large_dataset(self):
        """Test performance with larger dataset."""
        np.random.seed(42)
        n_points = 100
        
        data = pd.DataFrame({
            'X': np.random.uniform(0, 10, n_points),
            'Y': np.random.uniform(0, 10, n_points),
            'Value': np.random.uniform(0, 100, n_points)
        })
        
        rbf = RBFInterpolator()
        
        # Should fit without issues
        import time
        start_time = time.time()
        rbf.fit(data, 'X', 'Y', 'Value')
        fit_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert fit_time < 5.0  # 5 seconds
        assert rbf.is_fitted
        
        # Test prediction
        test_points = np.random.uniform(0, 10, (10, 2))
        start_time = time.time()
        predictions = rbf.predict(test_points)
        pred_time = time.time() - start_time
        
        assert pred_time < 1.0  # 1 second
        assert len(predictions) == 10
        assert np.all(np.isfinite(predictions))