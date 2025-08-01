"""
Unit tests for Kriging interpolator implementation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

from src.core.interpolation.kriging import (
    KrigingInterpolator, KrigingParameters, VariogramModel, KrigingType,
    VariogramModels
)
from src.core.interpolation.base import SearchParameters, FittingError, PredictionError


class TestVariogramModel:
    """Test variogram model enum."""
    
    def test_model_values(self):
        """Test that variogram model enum has expected values."""
        assert VariogramModel.SPHERICAL.value == "spherical"
        assert VariogramModel.EXPONENTIAL.value == "exponential"
        assert VariogramModel.GAUSSIAN.value == "gaussian"
        assert VariogramModel.LINEAR.value == "linear"
        assert VariogramModel.POWER.value == "power"
        assert VariogramModel.NUGGET.value == "nugget"


class TestKrigingType:
    """Test kriging type enum."""
    
    def test_type_values(self):
        """Test that kriging type enum has expected values."""
        assert KrigingType.ORDINARY.value == "ordinary"
        assert KrigingType.SIMPLE.value == "simple"
        assert KrigingType.UNIVERSAL.value == "universal"


class TestKrigingParameters:
    """Test Kriging parameters dataclass."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = KrigingParameters()
        assert params.kriging_type == KrigingType.ORDINARY
        assert params.variogram_model == VariogramModel.SPHERICAL
        assert params.nugget == 0.0
        assert params.sill == 1.0
        assert params.range_param == 1000.0
        assert params.use_global == True
        assert params.auto_fit_variogram == True
    
    def test_custom_parameters(self):
        """Test setting custom parameters."""
        params = KrigingParameters(
            kriging_type=KrigingType.SIMPLE,
            variogram_model=VariogramModel.EXPONENTIAL,
            nugget=0.1,
            sill=2.5,
            range_param=500.0,
            use_global=False,
            mean_value=10.0
        )
        assert params.kriging_type == KrigingType.SIMPLE
        assert params.variogram_model == VariogramModel.EXPONENTIAL
        assert params.nugget == 0.1
        assert params.sill == 2.5
        assert params.range_param == 500.0
        assert params.use_global == False
        assert params.mean_value == 10.0


class TestVariogramModels:
    """Test variogram model functions."""
    
    @pytest.fixture
    def distances(self):
        """Sample distance array."""
        return np.array([0.0, 100.0, 500.0, 1000.0, 2000.0])
    
    def test_spherical_model(self, distances):
        """Test spherical variogram model."""
        nugget, sill, range_param = 0.1, 1.0, 1000.0
        result = VariogramModels.spherical(distances, nugget, sill, range_param)
        
        # Test boundary conditions
        assert result[0] == 0.0  # Zero at distance 0
        assert np.isclose(result[-1], nugget + sill)  # Sill at large distances
        
        # Test monotonic increase up to range
        within_range = distances <= range_param
        within_range_diffs = np.diff(result[within_range])
        assert np.all(within_range_diffs >= 0)  # Non-decreasing
    
    def test_exponential_model(self, distances):
        """Test exponential variogram model."""
        nugget, sill, range_param = 0.1, 1.0, 1000.0
        result = VariogramModels.exponential(distances, nugget, sill, range_param)
        
        assert result[0] == 0.0  # Zero at distance 0
        assert result[-1] < nugget + sill  # Approaches but doesn't reach sill
        
        # Test monotonic increase
        assert np.all(np.diff(result) >= 0)
    
    def test_gaussian_model(self, distances):
        """Test Gaussian variogram model."""
        nugget, sill, range_param = 0.1, 1.0, 1000.0
        result = VariogramModels.gaussian(distances, nugget, sill, range_param)
        
        assert result[0] == 0.0  # Zero at distance 0
        assert result[-1] < nugget + sill  # Approaches but doesn't reach sill
        
        # Test monotonic increase
        assert np.all(np.diff(result) >= 0)
    
    def test_linear_model(self, distances):
        """Test linear variogram model."""
        nugget, slope = 0.1, 0.001
        result = VariogramModels.linear(distances, nugget, slope, 0)  # range not used
        
        assert result[0] == 0.0  # Zero at distance 0
        expected = nugget + slope * distances
        expected[0] = 0.0
        np.testing.assert_allclose(result, expected)
    
    def test_power_model(self, distances):
        """Test power variogram model."""
        nugget, scaling, power = 0.1, 0.001, 1.5
        result = VariogramModels.power(distances, nugget, scaling, power)
        
        assert result[0] == 0.0  # Zero at distance 0
        assert np.all(np.diff(result) >= 0)  # Monotonic increase
    
    def test_nugget_model(self, distances):
        """Test pure nugget variogram model."""
        nugget = 1.0
        result = VariogramModels.nugget(distances, nugget, 0, 0)
        
        assert result[0] == 0.0  # Zero at distance 0
        assert np.all(result[1:] == nugget)  # Nugget value for all h > 0


class TestKrigingInterpolator:
    """Test Kriging interpolator class."""
    
    @pytest.fixture
    def sample_data_2d(self):
        """Sample 2D training data."""
        np.random.seed(42)
        n_points = 25
        x = np.random.uniform(0, 1000, n_points)
        y = np.random.uniform(0, 1000, n_points)
        # Simple spatial function with some noise
        z = 0.001 * (x + y) + 0.0005 * x * y / 1000 + np.random.normal(0, 0.1, n_points)
        
        return pd.DataFrame({
            'X': x,
            'Y': y,
            'Value': z
        })
    
    @pytest.fixture
    def sample_data_3d(self):
        """Sample 3D training data."""
        np.random.seed(42)
        n_points = 20
        x = np.random.uniform(0, 500, n_points)
        y = np.random.uniform(0, 500, n_points)
        z_coord = np.random.uniform(0, 100, n_points)
        values = 0.002 * (x + y + z_coord) + np.random.normal(0, 0.05, n_points)
        
        return pd.DataFrame({
            'X': x,
            'Y': y,
            'Z': z_coord,
            'Value': values
        })
    
    def test_init_default(self):
        """Test Kriging interpolator initialization with defaults."""
        kriging = KrigingInterpolator()
        assert isinstance(kriging.search_params, SearchParameters)
        assert isinstance(kriging.kriging_params, KrigingParameters)
        assert kriging.kriging_params.kriging_type == KrigingType.ORDINARY
        assert kriging.kriging_params.variogram_model == VariogramModel.SPHERICAL
        assert not kriging.is_fitted
    
    def test_init_custom_params(self):
        """Test Kriging interpolator initialization with custom parameters."""
        search_params = SearchParameters(search_radius=500.0, max_points=20)
        kriging_params = KrigingParameters(
            kriging_type=KrigingType.SIMPLE,
            variogram_model=VariogramModel.EXPONENTIAL,
            nugget=0.2,
            sill=2.0,
            range_param=300.0
        )
        
        kriging = KrigingInterpolator(search_params, kriging_params)
        assert kriging.search_params.search_radius == 500.0
        assert kriging.kriging_params.kriging_type == KrigingType.SIMPLE
        assert kriging.kriging_params.variogram_model == VariogramModel.EXPONENTIAL
        assert kriging.kriging_params.nugget == 0.2
    
    def test_get_method_name(self):
        """Test getting method name."""
        kriging = KrigingInterpolator()
        name = kriging.get_method_name()
        assert "Ordinary Kriging" in name
        assert "Spherical" in name
        
        # Test with different parameters
        kriging.kriging_params.kriging_type = KrigingType.SIMPLE
        kriging.kriging_params.variogram_model = VariogramModel.EXPONENTIAL
        kriging.variogram_function = kriging._get_variogram_function()
        name = kriging.get_method_name()
        assert "Simple Kriging" in name
        assert "Exponential" in name
    
    def test_get_parameters(self):
        """Test getting parameters."""
        kriging_params = KrigingParameters(
            kriging_type=KrigingType.SIMPLE,
            variogram_model=VariogramModel.GAUSSIAN,
            nugget=0.15,
            sill=1.5,
            range_param=200.0,
            use_global=False
        )
        search_params = SearchParameters(search_radius=150.0, max_points=15)
        
        kriging = KrigingInterpolator(search_params, kriging_params)
        params = kriging.get_parameters()
        
        assert params['kriging_type'] == 'simple'
        assert params['variogram_model'] == 'gaussian'
        assert params['nugget'] == 0.15
        assert params['sill'] == 1.5
        assert params['range'] == 200.0
        assert params['use_global'] == False
        assert params['search_radius'] == 150.0
    
    def test_set_parameters(self):
        """Test setting parameters."""
        kriging = KrigingInterpolator()
        
        kriging.set_parameters(
            kriging_type='simple',
            variogram_model='exponential',
            nugget=0.25,
            sill=2.5,
            range_param=300.0,
            search_radius=400.0
        )
        
        assert kriging.kriging_params.kriging_type == KrigingType.SIMPLE
        assert kriging.kriging_params.variogram_model == VariogramModel.EXPONENTIAL
        assert kriging.kriging_params.nugget == 0.25
        assert kriging.kriging_params.sill == 2.5
        assert kriging.kriging_params.range_param == 300.0
        assert kriging.search_params.search_radius == 400.0
    
    def test_set_parameters_invalid_type(self):
        """Test setting invalid kriging type parameter."""
        kriging = KrigingInterpolator()
        
        with pytest.raises(ValueError, match="Invalid kriging type"):
            kriging.set_parameters(kriging_type='invalid_type')
    
    def test_set_parameters_invalid_model(self):
        """Test setting invalid variogram model parameter."""
        kriging = KrigingInterpolator()
        
        with pytest.raises(ValueError, match="Invalid variogram model"):
            kriging.set_parameters(variogram_model='invalid_model')
    
    def test_get_variogram_function(self):
        """Test getting variogram function."""
        kriging = KrigingInterpolator()
        
        # Test all variogram models
        model_tests = [
            (VariogramModel.SPHERICAL, VariogramModels.spherical),
            (VariogramModel.EXPONENTIAL, VariogramModels.exponential),
            (VariogramModel.GAUSSIAN, VariogramModels.gaussian),
            (VariogramModel.LINEAR, VariogramModels.linear),
            (VariogramModel.POWER, VariogramModels.power),
            (VariogramModel.NUGGET, VariogramModels.nugget),
        ]
        
        for model_enum, expected_func in model_tests:
            kriging.kriging_params.variogram_model = model_enum
            func = kriging._get_variogram_function()
            assert func == expected_func
    
    def test_fit_2d_data(self, sample_data_2d):
        """Test fitting with 2D data."""
        kriging = KrigingInterpolator()
        result = kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        assert result is kriging  # Method chaining
        assert kriging.is_fitted
        assert kriging.training_points.shape == (25, 2)
        assert kriging.training_values.shape == (25,)
        assert kriging.fitted_nugget is not None
        assert kriging.fitted_sill is not None
        assert kriging.fitted_range is not None
        assert kriging.data_mean is not None
    
    def test_fit_3d_data(self, sample_data_3d):
        """Test fitting with 3D data."""
        kriging = KrigingInterpolator()
        result = kriging.fit(sample_data_3d, 'X', 'Y', 'Value', z_col='Z')
        
        assert result is kriging
        assert kriging.is_fitted
        assert kriging.training_points.shape == (20, 3)
        assert kriging.coordinate_columns['Z'] == 'Z'
    
    def test_fit_without_auto_fit(self, sample_data_2d):
        """Test fitting without automatic variogram fitting."""
        kriging_params = KrigingParameters(
            auto_fit_variogram=False,
            nugget=0.1,
            sill=1.0,
            range_param=500.0
        )
        kriging = KrigingInterpolator(kriging_params=kriging_params)
        
        kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        assert kriging.is_fitted
        assert kriging.fitted_nugget == 0.1
        assert kriging.fitted_sill == 1.0
        assert kriging.fitted_range == 500.0
    
    def test_fit_invalid_data(self):
        """Test fitting with invalid data."""
        kriging = KrigingInterpolator()
        
        # Missing columns
        data = pd.DataFrame({'X': [1, 2], 'Y': [1, 2]})  # Missing 'Value'
        with pytest.raises(FittingError):
            kriging.fit(data, 'X', 'Y', 'Value')
        
        # Empty data
        empty_data = pd.DataFrame(columns=['X', 'Y', 'Value'])
        with pytest.raises(FittingError):
            kriging.fit(empty_data, 'X', 'Y', 'Value')
    
    def test_predict_not_fitted(self):
        """Test prediction before fitting."""
        kriging = KrigingInterpolator()
        points = np.array([[100, 100], [200, 200]])
        
        with pytest.raises(PredictionError, match="must be fitted"):
            kriging.predict(points)
    
    def test_predict_global_2d(self, sample_data_2d):
        """Test global prediction in 2D."""
        kriging = KrigingInterpolator()
        kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Predict at new points
        test_points = np.array([[500, 500], [250, 750]])
        predictions = kriging.predict(test_points)
        
        assert predictions.shape == (2,)
        assert np.all(np.isfinite(predictions))
    
    def test_predict_with_variance(self, sample_data_2d):
        """Test prediction with variance calculation."""
        kriging = KrigingInterpolator()
        kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        test_points = np.array([[500, 500], [250, 750]])
        predictions, variances = kriging.predict(test_points, return_variance=True)
        
        assert predictions.shape == (2,)
        assert variances.shape == (2,)
        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(variances))
        assert np.all(variances >= 0)  # Variance should be non-negative
    
    def test_predict_local_mode(self, sample_data_2d):
        """Test local prediction mode."""
        kriging_params = KrigingParameters(use_global=False)
        search_params = SearchParameters(search_radius=300.0, max_points=8)
        kriging = KrigingInterpolator(search_params, kriging_params)
        
        kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        test_points = np.array([[500, 500], [100, 100]])
        predictions = kriging.predict(test_points)
        
        assert predictions.shape == (2,)
        assert np.all(np.isfinite(predictions))
    
    def test_predict_different_input_formats(self, sample_data_2d):
        """Test prediction with different input formats."""
        kriging = KrigingInterpolator()
        kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Test with list of tuples
        points_list = [(500, 500), (250, 750)]
        pred1 = kriging.predict(points_list)
        
        # Test with numpy array
        points_array = np.array([[500, 500], [250, 750]])
        pred2 = kriging.predict(points_array)
        
        # Test with DataFrame
        points_df = pd.DataFrame({'X': [500, 250], 'Y': [500, 750]})
        pred3 = kriging.predict(points_df)
        
        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)
        np.testing.assert_allclose(pred1, pred3, rtol=1e-10)
    
    def test_different_kriging_types(self, sample_data_2d):
        """Test different kriging types."""
        # Ordinary kriging
        ok = KrigingInterpolator(kriging_params=KrigingParameters(kriging_type=KrigingType.ORDINARY))
        ok.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Simple kriging
        sk = KrigingInterpolator(kriging_params=KrigingParameters(
            kriging_type=KrigingType.SIMPLE,
            mean_value=sample_data_2d['Value'].mean()
        ))
        sk.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        test_point = np.array([[500, 500]])
        pred_ok = ok.predict(test_point)
        pred_sk = sk.predict(test_point)
        
        assert len(pred_ok) == 1
        assert len(pred_sk) == 1
        assert np.isfinite(pred_ok[0])
        assert np.isfinite(pred_sk[0])
    
    def test_different_variogram_models(self, sample_data_2d):
        """Test fitting and prediction with different variogram models."""
        models_to_test = [
            VariogramModel.SPHERICAL,
            VariogramModel.EXPONENTIAL,
            VariogramModel.GAUSSIAN,
            VariogramModel.LINEAR
        ]
        
        test_point = np.array([[500, 500]])
        
        for model in models_to_test:
            kriging_params = KrigingParameters(variogram_model=model)
            kriging = KrigingInterpolator(kriging_params=kriging_params)
            
            kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
            prediction = kriging.predict(test_point)
            
            assert len(prediction) == 1
            assert np.isfinite(prediction[0])
    
    def test_get_training_summary(self, sample_data_2d):
        """Test getting training summary."""
        kriging = KrigingInterpolator()
        
        # Before fitting
        summary = kriging.get_training_summary()
        assert summary == {}
        
        # After fitting
        kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
        summary = kriging.get_training_summary()
        
        assert 'n_points' in summary
        assert summary['n_points'] == 25
        assert 'fitted_nugget' in summary
        assert 'fitted_sill' in summary
        assert 'fitted_range' in summary
        assert 'kriging_type' in summary
        assert summary['kriging_type'] == 'ordinary'
        assert 'variogram_model' in summary
        assert summary['variogram_model'] == 'spherical'
    
    def test_get_variogram_values(self, sample_data_2d):
        """Test getting variogram values for plotting."""
        kriging = KrigingInterpolator()
        
        # Before fitting
        with pytest.raises(PredictionError, match="Must fit interpolator"):
            kriging.get_variogram_values()
        
        # After fitting
        kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
        distances, values = kriging.get_variogram_values(max_distance=1000.0, n_points=50)
        
        assert len(distances) == 50
        assert len(values) == 50
        assert distances[0] == 0.0
        assert distances[-1] == 1000.0
        assert values[0] == 0.0  # Variogram is 0 at distance 0
        assert np.all(np.isfinite(values))
    
    @patch('src.core.interpolation.kriging.minimize')
    def test_variogram_fitting_failure(self, mock_minimize, sample_data_2d):
        """Test fallback when variogram fitting fails."""
        # Mock optimization failure
        mock_result = MagicMock()
        mock_result.success = False
        mock_minimize.return_value = mock_result
        
        kriging = KrigingInterpolator()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kriging.fit(sample_data_2d, 'X', 'Y', 'Value')
        
        # Should still fit successfully with fallback parameters
        assert kriging.is_fitted
        assert kriging.fitted_nugget is not None
        assert kriging.fitted_sill is not None
        assert kriging.fitted_range is not None
    
    def test_singular_matrix_handling(self, sample_data_2d):
        """Test handling of singular kriging matrices."""
        # Create data with duplicate points to potentially cause singular matrix
        duplicate_data = sample_data_2d.copy()
        duplicate_data.loc[len(duplicate_data)] = duplicate_data.iloc[0]
        
        kriging = KrigingInterpolator()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kriging.fit(duplicate_data, 'X', 'Y', 'Value')
            
            # Should still work with fallback mechanisms
            test_point = np.array([[500, 500]])
            prediction = kriging.predict(test_point)
            
            assert len(prediction) == 1
            assert np.isfinite(prediction[0])
    
    def test_minimal_data(self):
        """Test with minimal dataset."""
        # Create minimal valid dataset
        data = pd.DataFrame({
            'X': [0, 1000, 500],
            'Y': [0, 0, 1000],
            'Value': [1.0, 2.0, 1.5]
        })
        
        kriging = KrigingInterpolator()
        kriging.fit(data, 'X', 'Y', 'Value')
        
        test_point = np.array([[250, 250]])
        prediction = kriging.predict(test_point)
        
        assert len(prediction) == 1
        assert np.isfinite(prediction[0])


class TestKrigingIntegration:
    """Integration tests for Kriging interpolator."""
    
    def test_exact_interpolation_property(self):
        """Test that Kriging provides exact interpolation at data points."""
        # Create simple regular dataset
        data = pd.DataFrame({
            'X': [0, 1000, 0, 1000],
            'Y': [0, 0, 1000, 1000],
            'Value': [1.0, 2.0, 3.0, 4.0]
        })
        
        kriging = KrigingInterpolator(kriging_params=KrigingParameters(auto_fit_variogram=False))
        kriging.fit(data, 'X', 'Y', 'Value')
        
        # Predict at training points
        train_points = data[['X', 'Y']].values
        predictions = kriging.predict(train_points)
        
        # Should be very close to training values (within numerical tolerance)
        np.testing.assert_allclose(predictions, data['Value'].values, rtol=1e-3)
    
    def test_uncertainty_quantification(self):
        """Test that kriging variance is reasonable."""
        # Create data with known spatial structure
        x = np.array([0, 500, 1000, 250, 750])
        y = np.array([0, 0, 0, 500, 500])
        values = x * 0.001 + y * 0.001  # Simple linear trend
        
        data = pd.DataFrame({'X': x, 'Y': y, 'Value': values})
        
        kriging = KrigingInterpolator()
        kriging.fit(data, 'X', 'Y', 'Value')
        
        # Predict at data points and between data points
        data_points = np.array([[0, 0], [1000, 0]])  # At data
        between_points = np.array([[500, 250]])      # Between data
        
        _, var_data = kriging.predict(data_points, return_variance=True)
        _, var_between = kriging.predict(between_points, return_variance=True)
        
        # Variance at data points should be lower (closer to nugget)
        # Variance between points should be higher
        assert np.all(var_data >= 0)
        assert np.all(var_between >= 0)
    
    def test_performance_reasonable_dataset(self):
        """Test performance with reasonable dataset size."""
        np.random.seed(42)
        n_points = 50
        
        data = pd.DataFrame({
            'X': np.random.uniform(0, 1000, n_points),
            'Y': np.random.uniform(0, 1000, n_points),
            'Value': np.random.uniform(0, 10, n_points)
        })
        
        kriging = KrigingInterpolator()
        
        # Should fit without issues
        import time
        start_time = time.time()
        kriging.fit(data, 'X', 'Y', 'Value')
        fit_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert fit_time < 10.0  # 10 seconds
        assert kriging.is_fitted
        
        # Test prediction
        test_points = np.random.uniform(0, 1000, (10, 2))
        start_time = time.time()
        predictions = kriging.predict(test_points)
        pred_time = time.time() - start_time
        
        assert pred_time < 2.0  # 2 seconds
        assert len(predictions) == 10
        assert np.all(np.isfinite(predictions))