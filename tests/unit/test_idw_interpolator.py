"""
Unit tests for IDW interpolator.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from src.core.interpolation import IDWInterpolator
from src.core.interpolation.idw import IDWParameters
from src.core.interpolation.base import SearchParameters, FittingError, PredictionError


class TestIDWInterpolator:
    """Test suite for IDW interpolator."""
    
    @pytest.fixture
    def simple_2d_data(self):
        """Simple 2D test data."""
        data = pd.DataFrame({
            'X': [0, 1, 0, 1, 0.5],
            'Y': [0, 0, 1, 1, 0.5],
            'Z': [0, 1, 2, 3, 1.5]
        })
        return data
    
    @pytest.fixture
    def simple_3d_data(self):
        """Simple 3D test data."""
        data = pd.DataFrame({
            'X': [0, 1, 0, 1, 0, 1, 0, 1],
            'Y': [0, 0, 1, 1, 0, 0, 1, 1],
            'Z': [0, 0, 0, 0, 1, 1, 1, 1],
            'Value': [0, 1, 2, 3, 4, 5, 6, 7]
        })
        return data
    
    @pytest.fixture
    def coal_data(self):
        """Realistic coal deposit data."""
        np.random.seed(42)
        n_points = 50
        x = np.random.uniform(0, 1000, n_points)
        y = np.random.uniform(0, 1000, n_points)
        # Simulate ash content with spatial correlation
        ash = 15 + 5 * np.sin(x / 200) + 3 * np.cos(y / 300) + np.random.normal(0, 2, n_points)
        
        data = pd.DataFrame({
            'Easting': x,
            'Northing': y,
            'Ash_Content': ash
        })
        return data
    
    def test_idw_initialization(self):
        """Test IDW interpolator initialization."""
        # Default initialization
        idw = IDWInterpolator()
        assert idw.idw_params.power == 2.0
        assert idw.idw_params.smoothing == 0.0
        assert idw.search_params.search_radius == 1000.0
        
        # Custom parameters
        search_params = SearchParameters(search_radius=500, max_points=10)
        idw_params = IDWParameters(power=3.0, smoothing=0.1)
        idw = IDWInterpolator(search_params=search_params, idw_params=idw_params)
        
        assert idw.idw_params.power == 3.0
        assert idw.idw_params.smoothing == 0.1
        assert idw.search_params.search_radius == 500.0
        assert idw.search_params.max_points == 10
    
    def test_idw_fit(self, simple_2d_data):
        """Test fitting IDW interpolator."""
        idw = IDWInterpolator()
        
        # Fit the interpolator
        idw.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        
        assert idw.is_fitted
        assert idw.training_data is not None
        assert len(idw.training_data) == 5
        assert idw.coordinate_columns == {'X': 'X', 'Y': 'Y'}
        assert idw.value_column == 'Z'
        
        # Test fitting with additional parameters
        idw2 = IDWInterpolator()
        idw2.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z', power=3.0)
        assert idw2.idw_params.power == 3.0
    
    def test_idw_fit_3d(self, simple_3d_data):
        """Test fitting IDW interpolator with 3D data."""
        idw = IDWInterpolator()
        idw.fit(simple_3d_data, x_col='X', y_col='Y', value_col='Value', z_col='Z')
        
        assert idw.is_fitted
        assert idw.coordinate_columns == {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
        assert idw.training_points.shape == (8, 3)
    
    def test_idw_predict_single_point(self, simple_2d_data):
        """Test predicting a single point."""
        idw = IDWInterpolator()
        idw.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        
        # Predict at center point
        result = idw.predict([[0.5, 0.5]])
        assert len(result) == 1
        # Should be close to 1.5 (average of surrounding points)
        assert 1.0 < result[0] < 2.0
    
    def test_idw_predict_multiple_points(self, simple_2d_data):
        """Test predicting multiple points."""
        idw = IDWInterpolator()
        idw.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        
        # Predict at multiple points
        points = [[0.25, 0.25], [0.75, 0.75], [0.5, 0.0]]
        results = idw.predict(points)
        
        assert len(results) == 3
        assert all(not np.isnan(r) for r in results)
    
    def test_idw_exact_interpolation(self, simple_2d_data):
        """Test that IDW returns exact values at training points."""
        idw = IDWInterpolator()
        idw.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        
        # Predict at training points
        training_points = simple_2d_data[['X', 'Y']].values
        results = idw.predict(training_points)
        expected = simple_2d_data['Z'].values
        
        assert_array_almost_equal(results, expected, decimal=10)
    
    def test_idw_power_parameter(self, simple_2d_data):
        """Test effect of power parameter."""
        # Lower power (more smoothing)
        idw1 = IDWInterpolator(idw_params=IDWParameters(power=1.0))
        idw1.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        result1 = idw1.predict([[0.5, 0.5]])
        
        # Higher power (less smoothing, more local)
        idw2 = IDWInterpolator(idw_params=IDWParameters(power=4.0))
        idw2.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        result2 = idw2.predict([[0.5, 0.5]])
        
        # Results should be different
        assert abs(result1[0] - result2[0]) > 0.01
    
    def test_idw_search_radius(self, coal_data):
        """Test search radius parameter."""
        idw = IDWInterpolator(search_params=SearchParameters(search_radius=100))
        idw.fit(coal_data, x_col='Easting', y_col='Northing', value_col='Ash_Content')
        
        # Predict at a point far from all training data
        result = idw.predict([[2000, 2000]])
        # Should use nearest points even if outside radius
        assert not np.isnan(result[0])
    
    def test_idw_max_points_limit(self, coal_data):
        """Test maximum points limitation."""
        # Use only 3 nearest points
        idw = IDWInterpolator(search_params=SearchParameters(max_points=3))
        idw.fit(coal_data, x_col='Easting', y_col='Northing', value_col='Ash_Content')
        
        result = idw.predict([[500, 500]])
        assert not np.isnan(result[0])
    
    def test_idw_sectoral_search(self, simple_2d_data):
        """Test sectoral search functionality."""
        search_params = SearchParameters(
            use_sectors=True,
            n_sectors=4,
            max_per_sector=1
        )
        idw = IDWInterpolator(search_params=search_params)
        idw.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        
        # Predict at center - should use one point from each quadrant
        result = idw.predict([[0.5, 0.5]])
        assert not np.isnan(result[0])
    
    def test_idw_anisotropy(self, coal_data):
        """Test anisotropy support."""
        # Create anisotropic search
        search_params = SearchParameters(
            anisotropy_ratio=0.5,  # Compress in one direction
            anisotropy_angle=45    # Rotate 45 degrees
        )
        idw = IDWInterpolator(search_params=search_params)
        idw.fit(coal_data, x_col='Easting', y_col='Northing', value_col='Ash_Content')
        
        result = idw.predict([[500, 500]])
        assert not np.isnan(result[0])
    
    def test_idw_predict_grid(self, simple_2d_data):
        """Test grid prediction."""
        idw = IDWInterpolator()
        idw.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        
        # Create a grid
        X, Y, Z = idw.predict_grid(
            x_range=(0, 1),
            y_range=(0, 1),
            grid_size=10
        )
        
        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert Z.shape == (10, 10)
        assert not np.any(np.isnan(Z))
    
    def test_idw_with_missing_data(self):
        """Test handling of missing data."""
        data = pd.DataFrame({
            'X': [0, 1, 0, 1, np.nan],
            'Y': [0, 0, 1, np.nan, 0.5],
            'Z': [0, 1, 2, 3, 1.5]
        })
        
        idw = IDWInterpolator()
        with pytest.warns(UserWarning, match="Removed 2 rows containing NaN"):
            idw.fit(data, x_col='X', y_col='Y', value_col='Z')
        
        assert len(idw.training_data) == 3
    
    def test_idw_with_duplicate_coordinates(self):
        """Test handling of duplicate coordinates."""
        data = pd.DataFrame({
            'X': [0, 1, 0, 1, 0],  # Duplicate (0, 0)
            'Y': [0, 0, 1, 1, 0],
            'Z': [0, 1, 2, 3, 4]
        })
        
        idw = IDWInterpolator()
        with pytest.warns(UserWarning, match="Found 1 duplicate coordinates"):
            idw.fit(data, x_col='X', y_col='Y', value_col='Z')
        
        assert len(idw.training_data) == 4
    
    def test_idw_insufficient_data(self):
        """Test error with insufficient data."""
        data = pd.DataFrame({
            'X': [0, 1],
            'Y': [0, 0],
            'Z': [0, 1]
        })
        
        idw = IDWInterpolator()
        with pytest.raises(FittingError, match="Insufficient data points"):
            idw.fit(data, x_col='X', y_col='Y', value_col='Z')
    
    def test_idw_predict_before_fit(self):
        """Test prediction before fitting."""
        idw = IDWInterpolator()
        
        with pytest.raises(PredictionError, match="must be fitted"):
            idw.predict([[0, 0]])
    
    def test_idw_smoothing_parameter(self):
        """Test smoothing parameter to avoid division by zero."""
        data = pd.DataFrame({
            'X': [0, 1, 0, 1],
            'Y': [0, 0, 1, 1],
            'Z': [0, 1, 2, 3]
        })
        
        # With smoothing
        idw = IDWInterpolator(idw_params=IDWParameters(smoothing=0.1))
        idw.fit(data, x_col='X', y_col='Y', value_col='Z')
        
        # Predict exactly at a training point
        result = idw.predict([[0, 0]])
        # Should be very close but not exactly 0 due to smoothing
        assert abs(result[0] - 0.0) < 0.1
    
    def test_idw_performance_large_dataset(self):
        """Test performance with larger dataset."""
        # Create larger dataset
        np.random.seed(42)
        n_points = 1000
        data = pd.DataFrame({
            'X': np.random.uniform(0, 1000, n_points),
            'Y': np.random.uniform(0, 1000, n_points),
            'Z': np.random.uniform(0, 100, n_points)
        })
        
        idw = IDWInterpolator()
        idw.fit(data, x_col='X', y_col='Y', value_col='Z')
        
        # Predict on a grid
        import time
        start_time = time.time()
        X, Y, Z = idw.predict_grid(
            x_range=(0, 1000),
            y_range=(0, 1000),
            grid_size=50
        )
        elapsed = time.time() - start_time
        
        assert Z.shape == (50, 50)
        # Should complete in reasonable time
        assert elapsed < 10.0  # 10 seconds max
    
    def test_idw_get_set_parameters(self, simple_2d_data):
        """Test getting and setting parameters."""
        idw = IDWInterpolator()
        idw.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        
        # Get parameters
        params = idw.get_parameters()
        assert params['power'] == 2.0
        assert params['search_radius'] == 1000.0
        
        # Set parameters
        idw.set_parameters(power=3.0, search_radius=500.0)
        params = idw.get_parameters()
        assert params['power'] == 3.0
        assert params['search_radius'] == 500.0
    
    def test_idw_method_name(self):
        """Test method name."""
        idw = IDWInterpolator()
        assert idw.get_method_name() == "Inverse Distance Weighted (IDW)"
    
    def test_idw_training_summary(self, simple_2d_data):
        """Test training data summary."""
        idw = IDWInterpolator()
        idw.fit(simple_2d_data, x_col='X', y_col='Y', value_col='Z')
        
        summary = idw.get_training_summary()
        assert summary['n_points'] == 5
        assert summary['method'] == "Inverse Distance Weighted (IDW)"
        assert 'X_range' in summary
        assert 'Y_range' in summary
        assert 'value_mean' in summary