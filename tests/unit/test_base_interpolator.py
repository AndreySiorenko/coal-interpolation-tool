"""
Unit tests for base interpolator functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.core.interpolation.base import (
    BaseInterpolator, SearchParameters, InterpolationParameters,
    InterpolationError, FittingError, PredictionError
)


class MockInterpolator(BaseInterpolator):
    """Mock interpolator for testing base functionality."""
    
    def fit(self, data, x_col, y_col, value_col, z_col=None, **kwargs):
        # Validate data using parent method
        clean_data = self._validate_training_data(data, x_col, y_col, value_col, z_col)
        
        # Store training data and parameters
        self.training_data = clean_data
        self.coordinate_columns = {'X': x_col, 'Y': y_col}
        if z_col:
            self.coordinate_columns['Z'] = z_col
        self.value_column = value_col
        self.is_fitted = True
        
        return self
    
    def predict(self, points, **kwargs):
        if not self.is_fitted:
            raise PredictionError("Must fit before prediction")
        
        # Parse points using parent method
        point_array = self._parse_prediction_points(points)
        
        # Mock prediction: return mean of training values
        mean_value = self.training_data[self.value_column].mean()
        return np.full(len(point_array), mean_value)
    
    def get_method_name(self):
        return "Mock Interpolator"
    
    def get_parameters(self):
        return {
            'search_radius': self.search_params.search_radius,
            'max_points': self.search_params.max_points,
            'min_points': self.search_params.min_points,
        }


class TestSearchParameters:
    """Test SearchParameters dataclass."""
    
    def test_default_initialization(self):
        """Test default parameter values."""
        params = SearchParameters()
        
        assert params.search_radius == 1000.0
        assert params.min_points == 1
        assert params.max_points == 12
        assert params.use_sectors is False
        assert params.n_sectors == 4
        assert params.anisotropy_ratio == 1.0
        assert params.anisotropy_angle == 0.0
    
    def test_custom_initialization(self):
        """Test initialization with custom values."""
        params = SearchParameters(
            search_radius=500.0,
            max_points=8,
            use_sectors=True,
            anisotropy_ratio=2.0
        )
        
        assert params.search_radius == 500.0
        assert params.max_points == 8
        assert params.use_sectors is True
        assert params.anisotropy_ratio == 2.0


class TestBaseInterpolator:
    """Test BaseInterpolator base class."""
    
    def test_initialization(self):
        """Test interpolator initialization."""
        interpolator = MockInterpolator()
        
        assert interpolator.is_fitted is False
        assert interpolator.training_data is None
        assert isinstance(interpolator.search_params, SearchParameters)
        assert isinstance(interpolator.coordinate_columns, dict)
        assert interpolator.value_column is None
        assert isinstance(interpolator.metadata, dict)
    
    def test_initialization_with_search_params(self):
        """Test initialization with custom search parameters."""
        custom_params = SearchParameters(search_radius=500.0, max_points=8)
        interpolator = MockInterpolator(search_params=custom_params)
        
        assert interpolator.search_params.search_radius == 500.0
        assert interpolator.search_params.max_points == 8
    
    def test_set_parameters(self):
        """Test parameter setting."""
        interpolator = MockInterpolator()
        
        result = interpolator.set_parameters(search_radius=800.0, max_points=10)
        
        assert result is interpolator  # Should return self for chaining
        assert interpolator.search_params.search_radius == 800.0
        assert interpolator.search_params.max_points == 10
    
    def test_validate_training_data_valid(self, sample_coal_data):
        """Test data validation with valid data."""
        interpolator = MockInterpolator()
        
        validated = interpolator._validate_training_data(
            sample_coal_data, 'X', 'Y', 'ASH', 'Z'
        )
        
        assert isinstance(validated, pd.DataFrame)
        assert len(validated) > 0
        assert 'X' in validated.columns
        assert 'Y' in validated.columns
        assert 'ASH' in validated.columns
        assert 'Z' in validated.columns
    
    def test_validate_training_data_missing_columns(self, sample_coal_data):
        """Test data validation with missing columns."""
        interpolator = MockInterpolator()
        
        with pytest.raises(FittingError, match="Missing columns"):
            interpolator._validate_training_data(
                sample_coal_data, 'X', 'MISSING', 'ASH'
            )
    
    def test_validate_training_data_non_numeric(self):
        """Test data validation with non-numeric columns."""
        bad_data = pd.DataFrame({
            'X': [100, 200, 300],
            'Y': [100, 200, 300],
            'VALUE': ['A', 'B', 'C']  # Non-numeric
        })
        
        interpolator = MockInterpolator()
        
        with pytest.raises(FittingError):
            interpolator._validate_training_data(bad_data, 'X', 'Y', 'VALUE')
    
    def test_validate_training_data_removes_nan(self):
        """Test that NaN values are removed during validation."""
        data_with_nan = pd.DataFrame({
            'X': [100, 200, np.nan, 400],
            'Y': [100, 200, 300, 400],
            'VALUE': [10, 20, 30, np.nan]
        })
        
        interpolator = MockInterpolator()
        validated = interpolator._validate_training_data(
            data_with_nan, 'X', 'Y', 'VALUE'
        )
        
        # Should remove rows with any NaN values
        assert len(validated) == 2  # Only first two rows are complete
        assert not validated.isnull().any().any()
    
    def test_validate_training_data_insufficient_points(self):
        """Test validation with too few data points."""
        minimal_data = pd.DataFrame({
            'X': [100],
            'Y': [200], 
            'VALUE': [10]
        })
        
        interpolator = MockInterpolator()
        
        with pytest.raises(FittingError, match="Insufficient data points"):
            interpolator._validate_training_data(minimal_data, 'X', 'Y', 'VALUE')
    
    def test_validate_training_data_removes_duplicates(self):
        """Test that duplicate coordinates are removed."""
        data_with_duplicates = pd.DataFrame({
            'X': [100, 200, 100, 300],  # Duplicate X,Y at 100,100
            'Y': [100, 200, 100, 300],
            'VALUE': [10, 20, 15, 30]  # Different value for duplicate coord
        })
        
        interpolator = MockInterpolator()
        
        with pytest.warns(UserWarning, match="duplicate coordinates"):
            validated = interpolator._validate_training_data(
                data_with_duplicates, 'X', 'Y', 'VALUE'
            )
        
        # Should keep only first occurrence of duplicate
        assert len(validated) == 3
        duplicate_removed = validated[(validated['X'] == 100) & (validated['Y'] == 100)]
        assert len(duplicate_removed) == 1
        assert duplicate_removed.iloc[0]['VALUE'] == 10  # First occurrence
    
    def test_parse_prediction_points_dataframe(self, sample_coal_data):
        """Test parsing DataFrame prediction points."""
        interpolator = MockInterpolator()
        # Need to set coordinate columns first
        interpolator.coordinate_columns = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
        
        points_df = sample_coal_data[['X', 'Y', 'Z']].head(5)
        parsed = interpolator._parse_prediction_points(points_df)
        
        assert isinstance(parsed, np.ndarray)
        assert parsed.shape[0] == 5
        assert parsed.shape[1] == 3  # X, Y, Z
    
    def test_parse_prediction_points_numpy_array(self):
        """Test parsing numpy array prediction points."""
        interpolator = MockInterpolator()
        
        points_array = np.array([[100, 200], [300, 400], [500, 600]])
        parsed = interpolator._parse_prediction_points(points_array)
        
        assert isinstance(parsed, np.ndarray)
        assert np.array_equal(parsed, points_array)
    
    def test_parse_prediction_points_list(self):
        """Test parsing list of tuples prediction points."""
        interpolator = MockInterpolator()
        
        points_list = [(100, 200), (300, 400), (500, 600)]
        parsed = interpolator._parse_prediction_points(points_list)
        
        assert isinstance(parsed, np.ndarray)
        assert parsed.shape == (3, 2)
        assert parsed[0, 0] == 100
        assert parsed[0, 1] == 200
    
    def test_parse_prediction_points_invalid_type(self):
        """Test parsing invalid prediction points type."""
        interpolator = MockInterpolator()
        
        with pytest.raises(PredictionError, match="Unsupported points type"):
            interpolator._parse_prediction_points("invalid")
    
    def test_parse_prediction_points_missing_columns(self):
        """Test parsing DataFrame without required coordinate columns."""
        interpolator = MockInterpolator()
        # Don't set coordinate columns - should raise error
        
        points_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        with pytest.raises(PredictionError, match="Coordinate column names not set"):
            interpolator._parse_prediction_points(points_df)
    
    def test_find_neighbors_2d(self):
        """Test neighbor finding in 2D."""
        interpolator = MockInterpolator()
        interpolator.search_params.search_radius = 300.0
        interpolator.search_params.max_points = 3
        
        training_points = np.array([
            [100, 100],
            [200, 200], 
            [300, 300],
            [1000, 1000]  # Far away point
        ])
        
        target_point = np.array([150, 150])
        
        indices, distances = interpolator._find_neighbors(target_point, training_points)
        
        assert len(indices) <= 3  # Respects max_points
        assert len(indices) == len(distances)
        assert all(dist <= 300.0 or len(indices) < interpolator.search_params.min_points 
                  for dist in distances)  # Within radius or using closest
    
    def test_find_neighbors_3d(self):
        """Test neighbor finding in 3D."""
        interpolator = MockInterpolator()
        interpolator.search_params.search_radius = 500.0
        
        training_points = np.array([
            [100, 100, 50],
            [200, 200, 60],
            [300, 300, 70]
        ])
        
        target_point = np.array([150, 150, 55])
        
        indices, distances = interpolator._find_neighbors(target_point, training_points)
        
        assert len(indices) > 0
        assert len(indices) == len(distances)
        # Check 3D distance calculation
        expected_dist_0 = np.sqrt((150-100)**2 + (150-100)**2 + (55-50)**2)
        assert abs(distances[0] - expected_dist_0) < 1e-10
    
    def test_fit_basic_functionality(self, sample_coal_data):
        """Test fitting functionality."""
        interpolator = MockInterpolator()
        
        result = interpolator.fit(sample_coal_data, 'X', 'Y', 'ASH', 'Z')
        
        assert result is interpolator  # Should return self
        assert interpolator.is_fitted is True
        assert interpolator.training_data is not None
        assert interpolator.coordinate_columns['X'] == 'X'
        assert interpolator.coordinate_columns['Y'] == 'Y'
        assert interpolator.coordinate_columns['Z'] == 'Z'
        assert interpolator.value_column == 'ASH'
    
    def test_predict_basic_functionality(self, sample_coal_data):
        """Test prediction functionality."""
        interpolator = MockInterpolator()
        interpolator.fit(sample_coal_data, 'X', 'Y', 'ASH')
        
        prediction_points = np.array([[101000, 201000], [102000, 202000]])
        predictions = interpolator.predict(prediction_points)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2
        assert not np.isnan(predictions).any()
    
    def test_predict_without_fitting(self):
        """Test prediction without fitting raises error."""
        interpolator = MockInterpolator()
        
        with pytest.raises(PredictionError, match="Must fit before prediction"):
            interpolator.predict([[100, 200]])
    
    def test_predict_grid(self, sample_coal_data):
        """Test grid prediction functionality."""
        interpolator = MockInterpolator()
        interpolator.fit(sample_coal_data, 'X', 'Y', 'ASH')
        
        x_range = (100000, 102000)
        y_range = (200000, 202000)
        grid_size = 5
        
        X, Y, Z = interpolator.predict_grid(x_range, y_range, grid_size)
        
        assert X.shape == (5, 5)
        assert Y.shape == (5, 5)
        assert Z.shape == (5, 5)
        assert X.min() == x_range[0]
        assert X.max() == x_range[1]
        assert Y.min() == y_range[0]
        assert Y.max() == y_range[1]
    
    def test_predict_grid_with_z(self, sample_coal_data):
        """Test grid prediction with fixed Z value."""
        interpolator = MockInterpolator()
        interpolator.fit(sample_coal_data, 'X', 'Y', 'ASH', 'Z')
        
        x_range = (100000, 102000)
        y_range = (200000, 202000)
        grid_size = 3
        z_value = 100.0
        
        X, Y, Z = interpolator.predict_grid(x_range, y_range, grid_size, z_value)
        
        assert X.shape == (3, 3)
        assert Y.shape == (3, 3)
        assert Z.shape == (3, 3)
    
    def test_predict_grid_different_dimensions(self, sample_coal_data):
        """Test grid prediction with different X and Y dimensions."""
        interpolator = MockInterpolator()
        interpolator.fit(sample_coal_data, 'X', 'Y', 'ASH')
        
        x_range = (100000, 102000)
        y_range = (200000, 202000)
        grid_size = (3, 4)  # 3x4 grid
        
        X, Y, Z = interpolator.predict_grid(x_range, y_range, grid_size)
        
        assert X.shape == (4, 3)  # Note: meshgrid creates (ny, nx) 
        assert Y.shape == (4, 3)
        assert Z.shape == (4, 3)
    
    def test_predict_grid_without_fitting(self):
        """Test grid prediction without fitting raises error."""
        interpolator = MockInterpolator()
        
        with pytest.raises(PredictionError, match="must be fitted"):
            interpolator.predict_grid((0, 100), (0, 100), 5)
    
    def test_get_training_summary(self, sample_coal_data):
        """Test training data summary."""
        interpolator = MockInterpolator()
        interpolator.fit(sample_coal_data, 'X', 'Y', 'ASH', 'Z')
        
        summary = interpolator.get_training_summary()
        
        assert 'n_points' in summary
        assert 'coordinate_columns' in summary
        assert 'value_column' in summary
        assert 'method' in summary
        assert 'parameters' in summary
        assert summary['n_points'] > 0
        assert summary['method'] == "Mock Interpolator"
        assert summary['value_column'] == 'ASH'
    
    def test_get_training_summary_unfitted(self):
        """Test training summary when not fitted."""
        interpolator = MockInterpolator()
        
        summary = interpolator.get_training_summary()
        
        assert summary == {}
    
    def test_get_method_name(self):
        """Test method name retrieval."""
        interpolator = MockInterpolator()
        
        assert interpolator.get_method_name() == "Mock Interpolator"
    
    def test_get_parameters(self):
        """Test parameter retrieval."""
        interpolator = MockInterpolator()
        interpolator.set_parameters(search_radius=800.0, max_points=10)
        
        params = interpolator.get_parameters()
        
        assert isinstance(params, dict)
        assert params['search_radius'] == 800.0
        assert params['max_points'] == 10


class TestInterpolationExceptions:
    """Test interpolation-specific exceptions."""
    
    def test_interpolation_error(self):
        """Test base InterpolationError."""
        with pytest.raises(InterpolationError):
            raise InterpolationError("Test error")
    
    def test_fitting_error(self):
        """Test FittingError inheritance."""
        with pytest.raises(InterpolationError):
            raise FittingError("Fitting failed")
        
        with pytest.raises(FittingError):
            raise FittingError("Fitting failed")
    
    def test_prediction_error(self):
        """Test PredictionError inheritance."""
        with pytest.raises(InterpolationError):
            raise PredictionError("Prediction failed")
        
        with pytest.raises(PredictionError):
            raise PredictionError("Prediction failed")