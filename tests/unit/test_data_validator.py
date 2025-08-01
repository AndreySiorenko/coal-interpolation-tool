"""
Unit tests for data validation functionality.
"""

import pytest
import pandas as pd
import numpy as np

from src.io.validators import DataValidator, ValidationResult


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult()
        
        assert result.is_valid is True
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list) 
        assert isinstance(result.statistics, dict)
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error(self):
        """Test adding error messages."""
        result = ValidationResult()
        result.add_error("Test error")
        
        assert result.is_valid is False
        assert "Test error" in result.errors  
        assert len(result.errors) == 1
    
    def test_add_warning(self):
        """Test adding warning messages."""
        result = ValidationResult()
        result.add_warning("Test warning")
        
        assert result.is_valid is True  # Warnings don't invalidate
        assert "Test warning" in result.warnings
        assert len(result.warnings) == 1
    
    def test_has_issues(self):
        """Test issue detection."""
        result = ValidationResult()
        assert result.has_issues() is False
        
        result.add_warning("Warning")
        assert result.has_issues() is True
        
        result = ValidationResult()
        result.add_error("Error") 
        assert result.has_issues() is True


class TestDataValidator:
    """Test DataValidator class."""
    
    def test_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()
        assert validator.tolerance > 0
    
    def test_validate_dataset_valid_data(self, sample_coal_data):
        """Test validation of valid dataset."""
        validator = DataValidator()
        result = validator.validate_dataset(
            sample_coal_data, 'X', 'Y', 'ASH', 'Z'
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert 'total_rows' in result.statistics
        assert 'coordinates' in result.statistics
        assert 'values' in result.statistics
        assert 'quality' in result.statistics
    
    def test_validate_dataset_minimal_data(self, minimal_data):
        """Test validation of minimal dataset."""
        validator = DataValidator()
        result = validator.validate_dataset(
            minimal_data, 'X', 'Y', 'VALUE'
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        # Should have warnings about few data points
        assert len(result.warnings) > 0
    
    def test_validate_dataset_empty_data(self):
        """Test validation of empty dataset."""
        empty_df = pd.DataFrame()
        validator = DataValidator()
        result = validator.validate_dataset(
            empty_df, 'X', 'Y', 'VALUE'
        )
        
        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()
    
    def test_validate_dataset_missing_columns(self, sample_coal_data):
        """Test validation with missing required columns."""
        validator = DataValidator()
        result = validator.validate_dataset(
            sample_coal_data, 'X', 'MISSING_COL', 'ASH'
        )
        
        assert result.is_valid is False
        assert any("missing" in error.lower() for error in result.errors)
    
    def test_validate_dataset_insufficient_rows(self):
        """Test validation with insufficient data rows."""
        small_df = pd.DataFrame({
            'X': [100], 
            'Y': [200],
            'VALUE': [10]
        })
        
        validator = DataValidator()
        result = validator.validate_dataset(
            small_df, 'X', 'Y', 'VALUE'
        )
        
        assert result.is_valid is False
        assert any("insufficient" in error.lower() for error in result.errors)
    
    def test_validate_non_numeric_coordinates(self):
        """Test validation with non-numeric coordinate columns."""
        bad_df = pd.DataFrame({
            'X': ['A', 'B', 'C'],
            'Y': [100, 200, 300],
            'VALUE': [10, 20, 30]
        })
        
        validator = DataValidator()
        result = validator.validate_dataset(
            bad_df, 'X', 'Y', 'VALUE'
        )
        
        assert result.is_valid is False
        assert any("non-numeric" in error.lower() for error in result.errors)
    
    def test_validate_non_numeric_values(self):
        """Test validation with non-numeric values."""
        bad_df = pd.DataFrame({
            'X': [100, 200, 300],
            'Y': [100, 200, 300], 
            'VALUE': ['A', 'B', 'C']
        })
        
        validator = DataValidator()
        result = validator.validate_dataset(
            bad_df, 'X', 'Y', 'VALUE'
        )
        
        assert result.is_valid is False
        assert any("non-numeric" in error.lower() for error in result.errors)
    
    def test_validate_with_null_values(self):
        """Test validation with null values."""
        null_df = pd.DataFrame({
            'X': [100, np.nan, 300],
            'Y': [100, 200, 300],
            'VALUE': [10, 20, np.nan]
        })
        
        validator = DataValidator()
        result = validator.validate_dataset(
            null_df, 'X', 'Y', 'VALUE'
        )
        
        # Should have warnings about null values
        assert len(result.warnings) > 0
        assert any("null" in warning.lower() for warning in result.warnings)
    
    def test_validate_with_infinite_values(self):
        """Test validation with infinite values."""
        inf_df = pd.DataFrame({
            'X': [100, 200, 300],
            'Y': [100, np.inf, 300],
            'VALUE': [10, 20, 30]
        })
        
        validator = DataValidator()
        result = validator.validate_dataset(
            inf_df, 'X', 'Y', 'VALUE'
        )
        
        assert result.is_valid is False
        assert any("infinite" in error.lower() for error in result.errors)
    
    def test_validate_duplicate_coordinates(self):
        """Test validation with duplicate coordinates."""
        dup_df = pd.DataFrame({
            'X': [100, 200, 100],  # Duplicate coordinate
            'Y': [100, 200, 100],  # Duplicate coordinate
            'VALUE': [10, 20, 30]
        })
        
        validator = DataValidator()
        result = validator.validate_dataset(
            dup_df, 'X', 'Y', 'VALUE'
        )
        
        # Should have warning about duplicates
        assert len(result.warnings) > 0
        assert any("duplicate" in warning.lower() for warning in result.warnings)
    
    def test_validate_outliers_detection(self):
        """Test outlier detection in values."""
        # Create data with clear outliers
        outlier_df = pd.DataFrame({
            'X': list(range(10)),
            'Y': list(range(10)),
            'VALUE': [10, 11, 12, 13, 14, 15, 16, 17, 1000, 19]  # 1000 is outlier
        })
        
        validator = DataValidator()
        result = validator.validate_dataset(
            outlier_df, 'X', 'Y', 'VALUE'
        )
        
        # Should detect outliers
        assert 'values' in result.statistics
        if 'outlier_count' in result.statistics['values']:
            assert result.statistics['values']['outlier_count'] > 0
    
    def test_validate_constant_values(self):
        """Test validation with constant values."""
        const_df = pd.DataFrame({
            'X': [100, 200, 300],
            'Y': [100, 200, 300],
            'VALUE': [10, 10, 10]  # All same value
        })
        
        validator = DataValidator()
        result = validator.validate_dataset(
            const_df, 'X', 'Y', 'VALUE'
        )
        
        # Should warn about little variation
        assert len(result.warnings) > 0
        assert any("variation" in warning.lower() for warning in result.warnings)  
    
    def test_validate_spatial_distribution(self, sample_coal_data):
        """Test spatial distribution validation."""
        validator = DataValidator()
        result = validator.validate_dataset(
            sample_coal_data, 'X', 'Y', 'ASH'
        )
        
        assert 'spatial' in result.statistics
        spatial_stats = result.statistics['spatial']
        
        assert 'x_range' in spatial_stats
        assert 'y_range' in spatial_stats
        assert 'area' in spatial_stats
        assert 'point_density' in spatial_stats
        assert spatial_stats['x_range'] > 0
        assert spatial_stats['y_range'] > 0
    
    def test_quality_score_calculation(self, sample_coal_data):
        """Test data quality score calculation."""
        validator = DataValidator()
        result = validator.validate_dataset(
            sample_coal_data, 'X', 'Y', 'ASH'
        )
        
        assert 'quality' in result.statistics
        quality = result.statistics['quality']
        
        assert 'quality_score' in quality
        assert 'quality_level' in quality
        assert 'completeness_percentage' in quality
        assert 0 <= quality['quality_score'] <= 100
        assert quality['quality_level'] in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']
    
    def test_recommendations_generation(self, problematic_data):
        """Test generation of data improvement recommendations."""
        validator = DataValidator()
        result = validator.validate_dataset(
            problematic_data, 'X', 'Y', 'VALUE'
        )
        
        assert 'recommendations' in result.statistics
        recommendations = result.statistics['recommendations']
        assert isinstance(recommendations, list)
        # Should have recommendations for problematic data
        assert len(recommendations) > 0


class TestMethodSpecificValidation:
    """Test validation for specific interpolation methods."""
    
    def test_validate_for_kriging(self, sample_coal_data):
        """Test kriging-specific validation."""
        validator = DataValidator()
        result = validator.validate_for_interpolation_method(
            sample_coal_data, 'X', 'Y', 'ASH', 'kriging'
        )
        
        assert isinstance(result, ValidationResult)
        # Should not have method-specific errors for good data
        
    def test_validate_for_kriging_few_points(self, minimal_data):
        """Test kriging validation with few points."""
        validator = DataValidator()
        result = validator.validate_for_interpolation_method(
            minimal_data, 'X', 'Y', 'VALUE', 'kriging'
        )
        
        # Should warn about few points for kriging
        assert len(result.warnings) > 0
    
    def test_validate_for_idw(self, sample_coal_data):
        """Test IDW-specific validation."""
        validator = DataValidator()
        result = validator.validate_for_interpolation_method(
            sample_coal_data, 'X', 'Y', 'ASH', 'idw'
        )
        
        assert isinstance(result, ValidationResult)
    
    def test_validate_for_idw_extreme_outliers(self):
        """Test IDW validation with extreme outliers."""
        # Create data with extreme outliers
        outlier_df = pd.DataFrame({
            'X': list(range(10)),
            'Y': list(range(10)),
            'VALUE': [10, 11, 12, 13, 14, 15, 16, 17, 10000, 19]  # Extreme outlier
        })
        
        validator = DataValidator()
        result = validator.validate_for_interpolation_method(
            outlier_df, 'X', 'Y', 'VALUE', 'idw'
        )
        
        # Should warn about extreme outliers affecting IDW
        assert len(result.warnings) > 0
    
    def test_validate_for_rbf(self, sample_coal_data):
        """Test RBF-specific validation."""
        validator = DataValidator()
        result = validator.validate_for_interpolation_method(
            sample_coal_data, 'X', 'Y', 'ASH', 'rbf'
        )
        
        assert isinstance(result, ValidationResult)
    
    def test_validate_for_rbf_scale_differences(self):
        """Test RBF validation with large coordinate scale differences."""
        # Create data with very different X and Y scales
        scale_df = pd.DataFrame({
            'X': [1, 2, 3, 4, 5],
            'Y': [100000, 200000, 300000, 400000, 500000],  # Much larger scale
            'VALUE': [10, 20, 30, 40, 50]
        })
        
        validator = DataValidator()
        result = validator.validate_for_interpolation_method(
            scale_df, 'X', 'Y', 'VALUE', 'rbf'
        )
        
        # Should warn about scale differences
        assert len(result.warnings) > 0
    
    def test_validate_unknown_method(self, sample_coal_data):
        """Test validation with unknown interpolation method."""
        validator = DataValidator()
        result = validator.validate_for_interpolation_method(
            sample_coal_data, 'X', 'Y', 'ASH', 'unknown_method'
        )
        
        # Should still do general validation
        assert isinstance(result, ValidationResult)


class TestValidationStatistics:
    """Test validation statistics calculation."""
    
    def test_coordinate_statistics(self, sample_coal_data):
        """Test coordinate statistics calculation."""
        validator = DataValidator()
        result = validator.validate_dataset(
            sample_coal_data, 'X', 'Y', 'ASH', 'Z'
        )
        
        coord_stats = result.statistics['coordinates']
        
        for coord in ['X', 'Y', 'Z']:
            assert coord in coord_stats
            stats = coord_stats[coord]
            assert 'count' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert 'mean' in stats
            assert 'std' in stats
            assert 'range' in stats
            assert stats['range'] == stats['max'] - stats['min']
    
    def test_value_statistics(self, sample_coal_data):
        """Test value statistics calculation."""
        validator = DataValidator()
        result = validator.validate_dataset(
            sample_coal_data, 'X', 'Y', 'ASH'
        )
        
        value_stats = result.statistics['values']
        
        required_stats = ['count', 'min', 'max', 'mean', 'median', 'std', 'skewness', 'kurtosis']
        for stat in required_stats:
            assert stat in value_stats
        
        # Check statistical properties
        assert value_stats['min'] <= value_stats['mean'] <= value_stats['max']
        assert value_stats['std'] >= 0
    
    def test_spatial_statistics_calculation(self, sample_coal_data):
        """Test spatial statistics calculation."""
        validator = DataValidator()
        result = validator.validate_dataset(
            sample_coal_data, 'X', 'Y', 'ASH'
        )
        
        spatial_stats = result.statistics['spatial']
        
        assert spatial_stats['x_range'] > 0
        assert spatial_stats['y_range'] > 0
        assert spatial_stats['area'] > 0
        assert spatial_stats['point_density'] > 0
        assert 'mean_nearest_distance' in spatial_stats
        assert spatial_stats['mean_nearest_distance'] > 0