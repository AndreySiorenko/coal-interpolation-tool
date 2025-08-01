"""
Unit tests for DataAnalyzer.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Mock imports for testing without dependencies
try:
    from src.core.recommendations.data_analyzer import DataAnalyzer, DataCharacteristics
except ImportError:
    pytest.skip("Recommendation modules not available", allow_module_level=True)


class TestDataAnalyzer:
    """Test cases for DataAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DataAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        n_points = 50
        
        # Generate coordinates
        x = np.random.uniform(0, 1000, n_points)
        y = np.random.uniform(0, 1000, n_points)
        
        # Generate values with some spatial correlation
        values = np.sin(x/200) + np.cos(y/200) + np.random.normal(0, 0.1, n_points)
        
        self.test_data = pd.DataFrame({
            'X': x,
            'Y': y, 
            'Value': values
        })
    
    def test_initialization(self):
        """Test DataAnalyzer initialization."""
        analyzer = DataAnalyzer()
        assert analyzer.data is None
        assert analyzer.coordinates is None
        assert analyzer.values is None
        assert analyzer.kdtree is None
    
    def test_analyze_basic(self):
        """Test basic analysis functionality."""
        result = self.analyzer.analyze(
            self.test_data, 'X', 'Y', 'Value'
        )
        
        assert isinstance(result, DataCharacteristics)
        assert result.n_points == 50
        assert result.dimensions == 2
        assert 'X' in result.bounds
        assert 'Y' in result.bounds
        assert result.density > 0
        assert 0 <= result.distribution_uniformity <= 1
        assert isinstance(result.has_trend, bool)
        assert result.anisotropy_ratio > 0
        assert 0 <= result.anisotropy_angle < 360
        assert isinstance(result.statistics, dict)
        assert isinstance(result.outlier_indices, list)
        assert 0 <= result.clustering_score <= 1
        assert isinstance(result.nearest_neighbor_stats, dict)
    
    def test_analyze_with_3d_data(self):
        """Test analysis with 3D data."""
        # Add Z column
        self.test_data['Z'] = np.random.uniform(0, 100, len(self.test_data))
        
        result = self.analyzer.analyze(
            self.test_data, 'X', 'Y', 'Value', 'Z'
        )
        
        assert result.dimensions == 3
        assert 'Z' in result.bounds
    
    def test_calculate_bounds(self):
        """Test bounds calculation."""
        self.analyzer.coordinates = self.test_data[['X', 'Y']].values
        bounds = self.analyzer._calculate_bounds(['X', 'Y'])
        
        assert 'X' in bounds
        assert 'Y' in bounds
        assert bounds['X'][0] < bounds['X'][1]  # min < max
        assert bounds['Y'][0] < bounds['Y'][1]  # min < max
    
    def test_calculate_density(self):
        """Test density calculation."""
        bounds = {'X': (0, 1000), 'Y': (0, 1000)}
        self.analyzer.data = self.test_data
        self.analyzer.dimensions = 2
        
        density = self.analyzer._calculate_density(bounds)
        
        expected_density = len(self.test_data) / (1000 * 1000)
        assert density == expected_density
    
    def test_analyze_distribution(self):
        """Test distribution uniformity analysis."""
        self.analyzer.coordinates = self.test_data[['X', 'Y']].values
        self.analyzer.data = self.test_data
        self.analyzer.dimensions = 2
        
        uniformity = self.analyzer._analyze_distribution()
        
        assert 0 <= uniformity <= 1
        assert isinstance(uniformity, float)
    
    def test_detect_trends(self):
        """Test trend detection."""
        self.analyzer.coordinates = self.test_data[['X', 'Y']].values
        self.analyzer.values = self.test_data['Value'].values
        self.analyzer.data = self.test_data
        self.analyzer.dimensions = 2
        
        trend_info = self.analyzer._detect_trends()
        
        assert 'has_trend' in trend_info
        assert 'trend_type' in trend_info
        assert 'r_squared' in trend_info
        assert isinstance(trend_info['has_trend'], bool)
        assert trend_info['trend_type'] in [None, 'linear', 'quadratic']
        assert 0 <= trend_info['r_squared'] <= 1
    
    def test_analyze_anisotropy(self):
        """Test anisotropy analysis."""
        self.analyzer.coordinates = self.test_data[['X', 'Y']].values
        self.analyzer.dimensions = 2
        self.analyzer.data = self.test_data
        
        ratio, angle = self.analyzer._analyze_anisotropy()
        
        assert 0 < ratio <= 1
        assert 0 <= angle < 180
    
    def test_calculate_statistics(self):
        """Test statistical calculations."""
        self.analyzer.values = self.test_data['Value'].values
        
        stats = self.analyzer._calculate_statistics()
        
        required_keys = ['mean', 'std', 'min', 'max', 'median', 
                        'q25', 'q75', 'iqr', 'cv', 'skewness', 'kurtosis']
        
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        # Add some obvious outliers
        outlier_data = self.test_data.copy()
        outlier_data.loc[0, 'Value'] = 1000  # Large outlier
        outlier_data.loc[1, 'Value'] = -1000  # Large outlier
        
        self.analyzer.values = outlier_data['Value'].values
        
        outliers = self.analyzer._detect_outliers()
        
        assert isinstance(outliers, list)
        assert 0 in outliers or 1 in outliers  # Should detect at least one outlier
    
    def test_analyze_clustering(self):
        """Test clustering analysis."""
        from scipy.spatial import cKDTree
        
        self.analyzer.coordinates = self.test_data[['X', 'Y']].values
        self.analyzer.data = self.test_data
        self.analyzer.dimensions = 2
        self.analyzer.kdtree = cKDTree(self.analyzer.coordinates)
        
        clustering = self.analyzer._analyze_clustering()
        
        assert 0 <= clustering <= 1
        assert isinstance(clustering, float)
    
    def test_nearest_neighbor_analysis(self):
        """Test nearest neighbor analysis."""
        from scipy.spatial import cKDTree
        
        self.analyzer.coordinates = self.test_data[['X', 'Y']].values
        self.analyzer.data = self.test_data
        self.analyzer.kdtree = cKDTree(self.analyzer.coordinates)
        
        nn_stats = self.analyzer._nearest_neighbor_analysis()
        
        required_keys = ['mean_distance', 'std_distance', 'min_distance', 'max_distance']
        for key in required_keys:
            assert key in nn_stats
            assert nn_stats[key] >= 0
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame(columns=['X', 'Y', 'Value'])
        
        with pytest.raises(Exception):  # Should raise some kind of error
            self.analyzer.analyze(empty_data, 'X', 'Y', 'Value')
    
    def test_invalid_columns(self):
        """Test handling of invalid column names."""
        with pytest.raises(Exception):
            self.analyzer.analyze(
                self.test_data, 'NonExistent', 'Y', 'Value'
            )
    
    @patch('scipy.spatial.cKDTree')
    def test_analyze_with_mock_kdtree(self, mock_kdtree):
        """Test analysis with mocked KDTree."""
        # Setup mock
        mock_tree = MagicMock()
        mock_tree.query.return_value = (
            np.array([[1.0, 2.0]]),  # distances
            np.array([[1, 2]])       # indices
        )
        mock_kdtree.return_value = mock_tree
        
        result = self.analyzer.analyze(
            self.test_data, 'X', 'Y', 'Value'
        )
        
        assert isinstance(result, DataCharacteristics)
        mock_kdtree.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])