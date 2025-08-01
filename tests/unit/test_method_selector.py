"""
Unit tests for MethodSelector.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

# Mock imports for testing without dependencies
try:
    from src.core.recommendations.method_selector import (
        MethodSelector, MethodScore, InterpolationMethod
    )
    from src.core.recommendations.data_analyzer import DataCharacteristics
except ImportError:
    pytest.skip("Recommendation modules not available", allow_module_level=True)


class TestMethodSelector:
    """Test cases for MethodSelector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = MethodSelector()
        
        # Create mock data characteristics
        self.mock_characteristics = DataCharacteristics(
            n_points=100,
            dimensions=2,
            bounds={'X': (0, 1000), 'Y': (0, 1000)},
            density=0.1,
            distribution_uniformity=0.7,
            has_trend=False,
            trend_type=None,
            anisotropy_ratio=0.9,
            anisotropy_angle=45.0,
            statistics={
                'mean': 10.0,
                'std': 2.0,
                'cv': 0.2,
                'skewness': 0.1,
                'kurtosis': 0.0
            },
            outlier_indices=[],
            clustering_score=0.3,
            nearest_neighbor_stats={'mean_distance': 100.0}
        )
    
    def test_initialization(self):
        """Test MethodSelector initialization."""
        selector = MethodSelector()
        assert selector.characteristics is None
        assert selector.scores == {}
    
    def test_recommend_method_basic(self):
        """Test basic method recommendation."""
        recommendations = self.selector.recommend_method(self.mock_characteristics)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) == 3  # IDW, Kriging, RBF
        
        # Check that all methods are evaluated
        methods = [rec.method for rec in recommendations]
        assert InterpolationMethod.IDW in methods
        assert InterpolationMethod.ORDINARY_KRIGING in methods
        assert InterpolationMethod.RBF in methods
        
        # Check that results are sorted by score (descending)
        scores = [rec.score for rec in recommendations]
        assert scores == sorted(scores, reverse=True)
    
    def test_recommend_method_with_preferences(self):
        """Test method recommendation with user preferences."""
        user_prefs = {
            'prioritize_speed': True,
            'need_uncertainty': False
        }
        
        recommendations = self.selector.recommend_method(
            self.mock_characteristics, user_prefs
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # IDW should score higher when speed is prioritized
        idw_score = next(r.score for r in recommendations 
                        if r.method == InterpolationMethod.IDW)
        assert idw_score > 50  # Should have a reasonable score
    
    def test_evaluate_idw(self):
        """Test IDW evaluation."""
        idw_score = self.selector._evaluate_idw({})
        
        assert isinstance(idw_score, MethodScore)
        assert idw_score.method == InterpolationMethod.IDW
        assert 0 <= idw_score.score <= 100
        assert isinstance(idw_score.reasons, list)
        assert isinstance(idw_score.pros, list)
        assert isinstance(idw_score.cons, list)
        assert isinstance(idw_score.suitable_conditions, list)
    
    def test_evaluate_idw_dense_uniform_data(self):
        """Test IDW evaluation with dense, uniform data."""
        # Modify characteristics for favorable IDW conditions
        self.mock_characteristics.density = 1.0  # High density
        self.mock_characteristics.distribution_uniformity = 0.9  # Very uniform
        self.mock_characteristics.has_trend = False  # No trend
        
        self.selector.characteristics = self.mock_characteristics
        idw_score = self.selector._evaluate_idw({})
        
        # Should score well for these conditions
        assert idw_score.score > 70
        assert "Dense data favors IDW performance" in idw_score.reasons or \
               "Uniform distribution is ideal for IDW" in idw_score.reasons
    
    def test_evaluate_idw_with_trend(self):
        """Test IDW evaluation with trend data."""
        # Set trend presence
        self.mock_characteristics.has_trend = True
        self.mock_characteristics.trend_type = 'linear'
        
        self.selector.characteristics = self.mock_characteristics
        idw_score = self.selector._evaluate_idw({})
        
        # Should score lower due to trend
        assert "trend" in " ".join(idw_score.reasons).lower() or \
               "trend" in " ".join(idw_score.cons).lower()
    
    def test_evaluate_kriging(self):
        """Test Kriging evaluation."""
        kriging_score = self.selector._evaluate_kriging({})
        
        assert isinstance(kriging_score, MethodScore)
        assert kriging_score.method == InterpolationMethod.ORDINARY_KRIGING
        assert 0 <= kriging_score.score <= 100
        assert isinstance(kriging_score.reasons, list)
        assert isinstance(kriging_score.pros, list)
        assert isinstance(kriging_score.cons, list)
    
    def test_evaluate_kriging_small_dataset(self):
        """Test Kriging evaluation with small dataset."""
        # Set small sample size
        self.mock_characteristics.n_points = 20
        
        self.selector.characteristics = self.mock_characteristics
        kriging_score = self.selector._evaluate_kriging({})
        
        # Should score lower for small datasets
        assert "30 points" in " ".join(kriging_score.reasons) or \
               "few points" in " ".join(kriging_score.cons).lower()
    
    def test_evaluate_kriging_with_uncertainty_preference(self):
        """Test Kriging evaluation when uncertainty is needed."""
        user_prefs = {'need_uncertainty': True}
        
        self.selector.characteristics = self.mock_characteristics
        kriging_score = self.selector._evaluate_kriging(user_prefs)
        
        # Should score higher when uncertainty is needed
        assert "uncertainty" in " ".join(kriging_score.reasons).lower()
    
    def test_evaluate_rbf(self):
        """Test RBF evaluation."""
        rbf_score = self.selector._evaluate_rbf({})
        
        assert isinstance(rbf_score, MethodScore)
        assert rbf_score.method == InterpolationMethod.RBF
        assert 0 <= rbf_score.score <= 100
        assert isinstance(rbf_score.reasons, list)
        assert isinstance(rbf_score.pros, list)
        assert isinstance(rbf_score.cons, list)
    
    def test_evaluate_rbf_smooth_surface_preference(self):
        """Test RBF evaluation when smooth surface is required."""
        user_prefs = {'require_smooth_surface': True}
        
        self.selector.characteristics = self.mock_characteristics
        rbf_score = self.selector._evaluate_rbf(user_prefs)
        
        # Should score higher when smoothness is required
        assert "smooth" in " ".join(rbf_score.reasons).lower()
    
    def test_evaluate_rbf_large_dataset(self):
        """Test RBF evaluation with large dataset."""
        # Set large sample size
        self.mock_characteristics.n_points = 5000
        
        self.selector.characteristics = self.mock_characteristics
        rbf_score = self.selector._evaluate_rbf({})
        
        # Should score lower for large datasets due to computational cost
        assert "slow" in " ".join(rbf_score.cons).lower() or \
               "many points" in " ".join(rbf_score.cons).lower()
    
    def test_evaluate_rbf_with_outliers(self):
        """Test RBF evaluation with outliers."""
        # Add outliers
        self.mock_characteristics.outlier_indices = [1, 5, 10, 15, 20, 25]  # 6% outliers
        
        self.selector.characteristics = self.mock_characteristics
        rbf_score = self.selector._evaluate_rbf({})
        
        # Should score lower due to outlier sensitivity
        assert "outlier" in " ".join(rbf_score.cons).lower() or \
               "oscillation" in " ".join(rbf_score.cons).lower()
    
    def test_get_method_comparison(self):
        """Test method comparison generation."""
        # First run recommendation to populate scores
        self.selector.recommend_method(self.mock_characteristics)
        
        comparison = self.selector.get_method_comparison()
        
        assert isinstance(comparison, dict)
        assert 'recommended' in comparison
        assert 'scores' in comparison
        assert 'details' in comparison
        
        # Check scores
        assert len(comparison['scores']) == 3
        for method_name, score in comparison['scores'].items():
            assert 0 <= score <= 100
        
        # Check details
        assert len(comparison['details']) == 3
        for method_name, details in comparison['details'].items():
            assert 'score' in details
            assert 'pros' in details
            assert 'cons' in details
            assert 'reasons' in details
            assert 'suitable_for' in details
    
    def test_method_scoring_consistency(self):
        """Test that method scoring is consistent."""
        # Run recommendation multiple times
        rec1 = self.selector.recommend_method(self.mock_characteristics)
        rec2 = self.selector.recommend_method(self.mock_characteristics)
        
        # Scores should be identical for same input
        scores1 = {r.method: r.score for r in rec1}
        scores2 = {r.method: r.score for r in rec2}
        
        assert scores1 == scores2
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal data
        minimal_chars = DataCharacteristics(
            n_points=3,
            dimensions=2,
            bounds={'X': (0, 1), 'Y': (0, 1)},
            density=3.0,
            distribution_uniformity=1.0,
            has_trend=False,
            trend_type=None,
            anisotropy_ratio=1.0,
            anisotropy_angle=0.0,
            statistics={'mean': 0, 'std': 1, 'cv': 1},
            outlier_indices=[],
            clustering_score=0.0,
            nearest_neighbor_stats={'mean_distance': 0.1}
        )
        
        recommendations = self.selector.recommend_method(minimal_chars)
        assert len(recommendations) == 3
        
        # All methods should handle minimal data gracefully
        for rec in recommendations:
            assert 0 <= rec.score <= 100
    
    def test_extreme_characteristics(self):
        """Test with extreme data characteristics."""
        # Test with very clustered data
        extreme_chars = DataCharacteristics(
            n_points=1000,
            dimensions=2,
            bounds={'X': (0, 10000), 'Y': (0, 10000)},
            density=0.01,
            distribution_uniformity=0.1,  # Very clustered
            has_trend=True,
            trend_type='quadratic',
            anisotropy_ratio=0.3,  # Strong anisotropy
            anisotropy_angle=135.0,
            statistics={'mean': 100, 'std': 50, 'cv': 2.0},  # High variability
            outlier_indices=list(range(0, 100, 10)),  # 10% outliers
            clustering_score=0.9,  # Highly clustered
            nearest_neighbor_stats={'mean_distance': 10.0}
        )
        
        recommendations = self.selector.recommend_method(extreme_chars)
        assert len(recommendations) == 3
        
        # Should still provide valid recommendations
        for rec in recommendations:
            assert 0 <= rec.score <= 100
            assert len(rec.reasons) > 0


if __name__ == "__main__":
    pytest.main([__file__])