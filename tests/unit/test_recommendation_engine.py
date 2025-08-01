"""
Unit tests for RecommendationEngine.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Mock imports for testing without dependencies
try:
    from src.core.recommendations.recommendation_engine import (
        RecommendationEngine, RecommendationReport
    )
    from src.core.recommendations.data_analyzer import DataCharacteristics
    from src.core.recommendations.method_selector import MethodScore, InterpolationMethod
    from src.core.recommendations.parameter_optimizer import OptimizationResult
except ImportError:
    pytest.skip("Recommendation modules not available", allow_module_level=True)


class TestRecommendationEngine:
    """Test cases for RecommendationEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RecommendationEngine()
        
        # Create sample data
        np.random.seed(42)
        n_points = 30
        
        self.test_data = pd.DataFrame({
            'X': np.random.uniform(0, 1000, n_points),
            'Y': np.random.uniform(0, 1000, n_points),
            'Value': np.random.normal(10, 2, n_points)
        })
    
    def test_initialization(self):
        """Test RecommendationEngine initialization."""
        engine = RecommendationEngine()
        assert engine.data_analyzer is not None
        assert engine.method_selector is not None
        assert engine.parameter_optimizer is not None
        assert engine.quality_evaluator is not None
        assert engine.data is None
        assert engine.characteristics is None
    
    @patch('src.core.recommendations.recommendation_engine.DataAnalyzer')
    @patch('src.core.recommendations.recommendation_engine.MethodSelector')
    @patch('src.core.recommendations.recommendation_engine.ParameterOptimizer')
    def test_analyze_and_recommend_basic(self, mock_optimizer, mock_selector, mock_analyzer):
        """Test basic analyze and recommend functionality."""
        # Setup mocks
        mock_characteristics = MagicMock()
        mock_characteristics.n_points = 30
        mock_characteristics.density = 0.03
        mock_characteristics.distribution_uniformity = 0.7
        mock_characteristics.has_trend = False
        mock_characteristics.anisotropy_ratio = 0.9
        mock_characteristics.clustering_score = 0.4
        
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = mock_characteristics
        mock_analyzer.return_value = mock_analyzer_instance
        
        mock_method_score = MagicMock()
        mock_method_score.method.value = 'IDW'
        mock_method_score.score = 85.0
        mock_method_score.reasons = ['Good for uniform data']
        
        mock_selector_instance = MagicMock()
        mock_selector_instance.recommend_method.return_value = [mock_method_score]
        mock_selector.return_value = mock_selector_instance
        
        mock_optimization = MagicMock()
        mock_optimization.parameters = {'power': 2.0, 'search_radius': 500}
        mock_optimization.reasoning = {'power': 'Standard power', 'search_radius': 'Balanced coverage'}
        
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.optimize_parameters.return_value = mock_optimization
        mock_optimizer.return_value = mock_optimizer_instance
        
        # Run analysis
        report = self.engine.analyze_and_recommend(
            self.test_data, 'X', 'Y', 'Value',
            evaluate_quality=False,
            quick_mode=True
        )
        
        # Verify results
        assert isinstance(report, RecommendationReport)
        assert report.data_characteristics == mock_characteristics
        assert len(report.method_scores) == 1
        assert report.recommended_method == 'IDW'
        assert report.optimal_parameters == mock_optimization.parameters
        assert report.parameter_reasoning == mock_optimization.reasoning
        assert isinstance(report.summary_text, str)
        assert isinstance(report.warnings, list)
        assert report.computation_time > 0
    
    def test_analyze_and_recommend_with_preferences(self):
        """Test analyze and recommend with user preferences."""
        user_prefs = {
            'prioritize_speed': True,
            'need_uncertainty': False
        }
        
        # Use real implementation with minimal mocking
        with patch.object(self.engine.data_analyzer, 'analyze') as mock_analyze:
            mock_characteristics = self._create_mock_characteristics()
            mock_analyze.return_value = mock_characteristics
            
            report = self.engine.analyze_and_recommend(
                self.test_data, 'X', 'Y', 'Value',
                user_preferences=user_prefs,
                evaluate_quality=False,
                quick_mode=True
            )
            
            assert isinstance(report, RecommendationReport)
            assert report.recommended_method in ['IDW', 'Ordinary Kriging', 'Radial Basis Functions']
    
    def test_analyze_and_recommend_with_quality_evaluation(self):
        """Test analyze and recommend with quality evaluation."""
        with patch.object(self.engine, '_create_interpolator') as mock_create:
            with patch.object(self.engine.quality_evaluator, 'evaluate') as mock_evaluate:
                # Setup mocks
                mock_create.return_value = MagicMock()
                mock_quality = MagicMock()
                mock_quality.metrics.rmse = 1.5
                mock_quality.metrics.r_squared = 0.85
                mock_evaluate.return_value = mock_quality
                
                with patch.object(self.engine.data_analyzer, 'analyze') as mock_analyze:
                    mock_analyze.return_value = self._create_mock_characteristics()
                    
                    report = self.engine.analyze_and_recommend(
                        self.test_data, 'X', 'Y', 'Value',
                        evaluate_quality=True,
                        quick_mode=False
                    )
                    
                    assert report.expected_quality == mock_quality
                    mock_evaluate.assert_called_once()
    
    def test_get_detailed_recommendations(self):
        """Test detailed recommendations generation."""
        # First run analysis to populate characteristics
        with patch.object(self.engine.data_analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = self._create_mock_characteristics()
            
            self.engine.analyze_and_recommend(
                self.test_data, 'X', 'Y', 'Value',
                evaluate_quality=False,
                quick_mode=True
            )
        
        recommendations = self.engine.get_detailed_recommendations()
        
        assert isinstance(recommendations, dict)
        assert 'data_insights' in recommendations
        assert 'method_comparison' in recommendations
        assert 'parameter_guidelines' in recommendations
        assert 'best_practices' in recommendations
        assert 'potential_issues' in recommendations
    
    def test_get_detailed_recommendations_no_analysis(self):
        """Test detailed recommendations without prior analysis."""
        recommendations = self.engine.get_detailed_recommendations()
        
        assert recommendations == {"error": "No analysis performed yet"}
    
    def test_generate_summary(self):
        """Test summary generation."""
        characteristics = self._create_mock_characteristics()
        
        method_score = MagicMock()
        method_score.method.value = 'IDW'
        method_score.score = 85.0
        method_score.reasons = ['Dense data', 'No trends', 'Fast computation']
        
        optimization = MagicMock()
        optimization.parameters = {
            'search_radius': 500,
            'power': 2.0,
            'max_points': 12
        }
        optimization.reasoning = {
            'search_radius': 'Balanced coverage',
            'power': 'Standard power',
            'max_points': 'Speed vs accuracy'
        }
        
        quality = MagicMock()
        quality.metrics.rmse = 1.2
        quality.metrics.r_squared = 0.88
        quality.metrics.mae = 0.9
        
        summary = self.engine._generate_summary(
            characteristics, method_score, optimization, quality
        )
        
        assert isinstance(summary, str)
        assert '30 points' in summary
        assert 'IDW' in summary
        assert '85.0/100' in summary
        assert 'RMSE: 1.2' in summary
    
    def test_check_data_quality(self):
        """Test data quality checking."""
        self.engine.characteristics = self._create_mock_characteristics()
        
        warnings = self.engine._check_data_quality()
        
        assert isinstance(warnings, list)
        # Should have warnings for small dataset
        assert any('few' in w.lower() or 'limited' in w.lower() for w in warnings)
    
    def test_describe_spatial_pattern(self):
        """Test spatial pattern description."""
        self.engine.characteristics = self._create_mock_characteristics()
        
        description = self.engine._describe_spatial_pattern()
        
        assert isinstance(description, str)
        assert len(description) > 0
    
    def test_describe_value_distribution(self):
        """Test value distribution description."""
        self.engine.characteristics = self._create_mock_characteristics()
        
        description = self.engine._describe_value_distribution()
        
        assert isinstance(description, str)
        assert len(description) > 0
    
    def test_assess_data_quality(self):
        """Test data quality assessment."""
        self.engine.characteristics = self._create_mock_characteristics()
        
        quality = self.engine._assess_data_quality()
        
        assert isinstance(quality, str)
        assert any(word in quality.lower() for word in ['excellent', 'good', 'fair', 'poor'])
    
    def test_identify_special_considerations(self):
        """Test special considerations identification."""
        # Test with trend
        characteristics = self._create_mock_characteristics()
        characteristics.has_trend = True
        characteristics.trend_type = 'linear'
        self.engine.characteristics = characteristics
        
        considerations = self.engine._identify_special_considerations()
        
        assert isinstance(considerations, list)
        assert any('trend' in c.lower() for c in considerations)
    
    def test_get_parameter_guidelines(self):
        """Test parameter guidelines generation."""
        guidelines = self.engine._get_parameter_guidelines()
        
        assert isinstance(guidelines, dict)
        assert 'search_radius' in guidelines
        assert 'power' in guidelines
        assert 'max_points' in guidelines
    
    def test_get_best_practices(self):
        """Test best practices generation."""
        self.engine.characteristics = self._create_mock_characteristics()
        
        practices = self.engine._get_best_practices()
        
        assert isinstance(practices, list)
        assert len(practices) > 0
        assert any('validation' in p.lower() for p in practices)
    
    def test_identify_potential_issues(self):
        """Test potential issues identification."""
        self.engine.characteristics = self._create_mock_characteristics()
        
        issues = self.engine._identify_potential_issues()
        
        assert isinstance(issues, list)
        # Should identify small dataset issue
        assert any('few' in i.lower() for i in issues)
    
    def test_create_minimal_characteristics(self):
        """Test minimal characteristics creation."""
        minimal = self.engine._create_minimal_characteristics(self.test_data)
        
        assert isinstance(minimal, DataCharacteristics)
        assert minimal.n_points == len(self.test_data)
        assert minimal.dimensions == 2
        assert minimal.density > 0
    
    def test_to_dict_conversion(self):
        """Test recommendation report to dictionary conversion."""
        with patch.object(self.engine.data_analyzer, 'analyze') as mock_analyze:
            mock_analyze.return_value = self._create_mock_characteristics()
            
            report = self.engine.analyze_and_recommend(
                self.test_data, 'X', 'Y', 'Value',
                evaluate_quality=False,
                quick_mode=True
            )
            
            report_dict = report.to_dict()
            
            assert isinstance(report_dict, dict)
            assert 'data_summary' in report_dict
            assert 'method_recommendation' in report_dict
            assert 'parameters' in report_dict
            assert 'parameter_reasoning' in report_dict
            assert 'summary' in report_dict
            assert 'warnings' in report_dict
            assert 'computation_time' in report_dict
    
    def test_error_handling(self):
        """Test error handling in analysis."""
        # Test with invalid data
        invalid_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        with patch.object(self.engine.data_analyzer, 'analyze', side_effect=Exception("Test error")):
            report = self.engine.analyze_and_recommend(
                invalid_data, 'X', 'Y', 'Value',
                evaluate_quality=False,
                quick_mode=True
            )
            
            # Should still return a report with minimal characteristics
            assert isinstance(report, RecommendationReport)
            assert len(report.warnings) > 0
    
    def _create_mock_characteristics(self):
        """Create mock data characteristics for testing."""
        return DataCharacteristics(
            n_points=30,
            dimensions=2,
            bounds={'X': (0, 1000), 'Y': (0, 1000)},
            density=0.03,
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
            clustering_score=0.4,
            nearest_neighbor_stats={'mean_distance': 100.0}
        )


if __name__ == "__main__":
    pytest.main([__file__])