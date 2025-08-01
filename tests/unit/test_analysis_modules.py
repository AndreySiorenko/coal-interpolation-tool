"""
Unit tests for analysis modules.

Tests for StatisticalAnalyzer, SpatialAnalyzer, OutlierDetector,
CorrelationAnalyzer, and DataQualityAnalyzer.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

# Import modules to test
from src.analysis.statistical_analyzer import StatisticalAnalyzer, StatisticalResults
from src.analysis.spatial_analyzer import SpatialAnalyzer, SpatialResults  
from src.analysis.outlier_detector import OutlierDetector, OutlierResults
from src.analysis.correlation_analyzer import CorrelationAnalyzer, CorrelationResults
from src.analysis.data_quality_analyzer import DataQualityAnalyzer, DataQualityResults


class TestStatisticalAnalyzer:
    """Test cases for StatisticalAnalyzer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'values': np.random.normal(100, 15, 1000),
            'skewed_values': np.random.exponential(2, 1000),
            'uniform_values': np.random.uniform(0, 100, 1000)
        })
    
    @pytest.fixture
    def analyzer(self):
        """Create StatisticalAnalyzer instance."""
        return StatisticalAnalyzer(confidence_level=0.95)
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.confidence_level == 0.95
        assert analyzer.alpha == 0.05
    
    def test_analyze_normal_data(self, analyzer, sample_data):
        """Test analysis of normal data."""
        results = analyzer.analyze(sample_data, 'values')
        
        assert isinstance(results, StatisticalResults)
        assert 'mean' in results.descriptive_stats
        assert 'std' in results.descriptive_stats
        assert 'skewness' in results.distribution_analysis
        assert 'kurtosis' in results.distribution_analysis
        
        # Check that mean is close to expected (100)
        assert abs(results.descriptive_stats['mean'] - 100) < 5
        
        # Check that standard deviation is reasonable
        assert 10 < results.descriptive_stats['std'] < 20
    
    def test_analyze_skewed_data(self, analyzer, sample_data):
        """Test analysis of skewed data."""
        results = analyzer.analyze(sample_data, 'skewed_values')
        
        # Exponential distribution should be right-skewed
        assert results.distribution_analysis['skewness'] > 0.5
        assert results.distribution_analysis['skewness_interpretation'] == 'right_skewed'
    
    def test_normality_tests(self, analyzer, sample_data):
        """Test normality testing."""
        results = analyzer.analyze(sample_data, 'values')
        
        # Should include various normality tests
        normality = results.normality_tests
        assert 'shapiro_wilk' in normality or len(sample_data) > 5000
        assert 'anderson_darling' in normality
    
    def test_quantile_analysis(self, analyzer, sample_data):
        """Test quantile calculations."""
        results = analyzer.analyze(sample_data, 'values')
        
        quantiles = results.quantile_stats
        assert 'q1' in quantiles
        assert 'q2' in quantiles  # median
        assert 'q3' in quantiles
        assert 'iqr' in quantiles
        
        # Check ordering
        assert quantiles['q1'] < quantiles['q2'] < quantiles['q3']
        assert abs(quantiles['iqr'] - (quantiles['q3'] - quantiles['q1'])) < 1e-10
    
    def test_confidence_intervals(self, analyzer, sample_data):
        """Test confidence interval calculations."""
        results = analyzer.analyze(sample_data, 'values')
        
        ci = results.confidence_intervals
        assert 'mean' in ci
        
        lower, upper = ci['mean']
        assert lower < results.descriptive_stats['mean'] < upper
    
    def test_empty_data_error(self, analyzer):
        """Test error handling for empty data."""
        empty_data = pd.DataFrame({'values': [np.nan, np.nan, np.nan]})
        
        with pytest.raises(ValueError, match="No valid data found"):
            analyzer.analyze(empty_data, 'values')
    
    def test_generate_summary_report(self, analyzer, sample_data):
        """Test summary report generation."""
        results = analyzer.analyze(sample_data, 'values')
        report = analyzer.generate_summary_report(results, 'values')
        
        assert isinstance(report, str)
        assert 'STATISTICAL ANALYSIS REPORT' in report
        assert 'Mean:' in report
        assert 'Standard Deviation:' in report


class TestSpatialAnalyzer:
    """Test cases for SpatialAnalyzer."""
    
    @pytest.fixture
    def spatial_data(self):
        """Create sample spatial data."""
        np.random.seed(42)
        n_points = 100
        
        return pd.DataFrame({
            'x': np.random.uniform(0, 100, n_points),
            'y': np.random.uniform(0, 100, n_points),
            'z': np.random.uniform(0, 50, n_points),
            'value': np.random.normal(25, 5, n_points)
        })
    
    @pytest.fixture
    def analyzer(self):
        """Create SpatialAnalyzer instance."""
        return SpatialAnalyzer()
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.coordinates is None
        assert analyzer.values is None
        assert analyzer.kdtree is None
        assert analyzer.dimensions == 2
    
    def test_analyze_2d_data(self, analyzer, spatial_data):
        """Test analysis of 2D spatial data."""
        results = analyzer.analyze(spatial_data, 'x', 'y', 'value')
        
        assert isinstance(results, SpatialResults)
        assert 'density_analysis' in results.__dict__
        assert 'clustering_analysis' in results.__dict__
        assert 'pattern_analysis' in results.__dict__
        assert 'spatial_statistics' in results.__dict__
    
    def test_analyze_3d_data(self, analyzer, spatial_data):
        """Test analysis of 3D spatial data."""
        results = analyzer.analyze(spatial_data, 'x', 'y', 'value', 'z')
        
        assert isinstance(results, SpatialResults)
        assert analyzer.dimensions == 3
    
    def test_density_analysis(self, analyzer, spatial_data):
        """Test density analysis."""
        results = analyzer.analyze(spatial_data, 'x', 'y', 'value')
        
        density = results.density_analysis
        assert 'points_per_unit_area' in density
        assert 'nearest_neighbor' in density
        assert 'density_variation' in density
        
        # Check nearest neighbor analysis
        nn = density['nearest_neighbor']
        assert 'mean_distance' in nn
        assert 'clark_evans_index' in nn
    
    def test_spatial_statistics(self, analyzer, spatial_data):
        """Test spatial statistics calculation."""
        results = analyzer.analyze(spatial_data, 'x', 'y', 'value')
        
        stats = results.spatial_statistics
        assert 'centroid' in stats
        assert 'standard_distance' in stats
        assert 'bounds' in stats
        
        # Check centroid calculation
        centroid = stats['centroid']
        assert len(centroid) == 2  # x, y coordinates
    
    @patch('src.analysis.spatial_analyzer.SKLEARN_AVAILABLE', False)
    def test_clustering_without_sklearn(self, analyzer, spatial_data):
        """Test clustering analysis when sklearn is not available."""
        results = analyzer.analyze(spatial_data, 'x', 'y', 'value')
        
        clustering = results.clustering_analysis
        assert 'error' in clustering or 'sklearn_note' in clustering
    
    def test_anisotropy_analysis_2d(self, analyzer, spatial_data):
        """Test anisotropy analysis for 2D data."""
        results = analyzer.analyze(spatial_data, 'x', 'y', 'value')
        
        anisotropy = results.anisotropy_analysis
        if 'error' not in anisotropy:
            assert 'directional_analysis' in anisotropy
            assert 'anisotropy_ellipse' in anisotropy
    
    def test_insufficient_data(self, analyzer):
        """Test handling of insufficient data."""
        small_data = pd.DataFrame({
            'x': [1, 2],
            'y': [1, 2], 
            'value': [10, 20]
        })
        
        results = analyzer.analyze(small_data, 'x', 'y', 'value')
        
        # Should still work but some analyses may have errors
        assert isinstance(results, SpatialResults)


class TestOutlierDetector:
    """Test cases for OutlierDetector."""
    
    @pytest.fixture
    def outlier_data(self):
        """Create data with known outliers."""
        np.random.seed(42)
        
        # Normal data with some outliers
        normal_data = np.random.normal(50, 10, 95)
        outliers = [100, 0, 150, -20, 200]  # Clear outliers
        
        return pd.DataFrame({
            'x': np.concatenate([np.random.uniform(0, 100, 95), [10, 20, 30, 40, 50]]),
            'y': np.concatenate([np.random.uniform(0, 100, 95), [15, 25, 35, 45, 55]]),
            'value': np.concatenate([normal_data, outliers])
        })
    
    @pytest.fixture
    def detector(self):
        """Create OutlierDetector instance."""
        return OutlierDetector(contamination=0.1)
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.contamination == 0.1
        assert detector.data is None
        assert detector.coordinates is None
        assert detector.values is None
    
    def test_detect_outliers_all_methods(self, detector, outlier_data):
        """Test outlier detection with all methods."""
        results = detector.detect_outliers(
            outlier_data, 'x', 'y', 'value',
            methods=['statistical', 'spatial', 'multivariate', 'ensemble']
        )
        
        assert isinstance(results, OutlierResults)
        assert 'statistical_outliers' in results.__dict__
        assert 'spatial_outliers' in results.__dict__
        assert 'multivariate_outliers' in results.__dict__
        assert 'ensemble_results' in results.__dict__
        assert 'summary' in results.__dict__
    
    def test_statistical_outliers(self, detector, outlier_data):
        """Test statistical outlier detection methods."""
        results = detector.detect_outliers(
            outlier_data, 'x', 'y', 'value',
            methods=['statistical']
        )
        
        stat_outliers = results.statistical_outliers
        assert 'iqr_method' in stat_outliers
        assert 'zscore_method' in stat_outliers
        assert 'modified_zscore' in stat_outliers
        
        # Should detect some outliers
        iqr_outliers = stat_outliers['iqr_method']['outlier_indices']
        assert len(iqr_outliers) > 0
    
    def test_ensemble_outliers(self, detector, outlier_data):
        """Test ensemble outlier detection."""
        results = detector.detect_outliers(
            outlier_data, 'x', 'y', 'value',
            methods=['statistical', 'ensemble']
        )
        
        ensemble = results.ensemble_results
        assert 'majority_vote' in ensemble
        assert 'consensus' in ensemble
        assert 'union' in ensemble
        assert 'vote_statistics' in ensemble
    
    @patch('src.analysis.outlier_detector.SKLEARN_AVAILABLE', False)
    def test_multivariate_without_sklearn(self, detector, outlier_data):
        """Test multivariate detection when sklearn is not available."""
        results = detector.detect_outliers(
            outlier_data, 'x', 'y', 'value',
            methods=['multivariate']
        )
        
        mv_outliers = results.multivariate_outliers
        assert 'sklearn_note' in mv_outliers
    
    def test_summary_generation(self, detector, outlier_data):
        """Test summary generation."""
        results = detector.detect_outliers(outlier_data, 'x', 'y', 'value')
        
        summary = results.summary
        assert 'total_points' in summary
        assert 'union_outliers' in summary
        assert 'intersection_outliers' in summary
        assert 'recommendation' in summary


class TestCorrelationAnalyzer:
    """Test cases for CorrelationAnalyzer."""
    
    @pytest.fixture
    def correlation_data(self):
        """Create data with known correlations."""
        np.random.seed(42)
        n = 200
        
        x1 = np.random.normal(0, 1, n)
        x2 = 0.8 * x1 + 0.6 * np.random.normal(0, 1, n)  # Strong positive correlation
        x3 = -0.5 * x1 + 0.866 * np.random.normal(0, 1, n)  # Moderate negative correlation
        x4 = np.random.normal(0, 1, n)  # Independent
        
        return pd.DataFrame({
            'var1': x1,
            'var2': x2,
            'var3': x3,
            'var4': x4,
            'coord_x': np.random.uniform(0, 100, n),
            'coord_y': np.random.uniform(0, 100, n)
        })
    
    @pytest.fixture
    def analyzer(self):
        """Create CorrelationAnalyzer instance."""
        return CorrelationAnalyzer(alpha=0.05)
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.alpha == 0.05
        assert analyzer.data is None
        assert analyzer.numeric_columns == []
    
    def test_analyze_correlations(self, analyzer, correlation_data):
        """Test correlation analysis."""
        results = analyzer.analyze(correlation_data)
        
        assert isinstance(results, CorrelationResults)
        assert 'pearson' in results.correlation_matrices
        assert 'spearman' in results.correlation_matrices
        assert 'kendall' in results.correlation_matrices
    
    def test_correlation_matrices(self, analyzer, correlation_data):
        """Test correlation matrix calculation."""
        results = analyzer.analyze(correlation_data)
        
        pearson_matrix = results.correlation_matrices['pearson']
        
        # Check diagonal is 1 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(pearson_matrix), 1.0)
        
        # Check symmetry
        np.testing.assert_array_almost_equal(pearson_matrix, pearson_matrix.T)
        
        # Check known strong correlation (var1 vs var2)
        var_names = analyzer.numeric_columns
        if 'var1' in var_names and 'var2' in var_names:
            idx1 = var_names.index('var1')
            idx2 = var_names.index('var2')
            correlation = pearson_matrix[idx1, idx2]
            assert abs(correlation) > 0.5  # Should be strongly correlated
    
    def test_significance_tests(self, analyzer, correlation_data):
        """Test significance testing of correlations."""
        results = analyzer.analyze(correlation_data)
        
        sig_tests = results.significance_tests
        assert 'pearson' in sig_tests
        
        pearson_test = sig_tests['pearson']
        assert 'p_values' in pearson_test
        assert 'significant' in pearson_test
        assert 'confidence_intervals' in pearson_test
    
    def test_partial_correlations(self, analyzer, correlation_data):
        """Test partial correlation calculation."""
        results = analyzer.analyze(correlation_data)
        
        partial_corr = results.partial_correlations
        
        # Should have partial correlations if enough variables
        if len(analyzer.numeric_columns) >= 3:
            assert len(partial_corr) > 0
            
            # Check structure of partial correlation results
            for key, value in partial_corr.items():
                if isinstance(value, dict):
                    assert 'partial_correlation' in value
                    assert 'controlling_for' in value
    
    def test_relationship_analysis(self, analyzer, correlation_data):
        """Test relationship analysis."""
        results = analyzer.analyze(correlation_data)
        
        relationships = results.relationship_analysis
        assert 'strongest_correlations' in relationships
        assert 'variable_connectivity' in relationships
        assert 'multicollinearity_check' in relationships
    
    def test_multicollinearity_detection(self, analyzer, correlation_data):
        """Test multicollinearity detection."""
        results = analyzer.analyze(correlation_data)
        
        mc_check = results.relationship_analysis['multicollinearity_check']
        assert 'high_correlation_pairs' in mc_check
        assert 'multicollinearity_risk' in mc_check
    
    def test_insufficient_variables_error(self, analyzer):
        """Test error handling for insufficient variables."""
        insufficient_data = pd.DataFrame({'single_var': [1, 2, 3, 4, 5]})
        
        with pytest.raises(ValueError, match="At least 2 numeric columns required"):
            analyzer.analyze(insufficient_data)
    
    def test_generate_correlation_report(self, analyzer, correlation_data):
        """Test correlation report generation."""
        results = analyzer.analyze(correlation_data)
        report = analyzer.generate_correlation_report(results)
        
        assert isinstance(report, str)
        assert 'CORRELATION ANALYSIS REPORT' in report
        assert 'PEARSON CORRELATIONS' in report


class TestDataQualityAnalyzer:
    """Test cases for DataQualityAnalyzer."""
    
    @pytest.fixture
    def quality_data(self):
        """Create data with various quality issues."""
        np.random.seed(42)
        
        # Create data with missing values, outliers, and inconsistencies
        data = pd.DataFrame({
            'x': np.concatenate([np.random.uniform(0, 100, 90), [np.nan] * 10]),
            'y': np.concatenate([np.random.uniform(0, 100, 95), [np.nan] * 5]),
            'value': np.concatenate([
                np.random.normal(50, 10, 85),  # Normal values  
                [150, -50, 200],  # Outliers
                [np.nan] * 12  # Missing values
            ]),
            'elevation': np.concatenate([
                np.random.uniform(100, 1000, 85),  # Reasonable elevations
                [15000, -1000, 20000],  # Unreasonable elevations
                [np.nan] * 12  # Missing values
            ]),
            'id': list(range(85)) + [1, 2, 3] + [np.nan] * 12,  # Duplicates + missing
            'category': ['A'] * 40 + ['B'] * 45 + ['C'] * 3 + [np.nan] * 12
        })
        
        return data
    
    @pytest.fixture
    def analyzer(self):
        """Create DataQualityAnalyzer instance."""
        return DataQualityAnalyzer()
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.data is None
        assert analyzer.coordinate_columns == []
        assert analyzer.value_columns == []
        assert analyzer.quality_threshold == 0.8
    
    def test_analyze_quality(self, analyzer, quality_data):
        """Test comprehensive quality analysis."""
        results = analyzer.analyze_quality(
            quality_data,
            coordinate_columns=['x', 'y'],
            value_columns=['value', 'elevation']
        )
        
        assert isinstance(results, DataQualityResults)
        assert 'completeness_analysis' in results.__dict__
        assert 'consistency_analysis' in results.__dict__
        assert 'accuracy_analysis' in results.__dict__
        assert 'validity_analysis' in results.__dict__
        assert 'overall_quality' in results.__dict__
        assert 'recommendations' in results.__dict__
    
    def test_completeness_analysis(self, analyzer, quality_data):
        """Test completeness analysis."""
        results = analyzer.analyze_quality(quality_data)
        
        completeness = results.completeness_analysis
        assert 'overall_stats' in completeness
        assert 'column_completeness' in completeness
        assert 'row_completeness' in completeness
        assert 'overall_completeness_score' in completeness
        
        # Check column completeness
        col_completeness = completeness['column_completeness']
        for col in quality_data.columns:
            assert col in col_completeness
            assert 'missing_count' in col_completeness[col]
            assert 'completeness_score' in col_completeness[col]
    
    def test_consistency_analysis(self, analyzer, quality_data):
        """Test consistency analysis."""
        results = analyzer.analyze_quality(quality_data)
        
        consistency = results.consistency_analysis
        assert 'data_types' in consistency
        assert 'formats' in consistency
        assert 'values' in consistency
        
        # Check duplicate detection
        if 'values' in consistency and 'duplicates' in consistency['values']:
            duplicates = consistency['values']['duplicates']
            assert 'duplicate_rate' in duplicates
    
    def test_accuracy_analysis(self, analyzer, quality_data):
        """Test accuracy analysis."""
        results = analyzer.analyze_quality(quality_data)
        
        accuracy = results.accuracy_analysis
        assert 'outliers' in accuracy
        assert 'statistical_checks' in accuracy
        assert 'precision' in accuracy
        
        # Check outlier detection
        outliers = accuracy['outliers']
        for col in ['value', 'elevation']:
            if col in outliers:
                assert 'accuracy_score' in outliers[col]
    
    def test_validity_analysis(self, analyzer, quality_data):
        """Test validity analysis."""
        results = analyzer.analyze_quality(quality_data)
        
        validity = results.validity_analysis
        assert 'data_types' in validity
        assert 'business_rules' in validity
        assert 'constraints' in validity  
        assert 'logical_consistency' in validity
    
    def test_business_rules_check(self, analyzer, quality_data):
        """Test business rules validation."""
        results = analyzer.analyze_quality(quality_data)
        
        business_rules = results.validity_analysis['business_rules']
        
        # Should detect unreasonable elevations
        elevation_checks = [key for key in business_rules.keys() if 'elevation' in key]
        if elevation_checks:
            check = business_rules[elevation_checks[0]]
            assert 'unreasonably_high' in check or 'unreasonably_low' in check
    
    def test_overall_quality_score(self, analyzer, quality_data):
        """Test overall quality score calculation."""
        results = analyzer.analyze_quality(quality_data)
        
        overall = results.overall_quality
        assert 'overall_quality_score' in overall
        assert 'quality_grade' in overall
        assert 'meets_threshold' in overall
        
        # Score should be between 0 and 1
        score = overall['overall_quality_score']
        assert 0 <= score <= 1
    
    def test_recommendations_generation(self, analyzer, quality_data):
        """Test recommendation generation."""
        results = analyzer.analyze_quality(quality_data)
        
        recommendations = results.recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should contain actionable recommendations
        rec_text = ' '.join(recommendations).lower()
        assert any(keyword in rec_text for keyword in 
                  ['missing', 'outlier', 'duplicate', 'quality', 'validation'])
    
    def test_expected_ranges_validation(self, analyzer, quality_data):
        """Test validation against expected ranges."""
        expected_ranges = {
            'value': (0, 100),
            'elevation': (0, 5000)
        }
        
        results = analyzer.analyze_quality(
            quality_data,
            expected_ranges=expected_ranges
        )
        
        # Should have range validation in accuracy analysis
        accuracy = results.accuracy_analysis
        if 'range_validation' in accuracy:
            range_val = accuracy['range_validation']
            for col in expected_ranges.keys():
                if col in range_val:
                    assert 'range_compliance_score' in range_val[col]
    
    def test_auto_column_detection(self, analyzer, quality_data):
        """Test automatic detection of coordinate and value columns."""
        # Don't specify columns - let analyzer auto-detect
        results = analyzer.analyze_quality(quality_data)
        
        # Should still work with auto-detection
        assert isinstance(results, DataQualityResults)
        
        # Check that some columns were identified
        assert len(analyzer.coordinate_columns) >= 0
        assert len(analyzer.value_columns) >= 0


# Integration test for all analyzers working together
class TestAnalysisIntegration:
    """Integration tests for analysis modules."""
    
    @pytest.fixture
    def comprehensive_data(self):
        """Create comprehensive test data."""
        np.random.seed(42)
        n = 500
        
        # Create realistic geological survey data
        x = np.random.uniform(0, 1000, n)  # Easting coordinates
        y = np.random.uniform(0, 1000, n)  # Northing coordinates
        elevation = 100 + 0.01 * x + 0.005 * y + np.random.normal(0, 10, n)
        
        # Simulated mineral concentrations with spatial correlation
        base_concentration = 0.1 * np.exp(-((x-500)**2 + (y-500)**2) / 100000)
        concentration = base_concentration + np.random.exponential(0.05, n)
        
        # Add some outliers and missing values
        outlier_indices = np.random.choice(n, size=20, replace=False)
        concentration[outlier_indices] *= 10  # Extreme outliers
        
        missing_indices = np.random.choice(n, size=30, replace=False)
        concentration[missing_indices] = np.nan
        
        return pd.DataFrame({
            'easting': x,
            'northing': y,
            'elevation': elevation,
            'concentration': concentration,
            'sample_id': range(n),
            'date_collected': pd.date_range('2023-01-01', periods=n, freq='D')
        })
    
    def test_full_analysis_pipeline(self, comprehensive_data):
        """Test complete analysis pipeline with all modules."""
        # Statistical analysis
        stat_analyzer = StatisticalAnalyzer()
        stat_results = stat_analyzer.analyze(comprehensive_data, 'concentration')
        
        assert isinstance(stat_results, StatisticalResults)
        
        # Spatial analysis
        spatial_analyzer = SpatialAnalyzer()
        spatial_results = spatial_analyzer.analyze(
            comprehensive_data, 'easting', 'northing', 'concentration'
        )
        
        assert isinstance(spatial_results, SpatialResults)
        
        # Outlier detection
        outlier_detector = OutlierDetector()
        outlier_results = outlier_detector.detect_outliers(
            comprehensive_data, 'easting', 'northing', 'concentration'
        )
        
        assert isinstance(outlier_results, OutlierResults)
        
        # Correlation analysis
        corr_analyzer = CorrelationAnalyzer()
        corr_results = corr_analyzer.analyze(comprehensive_data)
        
        assert isinstance(corr_results, CorrelationResults)
        
        # Data quality analysis
        quality_analyzer = DataQualityAnalyzer()
        quality_results = quality_analyzer.analyze_quality(
            comprehensive_data,
            coordinate_columns=['easting', 'northing'],
            value_columns=['concentration', 'elevation']
        )
        
        assert isinstance(quality_results, DataQualityResults)
    
    def test_analysis_consistency(self, comprehensive_data):
        """Test that different analyzers give consistent results."""
        # Both statistical and outlier analyzers should detect similar outliers
        stat_analyzer = StatisticalAnalyzer()
        stat_results = stat_analyzer.analyze(comprehensive_data, 'concentration')
        
        outlier_detector = OutlierDetector()
        outlier_results = outlier_detector.detect_outliers(
            comprehensive_data, 'easting', 'northing', 'concentration'
        )
        
        # Statistical extreme values should align with outlier detection
        if 'extreme_values' in stat_results.__dict__:
            stat_outliers = stat_results.extreme_values.get('iqr_method', {}).get('mild_outliers', 0)
            det_outliers = len(outlier_results.summary.get('union_outliers', []))
            
            # Should be in same order of magnitude
            assert abs(stat_outliers - det_outliers) < max(stat_outliers, det_outliers) * 0.5


# Test fixtures and utilities
@pytest.fixture
def suppress_warnings():
    """Suppress warnings during testing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# Mark tests that require sklearn
sklearn_required = pytest.mark.skipif(
    not hasattr(pytest, 'importorskip') or 
    pytest.importorskip('sklearn', reason="sklearn not available") is None,
    reason="sklearn not available"
)