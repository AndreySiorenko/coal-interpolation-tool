"""
Main recommendation engine for interpolation optimization.

Coordinates all recommendation system components to provide
comprehensive analysis and suggestions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import json

from .data_analyzer import DataAnalyzer, DataCharacteristics
from .method_selector import MethodSelector, MethodScore, InterpolationMethod
from .parameter_optimizer import ParameterOptimizer, OptimizationResult
from .quality_evaluator import QualityEvaluator, CrossValidationResult


@dataclass
class RecommendationReport:
    """Complete recommendation report."""
    # Data analysis
    data_characteristics: DataCharacteristics
    
    # Method selection
    method_scores: List[MethodScore]
    recommended_method: str
    
    # Parameter optimization
    optimal_parameters: Dict[str, Any]
    parameter_reasoning: Dict[str, str]
    
    # Quality assessment
    expected_quality: Optional[CrossValidationResult]
    
    # Summary
    summary_text: str
    warnings: List[str]
    computation_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'data_summary': {
                'n_points': self.data_characteristics.n_points,
                'density': self.data_characteristics.density,
                'distribution_uniformity': self.data_characteristics.distribution_uniformity,
                'has_trend': self.data_characteristics.has_trend,
                'anisotropy_ratio': self.data_characteristics.anisotropy_ratio,
                'clustering_score': self.data_characteristics.clustering_score
            },
            'method_recommendation': {
                'recommended': self.recommended_method,
                'scores': {m.method.value: m.score for m in self.method_scores},
                'reasoning': self.method_scores[0].reasons if self.method_scores else []
            },
            'parameters': self.optimal_parameters,
            'parameter_reasoning': self.parameter_reasoning,
            'expected_quality': {
                'rmse': self.expected_quality.metrics.rmse,
                'r_squared': self.expected_quality.metrics.r_squared
            } if self.expected_quality else None,
            'summary': self.summary_text,
            'warnings': self.warnings,
            'computation_time': self.computation_time
        }


class RecommendationEngine:
    """
    Main engine for providing interpolation recommendations.
    
    Integrates all analysis components to provide:
    - Comprehensive data analysis
    - Method selection with reasoning
    - Optimized parameters
    - Quality expectations
    - Human-readable recommendations
    """
    
    def __init__(self):
        """Initialize recommendation engine."""
        self.data_analyzer = DataAnalyzer()
        self.method_selector = MethodSelector()
        self.parameter_optimizer = ParameterOptimizer()
        self.quality_evaluator = QualityEvaluator()
        
        self.data: Optional[pd.DataFrame] = None
        self.characteristics: Optional[DataCharacteristics] = None
        
    def analyze_and_recommend(self,
                            data: pd.DataFrame,
                            x_col: str,
                            y_col: str,
                            value_col: str,
                            z_col: Optional[str] = None,
                            user_preferences: Optional[Dict[str, Any]] = None,
                            evaluate_quality: bool = True,
                            quick_mode: bool = False) -> RecommendationReport:
        """
        Perform complete analysis and provide recommendations.
        
        Args:
            data: Input data
            x_col, y_col, value_col: Column names
            z_col: Optional Z coordinate column
            user_preferences: User constraints/preferences
            evaluate_quality: Whether to run cross-validation
            quick_mode: Fast analysis with fewer evaluations
            
        Returns:
            Complete recommendation report
        """
        start_time = time.time()
        self.data = data
        warnings = []
        
        # 1. Analyze data characteristics
        try:
            self.characteristics = self.data_analyzer.analyze(
                data, x_col, y_col, value_col, z_col
            )
        except Exception as e:
            warnings.append(f"Data analysis warning: {str(e)}")
            # Create minimal characteristics
            self.characteristics = self._create_minimal_characteristics(data)
        
        # 2. Select best method
        method_scores = self.method_selector.recommend_method(
            self.characteristics,
            user_preferences
        )
        
        if not method_scores:
            raise ValueError("No suitable interpolation method found")
            
        recommended_method = method_scores[0].method.value
        
        # 3. Optimize parameters
        optimization_result = self.parameter_optimizer.optimize_parameters(
            recommended_method,
            self.characteristics,
            data,
            user_preferences
        )
        
        # 4. Evaluate expected quality (optional)
        expected_quality = None
        if evaluate_quality and not quick_mode:
            try:
                # Create interpolator with optimal parameters
                interpolator = self._create_interpolator(
                    recommended_method,
                    optimization_result.parameters
                )
                
                # Use k-fold for speed, or LOOCV for small datasets
                cv_method = 'kfold' if len(data) > 50 else 'loocv'
                n_folds = min(5, len(data)) if cv_method == 'kfold' else len(data)
                
                expected_quality = self.quality_evaluator.evaluate(
                    interpolator,
                    data,
                    x_col, y_col, value_col, z_col,
                    cv_method=cv_method,
                    n_folds=n_folds
                )
            except Exception as e:
                warnings.append(f"Quality evaluation failed: {str(e)}")
        
        # 5. Generate summary
        summary_text = self._generate_summary(
            self.characteristics,
            method_scores[0],
            optimization_result,
            expected_quality
        )
        
        # Add data quality warnings
        warnings.extend(self._check_data_quality())
        
        computation_time = time.time() - start_time
        
        return RecommendationReport(
            data_characteristics=self.characteristics,
            method_scores=method_scores,
            recommended_method=recommended_method,
            optimal_parameters=optimization_result.parameters,
            parameter_reasoning=optimization_result.reasoning,
            expected_quality=expected_quality,
            summary_text=summary_text,
            warnings=warnings,
            computation_time=computation_time
        )
    
    def get_detailed_recommendations(self) -> Dict[str, Any]:
        """
        Get detailed recommendations in structured format.
        
        Returns:
            Dictionary with detailed recommendations
        """
        if not self.characteristics:
            return {"error": "No analysis performed yet"}
            
        recommendations = {
            "data_insights": self._get_data_insights(),
            "method_comparison": self.method_selector.get_method_comparison(),
            "parameter_guidelines": self._get_parameter_guidelines(),
            "best_practices": self._get_best_practices(),
            "potential_issues": self._identify_potential_issues()
        }
        
        return recommendations
    
    def apply_recommendations(self, interpolator: Any) -> Any:
        """
        Apply recommended parameters to an interpolator.
        
        Args:
            interpolator: Interpolator instance to configure
            
        Returns:
            Configured interpolator
        """
        if not hasattr(self, 'optimal_parameters'):
            raise ValueError("No recommendations available. Run analyze_and_recommend first.")
            
        # Apply parameters
        interpolator.set_parameters(**self.optimal_parameters)
        
        return interpolator
    
    def _create_interpolator(self, method: str, parameters: Dict[str, Any]) -> Any:
        """Create interpolator instance with given parameters."""
        # Import here to avoid circular imports
        from ..interpolation.idw import IDWInterpolator, IDWParameters
        from ..interpolation.base import SearchParameters
        
        if method.upper() == 'IDW':
            # Extract search parameters
            search_params = SearchParameters(
                search_radius=parameters.get('search_radius', 1000),
                min_points=parameters.get('min_points', 2),
                max_points=parameters.get('max_points', 12),
                use_sectors=parameters.get('use_sectors', False),
                n_sectors=parameters.get('n_sectors', 4),
                anisotropy_ratio=parameters.get('anisotropy_ratio', 1.0),
                anisotropy_angle=parameters.get('anisotropy_angle', 0.0)
            )
            
            # Extract IDW parameters
            idw_params = IDWParameters(
                power=parameters.get('power', 2.0),
                smoothing=parameters.get('smoothing', 0.0)
            )
            
            return IDWInterpolator(search_params, idw_params)
        else:
            raise NotImplementedError(f"Interpolator creation for {method} not implemented")
    
    def _generate_summary(self,
                         characteristics: DataCharacteristics,
                         method_score: MethodScore,
                         optimization: OptimizationResult,
                         quality: Optional[CrossValidationResult]) -> str:
        """Generate human-readable summary."""
        summary_parts = []
        
        # Data summary
        summary_parts.append(f"**Data Analysis Summary**")
        summary_parts.append(f"- Dataset contains {characteristics.n_points} points")
        summary_parts.append(f"- Spatial density: {characteristics.density:.3f} points per unit area")
        summary_parts.append(f"- Distribution uniformity: {characteristics.distribution_uniformity:.2f} (0=clustered, 1=uniform)")
        
        if characteristics.has_trend:
            summary_parts.append(f"- Spatial trend detected: {characteristics.trend_type}")
            
        if characteristics.anisotropy_ratio < 0.8:
            summary_parts.append(f"- Directional variation detected (anisotropy ratio: {characteristics.anisotropy_ratio:.2f})")
        
        # Method recommendation
        summary_parts.append(f"\n**Recommended Method: {method_score.method.value}**")
        summary_parts.append(f"- Suitability score: {method_score.score:.1f}/100")
        summary_parts.append(f"- Key reasons:")
        for reason in method_score.reasons[:3]:  # Top 3 reasons
            summary_parts.append(f"  • {reason}")
        
        # Parameter highlights
        summary_parts.append(f"\n**Optimized Parameters**")
        key_params = ['search_radius', 'power', 'max_points']
        for param in key_params:
            if param in optimization.parameters:
                value = optimization.parameters[param]
                reason = optimization.reasoning.get(param, "")
                summary_parts.append(f"- {param}: {value} ({reason})")
        
        # Quality expectations
        if quality:
            summary_parts.append(f"\n**Expected Performance**")
            summary_parts.append(f"- RMSE: {quality.metrics.rmse:.3f}")
            summary_parts.append(f"- R²: {quality.metrics.r_squared:.3f}")
            summary_parts.append(f"- MAE: {quality.metrics.mae:.3f}")
        
        return "\n".join(summary_parts)
    
    def _check_data_quality(self) -> List[str]:
        """Check for data quality issues."""
        warnings = []
        
        if not self.characteristics:
            return warnings
            
        # Check sample size
        if self.characteristics.n_points < 10:
            warnings.append("Very few data points - results may be unreliable")
        elif self.characteristics.n_points < 30:
            warnings.append("Limited data points - consider collecting more data")
        
        # Check distribution
        if self.characteristics.distribution_uniformity < 0.3:
            warnings.append("Highly clustered data - consider using sectoral search")
        
        # Check outliers
        outlier_ratio = len(self.characteristics.outlier_indices) / self.characteristics.n_points
        if outlier_ratio > 0.1:
            warnings.append(f"High proportion of outliers ({outlier_ratio:.1%}) detected")
        
        # Check value range
        cv = self.characteristics.statistics.get('cv', 0)
        if cv > 2.0:
            warnings.append("Very high variability in values - results may be unstable")
        
        return warnings
    
    def _get_data_insights(self) -> Dict[str, Any]:
        """Get detailed data insights."""
        if not self.characteristics:
            return {}
            
        return {
            "spatial_pattern": self._describe_spatial_pattern(),
            "value_distribution": self._describe_value_distribution(),
            "data_quality": self._assess_data_quality(),
            "special_considerations": self._identify_special_considerations()
        }
    
    def _describe_spatial_pattern(self) -> str:
        """Describe spatial pattern of data."""
        uniformity = self.characteristics.distribution_uniformity
        clustering = self.characteristics.clustering_score
        
        if uniformity > 0.7:
            return "Data points are uniformly distributed"
        elif uniformity > 0.4:
            return "Data shows moderate clustering with some regular areas"
        elif clustering > 0.7:
            return "Data is highly clustered in specific regions"
        else:
            return "Data shows irregular spatial distribution"
    
    def _describe_value_distribution(self) -> str:
        """Describe distribution of values."""
        stats = self.characteristics.statistics
        skewness = stats.get('skewness', 0)
        cv = stats.get('cv', 0)
        
        parts = []
        
        if abs(skewness) < 0.5:
            parts.append("Values are approximately normally distributed")
        elif skewness > 1:
            parts.append("Values are strongly right-skewed")
        elif skewness < -1:
            parts.append("Values are strongly left-skewed")
        
        if cv < 0.3:
            parts.append("with low variability")
        elif cv < 1.0:
            parts.append("with moderate variability")
        else:
            parts.append("with high variability")
            
        return " ".join(parts)
    
    def _assess_data_quality(self) -> str:
        """Assess overall data quality."""
        score = 100
        
        # Deduct for issues
        if self.characteristics.n_points < 30:
            score -= 20
        if self.characteristics.distribution_uniformity < 0.3:
            score -= 15
        if len(self.characteristics.outlier_indices) > self.characteristics.n_points * 0.1:
            score -= 15
        if self.characteristics.statistics.get('cv', 0) > 1.5:
            score -= 10
            
        if score >= 80:
            return "Excellent - suitable for most interpolation methods"
        elif score >= 60:
            return "Good - suitable for interpolation with appropriate parameters"
        elif score >= 40:
            return "Fair - interpolation possible but results may be less reliable"
        else:
            return "Poor - consider data collection improvements"
    
    def _identify_special_considerations(self) -> List[str]:
        """Identify special considerations for interpolation."""
        considerations = []
        
        if self.characteristics.has_trend:
            considerations.append("Consider detrending or using universal kriging")
            
        if self.characteristics.anisotropy_ratio < 0.7:
            considerations.append("Strong anisotropy detected - use directional parameters")
            
        if self.characteristics.dimensions == 3:
            considerations.append("3D interpolation - ensure sufficient vertical sampling")
            
        bounds = list(self.characteristics.bounds.values())
        aspect_ratio = (bounds[1][1] - bounds[1][0]) / (bounds[0][1] - bounds[0][0])
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            considerations.append("Extreme aspect ratio - consider coordinate transformation")
            
        return considerations
    
    def _get_parameter_guidelines(self) -> Dict[str, str]:
        """Get parameter adjustment guidelines."""
        return {
            "search_radius": "Increase for sparse data, decrease for dense data",
            "power": "Increase for more local influence (1-4 range typical)",
            "max_points": "Balance between accuracy (more points) and speed",
            "use_sectors": "Enable for clustered or directional data",
            "anisotropy": "Adjust ratio and angle based on directional patterns"
        }
    
    def _get_best_practices(self) -> List[str]:
        """Get best practices for current data."""
        practices = []
        
        practices.append("Always validate results with cross-validation")
        practices.append("Visualize interpolation results to check for artifacts")
        
        if self.characteristics.n_points < 100:
            practices.append("With limited data, use conservative parameters")
            
        if len(self.characteristics.outlier_indices) > 0:
            practices.append("Consider removing or down-weighting outliers")
            
        if self.characteristics.has_trend:
            practices.append("Account for trends in your interpretation")
            
        return practices
    
    def _identify_potential_issues(self) -> List[str]:
        """Identify potential interpolation issues."""
        issues = []
        
        if self.characteristics.n_points < 20:
            issues.append("Too few points may lead to unreliable interpolation")
            
        edge_ratio = self._calculate_edge_point_ratio()
        if edge_ratio > 0.5:
            issues.append("Many points on edges - extrapolation may be unreliable")
            
        if self.characteristics.distribution_uniformity < 0.2:
            issues.append("Severe clustering may cause interpolation artifacts")
            
        return issues
    
    def _calculate_edge_point_ratio(self) -> float:
        """Calculate ratio of points near data edges."""
        if not self.data or not self.characteristics:
            return 0.0
            
        # Simple heuristic: points within 10% of bounds
        edge_margin = 0.1
        edge_count = 0
        
        for dim_bounds in self.characteristics.bounds.values():
            range_size = dim_bounds[1] - dim_bounds[0]
            margin = range_size * edge_margin
            # Count points near edges (simplified)
            edge_count += np.sum(
                (self.data.values[:, 0] < dim_bounds[0] + margin) |
                (self.data.values[:, 0] > dim_bounds[1] - margin)
            )
            
        return edge_count / (self.characteristics.n_points * len(self.characteristics.bounds))
    
    def _create_minimal_characteristics(self, data: pd.DataFrame) -> DataCharacteristics:
        """Create minimal characteristics when full analysis fails."""
        n_points = len(data)
        return DataCharacteristics(
            n_points=n_points,
            dimensions=2,
            bounds={'x': (0, 1), 'y': (0, 1)},
            density=n_points,
            distribution_uniformity=0.5,
            has_trend=False,
            trend_type=None,
            anisotropy_ratio=1.0,
            anisotropy_angle=0.0,
            statistics={'mean': 0, 'std': 1, 'cv': 1},
            outlier_indices=[],
            clustering_score=0.5,
            nearest_neighbor_stats={'mean_distance': 1.0}
        )