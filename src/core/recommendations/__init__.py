"""
Recommendation system for optimal interpolation.

This module provides automatic analysis of geological data
and recommends optimal interpolation methods and parameters.
"""

from .data_analyzer import DataAnalyzer
from .method_selector import MethodSelector
from .parameter_optimizer import ParameterOptimizer
from .quality_evaluator import QualityEvaluator
from .recommendation_engine import RecommendationEngine

__all__ = [
    'DataAnalyzer',
    'MethodSelector', 
    'ParameterOptimizer',
    'QualityEvaluator',
    'RecommendationEngine'
]