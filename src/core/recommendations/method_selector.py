"""
Method selector for choosing optimal interpolation algorithm.

Analyzes data characteristics to recommend the most suitable
interpolation method for geological survey data.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .data_analyzer import DataCharacteristics


class InterpolationMethod(Enum):
    """Available interpolation methods."""
    IDW = "IDW"
    ORDINARY_KRIGING = "Ordinary Kriging"
    RBF = "Radial Basis Functions"


@dataclass
class MethodScore:
    """Score and reasoning for an interpolation method."""
    method: InterpolationMethod
    score: float  # 0-100
    reasons: List[str]
    pros: List[str]
    cons: List[str]
    suitable_conditions: List[str]


class MethodSelector:
    """
    Selects optimal interpolation method based on data characteristics.
    
    Uses a rule-based scoring system considering:
    - Data density and distribution
    - Spatial correlation
    - Presence of trends
    - Data quality and outliers
    - Computational requirements
    """
    
    def __init__(self):
        """Initialize method selector."""
        self.characteristics: Optional[DataCharacteristics] = None
        self.scores: Dict[InterpolationMethod, MethodScore] = {}
        
    def recommend_method(self, 
                        characteristics: DataCharacteristics,
                        user_preferences: Optional[Dict[str, Any]] = None) -> List[MethodScore]:
        """
        Recommend interpolation methods ranked by suitability.
        
        Args:
            characteristics: Data analysis results
            user_preferences: Optional user preferences (speed, accuracy, etc.)
            
        Returns:
            List of MethodScore objects, sorted by score (descending)
        """
        self.characteristics = characteristics
        user_prefs = user_preferences or {}
        
        # Evaluate each method
        self.scores[InterpolationMethod.IDW] = self._evaluate_idw(user_prefs)
        self.scores[InterpolationMethod.ORDINARY_KRIGING] = self._evaluate_kriging(user_prefs)
        self.scores[InterpolationMethod.RBF] = self._evaluate_rbf(user_prefs)
        
        # Sort by score
        sorted_methods = sorted(
            self.scores.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_methods
    
    def _evaluate_idw(self, user_prefs: Dict[str, Any]) -> MethodScore:
        """Evaluate IDW method suitability."""
        score = 50.0  # Base score
        reasons = []
        pros = []
        cons = []
        
        # Pros of IDW
        pros.extend([
            "Simple and intuitive method",
            "Fast computation",
            "No assumptions about data distribution",
            "Works well for dense, regular data"
        ])
        
        # Data density evaluation
        if self.characteristics.density > 0.1:  # Dense data
            score += 10
            reasons.append("Dense data favors IDW performance")
        else:
            score -= 5
            cons.append("Sparse data may lead to artifacts")
        
        # Distribution uniformity
        if self.characteristics.distribution_uniformity > 0.7:
            score += 15
            reasons.append("Uniform distribution is ideal for IDW")
        elif self.characteristics.distribution_uniformity < 0.3:
            score -= 10
            cons.append("Highly clustered data reduces IDW effectiveness")
        
        # Trend handling
        if self.characteristics.has_trend:
            score -= 15
            cons.append("IDW cannot model global trends")
            reasons.append("Presence of trend reduces IDW suitability")
        else:
            score += 5
            pros.append("No trend - local interpolation sufficient")
        
        # Outliers
        outlier_ratio = len(self.characteristics.outlier_indices) / self.characteristics.n_points
        if outlier_ratio > 0.1:
            score -= 10
            cons.append("IDW sensitive to outliers")
        
        # Anisotropy
        if self.characteristics.anisotropy_ratio < 0.7:
            score += 5
            reasons.append("IDW can handle anisotropy with proper settings")
        
        # User preferences
        if user_prefs.get('prioritize_speed', False):
            score += 10
            reasons.append("IDW is computationally efficient")
        
        # Sample size
        if self.characteristics.n_points < 50:
            score += 5
            reasons.append("IDW works well with small datasets")
        elif self.characteristics.n_points > 10000:
            score -= 5
            cons.append("Large dataset may slow down IDW")
        
        suitable_conditions = [
            "Dense, uniformly distributed data",
            "No global trends",
            "Local variation is primary interest",
            "Need for fast computation",
            "Simple implementation required"
        ]
        
        return MethodScore(
            method=InterpolationMethod.IDW,
            score=np.clip(score, 0, 100),
            reasons=reasons,
            pros=pros,
            cons=cons,
            suitable_conditions=suitable_conditions
        )
    
    def _evaluate_kriging(self, user_prefs: Dict[str, Any]) -> MethodScore:
        """Evaluate Ordinary Kriging method suitability."""
        score = 50.0  # Base score
        reasons = []
        pros = []
        cons = []
        
        # Pros of Kriging
        pros.extend([
            "Provides uncertainty estimates",
            "Optimal linear unbiased estimator",
            "Can model spatial correlation",
            "Handles irregular sampling well"
        ])
        
        # Cons
        cons.extend([
            "Computationally intensive",
            "Requires variogram modeling",
            "Assumes stationarity"
        ])
        
        # Sample size requirements
        if self.characteristics.n_points < 30:
            score -= 20
            cons.append("Too few points for reliable variogram")
            reasons.append("Kriging requires at least 30 points")
        elif self.characteristics.n_points > 100:
            score += 10
            reasons.append("Sufficient data for variogram modeling")
        
        # Spatial correlation (estimated by clustering)
        if self.characteristics.clustering_score > 0.6:
            score += 15
            reasons.append("Spatial correlation detected - ideal for Kriging")
        
        # Distribution
        if self.characteristics.distribution_uniformity < 0.5:
            score += 10
            pros.append("Kriging handles irregular sampling well")
        
        # Trend handling
        if self.characteristics.has_trend:
            score -= 5  # OK can't handle trends directly
            cons.append("Requires detrending or Universal Kriging")
        
        # Data quality
        cv = self.characteristics.statistics.get('cv', 0)
        if cv > 1.0:  # High variability
            score += 5
            reasons.append("High variability benefits from Kriging's approach")
        
        # Outliers
        outlier_ratio = len(self.characteristics.outlier_indices) / self.characteristics.n_points
        if outlier_ratio > 0.05:
            score -= 5
            cons.append("Outliers affect variogram estimation")
        
        # User preferences
        if user_prefs.get('need_uncertainty', False):
            score += 20
            reasons.append("Kriging provides uncertainty estimates")
        
        if user_prefs.get('prioritize_speed', False):
            score -= 15
            reasons.append("Kriging is computationally expensive")
        
        suitable_conditions = [
            "Spatial correlation in data",
            "Need for uncertainty estimates",
            "Irregular sampling patterns",
            "Sufficient data points (>30)",
            "Stationary spatial process"
        ]
        
        return MethodScore(
            method=InterpolationMethod.ORDINARY_KRIGING,
            score=np.clip(score, 0, 100),
            reasons=reasons,
            pros=pros,
            cons=cons,
            suitable_conditions=suitable_conditions
        )
    
    def _evaluate_rbf(self, user_prefs: Dict[str, Any]) -> MethodScore:
        """Evaluate RBF method suitability."""
        score = 50.0  # Base score
        reasons = []
        pros = []
        cons = []
        
        # Pros of RBF
        pros.extend([
            "Creates smooth surfaces",
            "Exact interpolation at data points",
            "Flexible - various basis functions",
            "Can extrapolate beyond data range"
        ])
        
        # Cons
        cons.extend([
            "Can create unrealistic oscillations",
            "Sensitive to parameter choice",
            "Computationally intensive for large datasets"
        ])
        
        # Smoothness requirements
        if user_prefs.get('require_smooth_surface', False):
            score += 20
            reasons.append("RBF creates smooth surfaces")
        
        # Data characteristics
        if self.characteristics.n_points < 20:
            score -= 10
            cons.append("Too few points for stable RBF")
        elif self.characteristics.n_points > 1000:
            score -= 15
            cons.append("RBF becomes slow with many points")
        else:
            score += 10
            reasons.append("Good sample size for RBF")
        
        # Distribution
        if self.characteristics.distribution_uniformity > 0.7:
            score += 10
            reasons.append("Uniform distribution suits RBF")
        
        # Noise and outliers
        outlier_ratio = len(self.characteristics.outlier_indices) / self.characteristics.n_points
        if outlier_ratio > 0.05:
            score -= 15
            cons.append("RBF sensitive to outliers")
            reasons.append("Outliers can cause oscillations")
        
        # Value range
        value_range = self.characteristics.statistics['max'] - self.characteristics.statistics['min']
        if value_range > 0 and self.characteristics.statistics['std'] / value_range < 0.2:
            score += 10
            reasons.append("Smooth variation suits RBF")
        
        # Dimensionality
        if self.characteristics.dimensions == 3:
            score -= 5
            cons.append("RBF computation intensive in 3D")
        
        suitable_conditions = [
            "Need for smooth surfaces",
            "Moderate dataset size (20-1000 points)",
            "Low noise in data",
            "Regular or semi-regular sampling",
            "Interpolation and extrapolation needed"
        ]
        
        return MethodScore(
            method=InterpolationMethod.RBF,
            score=np.clip(score, 0, 100),
            reasons=reasons,
            pros=pros,
            cons=cons,
            suitable_conditions=suitable_conditions
        )
    
    def get_method_comparison(self) -> Dict[str, Any]:
        """
        Get detailed comparison of all methods.
        
        Returns:
            Dictionary with comparison data
        """
        if not self.scores:
            return {}
            
        comparison = {
            'recommended': max(self.scores.values(), key=lambda x: x.score).method.value,
            'scores': {
                method.value: score.score 
                for method, score in self.scores.items()
            },
            'details': {}
        }
        
        for method, score in self.scores.items():
            comparison['details'][method.value] = {
                'score': score.score,
                'pros': score.pros,
                'cons': score.cons,
                'reasons': score.reasons,
                'suitable_for': score.suitable_conditions
            }
            
        return comparison