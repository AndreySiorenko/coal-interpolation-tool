"""
Advanced correlation analysis for geological data.

Provides comprehensive correlation analysis including Pearson, Spearman,
Kendall correlations, partial correlations, and significance testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from itertools import combinations
import warnings


@dataclass
class CorrelationResults:
    """Container for correlation analysis results."""
    correlation_matrices: Dict[str, np.ndarray]
    significance_tests: Dict[str, Any]
    partial_correlations: Dict[str, Any]
    correlation_summary: Dict[str, Any]
    relationship_analysis: Dict[str, Any]


class CorrelationAnalyzer:
    """
    Advanced correlation analyzer for geological data.
    
    Provides comprehensive correlation analysis including:
    - Pearson, Spearman, and Kendall correlations
    - Significance testing for correlations
    - Partial correlation analysis
    - Cross-correlation with spatial lags
    - Correlation strength interpretation
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize correlation analyzer.
        
        Args:
            alpha: Significance level for statistical tests (default 0.05)
        """
        self.alpha = alpha
        self.data: Optional[pd.DataFrame] = None
        self.numeric_columns: List[str] = []
    
    def analyze(self,
                data: pd.DataFrame,
                variables: Optional[List[str]] = None,
                include_coordinates: bool = True) -> CorrelationResults:
        """
        Perform comprehensive correlation analysis.
        
        Args:
            data: Input DataFrame
            variables: List of variables to analyze (default: all numeric)
            include_coordinates: Whether to include coordinate columns
            
        Returns:
            CorrelationResults with complete analysis
        """
        self.data = data.copy()
        
        # Select numeric columns
        if variables is None:
            self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_columns = [col for col in variables if col in data.columns]
        
        if len(self.numeric_columns) < 2:
            raise ValueError("At least 2 numeric columns required for correlation analysis")
        
        # Remove columns with no variation
        for col in self.numeric_columns.copy():
            if data[col].nunique() <= 1:
                self.numeric_columns.remove(col)
                warnings.warn(f"Removed column '{col}' - no variation")
        
        if len(self.numeric_columns) < 2:
            raise ValueError("At least 2 variables with variation required")
        
        # Perform analyses
        correlation_matrices = self._calculate_correlation_matrices()
        significance_tests = self._perform_significance_tests()
        partial_correlations = self._calculate_partial_correlations()
        correlation_summary = self._generate_correlation_summary(correlation_matrices)
        relationship_analysis = self._analyze_relationships(correlation_matrices, significance_tests)
        
        return CorrelationResults(
            correlation_matrices=correlation_matrices,
            significance_tests=significance_tests,
            partial_correlations=partial_correlations,
            correlation_summary=correlation_summary,
            relationship_analysis=relationship_analysis
        )
    
    def _calculate_correlation_matrices(self) -> Dict[str, np.ndarray]:
        """Calculate different types of correlation matrices."""
        subset_data = self.data[self.numeric_columns]
        
        matrices = {}
        
        # Pearson correlation
        matrices['pearson'] = subset_data.corr(method='pearson').values
        
        # Spearman correlation (rank-based)
        matrices['spearman'] = subset_data.corr(method='spearman').values
        
        # Kendall correlation (tau)
        matrices['kendall'] = subset_data.corr(method='kendall').values
        
        return matrices
    
    def _perform_significance_tests(self) -> Dict[str, Any]:
        """Perform significance tests for correlations."""
        subset_data = self.data[self.numeric_columns].dropna()
        n_vars = len(self.numeric_columns)
        n_obs = len(subset_data)
        
        tests = {}
        
        # Test each pair of variables
        for method in ['pearson', 'spearman', 'kendall']:
            test_results = {
                'p_values': np.ones((n_vars, n_vars)),
                'significant': np.zeros((n_vars, n_vars), dtype=bool),
                'confidence_intervals': {}
            }
            
            for i, col1 in enumerate(self.numeric_columns):
                for j, col2 in enumerate(self.numeric_columns):
                    if i <= j:  # Only calculate upper triangle
                        if i == j:
                            test_results['p_values'][i, j] = 0.0
                            test_results['significant'][i, j] = True
                            continue
                        
                        x = subset_data[col1].values
                        y = subset_data[col2].values
                        
                        # Remove pairs with NaN
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean, y_clean = x[mask], y[mask]
                        
                        if len(x_clean) < 3:
                            continue
                        
                        try:
                            if method == 'pearson':
                                corr, p_value = stats.pearsonr(x_clean, y_clean)
                                # Confidence interval for Pearson
                                ci = self._pearson_confidence_interval(corr, len(x_clean))
                                test_results['confidence_intervals'][f'{col1}_{col2}'] = ci
                            elif method == 'spearman':
                                corr, p_value = stats.spearmanr(x_clean, y_clean)
                            else:  # kendall
                                corr, p_value = stats.kendalltau(x_clean, y_clean)
                            
                            test_results['p_values'][i, j] = p_value
                            test_results['p_values'][j, i] = p_value  # Symmetric
                            
                            is_significant = p_value < self.alpha
                            test_results['significant'][i, j] = is_significant
                            test_results['significant'][j, i] = is_significant
                            
                        except Exception as e:
                            warnings.warn(f"Error calculating {method} correlation for {col1}-{col2}: {e}")
            
            tests[method] = test_results
        
        return tests
    
    def _calculate_partial_correlations(self) -> Dict[str, Any]:
        """Calculate partial correlations."""
        if len(self.numeric_columns) < 3:
            return {'error': 'Need at least 3 variables for partial correlation'}
        
        subset_data = self.data[self.numeric_columns].dropna()
        
        if len(subset_data) < 10:
            return {'error': 'Insufficient data for partial correlation'}
        
        partial_results = {}
        
        # Calculate partial correlations for each pair, controlling for others
        for i, var1 in enumerate(self.numeric_columns):
            for j, var2 in enumerate(self.numeric_columns):
                if i >= j:
                    continue
                
                # Control variables (all others)
                control_vars = [var for var in self.numeric_columns if var not in [var1, var2]]
                
                if len(control_vars) == 0:
                    continue
                
                try:
                    partial_corr = self._calculate_partial_correlation_pair(
                        subset_data, var1, var2, control_vars
                    )
                    
                    partial_results[f'{var1}_{var2}'] = {
                        'partial_correlation': partial_corr,
                        'controlling_for': control_vars,
                        'n_observations': len(subset_data)
                    }
                    
                except Exception as e:
                    warnings.warn(f"Error calculating partial correlation for {var1}-{var2}: {e}")
        
        return partial_results
    
    def _calculate_partial_correlation_pair(self,
                                          data: pd.DataFrame,
                                          var1: str,
                                          var2: str,
                                          control_vars: List[str]) -> float:
        """Calculate partial correlation between two variables controlling for others."""
        # Method: Residual correlation after removing effect of control variables
        
        # Regress var1 on control variables
        X_control = data[control_vars].values
        y1 = data[var1].values
        y2 = data[var2].values
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X_control)), X_control])
        
        # Linear regression for var1
        try:
            coeffs1, _, _, _ = np.linalg.lstsq(X_with_intercept, y1, rcond=None)
            residuals1 = y1 - X_with_intercept @ coeffs1
        except:
            residuals1 = y1
        
        # Linear regression for var2
        try:
            coeffs2, _, _, _ = np.linalg.lstsq(X_with_intercept, y2, rcond=None)
            residuals2 = y2 - X_with_intercept @ coeffs2
        except:
            residuals2 = y2
        
        # Correlation of residuals
        if len(residuals1) > 1 and len(residuals2) > 1:
            corr, _ = stats.pearsonr(residuals1, residuals2)
            return float(corr)
        else:
            return 0.0
    
    def _generate_correlation_summary(self, matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate summary statistics for correlation matrices."""
        summary = {}
        
        for method, matrix in matrices.items():
            # Get upper triangle (excluding diagonal)
            n_vars = len(self.numeric_columns)
            upper_triangle = []
            
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    upper_triangle.append(matrix[i, j])
            
            upper_triangle = np.array(upper_triangle)
            upper_triangle = upper_triangle[~np.isnan(upper_triangle)]
            
            if len(upper_triangle) > 0:
                summary[method] = {
                    'mean_correlation': float(np.mean(np.abs(upper_triangle))),
                    'max_correlation': float(np.max(np.abs(upper_triangle))),
                    'min_correlation': float(np.min(np.abs(upper_triangle))),
                    'std_correlation': float(np.std(upper_triangle)),
                    'n_strong_correlations': int(np.sum(np.abs(upper_triangle) > 0.7)),
                    'n_moderate_correlations': int(np.sum((np.abs(upper_triangle) > 0.3) & (np.abs(upper_triangle) <= 0.7))),
                    'n_weak_correlations': int(np.sum(np.abs(upper_triangle) <= 0.3)),
                    'correlation_distribution': {
                        'q25': float(np.percentile(upper_triangle, 25)),
                        'q50': float(np.percentile(upper_triangle, 50)),
                        'q75': float(np.percentile(upper_triangle, 75))
                    }
                }
        
        return summary
    
    def _analyze_relationships(self,
                             matrices: Dict[str, np.ndarray],
                             significance_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between variables."""
        analysis = {
            'strongest_correlations': {},
            'variable_connectivity': {},
            'correlation_patterns': {},
            'multicollinearity_check': {}
        }
        
        # Find strongest correlations for each method
        for method, matrix in matrices.items():
            correlations = []
            n_vars = len(self.numeric_columns)
            
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not np.isnan(matrix[i, j]):
                        correlations.append({
                            'variables': (self.numeric_columns[i], self.numeric_columns[j]),
                            'correlation': float(matrix[i, j]),
                            'abs_correlation': float(abs(matrix[i, j])),
                            'significant': significance_tests[method]['significant'][i, j] if method in significance_tests else False
                        })
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            analysis['strongest_correlations'][method] = correlations[:10]  # Top 10
        
        # Variable connectivity (how many strong correlations each variable has)
        pearson_matrix = matrices.get('pearson', matrices[list(matrices.keys())[0]])
        connectivity = {}
        
        for i, var in enumerate(self.numeric_columns):
            strong_connections = 0
            for j in range(len(self.numeric_columns)):
                if i != j and not np.isnan(pearson_matrix[i, j]):
                    if abs(pearson_matrix[i, j]) > 0.5:  # Strong correlation threshold
                        strong_connections += 1
            connectivity[var] = strong_connections
        
        analysis['variable_connectivity'] = connectivity
        
        # Correlation patterns
        analysis['correlation_patterns'] = self._identify_correlation_patterns(matrices)
        
        # Multicollinearity check
        analysis['multicollinearity_check'] = self._check_multicollinearity(pearson_matrix)
        
        return analysis
    
    def _identify_correlation_patterns(self, matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Identify patterns in correlation matrices."""
        patterns = {}
        
        pearson_matrix = matrices.get('pearson')
        if pearson_matrix is None:
            return {'error': 'Pearson matrix not available'}
        
        n_vars = len(self.numeric_columns)
        
        # Hierarchical clustering of variables based on correlations
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(pearson_matrix)
            np.fill_diagonal(distance_matrix, 0)
            
            # Convert to condensed form for linkage
            condensed_distances = squareform(distance_matrix, checks=False)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Get clusters (try different numbers of clusters)
            clusters_2 = fcluster(linkage_matrix, 2, criterion='maxclust')
            clusters_3 = fcluster(linkage_matrix, min(3, n_vars), criterion='maxclust')
            
            patterns['clustering'] = {
                'linkage_matrix': linkage_matrix.tolist(),
                'clusters_2': {
                    'labels': clusters_2.tolist(),
                    'groups': {f'cluster_{i}': [self.numeric_columns[j] for j, label in enumerate(clusters_2) if label == i] 
                             for i in range(1, 3)}
                },
                'clusters_3': {
                    'labels': clusters_3.tolist(),
                    'groups': {f'cluster_{i}': [self.numeric_columns[j] for j, label in enumerate(clusters_3) if label == i] 
                             for i in range(1, min(4, n_vars + 1))}
                }
            }
            
        except ImportError:
            patterns['clustering'] = {'error': 'scipy.cluster not available'}
        except Exception as e:
            patterns['clustering'] = {'error': str(e)}
        
        # Block structure detection
        patterns['blocks'] = self._detect_correlation_blocks(pearson_matrix)
        
        return patterns
    
    def _detect_correlation_blocks(self, correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """Detect block structure in correlation matrix."""
        n_vars = len(self.numeric_columns)
        
        if n_vars < 4:
            return {'error': 'Need at least 4 variables for block detection'}
        
        # Simple block detection: find groups of highly correlated variables
        blocks = []
        used_vars = set()
        
        for i in range(n_vars):
            if i in used_vars:
                continue
            
            # Find variables highly correlated with variable i
            block = [i]
            for j in range(n_vars):
                if j != i and j not in used_vars:
                    if abs(correlation_matrix[i, j]) > 0.6:  # High correlation threshold
                        block.append(j)
                        used_vars.add(j)
            
            if len(block) > 1:
                block_vars = [self.numeric_columns[idx] for idx in block]
                blocks.append({
                    'variables': block_vars,
                    'size': len(block_vars),
                    'avg_correlation': float(np.mean([abs(correlation_matrix[idx1, idx2]) 
                                                    for idx1 in block for idx2 in block if idx1 != idx2]))
                })
                used_vars.update(block)
        
        return {
            'n_blocks': len(blocks),
            'blocks': blocks,
            'block_coverage': len(used_vars) / n_vars
        }
    
    def _check_multicollinearity(self, correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """Check for multicollinearity issues."""
        n_vars = len(self.numeric_columns)
        
        # High correlation pairs (potential multicollinearity)
        high_corr_pairs = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if not np.isnan(correlation_matrix[i, j]) and abs(correlation_matrix[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'variable_1': self.numeric_columns[i],
                        'variable_2': self.numeric_columns[j],
                        'correlation': float(correlation_matrix[i, j])
                    })
        
        # Determinant of correlation matrix (near zero indicates multicollinearity)
        try:
            det = np.linalg.det(correlation_matrix)
            condition_number = np.linalg.cond(correlation_matrix)
        except:
            det = np.nan
            condition_number = np.nan
        
        return {
            'high_correlation_pairs': high_corr_pairs,
            'n_high_correlations': len(high_corr_pairs),
            'matrix_determinant': float(det) if not np.isnan(det) else None,
            'condition_number': float(condition_number) if not np.isnan(condition_number) else None,
            'multicollinearity_risk': 'high' if len(high_corr_pairs) > 0 or (not np.isnan(condition_number) and condition_number > 30) else 'low'
        }
    
    def _pearson_confidence_interval(self, r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for Pearson correlation coefficient."""
        if n < 3:
            return (np.nan, np.nan)
        
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        
        # Critical value
        alpha = 1 - confidence
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval in z-space
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        
        # Transform back to correlation space
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (float(r_lower), float(r_upper))
    
    def generate_correlation_report(self, results: CorrelationResults) -> str:
        """Generate a human-readable correlation analysis report."""
        report = """
CORRELATION ANALYSIS REPORT
===========================

SUMMARY STATISTICS:
"""
        
        # Add summary for each correlation method
        for method, summary in results.correlation_summary.items():
            report += f"""
{method.upper()} CORRELATIONS:
• Mean absolute correlation: {summary['mean_correlation']:.3f}
• Strongest correlation: {summary['max_correlation']:.3f}
• Standard deviation: {summary['std_correlation']:.3f}
• Strong correlations (|r| > 0.7): {summary['n_strong_correlations']}
• Moderate correlations (0.3 < |r| ≤ 0.7): {summary['n_moderate_correlations']}
• Weak correlations (|r| ≤ 0.3): {summary['n_weak_correlations']}
"""
        
        # Top correlations
        if 'pearson' in results.correlation_matrices:
            pearson_strongest = results.relationship_analysis['strongest_correlations']['pearson'][:5]
            report += "\nSTRONGEST PEARSON CORRELATIONS:\n"
            for i, corr in enumerate(pearson_strongest, 1):
                var1, var2 = corr['variables']
                r = corr['correlation']
                sig = "***" if corr['significant'] else ""
                report += f"{i}. {var1} ↔ {var2}: r = {r:.3f} {sig}\n"
        
        # Multicollinearity warning
        mc_check = results.relationship_analysis['multicollinearity_check']
        if mc_check['multicollinearity_risk'] == 'high':
            report += f"""
⚠️  MULTICOLLINEARITY WARNING:
• {mc_check['n_high_correlations']} pairs with |r| > 0.8
• Consider removing redundant variables
"""
        
        # Variable connectivity
        connectivity = results.relationship_analysis['variable_connectivity']
        most_connected = max(connectivity.items(), key=lambda x: x[1])
        report += f"""
VARIABLE CONNECTIVITY:
• Most connected variable: {most_connected[0]} ({most_connected[1]} strong correlations)
"""
        
        return report