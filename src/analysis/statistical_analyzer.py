"""
Advanced statistical analysis for geological data.

Provides comprehensive statistical analysis including descriptive statistics,
distribution analysis, and statistical significance testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import warnings
import matplotlib.pyplot as plt
from typing import Union


@dataclass
class StatisticalResults:
    """Container for statistical analysis results."""
    descriptive_stats: Dict[str, float]
    distribution_analysis: Dict[str, Any]
    normality_tests: Dict[str, Any]
    quantile_stats: Dict[str, float]
    variability_measures: Dict[str, float]
    extreme_values: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]


class StatisticalAnalyzer:
    """
    Advanced statistical analyzer for geological data.
    
    Provides comprehensive statistical analysis including:
    - Descriptive statistics
    - Distribution analysis and normality testing
    - Quantile analysis and percentiles
    - Variability measures
    - Extreme value analysis
    - Confidence intervals
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def analyze(self, data: pd.DataFrame, value_col: str) -> StatisticalResults:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            data: Input DataFrame
            value_col: Column name containing values to analyze
            
        Returns:
            StatisticalResults object with complete analysis
        """
        values = data[value_col].dropna().values
        
        if len(values) == 0:
            raise ValueError(f"No valid data found in column '{value_col}'")
        
        # Perform all analyses
        descriptive = self._descriptive_statistics(values)
        distribution = self._distribution_analysis(values)
        normality = self._normality_tests(values)
        quantiles = self._quantile_analysis(values)
        variability = self._variability_measures(values)
        extremes = self._extreme_value_analysis(values)
        confidence = self._confidence_intervals(values)
        
        # Store raw values for enhanced analysis
        descriptive['_raw_values'] = values.tolist()
        
        return StatisticalResults(
            descriptive_stats=descriptive,
            distribution_analysis=distribution,
            normality_tests=normality,
            quantile_stats=quantiles,
            variability_measures=variability,
            extreme_values=extremes,
            confidence_intervals=confidence
        )
    
    def _descriptive_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive descriptive statistics."""
        n = len(values)
        
        stats_dict = {
            'count': float(n),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'mode': float(stats.mode(values, keepdims=True)[0][0]) if n > 0 else 0.0,
            'std': float(np.std(values, ddof=1)),
            'variance': float(np.var(values, ddof=1)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': float(np.ptp(values)),
            'sum': float(np.sum(values)),
        }
        
        # Robust statistics
        stats_dict.update({
            'median_absolute_deviation': float(stats.median_abs_deviation(values)),
            'trimmed_mean_10': float(stats.trim_mean(values, 0.1)),
            'trimmed_mean_20': float(stats.trim_mean(values, 0.2)),
            'harmonic_mean': float(stats.hmean(values)) if np.all(values > 0) else np.nan,
            'geometric_mean': float(stats.gmean(values)) if np.all(values > 0) else np.nan,
        })
        
        return stats_dict
    
    def _distribution_analysis(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze distribution characteristics."""
        analysis = {
            'skewness': float(stats.skew(values)),
            'kurtosis': float(stats.kurtosis(values)),
            'excess_kurtosis': float(stats.kurtosis(values, fisher=True)),
        }
        
        # Interpret skewness
        skew_val = analysis['skewness']
        if abs(skew_val) < 0.5:
            skew_interpretation = 'approximately_symmetric'
        elif skew_val < -0.5:
            skew_interpretation = 'left_skewed'
        else:
            skew_interpretation = 'right_skewed'
        analysis['skewness_interpretation'] = skew_interpretation
        
        # Interpret kurtosis
        kurt_val = analysis['excess_kurtosis']
        if abs(kurt_val) < 0.5:
            kurt_interpretation = 'mesokurtic'
        elif kurt_val < -0.5:
            kurt_interpretation = 'platykurtic'
        else:
            kurt_interpretation = 'leptokurtic'
        analysis['kurtosis_interpretation'] = kurt_interpretation
        
        return analysis
    
    def _normality_tests(self, values: np.ndarray) -> Dict[str, Any]:
        """Perform normality tests."""
        results = {}
        
        if len(values) < 3:
            return {'insufficient_data': True}
        
        # Shapiro-Wilk test (best for small samples)
        if len(values) <= 5000:
            try:
                statistic, p_value = stats.shapiro(values)
                results['shapiro_wilk'] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_normal': p_value > self.alpha
                }
            except Exception as e:
                results['shapiro_wilk'] = {'error': str(e)}
        
        # Anderson-Darling test
        try:
            result = stats.anderson(values, dist='norm')
            results['anderson_darling'] = {
                'statistic': float(result.statistic),
                'critical_values': result.critical_values.tolist(),
                'significance_levels': result.significance_level.tolist(),
                'is_normal': result.statistic < result.critical_values[2]  # 5% level
            }
        except Exception as e:
            results['anderson_darling'] = {'error': str(e)}
        
        # Kolmogorov-Smirnov test
        if len(values) >= 8:
            try:
                # Standardize values
                standardized = (values - np.mean(values)) / np.std(values)
                statistic, p_value = stats.kstest(standardized, 'norm')
                results['kolmogorov_smirnov'] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_normal': p_value > self.alpha
                }
            except Exception as e:
                results['kolmogorov_smirnov'] = {'error': str(e)}
        
        # D'Agostino's test (combines skewness and kurtosis)
        if len(values) >= 20:
            try:
                statistic, p_value = stats.normaltest(values)
                results['dagostino'] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_normal': p_value > self.alpha
                }
            except Exception as e:
                results['dagostino'] = {'error': str(e)}
        
        return results
    
    def _quantile_analysis(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate quantile statistics."""
        quantiles = {}
        
        # Standard percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            quantiles[f'p{p}'] = float(np.percentile(values, p))
        
        # Quartiles and IQR
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        quantiles.update({
            'q1': float(q1),
            'q2': float(q2),
            'q3': float(q3),
            'iqr': float(q3 - q1),
            'quartile_coefficient_dispersion': float((q3 - q1) / (q3 + q1)) if (q3 + q1) != 0 else 0.0
        })
        
        # Quintiles and deciles
        quintiles = np.percentile(values, [20, 40, 60, 80])
        for i, q in enumerate(quintiles, 1):
            quantiles[f'quintile_{i}'] = float(q)
        
        deciles = np.percentile(values, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        for i, d in enumerate(deciles, 1):
            quantiles[f'decile_{i}'] = float(d)
        
        return quantiles
    
    def _variability_measures(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate various measures of variability."""
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        measures = {
            'coefficient_variation': float(std_val / mean_val) if mean_val != 0 else np.inf,
            'relative_std': float(std_val / abs(mean_val)) if mean_val != 0 else np.inf,
            'quartile_deviation': float(np.percentile(values, 75) - np.percentile(values, 25)) / 2,
            'mean_absolute_deviation': float(np.mean(np.abs(values - mean_val))),
            'median_absolute_deviation': float(stats.median_abs_deviation(values)),
            'range_coefficient': float(np.ptp(values) / (np.max(values) + np.min(values))) if (np.max(values) + np.min(values)) != 0 else 0.0,
        }
        
        # Robust variability measures
        measures.update({
            'robust_cv': float(stats.median_abs_deviation(values) / np.median(values)) if np.median(values) != 0 else np.inf,
            'gini_coefficient': self._calculate_gini(values),
        })
        
        return measures
    
    def _extreme_value_analysis(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze extreme values and outliers."""
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        
        # IQR method outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mild_outliers = np.sum((values < lower_bound) | (values > upper_bound))
        
        # Extreme outliers
        extreme_lower = q1 - 3.0 * iqr
        extreme_upper = q3 + 3.0 * iqr
        extreme_outliers = np.sum((values < extreme_lower) | (values > extreme_upper))
        
        # Z-score outliers
        z_scores = np.abs(stats.zscore(values))
        z_outliers_2 = np.sum(z_scores > 2)
        z_outliers_3 = np.sum(z_scores > 3)
        
        analysis = {
            'iqr_method': {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'mild_outliers': int(mild_outliers),
                'extreme_outliers': int(extreme_outliers),
                'outlier_percentage': float(mild_outliers / len(values) * 100)
            },
            'z_score_method': {
                'outliers_2sigma': int(z_outliers_2),
                'outliers_3sigma': int(z_outliers_3),
                'max_z_score': float(np.max(z_scores)),
                'outlier_percentage_2sigma': float(z_outliers_2 / len(values) * 100)
            },
            'extreme_values': {
                'smallest_values': values[np.argsort(values)[:5]].tolist(),
                'largest_values': values[np.argsort(values)[-5:]].tolist(),
            }
        }
        
        return analysis
    
    def _confidence_intervals(self, values: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for various parameters."""
        n = len(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        intervals = {}
        
        # Confidence interval for mean
        if n >= 2:
            t_critical = stats.t.ppf(1 - self.alpha/2, df=n-1)
            margin_error = t_critical * std_val / np.sqrt(n)
            intervals['mean'] = (
                float(mean_val - margin_error),
                float(mean_val + margin_error)
            )
        
        # Bootstrap confidence interval for median
        if n >= 10:
            try:
                bootstrap_medians = []
                for _ in range(1000):
                    bootstrap_sample = np.random.choice(values, size=n, replace=True)
                    bootstrap_medians.append(np.median(bootstrap_sample))
                
                lower_percentile = (self.alpha/2) * 100
                upper_percentile = (1 - self.alpha/2) * 100
                intervals['median'] = (
                    float(np.percentile(bootstrap_medians, lower_percentile)),
                    float(np.percentile(bootstrap_medians, upper_percentile))
                )
            except Exception:
                pass
        
        # Confidence interval for standard deviation
        if n >= 2:
            chi2_lower = stats.chi2.ppf(self.alpha/2, df=n-1)
            chi2_upper = stats.chi2.ppf(1 - self.alpha/2, df=n-1)
            
            intervals['std'] = (
                float(np.sqrt((n-1) * std_val**2 / chi2_upper)),
                float(np.sqrt((n-1) * std_val**2 / chi2_lower))
            )
        
        return intervals
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient as a measure of inequality."""
        # Handle negative values by shifting
        shifted_values = values - np.min(values) + 1e-10
        
        n = len(shifted_values)
        index = np.arange(1, n + 1)
        sorted_values = np.sort(shifted_values)
        
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        
        return float(gini)
    
    def grubbs_test(self, values: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """Perform Grubbs test for outlier detection.
        
        Args:
            values: Data values
            alpha: Significance level (default 0.05)
            
        Returns:
            Dictionary with test results and outliers
        """
        n = len(values)
        if n < 3:
            return {'error': 'Insufficient data for Grubbs test (need n >= 3)'}
        
        results = {
            'outliers': [],
            'indices': [],
            'test_statistics': [],
            'critical_values': [],
            'n_iterations': 0
        }
        
        # Copy values to avoid modifying original
        test_values = values.copy()
        original_indices = np.arange(len(values))
        
        while len(test_values) > 2:
            n = len(test_values)
            mean_val = np.mean(test_values)
            std_val = np.std(test_values, ddof=1)
            
            if std_val == 0:
                break
            
            # Calculate test statistic for each value
            z_scores = np.abs(test_values - mean_val) / std_val
            max_idx = np.argmax(z_scores)
            G = z_scores[max_idx]
            
            # Critical value from t-distribution
            t_val = stats.t.ppf(1 - alpha/(2*n), n-2)
            G_critical = ((n-1) / np.sqrt(n)) * np.sqrt(t_val**2 / (n-2 + t_val**2))
            
            if G > G_critical:
                # Outlier detected
                outlier_val = test_values[max_idx]
                outlier_original_idx = original_indices[max_idx]
                
                results['outliers'].append(float(outlier_val))
                results['indices'].append(int(outlier_original_idx))
                results['test_statistics'].append(float(G))
                results['critical_values'].append(float(G_critical))
                
                # Remove outlier and continue
                test_values = np.delete(test_values, max_idx)
                original_indices = np.delete(original_indices, max_idx)
                results['n_iterations'] += 1
            else:
                # No more outliers
                break
        
        results['n_outliers'] = len(results['outliers'])
        results['outlier_percentage'] = float(results['n_outliers'] / len(values) * 100)
        
        return results
    
    def transform_data(self, values: np.ndarray, method: str = 'auto') -> Dict[str, Any]:
        """Apply data transformation to improve normality.
        
        Args:
            values: Data values
            method: Transformation method ('auto', 'log', 'sqrt', 'boxcox', 'yeojohnson')
            
        Returns:
            Dictionary with transformed data and transformation parameters
        """
        results = {
            'method': method,
            'transformed_values': None,
            'parameters': {},
            'normality_before': {},
            'normality_after': {},
            'success': False
        }
        
        # Check normality before transformation
        if len(values) >= 3:
            try:
                stat_before, p_before = stats.shapiro(values)
                results['normality_before'] = {
                    'shapiro_statistic': float(stat_before),
                    'p_value': float(p_before),
                    'is_normal': p_before > self.alpha
                }
            except:
                pass
        
        # Apply transformation
        try:
            if method == 'auto':
                # Try different methods and choose best
                methods_to_try = ['none', 'log', 'sqrt', 'yeojohnson']
                if np.all(values > 0):
                    methods_to_try.append('boxcox')
                
                best_p_value = 0
                best_method = 'none'
                best_transformed = values
                best_params = {}
                
                for m in methods_to_try:
                    try:
                        if m == 'none':
                            transformed = values
                            params = {}
                        elif m == 'log':
                            if np.all(values > 0):
                                transformed = np.log(values)
                                params = {}
                            else:
                                continue
                        elif m == 'sqrt':
                            if np.all(values >= 0):
                                transformed = np.sqrt(values)
                                params = {}
                            else:
                                continue
                        elif m == 'boxcox':
                            if np.all(values > 0):
                                transformed, lambda_param = boxcox(values)
                                params = {'lambda': float(lambda_param)}
                            else:
                                continue
                        elif m == 'yeojohnson':
                            transformed, lambda_param = yeojohnson(values)
                            params = {'lambda': float(lambda_param)}
                        
                        # Test normality
                        if len(transformed) >= 3:
                            stat, p_val = stats.shapiro(transformed)
                            if p_val > best_p_value:
                                best_p_value = p_val
                                best_method = m
                                best_transformed = transformed
                                best_params = params
                    except:
                        continue
                
                results['method'] = best_method
                results['transformed_values'] = best_transformed
                results['parameters'] = best_params
                
            elif method == 'log':
                if np.all(values > 0):
                    results['transformed_values'] = np.log(values)
                else:
                    # Log(x + c) transformation for non-positive values
                    c = abs(np.min(values)) + 1
                    results['transformed_values'] = np.log(values + c)
                    results['parameters'] = {'shift': float(c)}
                    
            elif method == 'sqrt':
                if np.all(values >= 0):
                    results['transformed_values'] = np.sqrt(values)
                else:
                    # Shift to make all values non-negative
                    c = abs(np.min(values))
                    results['transformed_values'] = np.sqrt(values + c)
                    results['parameters'] = {'shift': float(c)}
                    
            elif method == 'boxcox':
                if np.all(values > 0):
                    transformed, lambda_param = boxcox(values)
                    results['transformed_values'] = transformed
                    results['parameters'] = {'lambda': float(lambda_param)}
                else:
                    raise ValueError("Box-Cox requires all positive values")
                    
            elif method == 'yeojohnson':
                transformed, lambda_param = yeojohnson(values)
                results['transformed_values'] = transformed
                results['parameters'] = {'lambda': float(lambda_param)}
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['transformed_values'] = values  # Return original if transformation fails
        
        # Check normality after transformation
        if results['transformed_values'] is not None and len(results['transformed_values']) >= 3:
            try:
                stat_after, p_after = stats.shapiro(results['transformed_values'])
                results['normality_after'] = {
                    'shapiro_statistic': float(stat_after),
                    'p_value': float(p_after),
                    'is_normal': p_after > self.alpha
                }
            except:
                pass
        
        return results
    
    def inverse_transform(self, transformed_values: np.ndarray, method: str, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply inverse transformation to get back original scale.
        
        Args:
            transformed_values: Transformed data values
            method: Transformation method used
            parameters: Transformation parameters
            
        Returns:
            Original scale values
        """
        try:
            if method == 'none':
                return transformed_values
                
            elif method == 'log':
                values = np.exp(transformed_values)
                if 'shift' in parameters:
                    values = values - parameters['shift']
                return values
                
            elif method == 'sqrt':
                values = transformed_values ** 2
                if 'shift' in parameters:
                    values = values - parameters['shift']
                return values
                
            elif method == 'boxcox':
                lambda_param = parameters.get('lambda', 0)
                if lambda_param == 0:
                    return np.exp(transformed_values)
                else:
                    return (transformed_values * lambda_param + 1) ** (1 / lambda_param)
                    
            elif method == 'yeojohnson':
                # Inverse Yeo-Johnson transformation
                lambda_param = parameters.get('lambda', 0)
                
                # Handle the inverse transformation based on lambda and value sign
                result = np.zeros_like(transformed_values)
                
                for i, y in enumerate(transformed_values):
                    if lambda_param != 0:
                        if y >= 0:
                            result[i] = (y * lambda_param + 1) ** (1 / lambda_param) - 1
                        else:
                            result[i] = 1 - ((-y) * (2 - lambda_param) + 1) ** (1 / (2 - lambda_param))
                    else:
                        if y >= 0:
                            result[i] = np.exp(y) - 1
                        else:
                            result[i] = 1 - np.exp(-y)
                
                return result
                    
        except Exception:
            # Return transformed values if inverse fails
            return transformed_values
    
    def plot_qq(self, values: np.ndarray, ax=None, title: str = "Q-Q Plot"):
        """Create Q-Q plot for normality assessment.
        
        Args:
            values: Data values
            ax: Matplotlib axis (optional)
            title: Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_boxplot(self, values: np.ndarray, ax=None, title: str = "Box Plot", 
                     show_outliers: bool = True, outlier_method: str = 'iqr'):
        """Create box plot with outlier detection.
        
        Args:
            values: Data values  
            ax: Matplotlib axis (optional)
            title: Plot title
            show_outliers: Whether to highlight outliers
            outlier_method: Method for outlier detection ('iqr' or 'grubbs')
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create box plot
        bp = ax.boxplot(values, vert=True, patch_artist=True)
        
        # Customize box plot
        bp['boxes'][0].set_facecolor('lightblue')
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        
        # Add outliers if requested
        if show_outliers:
            if outlier_method == 'grubbs':
                grubbs_result = self.grubbs_test(values)
                outlier_indices = grubbs_result.get('indices', [])
                if outlier_indices:
                    outlier_values = values[outlier_indices]
                    ax.scatter(np.ones(len(outlier_values)), outlier_values, 
                              color='red', s=100, marker='o', label='Grubbs outliers')
            
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        if show_outliers and outlier_method == 'grubbs':
            ax.legend()
        
        return ax
    
    def generate_enhanced_report(self, results: StatisticalResults, value_col: str,
                               include_outliers: bool = True,
                               include_transformation: bool = True) -> str:
        """Generate enhanced statistical analysis report.
        
        Args:
            results: Statistical analysis results
            value_col: Name of analyzed column
            include_outliers: Include outlier analysis
            include_transformation: Include transformation analysis
            
        Returns:
            Formatted report string
        """
        report = self.generate_summary_report(results, value_col)
        
        if include_outliers:
            # Add Grubbs test results
            values = results.descriptive_stats.get('_raw_values', [])
            if len(values) > 0:
                grubbs_result = self.grubbs_test(np.array(values))
                if grubbs_result.get('n_outliers', 0) > 0:
                    report += f"""\nGRUBBS TEST OUTLIERS:
• Number of outliers: {grubbs_result['n_outliers']}
• Outlier values: {', '.join(f"{v:.3f}" for v in grubbs_result['outliers'][:5])}
• Outlier percentage: {grubbs_result['outlier_percentage']:.1f}%
"""
        
        if include_transformation:
            # Add transformation recommendation
            values = results.descriptive_stats.get('_raw_values', [])
            if len(values) > 0:
                transform_result = self.transform_data(np.array(values))
                if transform_result['success']:
                    report += f"""\nDATA TRANSFORMATION:
• Recommended method: {transform_result['method']}
• Normality before: p={transform_result['normality_before'].get('p_value', 0):.4f}
• Normality after: p={transform_result['normality_after'].get('p_value', 0):.4f}
"""
                    if transform_result['parameters']:
                        report += f"• Parameters: {transform_result['parameters']}\n"
        
        return report
    
    def generate_summary_report(self, results: StatisticalResults, value_col: str) -> str:
        """Generate a human-readable summary report."""
        stats = results.descriptive_stats
        dist = results.distribution_analysis
        
        report = f"""
STATISTICAL ANALYSIS REPORT - {value_col.upper()}
{'=' * 50}

BASIC STATISTICS:
• Count: {stats['count']:.0f}
• Mean: {stats['mean']:.3f}
• Median: {stats['median']:.3f}
• Standard Deviation: {stats['std']:.3f}
• Range: {stats['min']:.3f} to {stats['max']:.3f}

DISTRIBUTION CHARACTERISTICS:
• Skewness: {dist['skewness']:.3f} ({dist['skewness_interpretation']})
• Kurtosis: {dist['kurtosis']:.3f} ({dist['kurtosis_interpretation']})

VARIABILITY:
• Coefficient of Variation: {results.variability_measures['coefficient_variation']:.3f}
• Interquartile Range: {results.quantile_stats['iqr']:.3f}

OUTLIERS:
• IQR Method: {results.extreme_values['iqr_method']['mild_outliers']} outliers ({results.extreme_values['iqr_method']['outlier_percentage']:.1f}%)
• Z-Score Method: {results.extreme_values['z_score_method']['outliers_2sigma']} outliers (>2σ)

CONFIDENCE INTERVALS ({self.confidence_level*100:.0f}%):
"""
        
        for param, (lower, upper) in results.confidence_intervals.items():
            report += f"• {param.capitalize()}: [{lower:.3f}, {upper:.3f}]\n"
        
        return report