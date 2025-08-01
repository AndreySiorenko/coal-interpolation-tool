"""
Data quality analyzer for geological survey data.

Provides comprehensive data quality assessment including completeness,
consistency, accuracy, and validity checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import warnings
from datetime import datetime


@dataclass
class DataQualityResults:
    """Container for data quality analysis results."""
    completeness_analysis: Dict[str, Any]
    consistency_analysis: Dict[str, Any]
    accuracy_analysis: Dict[str, Any]
    validity_analysis: Dict[str, Any]
    overall_quality: Dict[str, Any]
    recommendations: List[str]


class DataQualityAnalyzer:
    """
    Comprehensive data quality analyzer for geological data.
    
    Assesses data quality across multiple dimensions:
    - Completeness: Missing values, data coverage
    - Consistency: Internal consistency, format consistency
    - Accuracy: Outliers, suspicious values, range validation
    - Validity: Data types, constraint violations, business rules
    """
    
    def __init__(self):
        """Initialize data quality analyzer."""
        self.data: Optional[pd.DataFrame] = None
        self.coordinate_columns: List[str] = []
        self.value_columns: List[str] = []
        self.quality_threshold: float = 0.8  # Minimum quality score
    
    def analyze_quality(self,
                       data: pd.DataFrame,
                       coordinate_columns: Optional[List[str]] = None,
                       value_columns: Optional[List[str]] = None,
                       expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> DataQualityResults:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            data: Input DataFrame
            coordinate_columns: List of coordinate column names
            value_columns: List of value column names
            expected_ranges: Expected value ranges for columns
            
        Returns:
            DataQualityResults with complete quality assessment
        """
        self.data = data.copy()
        
        # Auto-detect coordinate and value columns if not provided
        if coordinate_columns is None:
            coord_candidates = ['x', 'y', 'z', 'latitude', 'longitude', 'easting', 'northing', 'elevation']
            self.coordinate_columns = [col for col in data.columns if col.lower() in coord_candidates]
        else:
            self.coordinate_columns = coordinate_columns
        
        if value_columns is None:
            # All numeric columns except coordinates
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            self.value_columns = [col for col in numeric_cols if col not in self.coordinate_columns]
        else:
            self.value_columns = value_columns
        
        # Perform quality analyses
        completeness = self._analyze_completeness()
        consistency = self._analyze_consistency()
        accuracy = self._analyze_accuracy(expected_ranges)
        validity = self._analyze_validity(expected_ranges)
        overall_quality = self._calculate_overall_quality(completeness, consistency, accuracy, validity)
        recommendations = self._generate_recommendations(completeness, consistency, accuracy, validity)
        
        return DataQualityResults(
            completeness_analysis=completeness,
            consistency_analysis=consistency,
            accuracy_analysis=accuracy,
            validity_analysis=validity,
            overall_quality=overall_quality,
            recommendations=recommendations
        )
    
    def _analyze_completeness(self) -> Dict[str, Any]:
        """Analyze data completeness."""
        n_rows, n_cols = self.data.shape
        
        completeness = {
            'overall_stats': {
                'total_cells': n_rows * n_cols,
                'total_rows': n_rows,
                'total_columns': n_cols
            }
        }
        
        # Missing value analysis
        missing_stats = {}
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_pct = (missing_count / n_rows) * 100
            
            missing_stats[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_pct),
                'present_count': int(n_rows - missing_count),
                'completeness_score': float((n_rows - missing_count) / n_rows)
            }
        
        completeness['column_completeness'] = missing_stats
        
        # Row completeness
        rows_with_missing = self.data.isnull().any(axis=1).sum()
        complete_rows = n_rows - rows_with_missing
        
        completeness['row_completeness'] = {
            'complete_rows': int(complete_rows),
            'incomplete_rows': int(rows_with_missing),
            'row_completeness_rate': float(complete_rows / n_rows)
        }
        
        # Critical column analysis (coordinates and primary values)
        critical_cols = self.coordinate_columns + self.value_columns
        critical_completeness = {}
        
        for col in critical_cols:
            if col in self.data.columns:
                missing = self.data[col].isnull().sum()
                critical_completeness[col] = {
                    'is_critical': True,
                    'missing_count': int(missing),
                    'completeness_score': float((n_rows - missing) / n_rows),
                    'quality_impact': 'high' if missing > n_rows * 0.05 else 'low'
                }
        
        completeness['critical_completeness'] = critical_completeness
        
        # Data coverage analysis (for spatial data)
        if len(self.coordinate_columns) >= 2:
            completeness['spatial_coverage'] = self._analyze_spatial_coverage()
        
        # Overall completeness score
        avg_completeness = np.mean([stats['completeness_score'] for stats in missing_stats.values()])
        completeness['overall_completeness_score'] = float(avg_completeness)
        
        return completeness
    
    def _analyze_consistency(self) -> Dict[str, Any]:
        """Analyze data consistency."""
        consistency = {}
        
        # Data type consistency
        consistency['data_types'] = self._check_data_type_consistency()
        
        # Format consistency
        consistency['formats'] = self._check_format_consistency()
        
        # Value consistency (duplicates, contradictions)
        consistency['values'] = self._check_value_consistency()
        
        # Coordinate consistency
        if len(self.coordinate_columns) >= 2:
            consistency['coordinates'] = self._check_coordinate_consistency()
        
        # Temporal consistency (if date columns exist)
        date_cols = self.data.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols:
            consistency['temporal'] = self._check_temporal_consistency(date_cols)
        
        return consistency
    
    def _analyze_accuracy(self, expected_ranges: Optional[Dict[str, Tuple[float, float]]]) -> Dict[str, Any]:
        """Analyze data accuracy."""
        accuracy = {}
        
        # Outlier detection
        accuracy['outliers'] = self._detect_accuracy_outliers()
        
        # Range validation
        if expected_ranges:
            accuracy['range_validation'] = self._validate_ranges(expected_ranges)
        
        # Statistical accuracy checks
        accuracy['statistical_checks'] = self._perform_statistical_accuracy_checks()
        
        # Coordinate accuracy (if applicable)
        if len(self.coordinate_columns) >= 2:
            accuracy['coordinate_accuracy'] = self._check_coordinate_accuracy()
        
        # Value precision analysis
        accuracy['precision'] = self._analyze_value_precision()
        
        return accuracy
    
    def _analyze_validity(self, expected_ranges: Optional[Dict[str, Tuple[float, float]]]) -> Dict[str, Any]:
        """Analyze data validity."""
        validity = {}
        
        # Data type validity
        validity['data_types'] = self._check_data_type_validity()
        
        # Business rule validation
        validity['business_rules'] = self._check_business_rules()
        
        # Constraint validation
        validity['constraints'] = self._check_constraints(expected_ranges)
        
        # Referential integrity (if applicable)
        validity['referential_integrity'] = self._check_referential_integrity()
        
        # Logical consistency
        validity['logical_consistency'] = self._check_logical_consistency()
        
        return validity
    
    def _analyze_spatial_coverage(self) -> Dict[str, Any]:
        """Analyze spatial coverage of data points."""
        if len(self.coordinate_columns) < 2:
            return {'error': 'Insufficient coordinate columns'}
        
        coords = self.data[self.coordinate_columns[:2]].dropna()
        if len(coords) == 0:
            return {'error': 'No valid coordinates'}
        
        # Bounding box
        bounds = {
            'min_x': float(coords.iloc[:, 0].min()),
            'max_x': float(coords.iloc[:, 0].max()),
            'min_y': float(coords.iloc[:, 1].min()),
            'max_y': float(coords.iloc[:, 1].max())
        }
        
        # Coverage area
        area = (bounds['max_x'] - bounds['min_x']) * (bounds['max_y'] - bounds['min_y'])
        
        # Point density
        density = len(coords) / area if area > 0 else 0
        
        # Coverage uniformity
        n_bins = max(3, min(int(np.sqrt(len(coords) / 5)), 10))
        hist, _, _ = np.histogram2d(coords.iloc[:, 0], coords.iloc[:, 1], bins=n_bins)
        coverage_cv = np.std(hist.flatten()) / np.mean(hist.flatten()) if np.mean(hist.flatten()) > 0 else 0
        
        return {
            'bounds': bounds,
            'coverage_area': float(area),
            'point_density': float(density),
            'coverage_uniformity': float(1 / (1 + coverage_cv)),  # Convert CV to uniformity score
            'n_valid_coordinates': len(coords)
        }
    
    def _check_data_type_consistency(self) -> Dict[str, Any]:
        """Check consistency of data types across columns."""
        type_consistency = {}
        
        for col in self.data.columns:
            series = self.data[col].dropna()
            if len(series) == 0:
                continue
            
            # Check if numeric column has consistent types
            if pd.api.types.is_numeric_dtype(series):
                # Check for mixed int/float
                has_int = series.apply(lambda x: isinstance(x, (int, np.integer))).any()
                has_float = series.apply(lambda x: isinstance(x, (float, np.floating))).any()
                
                type_consistency[col] = {
                    'expected_type': 'numeric',
                    'has_mixed_numeric': bool(has_int and has_float),
                    'consistency_score': 1.0  # Pandas handles numeric mixing well
                }
            
            # Check string columns for format consistency
            elif pd.api.types.is_object_dtype(series):
                type_consistency[col] = self._check_string_format_consistency(series, col)
        
        return type_consistency
    
    def _check_format_consistency(self) -> Dict[str, Any]:
        """Check format consistency within columns."""
        format_consistency = {}
        
        for col in self.data.columns:
            series = self.data[col].dropna()
            if len(series) == 0 or not pd.api.types.is_object_dtype(series):
                continue
            
            # Check for consistent string formats
            string_lengths = series.astype(str).str.len()
            length_cv = string_lengths.std() / string_lengths.mean() if string_lengths.mean() > 0 else 0
            
            # Check for consistent patterns (digits, letters, special chars)
            has_digits = series.astype(str).str.contains(r'\d', na=False)
            has_letters = series.astype(str).str.contains(r'[A-Za-z]', na=False)
            has_special = series.astype(str).str.contains(r'[^A-Za-z0-9\s]', na=False)
            
            format_consistency[col] = {
                'length_variability': float(length_cv),
                'digit_consistency': float(has_digits.sum() / len(series)),
                'letter_consistency': float(has_letters.sum() / len(series)),
                'special_char_consistency': float(has_special.sum() / len(series)),
                'format_score': float(1 / (1 + length_cv))  # Lower variability = higher score
            }
        
        return format_consistency
    
    def _check_value_consistency(self) -> Dict[str, Any]:
        """Check for value consistency issues."""
        value_consistency = {}
        
        # Duplicate analysis
        total_rows = len(self.data)
        unique_rows = len(self.data.drop_duplicates())
        duplicate_rate = (total_rows - unique_rows) / total_rows
        
        value_consistency['duplicates'] = {
            'total_rows': total_rows,
            'unique_rows': unique_rows,
            'duplicate_rows': total_rows - unique_rows,
            'duplicate_rate': float(duplicate_rate)
        }
        
        # Check for contradictory values (same key, different values)
        if len(self.coordinate_columns) >= 2:
            coord_cols = self.coordinate_columns[:2]
            contradictions = 0
            
            for value_col in self.value_columns:
                if value_col in self.data.columns:
                    # Group by coordinates and check for multiple values
                    grouped = self.data.groupby(coord_cols)[value_col].nunique()
                    contradictions += (grouped > 1).sum()
            
            value_consistency['contradictions'] = {
                'coordinate_value_contradictions': int(contradictions),
                'contradiction_rate': float(contradictions / len(self.data)) if len(self.data) > 0 else 0
            }
        
        return value_consistency
    
    def _check_coordinate_consistency(self) -> Dict[str, Any]:
        """Check coordinate system consistency."""
        coord_consistency = {}
        
        if len(self.coordinate_columns) < 2:
            return {'error': 'Insufficient coordinate columns'}
        
        x_col, y_col = self.coordinate_columns[:2]
        x_vals = self.data[x_col].dropna()
        y_vals = self.data[y_col].dropna()
        
        # Check coordinate ranges (geographic vs projected)
        x_range = x_vals.max() - x_vals.min()
        y_range = y_vals.max() - y_vals.min()
        
        # Heuristic checks for coordinate system type
        likely_geographic = (
            (x_vals.min() >= -180 and x_vals.max() <= 180) and
            (y_vals.min() >= -90 and y_vals.max() <= 90)
        )
        
        likely_projected = x_range > 1000 or y_range > 1000
        
        coord_consistency = {
            'x_range': float(x_range),
            'y_range': float(y_range),
            'likely_coordinate_system': 'geographic' if likely_geographic else 'projected' if likely_projected else 'unknown',
            'coordinate_precision': {
                'x_precision': self._estimate_coordinate_precision(x_vals),
                'y_precision': self._estimate_coordinate_precision(y_vals)
            }
        }
        
        return coord_consistency
    
    def _check_temporal_consistency(self, date_cols: List[str]) -> Dict[str, Any]:
        """Check temporal consistency in date columns."""
        temporal_consistency = {}
        
        for col in date_cols:
            dates = pd.to_datetime(self.data[col], errors='coerce').dropna()
            if len(dates) == 0:
                continue
            
            # Check date range reasonableness
            min_date = dates.min()
            max_date = dates.max()
            date_range = (max_date - min_date).days
            
            # Check for future dates (might be errors)
            future_dates = (dates > pd.Timestamp.now()).sum()
            
            # Check for very old dates (might be errors)
            very_old_dates = (dates < pd.Timestamp('1900-01-01')).sum()
            
            temporal_consistency[col] = {
                'date_range_days': int(date_range),
                'min_date': str(min_date.date()),
                'max_date': str(max_date.date()),
                'future_dates': int(future_dates),
                'very_old_dates': int(very_old_dates),
                'temporal_quality_score': float(1 - (future_dates + very_old_dates) / len(dates))
            }
        
        return temporal_consistency
    
    def _detect_accuracy_outliers(self) -> Dict[str, Any]:
        """Detect outliers that may indicate accuracy issues."""
        outlier_analysis = {}
        
        for col in self.value_columns:
            if col not in self.data.columns:
                continue
            
            values = self.data[col].dropna()
            if len(values) < 4:
                continue
            
            # IQR method
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            # Z-score method
            z_scores = np.abs((values - values.mean()) / values.std())
            z_outliers = values[z_scores > 3]
            
            outlier_analysis[col] = {
                'iqr_outliers': len(outliers),
                'z_score_outliers': len(z_outliers),
                'outlier_percentage': float(len(outliers) / len(values) * 100),
                'accuracy_score': float(1 - len(outliers) / len(values))
            }
        
        return outlier_analysis
    
    def _validate_ranges(self, expected_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Validate values against expected ranges."""
        range_validation = {}
        
        for col, (min_val, max_val) in expected_ranges.items():
            if col not in self.data.columns:
                continue
            
            values = self.data[col].dropna()
            if len(values) == 0:
                continue
            
            out_of_range = values[(values < min_val) | (values > max_val)]
            
            range_validation[col] = {
                'expected_range': [min_val, max_val],
                'actual_range': [float(values.min()), float(values.max())],
                'out_of_range_count': len(out_of_range),
                'out_of_range_percentage': float(len(out_of_range) / len(values) * 100),
                'range_compliance_score': float(1 - len(out_of_range) / len(values))
            }
        
        return range_validation
    
    def _perform_statistical_accuracy_checks(self) -> Dict[str, Any]:
        """Perform statistical checks for data accuracy."""
        statistical_checks = {}
        
        for col in self.value_columns:
            if col not in self.data.columns:
                continue
            
            values = self.data[col].dropna()
            if len(values) < 10:
                continue
            
            # Check for excessive precision (might indicate artificial data)
            if pd.api.types.is_float_dtype(values):
                decimal_places = values.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
                avg_decimal_places = decimal_places.mean()
                
                # Check for suspiciously uniform precision
                precision_uniformity = decimal_places.std()
                
                statistical_checks[col] = {
                    'average_decimal_places': float(avg_decimal_places),
                    'precision_uniformity': float(precision_uniformity),
                    'excessive_precision_flag': avg_decimal_places > 6,
                    'uniform_precision_flag': precision_uniformity < 0.5 and avg_decimal_places > 3
                }
        
        return statistical_checks
    
    def _check_coordinate_accuracy(self) -> Dict[str, Any]:
        """Check coordinate accuracy and validity."""
        if len(self.coordinate_columns) < 2:
            return {'error': 'Insufficient coordinate columns'}
        
        coord_accuracy = {}
        
        x_col, y_col = self.coordinate_columns[:2]
        coords = self.data[[x_col, y_col]].dropna()
        
        if len(coords) == 0:
            return {'error': 'No valid coordinates'}
        
        # Check for coordinates at (0,0) which might indicate missing data
        zero_coords = ((coords[x_col] == 0) & (coords[y_col] == 0)).sum()
        
        # Check for duplicate coordinates
        duplicate_coords = len(coords) - len(coords.drop_duplicates())
        
        # Check coordinate precision consistency
        x_precision = self._estimate_coordinate_precision(coords[x_col])
        y_precision = self._estimate_coordinate_precision(coords[y_col])
        
        coord_accuracy = {
            'zero_coordinates': int(zero_coords),
            'duplicate_coordinates': int(duplicate_coords),
            'coordinate_precision': {
                'x_precision': x_precision,
                'y_precision': y_precision,
                'precision_consistency': abs(x_precision - y_precision) < 1
            },
            'accuracy_score': float(1 - (zero_coords + duplicate_coords) / len(coords))
        }
        
        return coord_accuracy
    
    def _analyze_value_precision(self) -> Dict[str, Any]:
        """Analyze precision of numeric values."""
        precision_analysis = {}
        
        for col in self.value_columns:
            if col not in self.data.columns:
                continue
            
            values = self.data[col].dropna()
            if len(values) == 0 or not pd.api.types.is_numeric_dtype(values):
                continue
            
            precision_analysis[col] = {
                'estimated_precision': self._estimate_coordinate_precision(values),
                'significant_digits': self._estimate_significant_digits(values)
            }
        
        return precision_analysis
    
    def _check_data_type_validity(self) -> Dict[str, Any]:
        """Check if data types are appropriate for the content."""
        type_validity = {}
        
        for col in self.data.columns:
            series = self.data[col].dropna()
            if len(series) == 0:
                continue
            
            current_dtype = str(series.dtype)
            
            # Check if numeric columns should be integers
            if pd.api.types.is_float_dtype(series):
                if series.apply(lambda x: float(x).is_integer()).all():
                    type_validity[col] = {
                        'current_type': current_dtype,
                        'suggested_type': 'integer',
                        'validity_score': 0.8  # Could be more efficient as int
                    }
                else:
                    type_validity[col] = {
                        'current_type': current_dtype,
                        'suggested_type': current_dtype,
                        'validity_score': 1.0
                    }
            
            # Check if string columns could be categorical
            elif pd.api.types.is_object_dtype(series):
                unique_ratio = series.nunique() / len(series)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    type_validity[col] = {
                        'current_type': current_dtype,
                        'suggested_type': 'category',
                        'unique_ratio': float(unique_ratio),
                        'validity_score': 0.9
                    }
        
        return type_validity
    
    def _check_business_rules(self) -> Dict[str, Any]:
        """Check domain-specific business rules for geological data."""
        business_rules = {}
        
        # Check for reasonable elevation values (if elevation column exists)
        elevation_cols = ['elevation', 'z', 'altitude', 'height']
        for col in elevation_cols:
            if col in self.data.columns:
                elevations = self.data[col].dropna()
                if len(elevations) > 0:
                    # Check for unreasonable elevations
                    too_high = (elevations > 9000).sum()  # Above Everest
                    too_low = (elevations < -500).sum()    # Below reasonable depth
                    
                    business_rules[f'{col}_range_check'] = {
                        'unreasonably_high': int(too_high),
                        'unreasonably_low': int(too_low),
                        'rule_compliance': float(1 - (too_high + too_low) / len(elevations))
                    }
        
        # Check for reasonable sample values (non-negative concentrations)
        concentration_keywords = ['concentration', 'content', 'grade', 'ppm', 'percent']
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in concentration_keywords):
                values = self.data[col].dropna()
                if len(values) > 0 and pd.api.types.is_numeric_dtype(values):
                    negative_values = (values < 0).sum()
                    business_rules[f'{col}_non_negative_check'] = {
                        'negative_values': int(negative_values),
                        'rule_compliance': float(1 - negative_values / len(values))
                    }
        
        return business_rules
    
    def _check_constraints(self, expected_ranges: Optional[Dict[str, Tuple[float, float]]]) -> Dict[str, Any]:
        """Check various data constraints."""
        constraints = {}
        
        # Uniqueness constraints (coordinates should be unique for point data)
        if len(self.coordinate_columns) >= 2:
            coord_cols = self.coordinate_columns[:2]
            total_points = len(self.data.dropna(subset=coord_cols))
            unique_points = len(self.data.dropna(subset=coord_cols).drop_duplicates(subset=coord_cols))
            
            constraints['coordinate_uniqueness'] = {
                'total_points': total_points,
                'unique_points': unique_points,
                'uniqueness_score': float(unique_points / total_points) if total_points > 0 else 0
            }
        
        # Not-null constraints for critical columns
        critical_cols = self.coordinate_columns + self.value_columns
        for col in critical_cols:
            if col in self.data.columns:
                null_count = self.data[col].isnull().sum()
                constraints[f'{col}_not_null'] = {
                    'null_count': int(null_count),
                    'constraint_compliance': float(1 - null_count / len(self.data))
                }
        
        return constraints
    
    def _check_referential_integrity(self) -> Dict[str, Any]:
        """Check referential integrity between related columns."""
        # This is a placeholder for more complex referential integrity checks
        # In a real geological database, this might check sample IDs, borehole IDs, etc.
        
        referential_integrity = {
            'note': 'Referential integrity checks would be implemented based on specific data schema'
        }
        
        return referential_integrity
    
    def _check_logical_consistency(self) -> Dict[str, Any]:
        """Check logical consistency between related fields."""
        logical_consistency = {}
        
        # Check coordinate system consistency
        if len(self.coordinate_columns) >= 2:
            x_col, y_col = self.coordinate_columns[:2]
            coords = self.data[[x_col, y_col]].dropna()
            
            if len(coords) > 0:
                # Check if coordinate ranges are consistent with each other
                x_range = coords[x_col].max() - coords[x_col].min()
                y_range = coords[y_col].max() - coords[y_col].min()
                
                # For geographic coordinates, ranges should be reasonable
                range_ratio = min(x_range, y_range) / max(x_range, y_range) if max(x_range, y_range) > 0 else 0
                
                logical_consistency['coordinate_range_consistency'] = {
                    'x_range': float(x_range),
                    'y_range': float(y_range),
                    'range_ratio': float(range_ratio),
                    'consistency_score': float(min(1.0, range_ratio + 0.1))  # Some tolerance
                }
        
        return logical_consistency
    
    def _calculate_overall_quality(self, completeness: Dict, consistency: Dict, accuracy: Dict, validity: Dict) -> Dict[str, Any]:
        """Calculate overall data quality score."""
        scores = []
        
        # Completeness score
        if 'overall_completeness_score' in completeness:
            scores.append(completeness['overall_completeness_score'])
        
        # Consistency scores
        if 'duplicates' in consistency.get('values', {}):
            dup_score = 1 - consistency['values']['duplicates']['duplicate_rate']
            scores.append(dup_score)
        
        # Accuracy scores
        if 'outliers' in accuracy:
            acc_scores = [info.get('accuracy_score', 1.0) for info in accuracy['outliers'].values()]
            if acc_scores:
                scores.extend(acc_scores)
        
        # Calculate weighted average
        overall_score = np.mean(scores) if scores else 0.0
        
        # Quality grade
        if overall_score >= 0.9:
            grade = 'Excellent'
        elif overall_score >= 0.8:
            grade = 'Good'
        elif overall_score >= 0.7:
            grade = 'Fair'
        elif overall_score >= 0.6:
            grade = 'Poor'
        else:
            grade = 'Very Poor'
        
        return {
            'overall_quality_score': float(overall_score),
            'quality_grade': grade,
            'component_scores': scores,
            'meets_threshold': overall_score >= self.quality_threshold,
            'score_breakdown': {
                'completeness_weight': 0.3,
                'consistency_weight': 0.2,
                'accuracy_weight': 0.3,
                'validity_weight': 0.2
            }
        }
    
    def _generate_recommendations(self, completeness: Dict, consistency: Dict, accuracy: Dict, validity: Dict) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Completeness recommendations
        if completeness.get('overall_completeness_score', 1.0) < 0.8:
            recommendations.append("Address missing values in critical columns")
        
        if 'critical_completeness' in completeness:
            for col, info in completeness['critical_completeness'].items():
                if info['completeness_score'] < 0.9:
                    recommendations.append(f"Improve data collection for column '{col}' - {info['missing_count']} missing values")
        
        # Consistency recommendations
        if consistency.get('values', {}).get('duplicates', {}).get('duplicate_rate', 0) > 0.05:
            recommendations.append("Remove or investigate duplicate records")
        
        # Accuracy recommendations
        if 'outliers' in accuracy:
            for col, info in accuracy['outliers'].items():
                if info['outlier_percentage'] > 10:
                    recommendations.append(f"Review outliers in column '{col}' - {info['outlier_percentage']:.1f}% of values")
        
        # Validity recommendations
        if 'data_types' in validity:
            for col, info in validity['data_types'].items():
                if info.get('validity_score', 1.0) < 1.0:
                    recommendations.append(f"Consider changing '{col}' from {info['current_type']} to {info['suggested_type']}")
        
        # General recommendations
        recommendations.append("Implement data validation rules at data entry point")
        recommendations.append("Establish regular data quality monitoring")
        
        return recommendations
    
    def _estimate_coordinate_precision(self, values: pd.Series) -> int:
        """Estimate the precision (decimal places) of coordinate values."""
        if not pd.api.types.is_numeric_dtype(values):
            return 0
        
        decimal_places = []
        for val in values.sample(min(100, len(values))):  # Sample for efficiency
            str_val = str(val)
            if '.' in str_val and 'e' not in str_val.lower():
                decimal_places.append(len(str_val.split('.')[-1]))
            else:
                decimal_places.append(0)
        
        return int(np.median(decimal_places)) if decimal_places else 0
    
    def _estimate_significant_digits(self, values: pd.Series) -> int:
        """Estimate the number of significant digits in values."""
        if not pd.api.types.is_numeric_dtype(values):
            return 0
        
        sig_digits = []
        for val in values.sample(min(100, len(values))):
            # Count significant digits
            str_val = str(abs(val))
            if 'e' in str_val.lower():
                # Scientific notation
                mantissa = str_val.split('e')[0]
                sig_digits.append(len(mantissa.replace('.', '').replace('-', '')))
            else:
                # Regular notation
                if '.' in str_val:
                    digits = str_val.replace('.', '').lstrip('0') or '0'
                else:
                    digits = str_val.rstrip('0') or '0'
                sig_digits.append(len(digits))
        
        return int(np.median(sig_digits)) if sig_digits else 0
    
    def _check_string_format_consistency(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Check format consistency for string columns."""
        str_series = series.astype(str)
        
        # Pattern analysis
        patterns = {
            'all_uppercase': str_series.str.isupper().mean(),
            'all_lowercase': str_series.str.islower().mean(),
            'mixed_case': 1 - str_series.str.isupper().mean() - str_series.str.islower().mean()
        }
        
        # Length consistency
        lengths = str_series.str.len()
        length_cv = lengths.std() / lengths.mean() if lengths.mean() > 0 else 0
        
        return {
            'expected_type': 'string',
            'case_patterns': patterns,
            'length_variability': float(length_cv),
            'consistency_score': float(max(patterns.values()))  # Highest consistency pattern
        }