"""
Data validation utilities for coal deposit interpolation data.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import warnings
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    statistics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.statistics is None:
            self.statistics = {}
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return len(self.errors) > 0 or len(self.warnings) > 0


class DataValidator:
    """
    Comprehensive data validator for geological survey data.
    
    Validates data quality, coordinates, values, and provides
    recommendations for interpolation.
    """
    
    def __init__(self):
        """Initialize validator."""
        self.tolerance = 1e-10  # Tolerance for floating point comparisons
        
    def validate_dataset(self, 
                        data: pd.DataFrame,
                        x_col: str,
                        y_col: str,
                        value_col: str,
                        z_col: Optional[str] = None) -> ValidationResult:
        """
        Comprehensive validation of interpolation dataset.
        
        Args:
            data: Input DataFrame
            x_col: X coordinate column name
            y_col: Y coordinate column name
            value_col: Value column name
            z_col: Z coordinate column name (optional)
            
        Returns:
            ValidationResult with all validation findings
        """
        result = ValidationResult()
        
        # Basic structure validation
        self._validate_structure(data, x_col, y_col, value_col, z_col, result)
        
        if not result.is_valid:
            return result
        
        # Data type validation
        self._validate_data_types(data, x_col, y_col, value_col, z_col, result)
        
        # Coordinate validation
        self._validate_coordinates(data, x_col, y_col, z_col, result)
        
        # Value validation
        self._validate_values(data, value_col, result)
        
        # Spatial distribution validation
        self._validate_spatial_distribution(data, x_col, y_col, result)
        
        # Data quality assessment
        self._assess_data_quality(data, x_col, y_col, value_col, z_col, result)
        
        return result
    
    def _validate_structure(self, 
                          data: pd.DataFrame,
                          x_col: str,
                          y_col: str, 
                          value_col: str,
                          z_col: Optional[str],
                          result: ValidationResult):
        """Validate basic DataFrame structure."""
        
        # Check if DataFrame is empty
        if data.empty:
            result.add_error("Dataset is empty")
            return
        
        # Check required columns exist
        required_cols = [x_col, y_col, value_col]
        if z_col:
            required_cols.append(z_col)
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            result.add_error(f"Missing required columns: {missing_cols}")
            return
        
        # Check for minimum number of rows
        if len(data) < 3:
            result.add_error(f"Insufficient data points. Need at least 3, got {len(data)}")
        elif len(data) < 10:
            result.add_warning(f"Very few data points ({len(data)}). Results may be unreliable.")
        
        result.statistics['total_rows'] = len(data)
        result.statistics['total_columns'] = len(data.columns)
    
    def _validate_data_types(self,
                           data: pd.DataFrame,
                           x_col: str,
                           y_col: str,
                           value_col: str, 
                           z_col: Optional[str],
                           result: ValidationResult):
        """Validate data types of coordinate and value columns."""
        
        columns_to_check = [(x_col, 'X coordinate'), (y_col, 'Y coordinate'), (value_col, 'Value')]
        if z_col:
            columns_to_check.append((z_col, 'Z coordinate'))
        
        for col, description in columns_to_check:
            col_data = data[col]
            
            # Check if numeric
            if not pd.api.types.is_numeric_dtype(col_data):
                # Try to convert
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    non_numeric_count = numeric_data.isnull().sum() - col_data.isnull().sum()
                    
                    if non_numeric_count > 0:
                        result.add_error(f"{description} column '{col}' contains {non_numeric_count} non-numeric values")
                    else:
                        result.add_warning(f"{description} column '{col}' was converted to numeric")
                        
                except Exception:
                    result.add_error(f"{description} column '{col}' cannot be converted to numeric")
    
    def _validate_coordinates(self,
                            data: pd.DataFrame,
                            x_col: str,
                            y_col: str,
                            z_col: Optional[str],
                            result: ValidationResult):
        """Validate coordinate data quality."""
        
        coord_stats = {}
        
        # Validate each coordinate column
        for coord_name, col_name in [('X', x_col), ('Y', y_col), ('Z', z_col)]:
            if not col_name:
                continue
                
            coord_data = pd.to_numeric(data[col_name], errors='coerce')
            
            # Check for null values
            null_count = coord_data.isnull().sum()
            if null_count > 0:
                if null_count > len(data) * 0.1:  # More than 10% null
                    result.add_error(f"{coord_name} coordinate has too many null values ({null_count})")
                else:
                    result.add_warning(f"{coord_name} coordinate has {null_count} null values")
            
            # Check for infinite values
            inf_count = np.isinf(coord_data).sum()
            if inf_count > 0:
                result.add_error(f"{coord_name} coordinate has {inf_count} infinite values")
            
            # Valid data statistics
            valid_data = coord_data.dropna()
            if len(valid_data) > 0:
                coord_stats[coord_name] = {
                    'count': len(valid_data),
                    'null_count': null_count,
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'range': float(valid_data.max() - valid_data.min())
                }
                
                # Check for suspicious coordinate ranges
                coord_range = valid_data.max() - valid_data.min()
                if coord_range < self.tolerance:
                    result.add_warning(f"{coord_name} coordinates have very small range ({coord_range:.2e})")
        
        result.statistics['coordinates'] = coord_stats
        
        # Check for duplicate coordinates
        coord_cols = [x_col, y_col]
        if z_col:
            coord_cols.append(z_col)
        
        # Convert to numeric for duplicate check
        coord_data = data[coord_cols].apply(pd.to_numeric, errors='coerce')
        duplicate_mask = coord_data.duplicated()
        duplicate_count = duplicate_mask.sum()
        
        if duplicate_count > 0:
            result.add_warning(f"Found {duplicate_count} duplicate coordinate points")
            result.statistics['duplicate_coordinates'] = duplicate_count
    
    def _validate_values(self,
                        data: pd.DataFrame,
                        value_col: str,
                        result: ValidationResult):
        """Validate interpolation values."""
        
        value_data = pd.to_numeric(data[value_col], errors='coerce')
        
        # Null values
        null_count = value_data.isnull().sum()
        if null_count > 0:
            if null_count > len(data) * 0.2:  # More than 20% null
                result.add_error(f"Value column has too many null values ({null_count})")
            else:
                result.add_warning(f"Value column has {null_count} null values")
        
        # Infinite values
        inf_count = np.isinf(value_data).sum()
        if inf_count > 0:
            result.add_error(f"Value column has {inf_count} infinite values")
        
        # Valid data analysis
        valid_values = value_data.dropna()
        if len(valid_values) > 0:
            
            # Basic statistics
            value_stats = {
                'count': len(valid_values),
                'null_count': null_count,
                'min': float(valid_values.min()),
                'max': float(valid_values.max()),
                'mean': float(valid_values.mean()),
                'median': float(valid_values.median()),
                'std': float(valid_values.std()),
                'skewness': float(valid_values.skew()),
                'kurtosis': float(valid_values.kurtosis())
            }
            
            # Outlier detection using IQR method
            Q1 = valid_values.quantile(0.25)
            Q3 = valid_values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = valid_values[(valid_values < lower_bound) | (valid_values > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_percentage = (outlier_count / len(valid_values)) * 100
                if outlier_percentage > 10:
                    result.add_warning(f"High number of outliers detected: {outlier_count} ({outlier_percentage:.1f}%)")
                else:
                    result.add_warning(f"Outliers detected: {outlier_count} ({outlier_percentage:.1f}%)")
                    
                value_stats['outlier_count'] = outlier_count
                value_stats['outlier_percentage'] = outlier_percentage
            
            # Check for constant values
            if valid_values.std() < self.tolerance:
                result.add_warning("Values have very little variation (nearly constant)")
            
            result.statistics['values'] = value_stats
    
    def _validate_spatial_distribution(self,
                                     data: pd.DataFrame,
                                     x_col: str,
                                     y_col: str,
                                     result: ValidationResult):
        """Validate spatial distribution of data points."""
        
        # Convert coordinates to numeric
        x_data = pd.to_numeric(data[x_col], errors='coerce')
        y_data = pd.to_numeric(data[y_col], errors='coerce')
        
        # Remove null coordinates
        valid_mask = ~(x_data.isnull() | y_data.isnull())
        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        
        if len(x_valid) < 3:
            result.add_error("Insufficient valid coordinate points for spatial analysis")
            return
        
        # Calculate point density
        x_range = x_valid.max() - x_valid.min()
        y_range = y_valid.max() - y_valid.min()
        area = x_range * y_range
        
        if area > 0:
            point_density = len(x_valid) / area
            result.statistics['spatial'] = {
                'x_range': float(x_range),
                'y_range': float(y_range),
                'area': float(area),
                'point_density': float(point_density),
                'points_per_unit_area': float(point_density)
            }
        
        # Check for clustering using simple nearest neighbor analysis
        points = np.column_stack([x_valid, y_valid])
        
        if len(points) > 1:
            # Calculate distances to nearest neighbors
            from scipy.spatial.distance import cdist
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)  # Ignore self-distances
            nearest_distances = np.min(distances, axis=1)
            
            mean_nearest_distance = np.mean(nearest_distances)
            std_nearest_distance = np.std(nearest_distances)
            
            # Check for very close points
            very_close_threshold = mean_nearest_distance * 0.1
            very_close_count = np.sum(nearest_distances < very_close_threshold)
            
            if very_close_count > len(points) * 0.1:
                result.add_warning(f"Many points are very close together ({very_close_count} points)")
            
            result.statistics['spatial'].update({
                'mean_nearest_distance': float(mean_nearest_distance),
                'std_nearest_distance': float(std_nearest_distance),
                'very_close_points': int(very_close_count)
            })
    
    def _assess_data_quality(self,
                           data: pd.DataFrame,
                           x_col: str,
                           y_col: str,
                           value_col: str,
                           z_col: Optional[str],
                           result: ValidationResult):
        """Assess overall data quality and provide recommendations."""
        
        # Calculate data completeness
        required_cols = [x_col, y_col, value_col]
        if z_col:
            required_cols.append(z_col)
        
        total_cells = len(data) * len(required_cols)
        null_cells = data[required_cols].isnull().sum().sum()
        completeness = ((total_cells - null_cells) / total_cells) * 100
        
        # Quality assessment
        quality_score = 100
        
        # Deduct for missing data
        if completeness < 100:
            quality_score -= (100 - completeness) * 2
        
        # Deduct for duplicates
        if 'duplicate_coordinates' in result.statistics:
            duplicate_percentage = (result.statistics['duplicate_coordinates'] / len(data)) * 100
            quality_score -= duplicate_percentage * 1.5
        
        # Deduct for outliers
        if 'values' in result.statistics and 'outlier_percentage' in result.statistics['values']:
            outlier_percentage = result.statistics['values']['outlier_percentage']
            quality_score -= outlier_percentage * 0.5
        
        quality_score = max(0, quality_score)
        
        # Quality assessment
        if quality_score >= 90:
            quality_level = "Excellent"
        elif quality_score >= 75:
            quality_level = "Good"
        elif quality_score >= 60:
            quality_level = "Fair"
        elif quality_score >= 40:
            quality_level = "Poor"
        else:
            quality_level = "Very Poor"
        
        result.statistics['quality'] = {
            'completeness_percentage': float(completeness),
            'quality_score': float(quality_score),
            'quality_level': quality_level
        }
        
        # Recommendations
        recommendations = []
        
        if completeness < 95:
            recommendations.append("Consider removing or imputing missing values")
        
        if 'duplicate_coordinates' in result.statistics and result.statistics['duplicate_coordinates'] > 0:
            recommendations.append("Remove or average duplicate coordinate points")
        
        if len(data) < 50:
            recommendations.append("Consider collecting more data points for better interpolation accuracy")
        
        if 'values' in result.statistics:
            skewness = abs(result.statistics['values'].get('skewness', 0))
            if skewness > 2:
                recommendations.append("Data is highly skewed - consider log transformation")
        
        result.statistics['recommendations'] = recommendations
    
    def validate_for_interpolation_method(self,
                                        data: pd.DataFrame,
                                        x_col: str,
                                        y_col: str,
                                        value_col: str,
                                        method: str,
                                        z_col: Optional[str] = None) -> ValidationResult:
        """
        Validate data suitability for specific interpolation method.
        
        Args:
            data: Input DataFrame
            x_col: X coordinate column
            y_col: Y coordinate column
            value_col: Value column
            method: Interpolation method name
            z_col: Z coordinate column (optional)
            
        Returns:
            ValidationResult with method-specific validation
        """
        # Start with general validation
        result = self.validate_dataset(data, x_col, y_col, value_col, z_col)
        
        if not result.is_valid:
            return result
        
        # Method-specific validation
        if method.lower() in ['kriging', 'ordinary_kriging', 'simple_kriging']:
            self._validate_for_kriging(data, x_col, y_col, value_col, result)
        elif method.lower() == 'idw':
            self._validate_for_idw(data, x_col, y_col, value_col, result)
        elif method.lower() in ['rbf', 'radial_basis_function']:
            self._validate_for_rbf(data, x_col, y_col, value_col, result)
        
        return result
    
    def _validate_for_kriging(self,
                             data: pd.DataFrame,
                             x_col: str,
                             y_col: str,
                             value_col: str,
                             result: ValidationResult):
        """Validate data for kriging interpolation."""
        
        # Kriging typically needs more points
        if len(data) < 20:
            result.add_warning("Kriging typically performs better with more data points (>20)")
        
        # Check for stationarity (basic check)
        value_data = pd.to_numeric(data[value_col], errors='coerce').dropna()
        
        if len(value_data) > 10:
            # Simple check for trend
            x_data = pd.to_numeric(data[x_col], errors='coerce')
            y_data = pd.to_numeric(data[y_col], errors='coerce')
            
            # Check correlation with coordinates (indicating trend)
            x_corr = abs(np.corrcoef(x_data.dropna(), value_data[:len(x_data.dropna())])[0, 1])
            y_corr = abs(np.corrcoef(y_data.dropna(), value_data[:len(y_data.dropna())])[0, 1])
            
            if x_corr > 0.7 or y_corr > 0.7:
                result.add_warning("Strong spatial trend detected - consider detrending or universal kriging")
    
    def _validate_for_idw(self,
                         data: pd.DataFrame,
                         x_col: str,
                         y_col: str,
                         value_col: str,
                         result: ValidationResult):
        """Validate data for IDW interpolation."""
        
        # IDW is generally robust, but check for extreme values
        value_data = pd.to_numeric(data[value_col], errors='coerce').dropna()
        
        if len(value_data) > 0:
            value_range = value_data.max() - value_data.min()
            if value_range == 0:
                result.add_warning("All values are identical - IDW will produce constant surface")
            
            # Check for extreme outliers that might dominate IDW
            Q1, Q3 = value_data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            extreme_outliers = value_data[(value_data < Q1 - 3*IQR) | (value_data > Q3 + 3*IQR)]
            
            if len(extreme_outliers) > 0:
                result.add_warning(f"Extreme outliers detected ({len(extreme_outliers)}) - may dominate IDW interpolation")
    
    def _validate_for_rbf(self,
                         data: pd.DataFrame,
                         x_col: str,
                         y_col: str,
                         value_col: str,
                         result: ValidationResult):
        """Validate data for RBF interpolation."""
        
        # RBF can be sensitive to scaling
        x_data = pd.to_numeric(data[x_col], errors='coerce').dropna()
        y_data = pd.to_numeric(data[y_col], errors='coerce').dropna()
        value_data = pd.to_numeric(data[value_col], errors='coerce').dropna()
        
        # Check scale differences
        if len(x_data) > 0 and len(y_data) > 0:
            x_range = x_data.max() - x_data.min()
            y_range = y_data.max() - y_data.min()
            
            if x_range > 0 and y_range > 0:
                scale_ratio = max(x_range, y_range) / min(x_range, y_range)
                if scale_ratio > 1000:
                    result.add_warning("Large scale difference between X and Y coordinates - consider normalizing")
        
        # Check for sufficient point density
        if len(data) < 10:
            result.add_warning("RBF interpolation may be unstable with very few points")