#!/usr/bin/env python3
"""
Example demonstrating advanced data analysis capabilities.

This example shows how to use the comprehensive analysis modules
for geological survey data analysis.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis import (
    StatisticalAnalyzer,
    SpatialAnalyzer, 
    OutlierDetector,
    CorrelationAnalyzer,
    DataQualityAnalyzer
)


def create_sample_geological_data(n_points: int = 500) -> pd.DataFrame:
    """
    Create realistic sample geological survey data.
    
    Args:
        n_points: Number of data points to generate
        
    Returns:
        DataFrame with simulated geological survey data
    """
    np.random.seed(42)
    
    print(f"🔬 Generating {n_points} sample geological data points...")
    
    # Generate spatial coordinates
    x = np.random.uniform(0, 1000, n_points)  # Easting (meters)
    y = np.random.uniform(0, 1000, n_points)  # Northing (meters)
    
    # Generate elevation with spatial trend
    elevation = 100 + 0.05 * x + 0.02 * y + np.random.normal(0, 15, n_points)
    
    # Generate mineral concentrations with spatial correlation
    # Higher concentrations near center of survey area
    distance_from_center = np.sqrt((x - 500)**2 + (y - 500)**2)
    base_concentration = 10 * np.exp(-distance_from_center / 300)
    
    # Add noise and create realistic distribution
    coal_content = base_concentration + np.random.exponential(2, n_points)
    ash_content = 100 - coal_content * 0.8 + np.random.normal(0, 5, n_points)
    moisture = 5 + np.random.exponential(3, n_points)
    
    # Add some missing values (realistic scenario)
    missing_indices = np.random.choice(n_points, size=int(0.05 * n_points), replace=False)
    coal_content[missing_indices] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(n_points, size=int(0.02 * n_points), replace=False)
    coal_content[outlier_indices] *= 5  # Extreme values
    
    # Create additional variables
    borehole_depth = 50 + np.random.exponential(30, n_points)
    sample_quality = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 
                                     n_points, p=[0.3, 0.4, 0.2, 0.1])
    
    data = pd.DataFrame({
        'easting': x,
        'northing': y,
        'elevation': elevation,
        'coal_content': coal_content,
        'ash_content': ash_content,
        'moisture': moisture,
        'borehole_depth': borehole_depth,
        'sample_quality': sample_quality,
        'sample_id': [f'BH-{i+1:03d}' for i in range(n_points)],
        'date_collected': pd.date_range('2023-01-01', periods=n_points, freq='D')
    })
    
    print("✅ Sample data generated successfully!")
    return data


def demonstrate_statistical_analysis(data: pd.DataFrame):
    """Demonstrate statistical analysis capabilities."""
    print("\n" + "="*60)
    print("📊 STATISTICAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    analyzer = StatisticalAnalyzer(confidence_level=0.95)
    
    # Analyze coal content
    print("\n🔍 Analyzing coal content distribution...")
    results = analyzer.analyze(data, 'coal_content')
    
    # Generate and display report
    report = analyzer.generate_summary_report(results, 'coal_content')
    print(report)
    
    # Show specific findings
    desc = results.descriptive_stats
    print(f"\n📈 Key Statistics:")
    print(f"• Sample size: {desc['count']:.0f} measurements")
    print(f"• Mean coal content: {desc['mean']:.2f}%")
    print(f"• Standard deviation: {desc['std']:.2f}%")
    print(f"• Coefficient of variation: {results.variability_measures['coefficient_variation']:.3f}")
    
    # Distribution characteristics
    dist = results.distribution_analysis
    print(f"\n📐 Distribution Shape:")
    print(f"• Skewness: {dist['skewness']:.3f} ({dist['skewness_interpretation']})")
    print(f"• Kurtosis: {dist['kurtosis']:.3f} ({dist['kurtosis_interpretation']})")
    
    # Outlier information
    outliers = results.extreme_values['iqr_method']
    print(f"\n⚠️  Outlier Detection:")
    print(f"• IQR method found {outliers['mild_outliers']} outliers ({outliers['outlier_percentage']:.1f}%)")


def demonstrate_spatial_analysis(data: pd.DataFrame):
    """Demonstrate spatial analysis capabilities."""
    print("\n" + "="*60)
    print("🗺️  SPATIAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    analyzer = SpatialAnalyzer()
    
    print("\n🔍 Analyzing spatial patterns in coal content...")
    results = analyzer.analyze(data, 'easting', 'northing', 'coal_content')
    
    # Density analysis
    density = results.density_analysis
    print(f"\n📍 Spatial Density:")
    print(f"• Point density: {density['points_per_unit_area']:.6f} points/m²")
    
    if 'nearest_neighbor' in density:
        nn = density['nearest_neighbor']
        print(f"• Mean nearest neighbor distance: {nn['mean_distance']:.2f} m")
        print(f"• Clark-Evans clustering index: {nn['clark_evans_index']:.3f}")
        
        if nn['clark_evans_index'] < 1:
            print("  → Points show clustering tendency")
        elif nn['clark_evans_index'] > 1:
            print("  → Points show regular spacing tendency")
        else:
            print("  → Points show random distribution")
    
    # Spatial statistics
    stats = results.spatial_statistics
    print(f"\n📐 Spatial Statistics:")
    print(f"• Survey area centroid: ({stats['centroid'][0]:.1f}, {stats['centroid'][1]:.1f})")
    print(f"• Standard distance: {stats['standard_distance']:.2f} m")
    
    if 'bounding_box_area' in stats:
        print(f"• Survey area: {stats['bounding_box_area']:.0f} m²")
        if 'convex_hull_area' in stats:
            coverage_efficiency = stats['convex_hull_area'] / stats['bounding_box_area']
            print(f"• Area coverage efficiency: {coverage_efficiency:.1%}")
    
    # Anisotropy analysis
    if 'anisotropy_analysis' in results.__dict__ and 'directional_analysis' in results.anisotropy_analysis:
        aniso = results.anisotropy_analysis['directional_analysis']
        print(f"\n🧭 Directional Analysis:")
        print(f"• Maximum range direction: {aniso['max_range_direction']:.0f}°")
        print(f"• Anisotropy ratio: {aniso['anisotropy_ratio']:.3f}")
        
        if aniso['anisotropy_ratio'] < 0.7:
            print("  → Strong directional preference detected")
        elif aniso['anisotropy_ratio'] < 0.9:
            print("  → Moderate directional preference")
        else:
            print("  → Nearly isotropic distribution")


def demonstrate_outlier_detection(data: pd.DataFrame):
    """Demonstrate outlier detection capabilities."""
    print("\n" + "="*60)
    print("🎯 OUTLIER DETECTION DEMONSTRATION")
    print("="*60)
    
    detector = OutlierDetector(contamination=0.05)
    
    print("\n🔍 Detecting outliers in coal content using multiple methods...")
    results = detector.detect_outliers(data, 'easting', 'northing', 'coal_content')
    
    # Statistical outliers
    if 'statistical_outliers' in results.__dict__:
        stat_outliers = results.statistical_outliers
        
        print(f"\n📊 Statistical Outlier Methods:")
        if 'iqr_method' in stat_outliers:
            iqr = stat_outliers['iqr_method']
            print(f"• IQR method: {iqr['n_outliers']} outliers ({iqr['outlier_percentage']:.1f}%)")
        
        if 'zscore_method' in stat_outliers:
            zscore = stat_outliers['zscore_method']
            print(f"• Z-score method (3σ): {zscore['n_outliers']} outliers")
            print(f"• Maximum Z-score: {zscore['max_z_score']:.2f}")
    
    # Spatial outliers
    if 'spatial_outliers' in results.__dict__:
        spatial_outliers = results.spatial_outliers
        
        print(f"\n🗺️  Spatial Outlier Methods:")
        if 'distance_based' in spatial_outliers:
            dist_based = spatial_outliers['distance_based']
            if 'n_outliers' in dist_based:
                print(f"• Distance-based: {dist_based['n_outliers']} outliers")
    
    # Ensemble results
    if 'ensemble_results' in results.__dict__:
        ensemble = results.ensemble_results
        
        print(f"\n🎭 Ensemble Outlier Detection:")
        if 'majority_vote' in ensemble:
            majority = ensemble['majority_vote']
            print(f"• Majority vote consensus: {majority['n_outliers']} outliers")
        
        if 'union' in ensemble:
            union = ensemble['union']
            print(f"• Any method flagged: {union['n_outliers']} outliers")
        
        if 'consensus' in ensemble:
            consensus = ensemble['consensus']
            print(f"• All methods agree: {consensus['n_outliers']} outliers")
    
    # Summary and recommendations
    summary = results.summary
    print(f"\n💡 Summary:")
    print(f"• Total data points analyzed: {summary['total_points']}")
    print(f"• Methods used: {summary['methods_used']}")
    print(f"• Method agreement rate: {summary['method_agreement']:.1%}")
    print(f"• Recommendation: {summary['recommendation']}")


def demonstrate_correlation_analysis(data: pd.DataFrame):
    """Demonstrate correlation analysis capabilities."""
    print("\n" + "="*60)
    print("🔗 CORRELATION ANALYSIS DEMONSTRATION")
    print("="*60)
    
    analyzer = CorrelationAnalyzer(alpha=0.05)
    
    # Focus on numeric variables
    numeric_vars = ['coal_content', 'ash_content', 'moisture', 'borehole_depth', 'elevation']
    
    print(f"\n🔍 Analyzing correlations between {len(numeric_vars)} variables...")
    results = analyzer.analyze(data, variables=numeric_vars)
    
    # Generate and display report
    report = analyzer.generate_correlation_report(results)
    print(report)
    
    # Show correlation matrix highlights
    if 'pearson' in results.correlation_matrices:
        print(f"\n🎯 Key Correlations (Pearson):")
        
        # Get strongest correlations
        strongest = results.relationship_analysis['strongest_correlations']['pearson'][:5]
        
        for i, corr_info in enumerate(strongest, 1):
            var1, var2 = corr_info['variables']
            r = corr_info['correlation']
            sig_marker = " ***" if corr_info['significant'] else ""
            
            strength = "very strong" if abs(r) > 0.8 else "strong" if abs(r) > 0.6 else "moderate" if abs(r) > 0.4 else "weak"
            direction = "positive" if r > 0 else "negative"
            
            print(f"{i}. {var1} ↔ {var2}: r = {r:.3f}{sig_marker}")
            print(f"   → {strength} {direction} correlation")
    
    # Multicollinearity check
    mc_check = results.relationship_analysis['multicollinearity_check']
    print(f"\n⚠️  Multicollinearity Assessment:")
    print(f"• High correlation pairs (|r| > 0.8): {mc_check['n_high_correlations']}")
    print(f"• Risk level: {mc_check['multicollinearity_risk']}")
    
    if mc_check['multicollinearity_risk'] == 'high':
        print("• Recommendation: Consider removing redundant variables before modeling")


def demonstrate_data_quality_analysis(data: pd.DataFrame):
    """Demonstrate data quality analysis capabilities."""
    print("\n" + "="*60)
    print("🏆 DATA QUALITY ANALYSIS DEMONSTRATION")
    print("="*60)
    
    analyzer = DataQualityAnalyzer()
    
    print("\n🔍 Performing comprehensive data quality assessment...")
    results = analyzer.analyze_quality(
        data,
        coordinate_columns=['easting', 'northing'],
        value_columns=['coal_content', 'ash_content', 'moisture', 'borehole_depth']
    )
    
    # Overall quality score
    overall = results.overall_quality
    print(f"\n🎖️  Overall Data Quality:")
    print(f"• Quality Score: {overall['overall_quality_score']:.3f}/1.000")
    print(f"• Quality Grade: {overall['quality_grade']}")
    print(f"• Meets Threshold (0.8): {'✅ Yes' if overall['meets_threshold'] else '❌ No'}")
    
    # Completeness analysis
    completeness = results.completeness_analysis
    print(f"\n📋 Completeness Assessment:")
    print(f"• Overall completeness: {completeness['overall_completeness_score']:.1%}")
    print(f"• Complete rows: {completeness['row_completeness']['complete_rows']}/{completeness['row_completeness']['complete_rows'] + completeness['row_completeness']['incomplete_rows']}")
    
    # Show completeness by critical columns
    if 'critical_completeness' in completeness:
        for col, info in completeness['critical_completeness'].items():
            status = "✅" if info['completeness_score'] > 0.95 else "⚠️" if info['completeness_score'] > 0.9 else "❌"
            print(f"• {col}: {info['completeness_score']:.1%} complete {status}")
    
    # Accuracy analysis
    if 'outliers' in results.accuracy_analysis:
        outliers = results.accuracy_analysis['outliers']
        print(f"\n🎯 Accuracy Assessment:")
        
        for col, info in outliers.items():
            accuracy_pct = info['accuracy_score'] * 100
            status = "✅" if accuracy_pct > 95 else "⚠️" if accuracy_pct > 90 else "❌"
            print(f"• {col}: {accuracy_pct:.1f}% accuracy {status} ({info['iqr_outliers']} outliers)")
    
    # Business rules validation
    if 'business_rules' in results.validity_analysis:
        business_rules = results.validity_analysis['business_rules']
        if business_rules:
            print(f"\n📏 Business Rules Validation:")
            for rule, info in business_rules.items():
                if 'rule_compliance' in info:
                    compliance_pct = info['rule_compliance'] * 100
                    status = "✅" if compliance_pct > 95 else "⚠️" if compliance_pct > 90 else "❌"
                    print(f"• {rule}: {compliance_pct:.1f}% compliance {status}")
    
    # Recommendations
    recommendations = results.recommendations
    if recommendations:
        print(f"\n💡 Quality Improvement Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"{i}. {rec}")


def demonstrate_integrated_analysis(data: pd.DataFrame):
    """Demonstrate integrated analysis workflow."""
    print("\n" + "="*60)
    print("🔬 INTEGRATED ANALYSIS WORKFLOW")
    print("="*60)
    
    print("\n🚀 Running complete analysis workflow...")
    
    # Step 1: Data Quality Check
    print("\n1️⃣  Data Quality Assessment...")
    quality_analyzer = DataQualityAnalyzer()
    quality_results = quality_analyzer.analyze_quality(data)
    quality_score = quality_results.overall_quality['overall_quality_score']
    
    if quality_score < 0.7:
        print(f"⚠️  Warning: Data quality score ({quality_score:.2f}) is below recommended threshold")
        print("   Consider addressing quality issues before analysis")
    else:
        print(f"✅ Data quality acceptable ({quality_score:.2f})")
    
    # Step 2: Outlier Detection and Cleaning
    print("\n2️⃣  Outlier Detection...")
    outlier_detector = OutlierDetector(contamination=0.05)
    outlier_results = outlier_detector.detect_outliers(data, 'easting', 'northing', 'coal_content')
    
    consensus_outliers = outlier_results.ensemble_results.get('consensus', {}).get('outlier_indices', [])
    print(f"🎯 Identified {len(consensus_outliers)} high-confidence outliers")
    
    # Step 3: Statistical Characterization
    print("\n3️⃣  Statistical Characterization...")
    stat_analyzer = StatisticalAnalyzer()
    stat_results = stat_analyzer.analyze(data, 'coal_content')
    
    distribution_type = stat_results.distribution_analysis['skewness_interpretation']
    print(f"📊 Coal content follows {distribution_type} distribution")
    
    # Step 4: Spatial Pattern Analysis
    print("\n4️⃣  Spatial Pattern Analysis...")
    spatial_analyzer = SpatialAnalyzer()
    spatial_results = spatial_analyzer.analyze(data, 'easting', 'northing', 'coal_content')
    
    if 'nearest_neighbor' in spatial_results.density_analysis:
        ce_index = spatial_results.density_analysis['nearest_neighbor']['clark_evans_index']
        if ce_index < 0.8:
            spatial_pattern = "clustered"
        elif ce_index > 1.2:
            spatial_pattern = "regular"
        else:
            spatial_pattern = "random"
        print(f"🗺️  Spatial pattern: {spatial_pattern} (CE index: {ce_index:.3f})")
    
    # Step 5: Variable Relationships
    print("\n5️⃣  Variable Relationship Analysis...")
    corr_analyzer = CorrelationAnalyzer()
    corr_results = corr_analyzer.analyze(data)
    
    if 'pearson' in corr_results.correlation_matrices:
        strongest_corr = corr_results.relationship_analysis['strongest_correlations']['pearson'][0]
        var1, var2 = strongest_corr['variables']
        r_value = strongest_corr['correlation']
        print(f"🔗 Strongest correlation: {var1} ↔ {var2} (r = {r_value:.3f})")
    
    # Integration Summary
    print(f"\n🎓 Analysis Summary:")
    print(f"• Data quality: {quality_score:.2f}/1.0 ({quality_results.overall_quality['quality_grade']})")
    print(f"• Distribution: {distribution_type}")
    print(f"• Spatial pattern: {spatial_pattern}")
    print(f"• High-confidence outliers: {len(consensus_outliers)}")
    print(f"• Analysis completed successfully! ✅")


def main():
    """Main demonstration function."""
    print("🎯 ADVANCED GEOLOGICAL DATA ANALYSIS DEMONSTRATION")
    print("=" * 70)
    print("This example demonstrates the comprehensive analysis capabilities")
    print("of the Coal Deposit Interpolation Tool's analysis modules.")
    print("=" * 70)
    
    try:
        # Generate sample data
        data = create_sample_geological_data(n_points=500)
        
        # Show data overview
        print(f"\n📋 Dataset Overview:")
        print(f"• Shape: {data.shape[0]} rows × {data.shape[1]} columns")
        print(f"• Columns: {', '.join(data.columns.tolist())}")
        print(f"• Memory usage: {data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Run individual analysis demonstrations
        demonstrate_statistical_analysis(data)
        demonstrate_spatial_analysis(data)
        demonstrate_outlier_detection(data)
        demonstrate_correlation_analysis(data)
        demonstrate_data_quality_analysis(data)
        
        # Run integrated workflow
        demonstrate_integrated_analysis(data)
        
        print(f"\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("All analysis modules are working correctly and provide")
        print("comprehensive insights into geological survey data.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error occurred during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)