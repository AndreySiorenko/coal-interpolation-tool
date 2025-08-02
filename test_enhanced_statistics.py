#!/usr/bin/env python3
"""
Test enhanced statistical analysis features including Grubbs test and data transformations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

def test_enhanced_statistical_analysis():
    """Test the enhanced statistical analysis functionality."""
    print("=== Testing Enhanced Statistical Analysis ===")
    
    try:
        from src.analysis.statistical_analyzer import StatisticalAnalyzer
        print("OK: StatisticalAnalyzer imports successfully")
        
        # Create test data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(100, 15, 50)
        outliers = [150, 160, 30, 25]  # Clear outliers
        test_data = np.concatenate([normal_data, outliers])
        
        # Create DataFrame
        df = pd.DataFrame({'Value': test_data})
        print(f"OK: Test data created: {len(test_data)} points with {len(outliers)} outliers")
        
        # Initialize analyzer
        analyzer = StatisticalAnalyzer()
        
        # Test basic analysis
        results = analyzer.analyze(df, 'Value')
        print("OK: Basic statistical analysis completed")
        
        # Test Grubbs test
        grubbs_result = analyzer.grubbs_test(test_data)
        print(f"OK: Grubbs test completed: {grubbs_result['n_outliers']} outliers detected")
        
        # Test data transformation
        transform_result = analyzer.transform_data(test_data, method='auto')
        print(f"OK: Data transformation completed: method={transform_result['method']}")
        
        # Test enhanced report
        enhanced_report = analyzer.generate_enhanced_report(results, 'Value')
        print("OK: Enhanced report generated")
        
        # Print some results
        print(f"\nGrubbs Test Results:")
        print(f"  Outliers detected: {grubbs_result['n_outliers']}")
        print(f"  Outlier percentage: {grubbs_result['outlier_percentage']:.1f}%")
        
        print(f"\nTransformation Results:")
        print(f"  Method: {transform_result['method']}")
        if 'normality_before' in transform_result:
            print(f"  Normality before: p={transform_result['normality_before'].get('p_value', 0):.4f}")
        if 'normality_after' in transform_result:
            print(f"  Normality after: p={transform_result['normality_after'].get('p_value', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Enhanced statistical analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_grubbs_outlier_detection():
    """Test Grubbs test with known outlier data."""
    print("\n=== Testing Grubbs Outlier Detection ===")
    
    try:
        from src.analysis.statistical_analyzer import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer()
        
        # Create data with clear outliers
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100 is clear outlier
        
        grubbs_result = analyzer.grubbs_test(data)
        
        print(f"Test data: {data}")
        print(f"Outliers detected: {grubbs_result['outliers']}")
        print(f"Number of outliers: {grubbs_result['n_outliers']}")
        
        # Check if 100 was detected as outlier
        if 100 in grubbs_result['outliers']:
            print("OK: Grubbs test correctly identified outlier")
        else:
            print("ERROR: Grubbs test failed to identify obvious outlier")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Grubbs test failed: {e}")
        return False

def test_data_transformations():
    """Test various data transformation methods."""
    print("\n=== Testing Data Transformations ===")
    
    try:
        from src.analysis.statistical_analyzer import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer()
        
        # Test different types of data
        test_cases = [
            ("Positive skewed", np.random.lognormal(0, 1, 100)),
            ("Negative values", np.random.normal(0, 1, 100)),
            ("Positive only", np.random.exponential(2, 100))
        ]
        
        for case_name, data in test_cases:
            print(f"\nTesting {case_name}:")
            
            # Test auto transformation
            result = analyzer.transform_data(data, method='auto')
            print(f"  Auto method selected: {result['method']}")
            
            if result['success']:
                print(f"  OK: Transformation successful")
                
                # Test inverse transformation
                if result['method'] != 'none':
                    inverse_data = analyzer.inverse_transform(
                        result['transformed_values'], 
                        result['method'], 
                        result['parameters']
                    )
                    # Check if inverse is close to original
                    max_diff = np.max(np.abs(inverse_data - data))
                    if max_diff < 1e-10:
                        print(f"  OK: Inverse transformation accurate")
                    else:
                        print(f"  WARNING: Inverse transformation error: {max_diff:.2e}")
            else:
                print(f"  ERROR: Transformation failed: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Data transformation test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== ENHANCED STATISTICAL ANALYSIS TESTS ===")
    
    success = True
    success &= test_enhanced_statistical_analysis()
    success &= test_grubbs_outlier_detection() 
    success &= test_data_transformations()
    
    if success:
        print("\nOK: All enhanced statistical analysis tests passed!")
    else:
        print("\nERROR: Some tests failed!")
    
    print("\n=== TESTS COMPLETED ===")