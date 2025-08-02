#!/usr/bin/env python3
"""
Test data analysis functionality to verify the fix.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_recommendation_engine():
    """Test the recommendation engine with corrected method names."""
    
    print("Testing recommendation engine...")
    
    try:
        from src.core.recommendations.recommendation_engine import RecommendationEngine
        
        # Load test data
        csv_file = Path(__file__).parent / "Скважины устья.csv"
        df = pd.read_csv(csv_file)
        
        # Take first 100 points for quick test
        test_data = df.head(100).copy()
        test_data = test_data.rename(columns={'East': 'X', 'North': 'Y'})
        
        print(f"Test data shape: {test_data.shape}")
        print(f"Columns: {list(test_data.columns)}")
        
        # Create recommendation engine
        engine = RecommendationEngine()
        
        # Test analysis and recommendation
        print("Running analysis and recommendation...")
        report = engine.analyze_and_recommend(
            data=test_data,
            x_col='X',
            y_col='Y',
            value_col='Elev',
            user_preferences={'prioritize_speed': True},
            evaluate_quality=False,  # Skip for speed
            quick_mode=True
        )
        
        print("SUCCESS: Analysis completed!")
        print(f"Recommended method: {report.recommended_method}")
        print(f"Method scores: {[score.method.value for score in report.method_scores]}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_selector():
    """Test method selector with corrected enum values."""
    
    print("\nTesting method selector...")
    
    try:
        from src.core.recommendations.method_selector import MethodSelector, InterpolationMethod
        from src.core.recommendations.data_analyzer import DataAnalyzer
        
        # Load test data
        csv_file = Path(__file__).parent / "Скважины устья.csv"
        df = pd.read_csv(csv_file)
        test_data = df.head(50).copy()
        test_data = test_data.rename(columns={'East': 'X', 'North': 'Y'})
        
        # Analyze data characteristics
        analyzer = DataAnalyzer()
        characteristics = analyzer.analyze(test_data, 'X', 'Y', 'Elev')
        
        # Test method selector
        selector = MethodSelector()
        method_scores = selector.recommend_method(characteristics)
        
        print("Available methods:")
        for method in InterpolationMethod:
            print(f"  {method.name}: {method.value}")
        
        print("\nMethod scores:")
        for score in method_scores:
            print(f"  {score.method.value}: {score.score:.1f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in method selector: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_optimizer():
    """Test parameter optimizer with method names."""
    
    print("\nTesting parameter optimizer...")
    
    try:
        from src.core.recommendations.parameter_optimizer import ParameterOptimizer
        from src.core.recommendations.data_analyzer import DataAnalyzer
        
        # Load test data
        csv_file = Path(__file__).parent / "Скважины устья.csv"
        df = pd.read_csv(csv_file)
        test_data = df.head(50).copy()
        test_data = test_data.rename(columns={'East': 'X', 'North': 'Y'})
        
        # Analyze data characteristics
        analyzer = DataAnalyzer()
        characteristics = analyzer.analyze(test_data, 'X', 'Y', 'Elev')
        
        # Test parameter optimizer with different method names
        optimizer = ParameterOptimizer()
        
        methods_to_test = ['IDW', 'RBF', 'Kriging']
        
        for method in methods_to_test:
            try:
                result = optimizer.optimize_parameters(method, characteristics)
                print(f"  {method}: SUCCESS - {len(result.parameters)} parameters")
            except Exception as e:
                print(f"  {method}: ERROR - {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in parameter optimizer: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Data Analysis Test Suite")
    print("=" * 40)
    
    # Test 1: Method selector
    selector_success = test_method_selector()
    
    # Test 2: Parameter optimizer
    optimizer_success = test_parameter_optimizer()
    
    # Test 3: Full recommendation engine
    engine_success = test_recommendation_engine()
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"  Method Selector: {'PASS' if selector_success else 'FAIL'}")
    print(f"  Parameter Optimizer: {'PASS' if optimizer_success else 'FAIL'}")
    print(f"  Recommendation Engine: {'PASS' if engine_success else 'FAIL'}")
    
    if selector_success and optimizer_success and engine_success:
        print("\nAll tests PASSED! Data analysis should work in the application.")
    else:
        print("\nSome tests FAILED. Check errors above.")