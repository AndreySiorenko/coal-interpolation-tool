#!/usr/bin/env python3
"""
Test script to verify Russian translations are working correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.i18n import set_language, get_current_language, _

def test_translations():
    """Test Russian translations for various UI elements."""
    
    print("=== Testing Translation System ===")
    
    # Set to Russian
    set_language('ru')
    current = get_current_language()
    print(f"Current language: {current}")
    
    # Test basic UI elements
    print("\n=== Basic UI Elements ===")
    test_items = [
        "File", "Parameters", "Results", "Statistics", 
        "Grid Information", "Sample Results"
    ]
    
    for item in test_items:
        translation = _(item)
        print(f"{item:20} -> {translation}")
    
    # Test Results panel specific items
    print("\n=== Results Panel ===")
    results_items = [
        "Method:", "Min Value:", "Max Value:", "Mean Value:",
        "Grid Points:", "Cell Size:", "Grid Extent:",
        "View All Results", "Export Results", "Quality Report"
    ]
    
    for item in results_items:
        translation = _(item)
        print(f"{item:20} -> {translation}")
    
    # Test Analysis panel items
    print("\n=== Analysis and Recommendations ===")
    analysis_items = [
        "Analysis and Recommendations", "Summary", 
        "Method Comparison", "Optimal Parameters",
        "Analyze Data", "Apply Recommendations"
    ]
    
    for item in analysis_items:
        translation = _(item)
        print(f"{item:30} -> {translation}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_translations()