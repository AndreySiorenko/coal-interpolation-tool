"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_coal_data():
    """Create sample coal deposit data for testing."""
    np.random.seed(42)
    
    # Generate sample coordinates
    n_points = 50
    x = np.random.uniform(100000, 105000, n_points)
    y = np.random.uniform(200000, 205000, n_points)
    z = np.random.uniform(50, 150, n_points)
    
    # Generate correlated coal quality parameters
    ash_content = np.random.normal(15, 5, n_points)  # Ash content %
    sulfur_content = np.random.normal(2.5, 1.0, n_points)  # Sulfur content %
    calorific_value = 30 - 0.5 * ash_content + np.random.normal(0, 2, n_points)  # MJ/kg
    
    # Ensure realistic ranges
    ash_content = np.clip(ash_content, 5, 40)
    sulfur_content = np.clip(sulfur_content, 0.1, 8.0)
    calorific_value = np.clip(calorific_value, 15, 35)
    
    data = pd.DataFrame({
        'X': x,
        'Y': y,
        'Z': z,
        'ASH': ash_content,
        'SULFUR': sulfur_content,
        'CALORIFIC': calorific_value,
        'HOLE_ID': [f'BH{i:03d}' for i in range(n_points)]
    })
    
    return data


@pytest.fixture
def minimal_data():
    """Create minimal dataset for edge case testing."""
    return pd.DataFrame({
        'X': [100, 200, 300],
        'Y': [100, 200, 300], 
        'VALUE': [10, 20, 30]
    })


@pytest.fixture
def problematic_data():
    """Create dataset with various data quality issues."""
    return pd.DataFrame({
        'X': [100, 200, np.nan, 400, 200],  # Missing coordinate
        'Y': [100, 200, 300, 400, 200],     # Duplicate coordinate
        'VALUE': [10, np.inf, 30, -999, 20], # Infinite and extreme values
        'TEXT_COL': ['A', 'B', 'C', 'D', 'E']  # Non-numeric column
    })


@pytest.fixture
def temp_csv_file(sample_coal_data):
    """Create temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_coal_data.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def temp_csv_with_issues():
    """Create temporary CSV file with data quality issues."""
    problematic_csv = """X,Y,VALUE,NOTES
100,200,10.5,Good point
200,,15.2,Missing Y coordinate
300,400,INF,Infinite value
400,500,25.8,Normal point
500,600,-999,Extreme value
600,700,30.1,Another good point"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(problematic_csv)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture  
def temp_directory():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def different_delimiter_csv():
    """Create CSV file with semicolon delimiter."""
    csv_content = """X;Y;ASH;SULFUR
100;200;15.5;2.1
200;300;18.2;1.8
300;400;12.8;2.5
400;500;20.1;3.2"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def empty_csv_file():
    """Create empty CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('')
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def csv_with_header_issues():
    """Create CSV with header detection issues."""
    csv_content = """Coal Deposit Survey Data
Project: Test Mine
Date: 2023-01-01

X,Y,Z,ASH_CONTENT,SULFUR_CONTENT
100,200,50,15.5,2.1
200,300,60,18.2,1.8
300,400,70,12.8,2.5"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)