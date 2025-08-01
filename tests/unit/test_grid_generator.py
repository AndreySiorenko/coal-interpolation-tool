"""
Unit tests for GridGenerator class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.core.grid import GridGenerator, GridParameters, GridGenerationError


class TestGridGeneratorInitialization:
    """Test GridGenerator initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        generator = GridGenerator()
        assert generator.params is not None
        assert isinstance(generator.params, GridParameters)
    
    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        params = GridParameters(
            bounds=(0, 0, 100, 100),
            cell_size=10.0,
            buffer=0.1
        )
        generator = GridGenerator(params)
        assert generator.params.bounds == (0, 0, 100, 100)
        assert generator.params.cell_size == 10.0
        assert generator.params.buffer == 0.1


class TestBoundsCalculation:
    """Test bounds calculation methods."""
    
    def test_calculate_bounds_from_explicit_bounds(self):
        """Test bounds calculation from explicit parameters."""
        generator = GridGenerator()
        bounds = generator._calculate_bounds(None, (0, 10, 100, 110))
        assert bounds == (0, 10, 100, 110)
    
    def test_calculate_bounds_from_data(self):
        """Test bounds calculation from data."""
        data = pd.DataFrame({
            'X': [10, 20, 30, 40],
            'Y': [15, 25, 35, 45]
        })
        generator = GridGenerator()
        bounds = generator._calculate_bounds(data, None)
        assert bounds == (10, 15, 40, 45)
    
    def test_calculate_bounds_from_params(self):
        """Test bounds calculation from initialization parameters."""
        params = GridParameters(bounds=(5, 5, 95, 95))
        generator = GridGenerator(params)
        bounds = generator._calculate_bounds(None, None)
        assert bounds == (5, 5, 95, 95)
    
    def test_calculate_bounds_with_buffer(self):
        """Test bounds calculation with buffer."""
        data = pd.DataFrame({
            'X': [10, 40],  # range = 30
            'Y': [20, 50]   # range = 30
        })
        params = GridParameters(buffer=0.1)  # 10% buffer
        generator = GridGenerator(params)
        bounds = generator._calculate_bounds(data, None)
        
        # Expected: buffer = 30 * 0.1 = 3
        expected = (10 - 3, 20 - 3, 40 + 3, 50 + 3)
        assert bounds == expected
    
    def test_calculate_bounds_with_absolute_buffer(self):
        """Test bounds calculation with absolute buffer."""
        data = pd.DataFrame({
            'X': [10, 40],
            'Y': [20, 50]
        })
        params = GridParameters(buffer=5.0)  # Absolute buffer
        generator = GridGenerator(params)
        bounds = generator._calculate_bounds(data, None)
        
        expected = (5, 15, 45, 55)
        assert bounds == expected
    
    def test_calculate_bounds_missing_columns(self):
        """Test error when data missing required columns."""
        data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        generator = GridGenerator()
        
        with pytest.raises(GridGenerationError, match="must contain 'X' and 'Y' columns"):
            generator._calculate_bounds(data, None)
    
    def test_calculate_bounds_no_input(self):
        """Test error when no bounds source provided."""
        generator = GridGenerator()
        
        with pytest.raises(GridGenerationError, match="Must provide either data or bounds"):
            generator._calculate_bounds(None, None)
    
    def test_calculate_bounds_invalid_bounds(self):
        """Test error with invalid bounds."""
        generator = GridGenerator()
        
        with pytest.raises(GridGenerationError, match="Invalid bounds"):
            generator._calculate_bounds(None, (100, 50, 50, 100))  # min > max


class TestDimensionsCalculation:
    """Test grid dimensions calculation."""
    
    def test_calculate_dimensions_explicit_nx_ny(self):
        """Test dimension calculation with explicit nx, ny."""
        generator = GridGenerator()
        bounds = (0, 0, 100, 100)
        nx, ny, cell_size = generator._calculate_dimensions(bounds, None, 50, 50)
        
        assert nx == 50
        assert ny == 50
        assert cell_size == 2.0  # min(100/50, 100/50)
    
    def test_calculate_dimensions_from_cell_size(self):
        """Test dimension calculation from cell size."""
        generator = GridGenerator()
        bounds = (0, 0, 100, 100)
        nx, ny, cell_size = generator._calculate_dimensions(bounds, 10.0, None, None)
        
        assert nx == 10  # ceil(100/10)
        assert ny == 10  # ceil(100/10)
        assert cell_size == 10.0
    
    def test_calculate_dimensions_from_params(self):
        """Test dimension calculation from initialization parameters."""
        params = GridParameters(nx=20, ny=25)
        generator = GridGenerator(params)
        bounds = (0, 0, 100, 100)
        nx, ny, cell_size = generator._calculate_dimensions(bounds, None, None, None)
        
        assert nx == 20
        assert ny == 25
        assert cell_size == 4.0  # min(100/20, 100/25)
    
    def test_calculate_dimensions_auto(self):
        """Test automatic dimension calculation."""
        generator = GridGenerator()
        bounds = (0, 0, 100, 200)  # 1:2 aspect ratio
        nx, ny, cell_size = generator._calculate_dimensions(bounds, None, None, None)
        
        # Should auto-calculate reasonable grid
        assert nx > 0 and ny > 0
        assert ny > nx  # Should respect aspect ratio
        assert cell_size > 0


class TestGridValidation:
    """Test grid parameter validation."""
    
    def test_validate_valid_parameters(self):
        """Test validation of valid parameters."""
        generator = GridGenerator()
        bounds = (0, 0, 100, 100)
        
        # Should not raise exception
        generator._validate_grid_parameters(bounds, 50, 50)
    
    def test_validate_invalid_bounds(self):
        """Test validation with invalid bounds."""
        generator = GridGenerator()
        bounds = (0, 0, np.inf, 100)
        
        with pytest.raises(GridGenerationError, match="bounds must be finite"):
            generator._validate_grid_parameters(bounds, 50, 50)
    
    def test_validate_invalid_dimensions(self):
        """Test validation with invalid dimensions."""
        generator = GridGenerator()
        bounds = (0, 0, 100, 100)
        
        with pytest.raises(GridGenerationError, match="must be positive"):
            generator._validate_grid_parameters(bounds, 0, 50)
    
    def test_validate_large_grid_error(self):
        """Test validation error for excessively large grids."""
        generator = GridGenerator()
        bounds = (0, 0, 100, 100)
        
        with pytest.raises(GridGenerationError, match="Grid too large"):
            generator._validate_grid_parameters(bounds, 2000, 2000)  # 4M points
    
    def test_validate_large_grid_warning(self):
        """Test warning for large grids."""
        generator = GridGenerator()
        bounds = (0, 0, 100, 100)
        
        with pytest.warns(UserWarning, match="Large grid"):
            generator._validate_grid_parameters(bounds, 500, 500)  # 250k points


class TestGridGeneration:
    """Test actual grid generation."""
    
    def test_generate_basic_grid(self):
        """Test basic grid generation."""
        generator = GridGenerator()
        bounds = (0, 0, 10, 10)
        
        grid = generator._generate_grid_points(bounds, 3, 3)
        
        assert isinstance(grid, pd.DataFrame)
        assert len(grid) == 9  # 3x3 grid
        assert list(grid.columns) == ['X', 'Y']
        
        # Check coordinate values
        expected_x = [0, 5, 10, 0, 5, 10, 0, 5, 10]
        expected_y = [0, 0, 0, 5, 5, 5, 10, 10, 10]
        
        np.testing.assert_array_almost_equal(grid['X'].values, expected_x)
        np.testing.assert_array_almost_equal(grid['Y'].values, expected_y)
    
    def test_generate_single_point_grid(self):
        """Test generation of 1x1 grid."""
        generator = GridGenerator()
        bounds = (5, 10, 5, 10)
        
        grid = generator._generate_grid_points(bounds, 1, 1)
        
        assert len(grid) == 1
        assert grid.iloc[0]['X'] == 5
        assert grid.iloc[0]['Y'] == 10


class TestFullGridCreation:
    """Test complete grid creation workflow."""
    
    def test_create_regular_grid_with_data(self):
        """Test creating regular grid from data."""
        data = pd.DataFrame({
            'X': [0, 10, 20],
            'Y': [0, 10, 20],
            'value': [1, 2, 3]
        })
        
        generator = GridGenerator()
        grid = generator.create_regular_grid(data=data, cell_size=5.0)
        
        assert isinstance(grid, pd.DataFrame)
        assert 'X' in grid.columns and 'Y' in grid.columns
        assert len(grid) > 0
        
        # Check bounds are reasonable
        assert grid['X'].min() >= 0
        assert grid['X'].max() <= 20
        assert grid['Y'].min() >= 0
        assert grid['Y'].max() <= 20
    
    def test_create_regular_grid_with_bounds(self):
        """Test creating regular grid with explicit bounds."""
        generator = GridGenerator()
        grid = generator.create_regular_grid(
            bounds=(0, 0, 100, 50),
            nx=11,
            ny=6
        )
        
        assert len(grid) == 66  # 11x6
        assert grid['X'].min() == 0
        assert grid['X'].max() == 100
        assert grid['Y'].min() == 0
        assert grid['Y'].max() == 50
    
    def test_create_regular_grid_insufficient_params(self):
        """Test error with insufficient parameters."""
        generator = GridGenerator()
        
        with pytest.raises(GridGenerationError):
            generator.create_regular_grid()  # No data or bounds


class TestGridInfo:
    """Test grid information method."""
    
    def test_get_grid_info_success(self):
        """Test successful grid info retrieval."""
        data = pd.DataFrame({
            'X': [0, 100],
            'Y': [0, 50]
        })
        
        generator = GridGenerator()
        info = generator.get_grid_info(data=data, cell_size=10)
        
        assert 'bounds' in info
        assert 'dimensions' in info
        assert 'cell_size' in info
        assert 'memory_estimate_mb' in info
        
        assert info['bounds']['min_x'] == 0
        assert info['bounds']['max_x'] == 100
        assert info['dimensions']['nx'] == 10
        assert info['dimensions']['ny'] == 5
    
    def test_get_grid_info_error(self):
        """Test grid info with error condition."""
        generator = GridGenerator()
        info = generator.get_grid_info()  # No parameters
        
        assert 'error' in info
        assert isinstance(info['error'], str)


class TestIntegrationWithInterpolation:
    """Integration tests with interpolation workflow."""
    
    def test_grid_compatible_with_idw(self):
        """Test that generated grid works with IDW interpolation."""
        # Create sample data
        np.random.seed(42)
        n_points = 20
        data = pd.DataFrame({
            'X': np.random.uniform(0, 100, n_points),
            'Y': np.random.uniform(0, 100, n_points),
            'value': np.random.uniform(10, 50, n_points)
        })
        
        # Generate grid
        generator = GridGenerator()
        grid = generator.create_regular_grid(data=data, cell_size=10)
        
        # Test that grid has proper format for interpolation
        assert isinstance(grid, pd.DataFrame)
        assert 'X' in grid.columns and 'Y' in grid.columns
        assert len(grid) > 0
        assert not grid['X'].isna().any()
        assert not grid['Y'].isna().any()
        
        # Test grid bounds make sense relative to data
        data_bounds = (data['X'].min(), data['Y'].min(), data['X'].max(), data['Y'].max())
        grid_bounds = (grid['X'].min(), grid['Y'].min(), grid['X'].max(), grid['Y'].max())
        
        # Grid should encompass data (allowing for small floating point differences)
        assert grid_bounds[0] <= data_bounds[0] + 1e-10
        assert grid_bounds[1] <= data_bounds[1] + 1e-10
        assert grid_bounds[2] >= data_bounds[2] - 1e-10
        assert grid_bounds[3] >= data_bounds[3] - 1e-10