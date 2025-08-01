"""
Tests for grid generation and integration components.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.grid.generator import GridGenerator, GridParameters, GridBounds, Grid
from core.interpolation.idw import IDWInterpolator


class TestGridIntegration:
    """Test grid generation integration with interpolation."""
    
    def setup_method(self):
        """Set up test data."""
        self.generator = GridGenerator()
        
        # Create sample data points
        np.random.seed(42)
        self.data = pd.DataFrame({
            'X': np.random.uniform(0, 100, 20),
            'Y': np.random.uniform(0, 100, 20),
            'Value': np.random.uniform(10, 50, 20)
        })
        
        self.bounds = GridBounds(0, 100, 0, 100)
        self.params = GridParameters(
            bounds=self.bounds,
            cell_size=10.0,
            buffer_factor=0.1
        )
    
    def test_grid_interpolation_workflow(self):
        """Test complete grid generation and interpolation workflow."""
        # Generate grid
        grid = self.generator.generate_regular_grid(self.params)
        
        assert isinstance(grid, Grid)
        assert grid.points.shape[1] == 2  # 2D points
        assert len(grid.points) > 0
        
        # Use grid for interpolation
        interpolator = IDWInterpolator()
        interpolator.fit(self.data, 'X', 'Y', 'Value')
        
        results = interpolator.predict(grid.points)
        
        assert len(results) == len(grid.points)
        assert not np.any(np.isnan(results))
        
        # Verify results are within reasonable range
        data_min = self.data['Value'].min()
        data_max = self.data['Value'].max()
        
        # Allow some extrapolation beyond data range
        assert np.all(results >= data_min - 10)
        assert np.all(results <= data_max + 10)
    
    def test_adaptive_grid_generation(self):
        """Test adaptive grid generation based on data density."""
        # Create non-uniform data distribution
        clustered_data = pd.DataFrame({
            'X': np.concatenate([
                np.random.uniform(10, 30, 15),  # Dense cluster
                np.random.uniform(70, 90, 5)    # Sparse area
            ]),
            'Y': np.concatenate([
                np.random.uniform(10, 30, 15),
                np.random.uniform(70, 90, 5)
            ]),
            'Value': np.random.uniform(10, 50, 20)
        })
        
        # Test regular grid with clustered data
        grid = self.generator.generate_regular_grid(self.params)
        
        interpolator = IDWInterpolator()
        interpolator.fit(clustered_data, 'X', 'Y', 'Value')
        
        results = interpolator.predict(grid.points)
        
        # Should still produce valid results despite uneven distribution
        assert not np.any(np.isnan(results))
        assert len(results) == len(grid.points)
    
    def test_grid_refinement(self):
        """Test grid refinement for better resolution."""
        # Generate coarse grid
        coarse_params = GridParameters(
            bounds=self.bounds,
            cell_size=20.0,
            buffer_factor=0.1
        )
        coarse_grid = self.generator.generate_regular_grid(coarse_params)
        
        # Generate fine grid
        fine_params = GridParameters(
            bounds=self.bounds,
            cell_size=5.0,
            buffer_factor=0.1
        )
        fine_grid = self.generator.generate_regular_grid(fine_params)
        
        # Fine grid should have more points
        assert len(fine_grid.points) > len(coarse_grid.points)
        
        # Test interpolation on both grids
        interpolator = IDWInterpolator()
        interpolator.fit(self.data, 'X', 'Y', 'Value')
        
        coarse_results = interpolator.predict(coarse_grid.points)
        fine_results = interpolator.predict(fine_grid.points)
        
        assert not np.any(np.isnan(coarse_results))
        assert not np.any(np.isnan(fine_results))
        assert len(fine_results) > len(coarse_results)
    
    def test_irregular_bounds_grid(self):
        """Test grid generation with irregular bounds."""
        # Create irregular bounds
        irregular_bounds = GridBounds(-50, 150, 25, 75)
        irregular_params = GridParameters(
            bounds=irregular_bounds,
            cell_size=15.0,
            buffer_factor=0.05
        )
        
        grid = self.generator.generate_regular_grid(irregular_params)
        
        # Verify grid points are within bounds (accounting for buffer)
        x_coords = grid.points[:, 0]
        y_coords = grid.points[:, 1]
        
        buffer_x = (irregular_bounds.x_max - irregular_bounds.x_min) * 0.05
        buffer_y = (irregular_bounds.y_max - irregular_bounds.y_min) * 0.05
        
        assert np.all(x_coords >= irregular_bounds.x_min - buffer_x)
        assert np.all(x_coords <= irregular_bounds.x_max + buffer_x)
        assert np.all(y_coords >= irregular_bounds.y_min - buffer_y)
        assert np.all(y_coords <= irregular_bounds.y_max + buffer_y)
    
    def test_grid_memory_efficiency(self):
        """Test memory efficiency with large grids."""
        # Create parameters for a larger grid
        large_params = GridParameters(
            bounds=GridBounds(0, 1000, 0, 1000),
            cell_size=50.0,
            buffer_factor=0.1
        )
        
        grid = self.generator.generate_regular_grid(large_params)
        
        # Check that grid was created successfully
        assert isinstance(grid, Grid)
        assert len(grid.points) > 100  # Should be reasonable size
        
        # Verify memory usage is reasonable (points should be float64)
        expected_memory = grid.points.nbytes
        assert expected_memory < 1e7  # Less than 10MB for this size
    
    def test_grid_coordinate_precision(self):
        """Test coordinate precision in generated grids."""
        # Use small cell size to test precision
        precise_params = GridParameters(
            bounds=GridBounds(0, 10, 0, 10),
            cell_size=0.1,
            buffer_factor=0.0  # No buffer for precision test
        )
        
        grid = self.generator.generate_regular_grid(precise_params)
        
        # Check that coordinates align with expected grid points
        x_coords = np.unique(grid.points[:, 0])
        y_coords = np.unique(grid.points[:, 1])
        
        # Should have coordinates at multiples of cell_size
        expected_x = np.arange(0, 10.1, 0.1)
        expected_y = np.arange(0, 10.1, 0.1)
        
        # Allow small floating point errors
        assert np.allclose(x_coords, expected_x, atol=1e-10)
        assert np.allclose(y_coords, expected_y, atol=1e-10)
    
    def test_grid_with_empty_data(self):
        """Test grid generation with minimal or empty data."""
        # Empty data
        empty_data = pd.DataFrame(columns=['X', 'Y', 'Value'])
        
        # Grid should still generate
        grid = self.generator.generate_regular_grid(self.params)
        assert len(grid.points) > 0
        
        # Interpolation with empty data should handle gracefully
        interpolator = IDWInterpolator()
        
        with pytest.raises(Exception):  # Should raise an error for empty data
            interpolator.fit(empty_data, 'X', 'Y', 'Value')
    
    def test_grid_bounds_validation(self):
        """Test validation of grid bounds."""
        # Test invalid bounds (x_min > x_max)
        with pytest.raises(ValueError):
            invalid_bounds = GridBounds(100, 0, 0, 100)
            GridParameters(bounds=invalid_bounds, cell_size=10)
        
        # Test invalid bounds (y_min > y_max)
        with pytest.raises(ValueError):
            invalid_bounds = GridBounds(0, 100, 100, 0)
            GridParameters(bounds=invalid_bounds, cell_size=10)
        
        # Test zero cell size
        with pytest.raises(ValueError):
            GridParameters(bounds=self.bounds, cell_size=0)
        
        # Test negative cell size
        with pytest.raises(ValueError):
            GridParameters(bounds=self.bounds, cell_size=-10)
    
    def test_grid_statistics(self):
        """Test grid statistics and properties."""
        grid = self.generator.generate_regular_grid(self.params)
        
        # Calculate grid statistics
        x_range = grid.points[:, 0].max() - grid.points[:, 0].min()
        y_range = grid.points[:, 1].max() - grid.points[:, 1].min()
        
        # Should approximately match expected range (with buffer)
        expected_x_range = (self.bounds.x_max - self.bounds.x_min) * (1 + 2 * self.params.buffer_factor)
        expected_y_range = (self.bounds.y_max - self.bounds.y_min) * (1 + 2 * self.params.buffer_factor)
        
        assert abs(x_range - expected_x_range) < self.params.cell_size
        assert abs(y_range - expected_y_range) < self.params.cell_size
        
        # Test point density
        area = x_range * y_range
        point_density = len(grid.points) / area
        expected_density = 1.0 / (self.params.cell_size ** 2)
        
        # Should be close to expected density
        assert abs(point_density - expected_density) / expected_density < 0.2  # Within 20%
    
    def test_multiple_grids_consistency(self):
        """Test consistency when generating multiple grids."""
        # Generate same grid multiple times
        grid1 = self.generator.generate_regular_grid(self.params)
        grid2 = self.generator.generate_regular_grid(self.params)
        
        # Should be identical
        assert np.array_equal(grid1.points, grid2.points)
        
        # Generate with different parameters
        params2 = GridParameters(
            bounds=self.bounds,
            cell_size=5.0,
            buffer_factor=0.1
        )
        grid3 = self.generator.generate_regular_grid(params2)
        
        # Should be different
        assert not np.array_equal(grid1.points, grid3.points)
        assert len(grid3.points) > len(grid1.points)  # Finer grid has more points


class TestGridUtils:
    """Test grid utility functions."""
    
    def test_grid_from_data_bounds(self):
        """Test automatic grid generation from data bounds."""
        # Create test data
        data = pd.DataFrame({
            'X': [10, 20, 30, 40, 50],
            'Y': [15, 25, 35, 45, 55],
            'Value': [1, 2, 3, 4, 5]
        })
        
        generator = GridGenerator()
        
        # Generate grid from data
        grid = generator.generate_from_data(data, 'X', 'Y', cell_size=5.0)
        
        # Verify grid encompasses data
        x_min, x_max = data['X'].min(), data['X'].max()
        y_min, y_max = data['Y'].min(), data['Y'].max()
        
        grid_x_min, grid_x_max = grid.points[:, 0].min(), grid.points[:, 0].max()
        grid_y_min, grid_y_max = grid.points[:, 1].min(), grid.points[:, 1].max()
        
        assert grid_x_min <= x_min
        assert grid_x_max >= x_max
        assert grid_y_min <= y_min
        assert grid_y_max >= y_max
    
    def test_grid_subset_extraction(self):
        """Test extracting grid subsets."""
        # Generate large grid
        large_params = GridParameters(
            bounds=GridBounds(0, 100, 0, 100),
            cell_size=5.0,
            buffer_factor=0.0
        )
        
        generator = GridGenerator()
        full_grid = generator.generate_regular_grid(large_params)
        
        # Extract subset
        subset_bounds = GridBounds(25, 75, 25, 75)
        
        # Find points within subset bounds
        x_coords = full_grid.points[:, 0]
        y_coords = full_grid.points[:, 1]
        
        mask = ((x_coords >= subset_bounds.x_min) & 
                (x_coords <= subset_bounds.x_max) &
                (y_coords >= subset_bounds.y_min) & 
                (y_coords <= subset_bounds.y_max))
        
        subset_points = full_grid.points[mask]
        
        # Verify subset properties
        assert len(subset_points) < len(full_grid.points)
        assert np.all(subset_points[:, 0] >= subset_bounds.x_min)
        assert np.all(subset_points[:, 0] <= subset_bounds.x_max)
        assert np.all(subset_points[:, 1] >= subset_bounds.y_min)
        assert np.all(subset_points[:, 1] <= subset_bounds.y_max)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])