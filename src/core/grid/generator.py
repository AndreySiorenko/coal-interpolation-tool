"""
Grid generator for creating regular grids for interpolation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class GridParameters:
    """Parameters for grid generation."""
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
    cell_size: Optional[float] = None                           # Size of grid cells
    nx: Optional[int] = None                                    # Number of cells in X direction
    ny: Optional[int] = None                                    # Number of cells in Y direction
    buffer: float = 0.0                                         # Buffer around data bounds
    coordinate_system: str = "cartesian"                        # Coordinate system type


class GridGenerationError(Exception):
    """Exception raised during grid generation."""
    pass


class GridGenerator:
    """
    Generator for creating regular grids for spatial interpolation.
    
    This class provides methods to create regular grids based on data bounds
    or user-specified parameters. It supports different grid configurations
    and can handle various coordinate systems.
    
    The generator can create grids by:
    1. Specifying cell size and calculating number of cells
    2. Specifying number of cells and calculating cell size
    3. Auto-detecting optimal grid from data bounds
    """
    
    def __init__(self, params: Optional[GridParameters] = None):
        """
        Initialize grid generator.
        
        Args:
            params: Grid generation parameters
        """
        self.params = params or GridParameters()
        
    def create_regular_grid(self, 
                           data: Optional[pd.DataFrame] = None,
                           bounds: Optional[Tuple[float, float, float, float]] = None,
                           cell_size: Optional[float] = None,
                           nx: Optional[int] = None,
                           ny: Optional[int] = None) -> pd.DataFrame:
        """
        Create a regular grid for interpolation.
        
        Args:
            data: Source data to determine bounds (optional)
            bounds: Explicit bounds (min_x, min_y, max_x, max_y)
            cell_size: Size of grid cells
            nx: Number of cells in X direction
            ny: Number of cells in Y direction
            
        Returns:
            DataFrame with X, Y coordinates of grid points
            
        Raises:
            GridGenerationError: If parameters are insufficient or invalid
        """
        # Determine bounds
        grid_bounds = self._calculate_bounds(data, bounds)
        
        # Determine grid dimensions
        grid_nx, grid_ny, actual_cell_size = self._calculate_dimensions(
            grid_bounds, cell_size, nx, ny
        )
        
        # Validate parameters before generation
        self._validate_grid_parameters(grid_bounds, grid_nx, grid_ny)
        
        # Generate grid points
        grid_points = self._generate_grid_points(grid_bounds, grid_nx, grid_ny)
        
        return grid_points
    
    def _calculate_bounds(self, 
                         data: Optional[pd.DataFrame],
                         bounds: Optional[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        """
        Calculate grid bounds from data or user input.
        
        Args:
            data: Source data with X, Y columns
            bounds: User-specified bounds
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if bounds is not None:
            min_x, min_y, max_x, max_y = bounds
        elif self.params.bounds is not None:
            min_x, min_y, max_x, max_y = self.params.bounds
        elif data is not None:
            if 'X' not in data.columns or 'Y' not in data.columns:
                raise GridGenerationError("Data must contain 'X' and 'Y' columns")
            
            min_x, max_x = data['X'].min(), data['X'].max()
            min_y, max_y = data['Y'].min(), data['Y'].max()
        else:
            raise GridGenerationError("Must provide either data or bounds")
        
        # Apply buffer if specified
        buffer = self.params.buffer
        if buffer > 0:
            range_x = max_x - min_x
            range_y = max_y - min_y
            buffer_x = range_x * buffer if buffer < 1.0 else buffer
            buffer_y = range_y * buffer if buffer < 1.0 else buffer
            
            min_x -= buffer_x
            max_x += buffer_x
            min_y -= buffer_y
            max_y += buffer_y
        
        # Validate bounds
        if min_x >= max_x or min_y >= max_y:
            raise GridGenerationError(f"Invalid bounds: ({min_x}, {min_y}, {max_x}, {max_y})")
        
        return min_x, min_y, max_x, max_y
    
    def _calculate_dimensions(self,
                            bounds: Tuple[float, float, float, float],
                            cell_size: Optional[float],
                            nx: Optional[int],
                            ny: Optional[int]) -> Tuple[int, int, float]:
        """
        Calculate grid dimensions (nx, ny) and actual cell size.
        
        Args:
            bounds: Grid bounds
            cell_size: Desired cell size
            nx: Number of cells in X direction
            ny: Number of cells in Y direction
            
        Returns:
            Tuple of (nx, ny, actual_cell_size)
        """
        min_x, min_y, max_x, max_y = bounds
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        # Priority: explicit nx/ny > cell_size > default params > auto-calculation
        if nx is not None and ny is not None:
            # Use explicit dimensions
            actual_cell_size = min(range_x / nx, range_y / ny)
            return nx, ny, actual_cell_size
        
        elif cell_size is not None or self.params.cell_size is not None:
            # Calculate dimensions from cell size
            cs = cell_size if cell_size is not None else self.params.cell_size
            nx = max(1, int(np.ceil(range_x / cs)))
            ny = max(1, int(np.ceil(range_y / cs)))
            actual_cell_size = cs
            return nx, ny, actual_cell_size
        
        elif self.params.nx is not None and self.params.ny is not None:
            # Use parameters from initialization
            nx = self.params.nx
            ny = self.params.ny
            actual_cell_size = min(range_x / nx, range_y / ny)
            return nx, ny, actual_cell_size
        
        else:
            # Auto-calculate reasonable grid size
            # Aim for approximately 50x50 to 200x200 grid depending on data range
            aspect_ratio = range_y / range_x
            
            if max(range_x, range_y) < 1000:
                target_cells = 50
            elif max(range_x, range_y) < 10000:
                target_cells = 100
            else:
                target_cells = 200
            
            nx = target_cells
            ny = max(1, int(target_cells * aspect_ratio))
            actual_cell_size = min(range_x / nx, range_y / ny)
            
            return nx, ny, actual_cell_size
    
    def _validate_grid_parameters(self,
                                bounds: Tuple[float, float, float, float],
                                nx: int,
                                ny: int) -> None:
        """
        Validate grid parameters before generation.
        
        Args:
            bounds: Grid bounds
            nx: Number of cells in X direction
            ny: Number of cells in Y direction
            
        Raises:
            GridGenerationError: If parameters are invalid
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Check bounds validity
        if not all(np.isfinite([min_x, min_y, max_x, max_y])):
            raise GridGenerationError("Grid bounds must be finite numbers")
        
        # Check grid dimensions
        if nx < 1 or ny < 1:
            raise GridGenerationError(f"Grid dimensions must be positive: nx={nx}, ny={ny}")
        
        # Check for excessively large grids
        total_points = nx * ny
        max_points = 1_000_000  # 1 million points
        
        if total_points > max_points:
            raise GridGenerationError(
                f"Grid too large: {total_points:,} points exceeds maximum of {max_points:,}. "
                f"Consider using larger cell size or smaller bounds."
            )
        
        # Warn for very large grids
        if total_points > 100_000:
            warnings.warn(
                f"Large grid with {total_points:,} points may consume significant memory",
                UserWarning
            )
    
    def _generate_grid_points(self,
                            bounds: Tuple[float, float, float, float],
                            nx: int,
                            ny: int) -> pd.DataFrame:
        """
        Generate the actual grid points.
        
        Args:
            bounds: Grid bounds
            nx: Number of cells in X direction
            ny: Number of cells in Y direction
            
        Returns:
            DataFrame with X, Y coordinates
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Create coordinate arrays
        x_coords = np.linspace(min_x, max_x, nx)
        y_coords = np.linspace(min_y, max_y, ny)
        
        # Create meshgrid
        X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Flatten to create point array
        x_flat = X.ravel()
        y_flat = Y.ravel()
        
        # Create DataFrame
        grid_df = pd.DataFrame({
            'X': x_flat,
            'Y': y_flat
        })
        
        return grid_df
    
    def get_grid_info(self, 
                     data: Optional[pd.DataFrame] = None,
                     bounds: Optional[Tuple[float, float, float, float]] = None,
                     cell_size: Optional[float] = None,
                     nx: Optional[int] = None,
                     ny: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about grid that would be generated with given parameters.
        
        Args:
            data: Source data to determine bounds (optional)
            bounds: Explicit bounds
            cell_size: Size of grid cells
            nx: Number of cells in X direction
            ny: Number of cells in Y direction
            
        Returns:
            Dictionary with grid information
        """
        try:
            grid_bounds = self._calculate_bounds(data, bounds)
            grid_nx, grid_ny, actual_cell_size = self._calculate_dimensions(
                grid_bounds, cell_size, nx, ny
            )
            
            min_x, min_y, max_x, max_y = grid_bounds
            
            return {
                'bounds': {
                    'min_x': min_x,
                    'min_y': min_y,
                    'max_x': max_x,
                    'max_y': max_y,
                    'range_x': max_x - min_x,
                    'range_y': max_y - min_y
                },
                'dimensions': {
                    'nx': grid_nx,
                    'ny': grid_ny,
                    'total_points': grid_nx * grid_ny
                },
                'cell_size': {
                    'actual': actual_cell_size,
                    'x_spacing': (max_x - min_x) / (grid_nx - 1) if grid_nx > 1 else 0,
                    'y_spacing': (max_y - min_y) / (grid_ny - 1) if grid_ny > 1 else 0
                },
                'memory_estimate_mb': (grid_nx * grid_ny * 16) / (1024 * 1024)  # Rough estimate
            }
            
        except Exception as e:
            return {'error': str(e)}