"""
Example of using GridGenerator for creating interpolation grids.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.core.grid import GridGenerator, GridParameters
from src.core.interpolation import IDWInterpolator


def main():
    """Demonstrate grid generation and interpolation workflow."""
    
    # Create sample coal deposit data
    np.random.seed(42)
    n_boreholes = 50
    
    # Generate random borehole locations
    x_coords = np.random.uniform(0, 1000, n_boreholes)  # meters
    y_coords = np.random.uniform(0, 800, n_boreholes)   # meters
    
    # Simulate ash content with spatial trend
    # Higher ash content towards the east (higher X values)
    base_ash = 15 + (x_coords / 1000) * 10  # 15-25% base trend
    noise = np.random.normal(0, 2, n_boreholes)  # random variation
    ash_content = np.clip(base_ash + noise, 5, 35)  # realistic range
    
    # Create data DataFrame
    borehole_data = pd.DataFrame({
        'X': x_coords,
        'Y': y_coords,
        'ash_content': ash_content,
        'borehole_id': [f'BH-{i:03d}' for i in range(n_boreholes)]
    })
    
    print("Sample borehole data:")
    print(borehole_data.head())
    print(f"Data bounds: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], "
          f"Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")
    
    # Example 1: Create grid with fixed cell size
    print("\n" + "="*50)
    print("Example 1: Grid with 50m cell size")
    print("="*50)
    
    generator = GridGenerator()
    grid1 = generator.create_regular_grid(
        data=borehole_data,
        cell_size=50.0
    )
    
    print(f"Generated grid with {len(grid1)} points")
    print(f"Grid bounds: X=[{grid1['X'].min():.1f}, {grid1['X'].max():.1f}], "
          f"Y=[{grid1['Y'].min():.1f}, {grid1['Y'].max():.1f}]")
    
    # Example 2: Create grid with specific dimensions
    print("\n" + "="*50)
    print("Example 2: Grid with 25x20 dimensions")
    print("="*50)
    
    grid2 = generator.create_regular_grid(
        data=borehole_data,
        nx=25,
        ny=20
    )
    
    print(f"Generated grid with {len(grid2)} points (25x20)")
    
    # Example 3: Create grid with buffer
    print("\n" + "="*50) 
    print("Example 3: Grid with 10% buffer around data")
    print("="*50)
    
    params_with_buffer = GridParameters(
        cell_size=75.0,
        buffer=0.1  # 10% buffer
    )
    generator_buffered = GridGenerator(params_with_buffer)
    
    grid3 = generator_buffered.create_regular_grid(data=borehole_data)
    print(f"Generated buffered grid with {len(grid3)} points")
    
    # Example 4: Get grid information before generation
    print("\n" + "="*50)
    print("Example 4: Grid information preview")
    print("="*50)
    
    info = generator.get_grid_info(
        data=borehole_data,
        cell_size=25.0
    )
    
    print("Grid information:")
    print(f"  Bounds: {info['bounds']}")
    print(f"  Dimensions: {info['dimensions']}")
    print(f"  Cell size: {info['cell_size']}")
    print(f"  Estimated memory: {info['memory_estimate_mb']:.2f} MB")
    
    # Example 5: Interpolation workflow
    print("\n" + "="*50)
    print("Example 5: Complete interpolation workflow")
    print("="*50)
    
    # Create coarser grid for interpolation
    interp_grid = generator.create_regular_grid(
        data=borehole_data,
        cell_size=100.0  # 100m cells
    )
    
    print(f"Created interpolation grid with {len(interp_grid)} points")
    
    # Perform IDW interpolation
    idw = IDWInterpolator()
    idw.fit(borehole_data[['X', 'Y']], borehole_data['ash_content'])
    
    # Predict on grid
    interpolated_values = idw.predict(interp_grid[['X', 'Y']])
    
    # Add results to grid
    interp_grid['ash_content_interp'] = interpolated_values
    
    print(f"Interpolation completed. Ash content range: "
          f"[{interpolated_values.min():.2f}, {interpolated_values.max():.2f}]%")
    
    # Example 6: Visualization
    print("\n" + "="*50)
    print("Example 6: Creating visualization")
    print("="*50)
    
    create_visualization(borehole_data, interp_grid)
    
    print("Visualization saved as 'grid_interpolation_example.png'")


def create_visualization(borehole_data, interp_grid):
    """Create visualization of borehole data and interpolation grid."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original borehole data
    scatter1 = ax1.scatter(borehole_data['X'], borehole_data['Y'], 
                          c=borehole_data['ash_content'], 
                          cmap='YlOrRd', s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title('Original Borehole Data\n(Ash Content %)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Ash Content (%)')
    
    # Plot 2: Interpolated grid
    # Reshape for contour plot
    x_unique = sorted(interp_grid['X'].unique())
    y_unique = sorted(interp_grid['Y'].unique())
    
    # Create meshgrid for contour plot
    X_mesh = interp_grid['X'].values.reshape(len(y_unique), len(x_unique))
    Y_mesh = interp_grid['Y'].values.reshape(len(y_unique), len(x_unique))
    Z_mesh = interp_grid['ash_content_interp'].values.reshape(len(y_unique), len(x_unique))
    
    contour = ax2.contourf(X_mesh, Y_mesh, Z_mesh, levels=15, cmap='YlOrRd')
    ax2.scatter(borehole_data['X'], borehole_data['Y'], 
               c='black', s=20, marker='+', linewidths=1.5, alpha=0.8)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('Interpolated Surface\n(IDW with 100m grid)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax2, label='Ash Content (%)')
    
    plt.tight_layout()
    plt.savefig('grid_interpolation_example.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()