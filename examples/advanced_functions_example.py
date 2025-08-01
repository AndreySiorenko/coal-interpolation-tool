"""
Example usage of advanced functions: data compositing and declustering.

This script demonstrates how to use the advanced functionality for:
1. Data compositing (interval-based, statistical, domain-based)
2. Declustering (cell-based, polygon-based, distance-based)
3. Quality assessment and validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import advanced functions
from src.advanced.data_compositor import DataCompositor
from src.advanced.declustering import CellDeclusterer, PolygonDeclusterer, DistanceDeclusterer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def create_sample_drill_data():
    """Create sample drill hole data for compositing demonstration."""
    np.random.seed(42)
    
    # Create multiple drill holes
    drill_holes = []
    hole_ids = ['DH001', 'DH002', 'DH003', 'DH004', 'DH005']
    
    for hole_id in hole_ids:
        # Random hole depth
        hole_depth = np.random.uniform(20, 50)
        
        # Create intervals (variable length)
        depths = [0]
        while depths[-1] < hole_depth:
            interval_length = np.random.uniform(0.5, 2.5)
            depths.append(min(depths[-1] + interval_length, hole_depth))
        
        # Create DataFrame for this hole
        n_intervals = len(depths) - 1
        hole_data = pd.DataFrame({
            'hole_id': [hole_id] * n_intervals,
            'from_depth': depths[:-1],
            'to_depth': depths[1:],
            'coal_thickness': np.random.lognormal(0, 0.3, n_intervals),
            'ash_content': np.random.normal(15, 3, n_intervals),
            'sulfur_content': np.random.normal(0.8, 0.2, n_intervals),
            'calorific_value': np.random.normal(6500, 500, n_intervals)
        })
        
        drill_holes.append(hole_data)
    
    return pd.concat(drill_holes, ignore_index=True)


def create_sample_spatial_data():
    """Create sample spatial data for statistical compositing and declustering."""
    np.random.seed(42)
    
    # Create clustered data to demonstrate declustering
    
    # Cluster 1: High-density coal seam area
    cluster1_n = 40
    cluster1_x = np.random.normal(25, 5, cluster1_n)
    cluster1_y = np.random.normal(25, 5, cluster1_n)
    cluster1_quality = np.random.normal(20, 2, cluster1_n)
    cluster1_thickness = np.random.normal(2.5, 0.5, cluster1_n)
    
    # Cluster 2: Medium-density area
    cluster2_n = 25
    cluster2_x = np.random.normal(75, 8, cluster2_n)
    cluster2_y = np.random.normal(75, 8, cluster2_n)
    cluster2_quality = np.random.normal(18, 3, cluster2_n)
    cluster2_thickness = np.random.normal(1.8, 0.4, cluster2_n)
    
    # Scattered points: Exploration data
    scatter_n = 20
    scatter_x = np.random.uniform(0, 100, scatter_n)
    scatter_y = np.random.uniform(0, 100, scatter_n)
    scatter_quality = np.random.normal(16, 4, scatter_n)
    scatter_thickness = np.random.normal(1.2, 0.6, scatter_n)
    
    # Combine all data
    spatial_data = pd.DataFrame({
        'x': np.concatenate([cluster1_x, cluster2_x, scatter_x]),
        'y': np.concatenate([cluster1_y, cluster2_y, scatter_y]),
        'z': np.random.uniform(0, 30, cluster1_n + cluster2_n + scatter_n),
        'coal_quality': np.concatenate([cluster1_quality, cluster2_quality, scatter_quality]),
        'thickness': np.concatenate([cluster1_thickness, cluster2_thickness, scatter_thickness]),
        'data_type': (['dense'] * cluster1_n + ['medium'] * cluster2_n + ['sparse'] * scatter_n)
    })
    
    return spatial_data


def demonstrate_interval_compositing():
    """Demonstrate interval-based compositing on drill hole data."""
    print("\n" + "="*60)
    print("INTERVAL-BASED COMPOSITING DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    drill_data = create_sample_drill_data()
    print(f"Original drill data: {len(drill_data)} intervals from {drill_data['hole_id'].nunique()} holes")
    
    # Initialize compositor
    compositor = DataCompositor(min_composite_length=1.0, max_composite_length=10.0)
    
    # Test different composite lengths
    composite_lengths = [1.5, 2.0, 3.0]
    
    for comp_length in composite_lengths:
        print(f"\n--- Compositing with {comp_length}m intervals ---")
        
        result = compositor.interval_based_compositing(
            data=drill_data,
            hole_id_col='hole_id',
            from_col='from_depth',
            to_col='to_depth',
            value_cols=['coal_thickness', 'ash_content', 'sulfur_content'],
            composite_length=comp_length,
            method='length_weighted',
            min_recovery=0.7
        )
        
        print(f"Original intervals: {result.original_count}")
        print(f"Composited intervals: {result.composited_count}")
        print(f"Reduction ratio: {result.quality_metrics['reduction_ratio']:.2f}")
        print(f"Average recovery: {result.quality_metrics['average_recovery']:.2f}")
        print(f"Rejection rate: {result.quality_metrics['rejection_rate']:.2f}")
        
        # Quality assessment
        quality = compositor.validate_compositing_quality(result)
        print(f"Quality assessment: {quality['overall_quality']} (score: {quality['quality_score']:.2f})")
        
        if quality['recommendations']:
            print("Recommendations:")
            for rec in quality['recommendations']:
                print(f"  - {rec}")


def demonstrate_statistical_compositing():
    """Demonstrate statistical compositing on spatial data."""
    print("\n" + "="*60)
    print("STATISTICAL COMPOSITING DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    spatial_data = create_sample_spatial_data()
    print(f"Original spatial data: {len(spatial_data)} points")
    
    # Initialize compositor
    compositor = DataCompositor()
    
    # Test different methods and radii
    methods = ['inverse_distance', 'average', 'median']
    radii = [8.0, 12.0, 16.0]
    
    for method in methods:
        print(f"\n--- Statistical compositing using {method} method ---")
        
        for radius in radii:
            result = compositor.statistical_compositing(
                data=spatial_data,
                x_col='x',
                y_col='y',
                value_cols=['coal_quality', 'thickness'],
                composite_radius=radius,
                method=method,
                min_samples=2,
                max_samples=8
            )
            
            print(f"Radius {radius}m: {result.original_count} → {result.composited_count} points "
                  f"(reduction: {result.quality_metrics['reduction_ratio']:.2f})")
            
            if len(result.composited_data) > 0:
                avg_samples = result.quality_metrics['average_samples_per_composite']
                print(f"  Average samples per composite: {avg_samples:.1f}")


def demonstrate_cell_declustering():
    """Demonstrate cell-based declustering."""
    print("\n" + "="*60)
    print("CELL-BASED DECLUSTERING DEMONSTRATION")
    print("="*60)
    
    # Create clustered data
    spatial_data = create_sample_spatial_data()
    print(f"Original data: {len(spatial_data)} points")
    
    # Show data distribution by type
    type_counts = spatial_data['data_type'].value_counts()
    print("Data distribution:")
    for data_type, count in type_counts.items():
        print(f"  {data_type}: {count} points")
    
    # Initialize declusterer
    declusterer = CellDeclusterer()
    
    # Perform declustering
    result = declusterer.decluster(
        data=spatial_data,
        x_col='x',
        y_col='y',
        value_col='coal_quality'
    )
    
    print(f"\n--- Declustering Results ---")
    print(f"Declustering method: {result.declustering_method}")
    print(f"Original count: {result.original_count}")
    print(f"Effective count: {result.effective_count:.1f}")
    print(f"Efficiency: {result.quality_metrics['efficiency']:.2f}")
    
    # Weight statistics
    weights = result.weights
    print(f"\nWeight statistics:")
    print(f"  Min weight: {weights.min():.3f}")
    print(f"  Max weight: {weights.max():.3f}")
    print(f"  Mean weight: {weights.mean():.3f}")
    print(f"  Weight CV: {result.quality_metrics['weight_cv']:.3f}")
    
    # Grid information
    info = result.declustering_info
    print(f"\nGrid information:")
    print(f"  Cell size: {info['cell_size']:.2f}")
    print(f"  Grid dimensions: {info['n_cells_x']} × {info['n_cells_y']}")
    print(f"  Total cells: {info['total_cells']}")
    print(f"  Occupied cells: {info['occupied_cells']}")
    
    return result


def demonstrate_polygon_declustering():
    """Demonstrate polygon-based (Voronoi) declustering."""
    print("\n" + "="*60)
    print("POLYGON-BASED DECLUSTERING DEMONSTRATION")
    print("="*60)
    
    # Create sample data (fewer points for stable Voronoi)
    np.random.seed(42)
    
    # Create less clustered data for better Voronoi results
    n_points = 25
    spatial_data = pd.DataFrame({
        'x': np.random.uniform(10, 90, n_points),
        'y': np.random.uniform(10, 90, n_points),
        'coal_quality': np.random.normal(18, 3, n_points)
    })
    
    print(f"Sample data: {len(spatial_data)} points")
    
    # Initialize declusterer
    declusterer = PolygonDeclusterer(polygon_method='voronoi')
    
    # Perform declustering
    result = declusterer.decluster(
        data=spatial_data,
        x_col='x',
        y_col='y',
        value_col='coal_quality'
    )
    
    print(f"\n--- Declustering Results ---")
    print(f"Declustering method: {result.declustering_method}")
    print(f"Original count: {result.original_count}")
    print(f"Effective count: {result.effective_count:.1f}")
    print(f"Efficiency: {result.quality_metrics['efficiency']:.2f}")
    
    # Weight statistics
    weights = result.weights
    print(f"\nWeight statistics:")
    print(f"  Min weight: {weights.min():.3f}")
    print(f"  Max weight: {weights.max():.3f}")
    print(f"  Mean weight: {weights.mean():.3f}")
    print(f"  Weight CV: {result.quality_metrics['weight_cv']:.3f}")


def demonstrate_distance_declustering():
    """Demonstrate distance-based declustering."""
    print("\n" + "="*60)
    print("DISTANCE-BASED DECLUSTERING DEMONSTRATION")
    print("="*60)
    
    # Create data with varying density
    spatial_data = create_sample_spatial_data()
    print(f"Original data: {len(spatial_data)} points")
    
    # Test different parameters
    neighbor_counts = [3, 5, 8]
    
    for n_neighbors in neighbor_counts:
        print(f"\n--- Using {n_neighbors} nearest neighbors ---")
        
        declusterer = DistanceDeclusterer(
            n_neighbors=n_neighbors,
            distance_power=2.0
        )
        
        result = declusterer.decluster(
            data=spatial_data,
            x_col='x',
            y_col='y',
            value_col='coal_quality'
        )
        
        print(f"Effective count: {result.effective_count:.1f}")
        print(f"Efficiency: {result.quality_metrics['efficiency']:.2f}")
        print(f"Weight range: {result.weights.min():.3f} - {result.weights.max():.3f}")


def create_visualization(cell_result):
    """Create visualization of declustering results."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        data = cell_result.declustered_data
        
        # Plot 1: Original data points colored by data type
        ax1 = axes[0]
        scatter_types = data['data_type'].unique()
        colors = ['red', 'blue', 'green']
        
        for i, data_type in enumerate(scatter_types):
            mask = data['data_type'] == data_type
            ax1.scatter(data.loc[mask, 'x'], data.loc[mask, 'y'], 
                       c=colors[i % len(colors)], label=data_type, alpha=0.7, s=50)
        
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Original Data Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Declustered weights
        ax2 = axes[1]
        scatter = ax2.scatter(data['x'], data['y'], c=data['decluster_weight'], 
                            cmap='viridis', s=60, alpha=0.8)
        plt.colorbar(scatter, ax=ax2, label='Decluster Weight')
        
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('Declustering Weights')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_functions_example.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'advanced_functions_example.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")


def main():
    """Main demonstration function."""
    print("COAL DEPOSIT INTERPOLATION TOOL")
    print("Advanced Functions Demonstration")
    print("=" * 80)
    
    try:
        # Demonstrate compositing methods
        demonstrate_interval_compositing()
        demonstrate_statistical_compositing()
        
        # Demonstrate declustering methods
        cell_result = demonstrate_cell_declustering()
        demonstrate_polygon_declustering()
        demonstrate_distance_declustering()
        
        # Create visualization
        create_visualization(cell_result)
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey takeaways:")
        print("1. Interval compositing reduces drill hole data while preserving geological information")
        print("2. Statistical compositing creates regular grids from scattered data")
        print("3. Cell declustering reduces spatial clustering bias using regular grids")
        print("4. Polygon declustering uses Voronoi tessellation for irregular boundaries")
        print("5. Distance declustering considers local point density")
        print("\nThese advanced functions improve interpolation quality by:")
        print("- Reducing data redundancy")
        print("- Correcting for spatial clustering bias")
        print("- Creating more representative datasets")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()