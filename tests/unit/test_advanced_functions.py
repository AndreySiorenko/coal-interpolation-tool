"""
Unit tests for advanced functions (compositing and declustering).
"""

import unittest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, MagicMock

# Import modules to test
from src.advanced.data_compositor import DataCompositor, CompositingResult
from src.advanced.declustering import CellDeclusterer, PolygonDeclusterer, DistanceDeclusterer, DeclusteringResult


class TestDataCompositor(unittest.TestCase):
    """Test cases for DataCompositor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.compositor = DataCompositor()
        
        # Create sample drill hole data
        self.drill_data = pd.DataFrame({
            'hole_id': ['DH001'] * 10 + ['DH002'] * 8,
            'from_depth': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5],
            'to_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12],
            'coal_thickness': [0.5, 0.8, 1.2, 0.9, 1.1, 0.7, 1.0, 0.6, 0.9, 1.3, 
                              0.6, 1.1, 0.8, 1.0, 0.9, 1.2, 0.7, 0.8],
            'ash_content': [15.2, 18.5, 12.3, 16.7, 14.1, 19.8, 13.9, 17.2, 15.6, 11.8,
                           16.1, 13.7, 18.9, 14.5, 16.3, 12.1, 17.8, 15.4]
        })
        
        # Create sample spatial data
        np.random.seed(42)
        n_points = 50
        self.spatial_data = pd.DataFrame({
            'x': np.random.uniform(0, 100, n_points),
            'y': np.random.uniform(0, 100, n_points),
            'z': np.random.uniform(0, 50, n_points),
            'coal_quality': np.random.uniform(10, 25, n_points),
            'thickness': np.random.uniform(0.5, 3.0, n_points)
        })
        
        # Create domain data
        self.domain_data = pd.DataFrame({
            'domain': ['A'] * 20 + ['B'] * 15 + ['C'] * 10,
            'value1': np.random.normal(10, 2, 45),
            'value2': np.random.normal(20, 5, 45)
        })
    
    def test_initialization(self):
        """Test DataCompositor initialization."""
        compositor = DataCompositor(
            min_composite_length=2.0,
            max_composite_length=15.0,
            quality_threshold=0.9
        )
        
        self.assertEqual(compositor.min_composite_length, 2.0)
        self.assertEqual(compositor.max_composite_length, 15.0)
        self.assertEqual(compositor.quality_threshold, 0.9)
    
    def test_interval_based_compositing_basic(self):
        """Test basic interval-based compositing."""
        result = self.compositor.interval_based_compositing(
            data=self.drill_data,
            hole_id_col='hole_id',
            from_col='from_depth',
            to_col='to_depth',
            value_cols=['coal_thickness', 'ash_content'],
            composite_length=2.0,
            method='length_weighted'
        )
        
        self.assertIsInstance(result, CompositingResult)
        self.assertEqual(result.compositing_method, 'interval_based')
        self.assertLess(result.composited_count, result.original_count)
        self.assertIn('coal_thickness', result.composited_data.columns)
        self.assertIn('ash_content', result.composited_data.columns)
        self.assertIn('recovery', result.composited_data.columns)
    
    def test_interval_based_compositing_methods(self):
        """Test different compositing methods."""
        # Test length-weighted method
        result_weighted = self.compositor.interval_based_compositing(
            data=self.drill_data,
            hole_id_col='hole_id',
            from_col='from_depth',
            to_col='to_depth',
            value_cols=['coal_thickness'],
            composite_length=3.0,
            method='length_weighted'
        )
        
        # Test simple average method
        result_average = self.compositor.interval_based_compositing(
            data=self.drill_data,
            hole_id_col='hole_id',
            from_col='from_depth',
            to_col='to_depth',
            value_cols=['coal_thickness'],
            composite_length=3.0,
            method='simple_average'
        )
        
        self.assertEqual(result_weighted.compositing_method, 'interval_based')
        self.assertEqual(result_average.compositing_method, 'interval_based')
        
        # Values should be different between methods
        if len(result_weighted.composited_data) > 0 and len(result_average.composited_data) > 0:
            # May be the same in some cases, but should have same structure
            self.assertEqual(len(result_weighted.composited_data.columns), 
                           len(result_average.composited_data.columns))
    
    def test_statistical_compositing(self):
        """Test statistical compositing."""
        result = self.compositor.statistical_compositing(
            data=self.spatial_data,
            x_col='x',
            y_col='y',
            value_cols=['coal_quality', 'thickness'],
            composite_radius=15.0,
            method='inverse_distance',
            min_samples=2
        )
        
        self.assertIsInstance(result, CompositingResult)
        self.assertEqual(result.compositing_method, 'statistical')
        self.assertIn('coal_quality', result.composited_data.columns)
        self.assertIn('n_samples', result.composited_data.columns)
    
    def test_statistical_compositing_methods(self):
        """Test different statistical compositing methods."""
        methods = ['inverse_distance', 'average', 'median']
        
        for method in methods:
            with self.subTest(method=method):
                result = self.compositor.statistical_compositing(
                    data=self.spatial_data,
                    x_col='x',
                    y_col='y',
                    value_cols=['coal_quality'],
                    composite_radius=20.0,
                    method=method,
                    min_samples=1
                )
                
                self.assertEqual(result.compositing_method, 'statistical')
                if len(result.composited_data) > 0:
                    self.assertIn('coal_quality', result.composited_data.columns)
    
    def test_domain_based_compositing(self):
        """Test domain-based compositing."""
        result = self.compositor.domain_based_compositing(
            data=self.domain_data,
            domain_col='domain',
            value_cols=['value1', 'value2'],
            method='domain_weighted'
        )
        
        self.assertIsInstance(result, CompositingResult)
        self.assertEqual(result.compositing_method, 'domain_based')
        self.assertEqual(len(result.composited_data), 3)  # 3 domains
        self.assertIn('domain', result.composited_data.columns)
        self.assertIn('value1', result.composited_data.columns)
    
    def test_validate_compositing_quality(self):
        """Test compositing quality validation."""
        result = self.compositor.interval_based_compositing(
            data=self.drill_data,
            hole_id_col='hole_id',
            from_col='from_depth',
            to_col='to_depth',
            value_cols=['coal_thickness'],
            composite_length=2.0
        )
        
        quality_assessment = self.compositor.validate_compositing_quality(result)
        
        self.assertIn('overall_quality', quality_assessment)
        self.assertIn('quality_score', quality_assessment)
        self.assertIn('issues', quality_assessment)
        self.assertIn('recommendations', quality_assessment)
        self.assertIsInstance(quality_assessment['quality_score'], float)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty data
        empty_data = pd.DataFrame(columns=['hole_id', 'from_depth', 'to_depth', 'value'])
        result = self.compositor.interval_based_compositing(
            data=empty_data,
            hole_id_col='hole_id',
            from_col='from_depth',
            to_col='to_depth',
            value_cols=['value'],
            composite_length=1.0
        )
        
        self.assertEqual(len(result.composited_data), 0)
        self.assertEqual(result.original_count, 0)
        
        # Single row data
        single_row = self.drill_data.iloc[:1].copy()
        result = self.compositor.interval_based_compositing(
            data=single_row,
            hole_id_col='hole_id',
            from_col='from_depth',
            to_col='to_depth',
            value_cols=['coal_thickness'],
            composite_length=2.0
        )
        
        self.assertIsInstance(result, CompositingResult)


class TestCellDeclusterer(unittest.TestCase):
    """Test cases for CellDeclusterer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.declusterer = CellDeclusterer()
        
        # Create clustered data
        np.random.seed(42)
        
        # Cluster 1: Dense cluster
        cluster1_x = np.random.normal(20, 2, 30)
        cluster1_y = np.random.normal(20, 2, 30)
        
        # Cluster 2: Medium density
        cluster2_x = np.random.normal(50, 5, 20)
        cluster2_y = np.random.normal(50, 5, 20)
        
        # Scattered points
        scatter_x = np.random.uniform(0, 100, 10)
        scatter_y = np.random.uniform(0, 100, 10)
        
        # Combine all points
        x = np.concatenate([cluster1_x, cluster2_x, scatter_x])
        y = np.concatenate([cluster1_y, cluster2_y, scatter_y])
        values = np.random.uniform(10, 30, len(x))
        
        self.clustered_data = pd.DataFrame({
            'x': x,
            'y': y,
            'value': values
        })
    
    def test_initialization(self):
        """Test CellDeclusterer initialization."""
        declusterer = CellDeclusterer(
            cell_size=5.0,
            min_cells_x=20,
            min_cells_y=15,
            max_cells=500
        )
        
        self.assertEqual(declusterer.cell_size, 5.0)
        self.assertEqual(declusterer.min_cells_x, 20)
        self.assertEqual(declusterer.min_cells_y, 15)
        self.assertEqual(declusterer.max_cells, 500)
    
    def test_cell_declustering_basic(self):
        """Test basic cell declustering."""
        result = self.declusterer.decluster(
            data=self.clustered_data,
            x_col='x',
            y_col='y',
            value_col='value'
        )
        
        self.assertIsInstance(result, DeclusteringResult)
        self.assertEqual(result.declustering_method, 'cell_based')
        self.assertEqual(len(result.weights), len(self.clustered_data))
        self.assertIn('decluster_weight', result.declustered_data.columns)
        self.assertIn('cell_id', result.declustered_data.columns)
        
        # Check weight properties
        self.assertAlmostEqual(result.weights.sum(), len(self.clustered_data), places=1)
        self.assertGreater(result.weights.min(), 0)
    
    def test_optimal_cell_size_calculation(self):
        """Test optimal cell size calculation."""
        x = self.clustered_data['x'].values
        y = self.clustered_data['y'].values
        
        cell_size = self.declusterer._calculate_optimal_cell_size(x, y)
        
        self.assertIsInstance(cell_size, float)
        self.assertGreater(cell_size, 0)
        
        # Should be reasonable relative to data extent
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        min_extent = min(x_range, y_range)
        
        self.assertLess(cell_size, min_extent / 2)  # Not too large
        self.assertGreater(cell_size, min_extent / 100)  # Not too small
    
    def test_fixed_cell_size(self):
        """Test declustering with fixed cell size."""
        declusterer = CellDeclusterer(cell_size=10.0)
        
        result = declusterer.decluster(
            data=self.clustered_data,
            x_col='x',
            y_col='y',
            value_col='value'
        )
        
        self.assertEqual(result.declustering_info['cell_size'], 10.0)
    
    def test_quality_metrics(self):
        """Test quality metrics calculation."""
        result = self.declusterer.decluster(
            data=self.clustered_data,
            x_col='x',
            y_col='y',
            value_col='value'
        )
        
        metrics = result.quality_metrics
        
        self.assertIn('min_weight', metrics)
        self.assertIn('max_weight', metrics)
        self.assertIn('mean_weight', metrics)
        self.assertIn('effective_sample_size', metrics)
        self.assertIn('efficiency', metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['efficiency'], 0)
        self.assertLessEqual(metrics['efficiency'], 1)
        self.assertGreater(metrics['effective_sample_size'], 0)


class TestPolygonDeclusterer(unittest.TestCase):
    """Test cases for PolygonDeclusterer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.declusterer = PolygonDeclusterer()
        
        # Create test data with known distribution
        np.random.seed(42)
        n_points = 20  # Smaller number for Voronoi stability
        
        self.test_data = pd.DataFrame({
            'x': np.random.uniform(10, 90, n_points),
            'y': np.random.uniform(10, 90, n_points),
            'value': np.random.uniform(15, 25, n_points)
        })
    
    def test_initialization(self):
        """Test PolygonDeclusterer initialization."""
        declusterer = PolygonDeclusterer(
            polygon_method='voronoi',
            boundary_buffer=0.2
        )
        
        self.assertEqual(declusterer.polygon_method, 'voronoi')
        self.assertEqual(declusterer.boundary_buffer, 0.2)
    
    def test_voronoi_declustering(self):
        """Test Voronoi-based declustering."""
        result = self.declusterer.decluster(
            data=self.test_data,
            x_col='x',
            y_col='y',
            value_col='value'
        )
        
        self.assertIsInstance(result, DeclusteringResult)
        self.assertEqual(result.declustering_method, 'polygon_based')
        self.assertEqual(len(result.weights), len(self.test_data))
        self.assertIn('decluster_weight', result.declustered_data.columns)
        
        # Check weight normalization
        self.assertAlmostEqual(result.weights.sum(), len(self.test_data), places=1)
    
    def test_polygon_area_calculation(self):
        """Test polygon area calculation."""
        # Simple triangle
        triangle = np.array([[0, 0], [1, 0], [0, 1]])
        area = self.declusterer._polygon_area(triangle)
        self.assertAlmostEqual(area, 0.5, places=5)
        
        # Square
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        area = self.declusterer._polygon_area(square)
        self.assertAlmostEqual(area, 1.0, places=5)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Very few points (less than 3)
        few_points = self.test_data.iloc[:2].copy()
        result = self.declusterer.decluster(
            data=few_points,
            x_col='x',
            y_col='y',
            value_col='value'
        )
        
        # Should handle gracefully
        self.assertEqual(len(result.weights), 2)
        self.assertTrue(np.allclose(result.weights, 1.0))


class TestDistanceDeclusterer(unittest.TestCase):
    """Test cases for DistanceDeclusterer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.declusterer = DistanceDeclusterer()
        
        # Create data with varying density
        np.random.seed(42)
        
        # Dense cluster
        dense_x = np.random.normal(25, 3, 25)
        dense_y = np.random.normal(25, 3, 25)
        
        # Sparse points
        sparse_x = np.random.uniform(50, 100, 10)
        sparse_y = np.random.uniform(50, 100, 10)
        
        x = np.concatenate([dense_x, sparse_x])
        y = np.concatenate([dense_y, sparse_y])
        values = np.random.uniform(10, 30, len(x))
        
        self.density_data = pd.DataFrame({
            'x': x,
            'y': y,
            'value': values
        })
    
    def test_initialization(self):
        """Test DistanceDeclusterer initialization."""
        declusterer = DistanceDeclusterer(
            n_neighbors=8,
            distance_power=1.5
        )
        
        self.assertEqual(declusterer.n_neighbors, 8)
        self.assertEqual(declusterer.distance_power, 1.5)
    
    def test_distance_declustering(self):
        """Test distance-based declustering."""
        result = self.declusterer.decluster(
            data=self.density_data,
            x_col='x',
            y_col='y',
            value_col='value'
        )
        
        self.assertIsInstance(result, DeclusteringResult)
        self.assertEqual(result.declustering_method, 'distance_based')
        self.assertEqual(len(result.weights), len(self.density_data))
        self.assertIn('decluster_weight', result.declustered_data.columns)
        
        # Check weight normalization
        self.assertAlmostEqual(result.weights.sum(), len(self.density_data), places=1)
        
        # Dense areas should have lower weights, sparse areas higher weights
        # This is a qualitative check - exact values depend on the random data
        self.assertGreater(result.weights.max(), result.weights.min())
    
    def test_few_points_handling(self):
        """Test handling of few points."""
        few_points = self.density_data.iloc[:3].copy()
        
        result = self.declusterer.decluster(
            data=few_points,
            x_col='x',
            y_col='y',
            value_col='value'
        )
        
        # Should handle gracefully with equal weights
        self.assertEqual(len(result.weights), 3)
        self.assertTrue(np.allclose(result.weights, 1.0))
    
    def test_different_parameters(self):
        """Test with different parameters."""
        # Test different number of neighbors
        declusterer_few = DistanceDeclusterer(n_neighbors=2)
        declusterer_many = DistanceDeclusterer(n_neighbors=10)
        
        result_few = declusterer_few.decluster(
            data=self.density_data,
            x_col='x',
            y_col='y',
            value_col='value'
        )
        
        result_many = declusterer_many.decluster(
            data=self.density_data,
            x_col='x',
            y_col='y',
            value_col='value'
        )
        
        # Both should work
        self.assertEqual(len(result_few.weights), len(self.density_data))
        self.assertEqual(len(result_many.weights), len(self.density_data))
        
        # Results may be different due to different neighbor counts
        self.assertEqual(result_few.declustering_info['n_neighbors'], 2)
        self.assertEqual(result_many.declustering_info['n_neighbors'], 10)


class TestIntegration(unittest.TestCase):
    """Integration tests for advanced functions."""
    
    def test_compositing_to_declustering_workflow(self):
        """Test workflow from compositing to declustering."""
        # Create synthetic drill hole data
        np.random.seed(42)
        
        drill_data = pd.DataFrame({
            'hole_id': ['DH001'] * 10,
            'from_depth': range(10),
            'to_depth': range(1, 11),
            'coal_thickness': np.random.uniform(0.5, 2.0, 10),
            'x': [25] * 10,  # Same X for all intervals in hole
            'y': [25] * 10,  # Same Y for all intervals in hole
        })
        
        # Step 1: Composite the data
        compositor = DataCompositor()
        composite_result = compositor.interval_based_compositing(
            data=drill_data,
            hole_id_col='hole_id',
            from_col='from_depth',
            to_col='to_depth',
            value_cols=['coal_thickness'],
            composite_length=2.0
        )
        
        # Add spatial coordinates to composited data
        composite_result.composited_data['x'] = 25
        composite_result.composited_data['y'] = 25
        
        # Step 2: Apply declustering to composited data
        declusterer = CellDeclusterer(cell_size=5.0)
        
        if len(composite_result.composited_data) > 0:
            decluster_result = declusterer.decluster(
                data=composite_result.composited_data,
                x_col='x',
                y_col='y',
                value_col='coal_thickness'
            )
            
            self.assertIsInstance(decluster_result, DeclusteringResult)
            self.assertIn('decluster_weight', decluster_result.declustered_data.columns)


if __name__ == '__main__':
    # Configure logging to reduce noise during tests
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)