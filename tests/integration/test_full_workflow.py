"""
Integration tests for full interpolation workflow.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from io.readers.csv_reader import CSVReader
from io.writers.csv_writer import CSVWriter, CSVExportOptions
from core.interpolation.idw import IDWInterpolator, IDWParameters
from core.interpolation.rbf import RBFInterpolator, RBFParameters, RBFKernel
from core.interpolation.kriging import KrigingInterpolator, KrigingParameters, KrigingType, VariogramModel
from core.grid.generator import GridGenerator, GridParameters
from core.recommendations.recommendation_engine import RecommendationEngine


class TestFullWorkflow:
    """Integration tests for complete interpolation workflow."""
    
    def setup_method(self):
        """Set up test data and temporary files."""
        # Create sample well data
        np.random.seed(42)
        n_wells = 50
        
        self.test_data = pd.DataFrame({
            'Well_ID': [f'WELL_{i+1:03d}' for i in range(n_wells)],
            'X': np.random.uniform(0, 1000, n_wells),
            'Y': np.random.uniform(0, 1000, n_wells),
            'Z': np.random.uniform(-100, -10, n_wells),
            'Ash_Content': np.random.uniform(10, 45, n_wells),
            'Sulfur_Content': np.random.uniform(0.5, 3.0, n_wells),
            'Calorific_Value': np.random.uniform(5500, 7500, n_wells)
        })
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.temp_dir, 'test_wells.csv')
        
        # Write test data to CSV
        self.test_data.to_csv(self.csv_file, index=False)
        
        # Grid parameters
        self.grid_params = GridParameters(
            bounds=(0, 1000, 0, 1000),
            cell_size=50.0,
            buffer_factor=0.1
        )
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_csv_read_interpolate_write_workflow(self):
        """Test complete workflow: CSV read → interpolate → write results."""
        # Step 1: Read data
        reader = CSVReader()
        data = reader.read(self.csv_file)
        
        assert len(data) == len(self.test_data)
        assert 'Ash_Content' in data.columns
        
        # Step 2: Generate grid
        grid_gen = GridGenerator()
        grid = grid_gen.generate_regular_grid(self.grid_params)
        
        assert grid.points.shape[1] == 2  # 2D coordinates
        assert len(grid.points) > 0
        
        # Step 3: IDW interpolation
        idw = IDWInterpolator()
        idw.fit(data, 'X', 'Y', 'Ash_Content')
        
        idw_results = idw.predict(grid.points)
        
        assert len(idw_results) == len(grid.points)
        assert not np.any(np.isnan(idw_results))
        
        # Step 4: Write results
        output_file = os.path.join(self.temp_dir, 'idw_results.csv')
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'X': grid.points[:, 0],
            'Y': grid.points[:, 1], 
            'Ash_Content_IDW': idw_results
        })
        
        writer = CSVWriter()
        # Mock the GridData for write_grid test
        from io.writers.base import GridData
        grid_data = GridData(
            coordinates=grid.points,
            values=idw_results,
            parameter_name='Ash_Content',
            bounds=self.grid_params.bounds,
            cell_size=self.grid_params.cell_size
        )
        
        writer.write_grid(grid_data, output_file)
        
        # Verify output file
        assert os.path.exists(output_file)
        
        # Read back and verify
        output_data = pd.read_csv(output_file)
        assert len(output_data) == len(grid.points)
        assert 'X' in output_data.columns
        assert 'Y' in output_data.columns
    
    def test_multiple_methods_comparison(self):
        """Test comparison of multiple interpolation methods."""
        # Read data
        reader = CSVReader()
        data = reader.read(self.csv_file)
        
        # Generate grid
        grid_gen = GridGenerator()
        grid = grid_gen.generate_regular_grid(self.grid_params)
        
        methods = {}
        results = {}
        
        # IDW interpolation
        idw_params = IDWParameters(power=2.0, smoothing=0.0)
        idw = IDWInterpolator(rbf_params=idw_params)
        idw.fit(data, 'X', 'Y', 'Ash_Content')
        results['IDW'] = idw.predict(grid.points)
        methods['IDW'] = idw
        
        # RBF interpolation
        rbf_params = RBFParameters(
            kernel=RBFKernel.MULTIQUADRIC,
            shape_parameter=0.1,
            regularization=1e-12
        )
        rbf = RBFInterpolator(rbf_params=rbf_params)
        rbf.fit(data, 'X', 'Y', 'Ash_Content')
        results['RBF'] = rbf.predict(grid.points)
        methods['RBF'] = rbf
        
        # Kriging interpolation
        kriging_params = KrigingParameters(
            kriging_type=KrigingType.ORDINARY,
            variogram_model=VariogramModel.SPHERICAL,
            auto_fit=True
        )
        kriging = KrigingInterpolator(kriging_params=kriging_params)
        kriging.fit(data, 'X', 'Y', 'Ash_Content')
        results['Kriging'] = kriging.predict(grid.points)
        methods['Kriging'] = kriging
        
        # Verify all methods produced results
        for method_name, result in results.items():
            assert len(result) == len(grid.points), f"{method_name} failed"
            assert not np.all(np.isnan(result)), f"{method_name} produced all NaN"
            
        # Check that results are different (methods should produce different outputs)
        assert not np.allclose(results['IDW'], results['RBF'], rtol=0.1)
        assert not np.allclose(results['IDW'], results['Kriging'], rtol=0.1)
        
        # Write comparison results
        comparison_file = os.path.join(self.temp_dir, 'method_comparison.csv')
        comparison_df = pd.DataFrame({
            'X': grid.points[:, 0],
            'Y': grid.points[:, 1],
            'IDW': results['IDW'],
            'RBF': results['RBF'],
            'Kriging': results['Kriging']
        })
        comparison_df.to_csv(comparison_file, index=False)
        
        assert os.path.exists(comparison_file)
    
    def test_recommendation_system_integration(self):
        """Test integration with recommendation system."""
        # Read data
        reader = CSVReader()
        data = reader.read(self.csv_file)
        
        # Get recommendations
        engine = RecommendationEngine()
        recommendations = engine.analyze_and_recommend(
            data, 'X', 'Y', 'Ash_Content'
        )
        
        assert 'recommended_method' in recommendations
        assert 'parameters' in recommendations
        assert 'data_analysis' in recommendations
        
        # Use recommended method
        recommended_method = recommendations['recommended_method']
        recommended_params = recommendations['parameters']
        
        # Create interpolator based on recommendation
        if recommended_method.lower() == 'idw':
            interpolator = IDWInterpolator()
        elif recommended_method.lower() == 'rbf':
            interpolator = RBFInterpolator()
        elif recommended_method.lower() == 'kriging':
            interpolator = KrigingInterpolator()
        else:
            # Default to IDW
            interpolator = IDWInterpolator()
        
        # Apply recommended parameters
        if hasattr(interpolator, 'set_parameters'):
            interpolator.set_parameters(**recommended_params)
        
        # Fit and predict
        interpolator.fit(data, 'X', 'Y', 'Ash_Content')
        
        grid_gen = GridGenerator()
        grid = grid_gen.generate_regular_grid(self.grid_params)
        
        results = interpolator.predict(grid.points)
        
        assert len(results) == len(grid.points)
        assert not np.all(np.isnan(results))
    
    def test_cross_validation_workflow(self):
        """Test cross-validation workflow."""
        # Read data
        reader = CSVReader()
        data = reader.read(self.csv_file)
        
        # Create interpolator
        idw = IDWInterpolator()
        idw.fit(data, 'X', 'Y', 'Ash_Content')
        
        # Perform cross-validation
        cv_results = idw.cross_validate(cv_folds=5)
        
        assert 'rmse' in cv_results
        assert 'mae' in cv_results
        assert 'r2' in cv_results
        
        # Verify reasonable values
        assert cv_results['rmse'] > 0
        assert cv_results['mae'] > 0
        assert -1 <= cv_results['r2'] <= 1
    
    def test_multi_parameter_workflow(self):
        """Test workflow with multiple parameters."""
        # Read data
        reader = CSVReader()
        data = reader.read(self.csv_file)
        
        parameters = ['Ash_Content', 'Sulfur_Content', 'Calorific_Value']
        grid_gen = GridGenerator()
        grid = grid_gen.generate_regular_grid(self.grid_params)
        
        all_results = {}
        
        # Interpolate each parameter
        for param in parameters:
            idw = IDWInterpolator()
            idw.fit(data, 'X', 'Y', param)
            results = idw.predict(grid.points)
            all_results[param] = results
        
        # Verify all parameters were processed
        assert len(all_results) == len(parameters)
        
        for param, results in all_results.items():
            assert len(results) == len(grid.points)
            assert not np.all(np.isnan(results))
        
        # Write combined results
        combined_file = os.path.join(self.temp_dir, 'multi_parameter_results.csv')
        combined_df = pd.DataFrame({
            'X': grid.points[:, 0],
            'Y': grid.points[:, 1]
        })
        
        for param, results in all_results.items():
            combined_df[f'{param}_interpolated'] = results
        
        combined_df.to_csv(combined_file, index=False)
        assert os.path.exists(combined_file)
    
    def test_error_handling_workflow(self):
        """Test error handling in workflow."""
        # Test with invalid file
        with pytest.raises(FileNotFoundError):
            reader = CSVReader()
            reader.read('non_existent_file.csv')
        
        # Test with insufficient data
        minimal_data = pd.DataFrame({
            'X': [0, 1],
            'Y': [0, 1],
            'Value': [10, 20]
        })
        
        idw = IDWInterpolator()
        idw.fit(minimal_data, 'X', 'Y', 'Value')
        
        # Should still work with minimal data
        test_points = np.array([[0.5, 0.5]])
        result = idw.predict(test_points)
        assert len(result) == 1
        assert not np.isnan(result[0])
    
    def test_large_dataset_workflow(self):
        """Test workflow with larger dataset."""
        # Create larger dataset
        np.random.seed(123)
        n_wells = 500
        
        large_data = pd.DataFrame({
            'X': np.random.uniform(0, 2000, n_wells),
            'Y': np.random.uniform(0, 2000, n_wells),
            'Value': np.random.uniform(0, 100, n_wells)
        })
        
        # Write to temporary file
        large_csv = os.path.join(self.temp_dir, 'large_wells.csv')
        large_data.to_csv(large_csv, index=False)
        
        # Read and process
        reader = CSVReader()
        data = reader.read(large_csv)
        
        assert len(data) == n_wells
        
        # Use smaller grid for performance
        small_grid_params = GridParameters(
            bounds=(0, 2000, 0, 2000),
            cell_size=200.0,
            buffer_factor=0.05
        )
        
        grid_gen = GridGenerator()
        grid = grid_gen.generate_regular_grid(small_grid_params)
        
        # Test IDW with large dataset
        idw = IDWInterpolator()
        idw.fit(data, 'X', 'Y', 'Value')
        
        results = idw.predict(grid.points)
        
        assert len(results) == len(grid.points)
        assert not np.all(np.isnan(results))
    
    def test_boundary_conditions_workflow(self):
        """Test workflow with boundary conditions and edge cases."""
        # Create data with extreme values
        boundary_data = pd.DataFrame({
            'X': [0, 0, 1000, 1000, 500],
            'Y': [0, 1000, 0, 1000, 500],
            'Value': [0, 100, 50, 200, 75]
        })
        
        boundary_csv = os.path.join(self.temp_dir, 'boundary_wells.csv')
        boundary_data.to_csv(boundary_csv, index=False)
        
        # Read and process
        reader = CSVReader()
        data = reader.read(boundary_csv)
        
        # Generate grid
        grid_gen = GridGenerator()
        grid = grid_gen.generate_regular_grid(self.grid_params)
        
        # Test interpolation
        idw = IDWInterpolator()
        idw.fit(data, 'X', 'Y', 'Value')
        
        results = idw.predict(grid.points)
        
        # Verify results are within reasonable bounds
        assert np.all(results >= -50)  # Allow some extrapolation
        assert np.all(results <= 250)  # Allow some extrapolation
        assert not np.any(np.isnan(results))
    
    def test_export_format_integration(self):
        """Test integration with different export formats."""
        # Read data and interpolate
        reader = CSVReader()
        data = reader.read(self.csv_file)
        
        grid_gen = GridGenerator()
        grid = grid_gen.generate_regular_grid(self.grid_params)
        
        idw = IDWInterpolator()
        idw.fit(data, 'X', 'Y', 'Ash_Content')
        results = idw.predict(grid.points)
        
        # Prepare grid data for export
        from io.writers.base import GridData
        grid_data = GridData(
            coordinates=grid.points,
            values=results,
            parameter_name='Ash_Content',
            bounds=self.grid_params.bounds,
            cell_size=self.grid_params.cell_size
        )
        
        # Test CSV export with different options
        csv_options = CSVExportOptions(
            delimiter=';',
            precision=2,
            include_metadata=True
        )
        
        csv_writer = CSVWriter(csv_options)
        csv_output = os.path.join(self.temp_dir, 'export_test.csv')
        csv_writer.write_grid(grid_data, csv_output)
        
        assert os.path.exists(csv_output)
        
        # Verify export format
        with open(csv_output, 'r') as f:
            first_line = f.readline()
            assert ';' in first_line  # Check delimiter
        
        exported_data = pd.read_csv(csv_output, delimiter=';')
        assert len(exported_data) == len(grid.points)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])