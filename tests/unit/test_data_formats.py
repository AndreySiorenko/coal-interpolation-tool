"""
Unit tests for data formats (geological formats, database connectors, exports, reports).
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import warnings

# Import modules to test
from src.io.geological_formats import (
    LASReader, ShapefileReader, KMLReader, 
    create_geological_reader, read_geological_file
)
from src.io.database_connectors import (
    PostgreSQLConnector, SQLiteConnector, ODBCConnector,
    DatabaseConfig, create_database_connector
)
from src.io.specialized_exports import (
    SurferExporter, NetCDFExporter, GoldenSoftwareExporter,
    create_exporter, export_data
)
from src.io.report_generators import (
    PDFReportGenerator, HTMLReportGenerator, OfficeReportGenerator,
    create_report_generator, generate_report
)


class TestGeologicalFormats(unittest.TestCase):
    """Test cases for geological format readers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_las_reader_initialization(self):
        """Test LAS reader initialization."""
        reader = LASReader()
        self.assertIsInstance(reader, LASReader)
        self.assertEqual(reader.supported_versions, ['2.0', '3.0'])
    
    def test_las_reader_validate_format(self):
        """Test LAS format validation."""
        reader = LASReader()
        
        # Create mock LAS file
        las_content = """~VERSION INFORMATION
VERS.                          2.0 :   CWLS LOG ASCII STANDARD -VERSION 2.0
WRAP.                          NO  :   ONE LINE PER DEPTH STEP
~WELL INFORMATION
STRT.M                        0.5000 :   START DEPTH
STOP.M                       99.5000 :   STOP DEPTH
STEP.M                        0.1250 :   STEP
~CURVE INFORMATION
DEPT.M                      :   1  DEPTH
GR  .GAPI                   :   2  GAMMA RAY
~ASCII
0.5000    45.45
0.6250    45.78
"""
        
        las_file = self.temp_dir_path / "test.las"
        with open(las_file, 'w') as f:
            f.write(las_content)
        
        self.assertTrue(reader.validate_format(str(las_file)))
        
        # Test invalid file
        invalid_file = self.temp_dir_path / "invalid.txt"
        with open(invalid_file, 'w') as f:
            f.write("This is not a LAS file")
        
        self.assertFalse(reader.validate_format(str(invalid_file)))
    
    def test_las_reader_parse_header_line(self):
        """Test LAS header line parsing."""
        reader = LASReader()
        header_info = {}
        
        # Test various header formats
        reader._parse_header_line("VERS.                          2.0 :   VERSION", header_info)
        self.assertEqual(header_info['VERS'], 2.0)
        
        reader._parse_header_line("COMP.                      COMPANY :   COMPANY NAME", header_info)
        self.assertEqual(header_info['COMP'], 'COMPANY')
        
        reader._parse_header_line("STRT.M                        0.5 :   START DEPTH", header_info)
        self.assertEqual(header_info['STRT'], 0.5)
    
    def test_las_reader_parse_curve_line(self):
        """Test LAS curve line parsing."""
        reader = LASReader()
        
        curve_dict = reader._parse_curve_line("DEPT.M                      :   1  DEPTH")
        expected = {
            'mnemonic': 'DEPT',
            'unit': 'M',
            'api_code': ':',
            'description': '1  DEPTH'
        }
        self.assertEqual(curve_dict, expected)
    
    def test_las_reader_identify_coal_curves(self):
        """Test coal curve identification."""
        reader = LASReader()
        
        columns = ['DEPT', 'GR', 'COAL_THICKNESS', 'ASH_CONTENT', 'SULFUR', 'DENSITY']
        coal_curves = reader._identify_coal_curves(columns)
        
        self.assertIn('COAL_THICKNESS', coal_curves)
        self.assertIn('ASH_CONTENT', coal_curves)
        self.assertIn('SULFUR', coal_curves)
        self.assertIn('DENSITY', coal_curves)
        self.assertNotIn('DEPT', coal_curves)
        self.assertNotIn('GR', coal_curves)
    
    def test_shapefile_reader_initialization(self):
        """Test Shapefile reader initialization."""
        reader = ShapefileReader()
        self.assertIsInstance(reader, ShapefileReader)
        # Dependencies check should work even if geopandas is not available
        self.assertIsInstance(reader.dependencies_available, bool)
    
    def test_shapefile_reader_validate_format(self):
        """Test Shapefile format validation."""
        reader = ShapefileReader()
        
        # Create mock shapefile structure
        shp_file = self.temp_dir_path / "test.shp"
        shx_file = self.temp_dir_path / "test.shx"
        dbf_file = self.temp_dir_path / "test.dbf"
        
        # Create empty files
        for file_path in [shp_file, shx_file, dbf_file]:
            file_path.touch()
        
        self.assertTrue(reader.validate_format(str(shp_file)))
        
        # Test missing companion files
        dbf_file.unlink()
        self.assertFalse(reader.validate_format(str(shp_file)))
    
    def test_kml_reader_initialization(self):
        """Test KML reader initialization."""
        reader = KMLReader()
        self.assertIsInstance(reader, KMLReader)
    
    def test_kml_reader_validate_format(self):
        """Test KML format validation."""
        reader = KMLReader()
        
        # Create mock KML file
        kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Test Point</name>
      <Point>
        <coordinates>-122.0822035425683,37.42228990140251,0</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>"""
        
        kml_file = self.temp_dir_path / "test.kml"
        with open(kml_file, 'w') as f:
            f.write(kml_content)
        
        self.assertTrue(reader.validate_format(str(kml_file)))
        
        # Test invalid file
        invalid_file = self.temp_dir_path / "invalid.txt"
        with open(invalid_file, 'w') as f:
            f.write("This is not a KML file")
        
        self.assertFalse(reader.validate_format(str(invalid_file)))
    
    def test_create_geological_reader(self):
        """Test geological reader factory function."""
        # Test LAS reader creation
        las_reader = create_geological_reader("test.las")
        self.assertIsInstance(las_reader, LASReader)
        
        # Test Shapefile reader creation
        shp_reader = create_geological_reader("test.shp")
        self.assertIsInstance(shp_reader, ShapefileReader)
        
        # Test KML reader creation
        kml_reader = create_geological_reader("test.kml")
        self.assertIsInstance(kml_reader, KMLReader)
        
        # Test unsupported format
        with self.assertRaises(ValueError):
            create_geological_reader("test.xyz")


class TestDatabaseConnectors(unittest.TestCase):
    """Test cases for database connectors."""
    
    def test_database_config(self):
        """Test DatabaseConfig dataclass."""
        config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='testdb',
            username='user',
            password='pass'
        )
        
        self.assertEqual(config.host, 'localhost')
        self.assertEqual(config.port, 5432)
        self.assertEqual(config.database, 'testdb')
    
    def test_postgresql_connector_initialization(self):
        """Test PostgreSQL connector initialization."""
        config = DatabaseConfig(database='testdb')
        connector = PostgreSQLConnector(config)
        
        self.assertEqual(connector.config.database, 'testdb')
        self.assertIsInstance(connector.dependencies_available, bool)
    
    def test_sqlite_connector_initialization(self):
        """Test SQLite connector initialization."""
        config = DatabaseConfig(database='test.db')
        connector = SQLiteConnector(config)
        
        self.assertEqual(connector.config.database, 'test.db')
        self.assertIsInstance(connector.dependencies_available, bool)
    
    def test_odbc_connector_initialization(self):
        """Test ODBC connector initialization."""
        config = DatabaseConfig(
            host='server',
            database='testdb',
            username='user',
            password='pass'
        )
        connector = ODBCConnector(config)
        
        self.assertEqual(connector.config.database, 'testdb')
        self.assertIsInstance(connector.dependencies_available, bool)
    
    def test_create_database_connector(self):
        """Test database connector factory function."""
        config = DatabaseConfig(database='testdb')
        
        # Test PostgreSQL connector creation
        pg_connector = create_database_connector('postgresql', config)
        self.assertIsInstance(pg_connector, PostgreSQLConnector)
        
        # Test SQLite connector creation
        sqlite_connector = create_database_connector('sqlite', config)
        self.assertIsInstance(sqlite_connector, SQLiteConnector)
        
        # Test ODBC connector creation
        odbc_connector = create_database_connector('odbc', config)
        self.assertIsInstance(odbc_connector, ODBCConnector)
        
        # Test unsupported database type
        with self.assertRaises(ValueError):
            create_database_connector('unsupported', config)


class TestSpecializedExports(unittest.TestCase):
    """Test cases for specialized export formats."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_path = Path(self.temp_dir)
        
        # Create sample data
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            'x': np.random.uniform(0, 100, 20),
            'y': np.random.uniform(0, 100, 20),
            'z': np.random.uniform(10, 30, 20)
        })
        
        self.sample_array = np.random.random((10, 10))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_surfer_exporter_initialization(self):
        """Test Surfer exporter initialization."""
        # Test ASCII format
        ascii_exporter = SurferExporter(format_type='ascii')
        self.assertEqual(ascii_exporter.format_type, 'ascii')
        
        # Test binary format
        binary_exporter = SurferExporter(format_type='binary')
        self.assertEqual(binary_exporter.format_type, 'binary')
        
        # Test invalid format
        with self.assertRaises(ValueError):
            SurferExporter(format_type='invalid')
    
    def test_surfer_exporter_validate_data(self):
        """Test Surfer data validation."""
        exporter = SurferExporter()
        
        # Test valid DataFrame
        self.assertTrue(exporter.validate_data(self.sample_df))
        
        # Test valid array
        self.assertTrue(exporter.validate_data(self.sample_array))
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        self.assertFalse(exporter.validate_data(empty_df))
        
        # Test empty array
        empty_array = np.array([])
        self.assertFalse(exporter.validate_data(empty_array))
    
    def test_surfer_exporter_prepare_grid_from_array(self):
        """Test grid preparation from array."""
        exporter = SurferExporter()
        grid_data = exporter._prepare_grid_from_array(self.sample_array)
        
        self.assertEqual(grid_data['nx'], self.sample_array.shape[1])
        self.assertEqual(grid_data['ny'], self.sample_array.shape[0])
        self.assertEqual(grid_data['x_min'], 0.0)
        self.assertEqual(grid_data['y_min'], 0.0)
        np.testing.assert_array_equal(grid_data['z'], self.sample_array)
    
    def test_surfer_export_ascii(self):
        """Test ASCII Surfer export."""
        exporter = SurferExporter(format_type='ascii')
        output_file = self.temp_dir_path / "test.grd"
        
        exporter.export(self.sample_array, str(output_file))
        
        # Check file was created
        self.assertTrue(output_file.exists())
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn('DSAA', content)  # ASCII identifier
    
    def test_surfer_export_binary(self):
        """Test binary Surfer export."""
        exporter = SurferExporter(format_type='binary')
        output_file = self.temp_dir_path / "test.grd"
        
        exporter.export(self.sample_array, str(output_file))
        
        # Check file was created
        self.assertTrue(output_file.exists())
        
        # Check file starts with binary identifier
        with open(output_file, 'rb') as f:
            header = f.read(4)
            self.assertEqual(header, b'DSBB')  # Binary identifier
    
    def test_netcdf_exporter_initialization(self):
        """Test NetCDF exporter initialization."""
        exporter = NetCDFExporter()
        self.assertIsInstance(exporter, NetCDFExporter)
        self.assertIsInstance(exporter.dependencies_available, bool)
    
    def test_netcdf_exporter_validate_data(self):
        """Test NetCDF data validation."""
        exporter = NetCDFExporter()
        
        # Test valid DataFrame
        self.assertTrue(exporter.validate_data(self.sample_df))
        
        # Test valid array
        self.assertTrue(exporter.validate_data(self.sample_array))
        
        # Test empty data
        self.assertFalse(exporter.validate_data(pd.DataFrame()))
        self.assertFalse(exporter.validate_data(np.array([])))
    
    def test_golden_software_exporter_initialization(self):
        """Test Golden Software exporter initialization."""
        # Test Voxler
        voxler_exporter = GoldenSoftwareExporter(software='voxler')
        self.assertEqual(voxler_exporter.software, 'voxler')
        
        # Test Grapher
        grapher_exporter = GoldenSoftwareExporter(software='grapher')
        self.assertEqual(grapher_exporter.software, 'grapher')
        
        # Test invalid software
        with self.assertRaises(ValueError):
            GoldenSoftwareExporter(software='invalid')
    
    def test_create_exporter(self):
        """Test exporter factory function."""
        # Test Surfer exporter creation
        surfer_exporter = create_exporter('surfer')
        self.assertIsInstance(surfer_exporter, SurferExporter)
        
        # Test NetCDF exporter creation
        netcdf_exporter = create_exporter('netcdf')
        self.assertIsInstance(netcdf_exporter, NetCDFExporter)
        
        # Test Golden Software exporter creation
        golden_exporter = create_exporter('golden_software')
        self.assertIsInstance(golden_exporter, GoldenSoftwareExporter)
        
        # Test unsupported format
        with self.assertRaises(ValueError):
            create_exporter('unsupported')


class TestReportGenerators(unittest.TestCase):
    """Test cases for report generators."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_path = Path(self.temp_dir)
        
        # Create sample report data
        self.sample_data = {
            'project_name': 'Test Coal Project',
            'interpolation_method': 'IDW',
            'data_count': 100,
            'extent': {
                'x_min': 0.0, 'x_max': 100.0,
                'y_min': 0.0, 'y_max': 100.0
            },
            'data_summary': {
                'statistics': {
                    'mean': 15.5,
                    'median': 14.8,
                    'std': 3.2,
                    'min': 8.1,
                    'max': 25.7,
                    'count': 100
                }
            },
            'interpolation_results': {
                'method_info': {
                    'name': 'Inverse Distance Weighting',
                    'parameters': {'power': 2.0, 'radius': 50.0}
                },
                'quality_metrics': {
                    'rmse': 2.34,
                    'r_squared': 0.87,
                    'mae': 1.89
                }
            },
            'validation_results': {
                'cross_validation': {
                    'leave_one_out': {
                        'rmse': 2.45,
                        'r_squared': 0.85,
                        'mae': 1.95
                    }
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_pdf_report_generator_initialization(self):
        """Test PDF report generator initialization."""
        generator = PDFReportGenerator()
        self.assertIsInstance(generator, PDFReportGenerator)
        self.assertIsInstance(generator.dependencies_available, bool)
    
    def test_html_report_generator_initialization(self):
        """Test HTML report generator initialization."""
        generator = HTMLReportGenerator()
        self.assertIsInstance(generator, HTMLReportGenerator)
    
    def test_html_report_generation(self):
        """Test HTML report generation."""
        generator = HTMLReportGenerator()
        output_file = self.temp_dir_path / "test_report.html"
        
        generator.generate_report(self.sample_data, str(output_file))
        
        # Check file was created
        self.assertTrue(output_file.exists())
        
        # Check file content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('<!DOCTYPE html>', content)
            self.assertIn('Test Coal Project', content)
            self.assertIn('Cross-Validation', content)
    
    def test_html_report_css_styles(self):
        """Test HTML report CSS styles."""
        generator = HTMLReportGenerator()
        css = generator._get_css_styles()
        
        self.assertIn('body', css)
        self.assertIn('container', css)
        self.assertIn('table', css)
    
    def test_html_report_sections(self):
        """Test HTML report section building."""
        generator = HTMLReportGenerator()
        
        # Test metadata section
        metadata_html = generator._build_metadata_section(self.sample_data)
        self.assertIn('Report Information', metadata_html)
        self.assertIn('Test Coal Project', metadata_html)
        
        # Test data summary section
        summary_html = generator._build_data_summary_section(self.sample_data['data_summary'])
        self.assertIn('Data Summary', summary_html)
        self.assertIn('15.500', summary_html)  # Mean value
        
        # Test interpolation section
        interp_html = generator._build_interpolation_section(self.sample_data['interpolation_results'])
        self.assertIn('Interpolation Results', interp_html)
        self.assertIn('Inverse Distance Weighting', interp_html)
    
    def test_office_report_generator_initialization(self):
        """Test Office report generator initialization."""
        generator = OfficeReportGenerator()
        self.assertIsInstance(generator, OfficeReportGenerator)
        self.assertIsInstance(generator.dependencies_available, bool)
    
    def test_base_report_generator_format_methods(self):
        """Test base report generator formatting methods."""
        generator = HTMLReportGenerator()  # Use concrete implementation
        
        # Test number formatting
        self.assertEqual(generator._format_number(15.12345, 2), "15.12")
        self.assertEqual(generator._format_number(np.nan), "N/A")
        
        # Test percentage formatting
        self.assertEqual(generator._format_percentage(85.7, 1), "85.7%")
        self.assertEqual(generator._format_percentage(np.nan), "N/A")
    
    def test_create_report_generator(self):
        """Test report generator factory function."""
        # Test PDF generator creation
        pdf_generator = create_report_generator('pdf')
        self.assertIsInstance(pdf_generator, PDFReportGenerator)
        
        # Test HTML generator creation
        html_generator = create_report_generator('html')
        self.assertIsInstance(html_generator, HTMLReportGenerator)
        
        # Test Excel generator creation
        excel_generator = create_report_generator('excel')
        self.assertIsInstance(excel_generator, OfficeReportGenerator)
        
        # Test Word generator creation
        word_generator = create_report_generator('word')
        self.assertIsInstance(word_generator, OfficeReportGenerator)
        
        # Test unsupported format
        with self.assertRaises(ValueError):
            create_report_generator('unsupported')
    
    def test_generate_report_convenience_function(self):
        """Test convenience report generation function."""
        output_file = self.temp_dir_path / "test_report.html"
        
        # This should work without raising exceptions
        generate_report(self.sample_data, str(output_file), 'html')
        
        # Check file was created
        self.assertTrue(output_file.exists())


class TestIntegration(unittest.TestCase):
    """Integration tests for data formats functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_data_flow_export_to_report(self):
        """Test data flow from export to report generation."""
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.uniform(0, 100, 20),
            'y': np.random.uniform(0, 100, 20),
            'coal_thickness': np.random.uniform(1, 5, 20)
        })
        
        # Export to Surfer format
        surfer_file = self.temp_dir_path / "test.grd"
        surfer_exporter = SurferExporter(format_type='ascii')
        surfer_exporter.export(data, str(surfer_file), z_col='coal_thickness')
        
        # Check export worked
        self.assertTrue(surfer_file.exists())
        
        # Generate report about the data
        report_data = {
            'project_name': 'Integration Test',
            'data_summary': {
                'statistics': {
                    'mean': data['coal_thickness'].mean(),
                    'std': data['coal_thickness'].std(),
                    'count': len(data)
                }
            }
        }
        
        report_file = self.temp_dir_path / "report.html"
        html_generator = HTMLReportGenerator()
        html_generator.generate_report(report_data, str(report_file))
        
        # Check report was generated
        self.assertTrue(report_file.exists())
        
        # Check report contains data
        with open(report_file, 'r') as f:
            content = f.read()
            self.assertIn('Integration Test', content)
    
    def test_format_chain_workflow(self):
        """Test workflow using multiple format converters."""
        # This would test a realistic workflow of:
        # 1. Reading geological data
        # 2. Exporting to specialized format
        # 3. Generating comprehensive report
        
        # For now, just test that all components can be instantiated together
        las_reader = LASReader()
        surfer_exporter = SurferExporter()
        html_generator = HTMLReportGenerator()
        
        self.assertIsInstance(las_reader, LASReader)
        self.assertIsInstance(surfer_exporter, SurferExporter)
        self.assertIsInstance(html_generator, HTMLReportGenerator)


if __name__ == '__main__':
    # Configure logging to reduce noise during tests
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    # Suppress pandas warnings
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Run tests
    unittest.main(verbosity=2)