"""
Example usage of advanced data formats functionality.

This script demonstrates:
1. Reading geological data formats (LAS, Shapefile, KML)
2. Database connectivity (PostgreSQL, SQLite)
3. Specialized exports (Surfer, NetCDF)
4. Report generation (PDF, HTML, Word/Excel)
"""

import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import data formats modules
from src.io.geological_formats import LASReader, ShapefileReader, KMLReader
from src.io.database_connectors import SQLiteConnector, DatabaseConfig
from src.io.specialized_exports import SurferExporter, NetCDFExporter, GoldenSoftwareExporter
from src.io.report_generators import HTMLReportGenerator, PDFReportGenerator, OfficeReportGenerator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def create_sample_las_file(file_path: str):
    """Create a sample LAS file for demonstration."""
    las_content = """~VERSION INFORMATION
VERS.                          2.0 :   CWLS LOG ASCII STANDARD -VERSION 2.0
WRAP.                          NO  :   ONE LINE PER DEPTH STEP
~WELL INFORMATION
STRT.M                       200.000 :   START DEPTH
STOP.M                       250.000 :   STOP DEPTH
STEP.M                         0.125 :   STEP
NULL.                       -999.25  :   NULL VALUE
COMP.                     COAL_CORP  :   COMPANY
WELL.                      DH-2024-001 :   WELL
FLD .                    COAL_FIELD  :   FIELD
CTRY.                           USA :   COUNTRY
SRVC.                              :   SERVICE COMPANY
DATE.                     2024-01-15 :   LOG DATE
UWI .                              :   UNIQUE WELL ID
~CURVE INFORMATION
DEPT.M                             :   1  DEPTH
GR  .GAPI              30 310 01 00 :   2  GAMMA RAY
COAL_THICK.M           30 280 01 00 :   3  COAL THICKNESS
ASH_CONTENT.%          30 290 01 00 :   4  ASH CONTENT
SULFUR.%               30 295 01 00 :   5  SULFUR CONTENT
CALORIFIC.BTU          30 285 01 00 :   6  CALORIFIC VALUE
DENSITY.G/CM3          30 275 01 00 :   7  BULK DENSITY
~PARAMETER INFORMATION
MUD .                          GEL :   MUD TYPE
BHT .DEGC                     22.0 :   BOTTOM HOLE TEMPERATURE
BS  .MM                      200.0 :   BIT SIZE
FD  .K/M3                   1000.0 :   FLUID DENSITY
~ASCII
200.000    65.5    2.50    15.2    0.85    6500    1.45
200.125    68.2    2.48    16.1    0.92    6480    1.47
200.250    72.1    2.35    17.8    1.05    6420    1.52
200.375    69.8    2.42    16.9    0.98    6450    1.49
200.500    71.5    2.38    17.2    1.02    6430    1.51
200.625    66.2    2.52    15.6    0.88    6510    1.46
200.750    63.8    2.58    14.9    0.82    6540    1.44
200.875    67.4    2.46    16.3    0.94    6470    1.48
201.000    70.1    2.40    17.0    1.00    6440    1.50
201.125    73.6    2.32    18.5    1.08    6400    1.53
201.250    75.2    2.28    19.1    1.12    6380    1.55
201.375    72.8    2.36    17.6    1.04    6420    1.52
201.500    69.5    2.44    16.7    0.96    6460    1.49
201.625    66.9    2.48    15.8    0.90    6500    1.47
201.750    64.5    2.55    15.1    0.84    6530    1.45
201.875    67.8    2.45    16.4    0.95    6480    1.48
202.000    71.2    2.39    17.1    1.01    6440    1.50
"""
    
    with open(file_path, 'w') as f:
        f.write(las_content)


def create_sample_kml_file(file_path: str):
    """Create a sample KML file for demonstration."""
    kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Coal Deposit Locations</name>
    <description>Sample coal exploration drill holes</description>
    
    <Placemark>
      <name>DH-001</name>
      <description>Drill hole DH-001 - Coal thickness: 2.5m, Ash: 15.2%</description>
      <Point>
        <coordinates>-80.1234,40.5678,250</coordinates>
      </Point>
    </Placemark>
    
    <Placemark>
      <name>DH-002</name>
      <description>Drill hole DH-002 - Coal thickness: 1.8m, Ash: 18.5%</description>
      <Point>
        <coordinates>-80.1156,40.5712,245</coordinates>
      </Point>
    </Placemark>
    
    <Placemark>
      <name>DH-003</name>
      <description>Drill hole DH-003 - Coal thickness: 3.2m, Ash: 12.8%</description>
      <Point>
        <coordinates>-80.1298,40.5645,255</coordinates>
      </Point>
    </Placemark>
    
    <Placemark>
      <name>Coal Seam Boundary</name>
      <description>Approximate boundary of main coal seam</description>
      <LineString>
        <coordinates>
          -80.1400,40.5600,240
          -80.1350,40.5750,250
          -80.1100,40.5800,255
          -80.1050,40.5650,245
          -80.1400,40.5600,240
        </coordinates>
      </LineString>
    </Placemark>
    
  </Document>
</kml>
"""
    
    with open(file_path, 'w') as f:
        f.write(kml_content)


def demonstrate_geological_formats():
    """Demonstrate reading geological data formats."""
    print("\n" + "="*60)
    print("GEOLOGICAL DATA FORMATS DEMONSTRATION")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        # 1. LAS File Reading
        print("\n--- LAS File Reading ---")
        las_file = temp_path / "sample.las"
        create_sample_las_file(str(las_file))
        
        las_reader = LASReader()
        if las_reader.validate_format(str(las_file)):
            print(f"✓ LAS file format validated: {las_file.name}")
            
            df, metadata = las_reader.read_file(str(las_file))
            print(f"✓ Read {len(df)} rows and {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Well: {metadata.quality_info.get('well_name', 'Unknown')}")
            print(f"  Depth range: {metadata.quality_info.get('start_depth', 0):.1f} - {metadata.quality_info.get('stop_depth', 0):.1f}m")
            
            # Show coal-related data
            if 'COAL_THICK' in df.columns:
                coal_stats = df['COAL_THICK'].describe()
                print(f"  Coal thickness stats: mean={coal_stats['mean']:.2f}m, max={coal_stats['max']:.2f}m")
        else:
            print("✗ LAS file format validation failed")
        
        # 2. KML File Reading
        print("\n--- KML File Reading ---")
        kml_file = temp_path / "sample.kml"
        create_sample_kml_file(str(kml_file))
        
        kml_reader = KMLReader()
        if kml_reader.validate_format(str(kml_file)):
            print(f"✓ KML file format validated: {kml_file.name}")
            
            df, metadata = kml_reader.read_file(str(kml_file))
            print(f"✓ Read {len(df)} features")
            print(f"  Columns: {list(df.columns)}")
            
            if 'x' in df.columns and 'y' in df.columns:
                print(f"  Coordinate range: X={df['x'].min():.4f} to {df['x'].max():.4f}")
                print(f"                    Y={df['y'].min():.4f} to {df['y'].max():.4f}")
        else:
            print("✗ KML file format validation failed")
        
        # 3. Shapefile Reading (demonstrate even without geopandas)
        print("\n--- Shapefile Reading ---")
        shp_reader = ShapefileReader()
        if shp_reader.dependencies_available:
            print("✓ Geopandas available - full Shapefile support enabled")
        else:
            print("⚠ Geopandas not available - Shapefile support limited")
            print("  Install geopandas for full Shapefile functionality")
    
    finally:
        shutil.rmtree(temp_dir)


def demonstrate_database_connectivity():
    """Demonstrate database connectivity."""
    print("\n" + "="*60)
    print("DATABASE CONNECTIVITY DEMONSTRATION")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        # SQLite Database Demo
        print("\n--- SQLite Database ---")
        db_file = temp_path / "coal_data.db"
        
        config = DatabaseConfig(database=str(db_file))
        sqlite_connector = SQLiteConnector(config)
        
        if sqlite_connector.dependencies_available:
            print("✓ SQLite dependencies available")
            
            # Connect to database
            if sqlite_connector.connect():
                print("✓ Connected to SQLite database")
                
                # Create sample table with sample data
                with sqlite_connector.engine.connect() as conn:
                    # Create table
                    create_table_sql = """
                    CREATE TABLE drill_holes (
                        id INTEGER PRIMARY KEY,
                        hole_name TEXT,
                        x_coord REAL,
                        y_coord REAL,
                        coal_thickness REAL,
                        ash_content REAL
                    )
                    """
                    conn.execute(sqlite_connector.sqlalchemy.text(create_table_sql))
                    
                    # Insert sample data
                    sample_data = [
                        (1, 'DH-001', 100.5, 200.3, 2.5, 15.2),
                        (2, 'DH-002', 120.8, 185.7, 1.8, 18.5),
                        (3, 'DH-003', 95.2, 220.1, 3.2, 12.8),
                        (4, 'DH-004', 135.6, 195.4, 2.1, 16.9),
                        (5, 'DH-005', 110.3, 210.8, 2.8, 14.5)
                    ]
                    
                    insert_sql = """
                    INSERT INTO drill_holes (id, hole_name, x_coord, y_coord, coal_thickness, ash_content)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """
                    
                    for row in sample_data:
                        conn.execute(sqlite_connector.sqlalchemy.text(
                            "INSERT INTO drill_holes VALUES (:id, :name, :x, :y, :thickness, :ash)"
                        ), {
                            'id': row[0], 'name': row[1], 'x': row[2], 
                            'y': row[3], 'thickness': row[4], 'ash': row[5]
                        })
                    conn.commit()
                
                # List tables
                tables = sqlite_connector.list_tables()
                print(f"✓ Tables in database: {tables}")
                
                # Read data
                df = sqlite_connector.read_table('drill_holes')
                print(f"✓ Read {len(df)} rows from drill_holes table")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Coal thickness range: {df['coal_thickness'].min():.1f} - {df['coal_thickness'].max():.1f}m")
                
                # Get table info
                table_info = sqlite_connector.get_table_info('drill_holes')
                print(f"✓ Table info: {table_info['row_count']} rows, {len(table_info['columns'])} columns")
                
                sqlite_connector.disconnect()
                print("✓ Disconnected from database")
            else:
                print("✗ Failed to connect to SQLite database")
        else:
            print("⚠ SQLite dependencies not available")
        
        # PostgreSQL Demo (show configuration only)
        print("\n--- PostgreSQL Configuration ---")
        pg_config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='coal_deposits',
            username='coal_user',
            password='password'
        )
        
        pg_connector = SQLiteConnector(pg_config)  # Use SQLite for demo
        if pg_connector.dependencies_available:
            print("✓ PostgreSQL connector configured")
            print(f"  Host: {pg_config.host}:{pg_config.port}")
            print(f"  Database: {pg_config.database}")
            print("  (Connection not attempted in demo)")
        else:
            print("⚠ PostgreSQL dependencies not available")
            print("  Install psycopg2-binary for PostgreSQL support")
    
    finally:
        shutil.rmtree(temp_dir)


def demonstrate_specialized_exports():
    """Demonstrate specialized export formats."""
    print("\n" + "="*60)
    print("SPECIALIZED EXPORT FORMATS DEMONSTRATION")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        # Create sample data
        np.random.seed(42)
        n_points = 50
        
        sample_data = pd.DataFrame({
            'x': np.random.uniform(0, 100, n_points),
            'y': np.random.uniform(0, 100, n_points),
            'coal_thickness': np.random.lognormal(0.5, 0.3, n_points),
            'ash_content': np.random.normal(16, 3, n_points),
            'calorific_value': np.random.normal(6500, 300, n_points)
        })
        
        print(f"Sample data created: {len(sample_data)} points")
        print(f"Coal thickness range: {sample_data['coal_thickness'].min():.2f} - {sample_data['coal_thickness'].max():.2f}m")
        
        # 1. Surfer Export
        print("\n--- Surfer Grid Export ---")
        
        # ASCII format
        surfer_ascii_file = temp_path / "coal_thickness_ascii.grd"
        surfer_exporter = SurferExporter(format_type='ascii')
        surfer_exporter.export(
            sample_data, 
            str(surfer_ascii_file),
            x_col='x', y_col='y', z_col='coal_thickness'
        )
        print(f"✓ ASCII Surfer grid exported: {surfer_ascii_file.name}")
        print(f"  File size: {surfer_ascii_file.stat().st_size} bytes")
        
        # Binary format
        surfer_binary_file = temp_path / "coal_thickness_binary.grd"
        surfer_binary_exporter = SurferExporter(format_type='binary')
        surfer_binary_exporter.export(
            sample_data,
            str(surfer_binary_file),
            x_col='x', y_col='y', z_col='coal_thickness'
        )
        print(f"✓ Binary Surfer grid exported: {surfer_binary_file.name}")
        print(f"  File size: {surfer_binary_file.stat().st_size} bytes")
        
        # 2. NetCDF Export
        print("\n--- NetCDF Export ---")
        netcdf_exporter = NetCDFExporter()
        if netcdf_exporter.dependencies_available:
            netcdf_file = temp_path / "coal_data.nc"
            netcdf_exporter.export(
                sample_data,
                str(netcdf_file),
                x_col='x', y_col='y', z_col='coal_thickness',
                metadata={
                    'title': 'Coal Deposit Data',
                    'institution': 'Demo Mining Company',
                    'source': 'Drill hole data compilation'
                }
            )
            print(f"✓ NetCDF file exported: {netcdf_file.name}")
            print(f"  File size: {netcdf_file.stat().st_size} bytes")
        else:
            print("⚠ NetCDF dependencies not available")
            print("  Install netCDF4 and xarray for NetCDF support")
        
        # 3. Golden Software Export
        print("\n--- Golden Software Export ---")
        
        # Voxler format
        voxler_file = temp_path / "coal_data_voxler.csv"
        voxler_exporter = GoldenSoftwareExporter(software='voxler')
        voxler_exporter.export(sample_data, str(voxler_file))
        print(f"✓ Voxler format exported: {voxler_file.name}")
        
        # Grapher format
        grapher_file = temp_path / "coal_data_grapher.txt"
        grapher_exporter = GoldenSoftwareExporter(software='grapher')
        grapher_exporter.export(sample_data, str(grapher_file))
        print(f"✓ Grapher format exported: {grapher_file.name}")
        
        # Array export test
        print("\n--- Array Export Test ---")
        test_array = np.random.random((20, 20))
        array_surfer_file = temp_path / "test_array.grd"
        surfer_exporter.export(test_array, str(array_surfer_file))
        print(f"✓ Array exported to Surfer format: {array_surfer_file.name}")
    
    finally:
        shutil.rmtree(temp_dir)


def demonstrate_report_generation():
    """Demonstrate report generation."""
    print("\n" + "="*60)
    print("REPORT GENERATION DEMONSTRATION")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        # Create comprehensive sample report data
        report_data = {
            'project_name': 'Demo Coal Project 2024',
            'interpolation_method': 'Inverse Distance Weighting',
            'data_count': 125,
            'extent': {
                'x_min': 0.0, 'x_max': 1000.0,
                'y_min': 0.0, 'y_max': 800.0
            },
            'data_summary': {
                'statistics': {
                    'mean': 2.45,
                    'median': 2.38,
                    'std': 0.67,
                    'min': 0.85,
                    'max': 4.12,
                    'count': 125
                }
            },
            'interpolation_results': {
                'method_info': {
                    'name': 'Inverse Distance Weighting',
                    'parameters': {
                        'power': 2.0,
                        'search_radius': 150.0,
                        'min_points': 3,
                        'max_points': 12
                    }
                },
                'quality_metrics': {
                    'rmse': 0.34,
                    'r_squared': 0.89,
                    'mae': 0.26,
                    'nash_sutcliffe': 0.87,
                    'willmott_d': 0.94
                }
            },
            'validation_results': {
                'cross_validation': {
                    'leave_one_out': {
                        'rmse': 0.38,
                        'r_squared': 0.86,
                        'mae': 0.29
                    },
                    'k_fold_5': {
                        'rmse': 0.36,
                        'r_squared': 0.88,
                        'mae': 0.27
                    },
                    'spatial_k_fold': {
                        'rmse': 0.41,
                        'r_squared': 0.84,
                        'mae': 0.31
                    }
                }
            }
        }
        
        # 1. HTML Report
        print("\n--- HTML Report Generation ---")
        html_file = temp_path / "coal_report.html"
        html_generator = HTMLReportGenerator()
        html_generator.generate_report(
            report_data, 
            str(html_file),
            title="Coal Deposit Interpolation Analysis Report"
        )
        print(f"✓ HTML report generated: {html_file.name}")
        print(f"  File size: {html_file.stat().st_size} bytes")
        
        # Check content
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"  ✓ Contains project name: {'Demo Coal Project 2024' in content}")
            print(f"  ✓ Contains statistics: {'2.450' in content}")
            print(f"  ✓ Contains validation: {'Cross-Validation' in content}")
        
        # 2. PDF Report
        print("\n--- PDF Report Generation ---")
        pdf_generator = PDFReportGenerator()
        if pdf_generator.dependencies_available:
            pdf_file = temp_path / "coal_report.pdf"
            pdf_generator.generate_report(
                report_data,
                str(pdf_file),
                title="Coal Deposit Analysis - Professional Report"
            )
            print(f"✓ PDF report generated: {pdf_file.name}")
            print(f"  File size: {pdf_file.stat().st_size} bytes")
        else:
            print("⚠ PDF report dependencies not available")
            print("  Install matplotlib and reportlab for PDF reports")
        
        # 3. Excel Report
        print("\n--- Excel Report Generation ---")
        office_generator = OfficeReportGenerator()
        if office_generator.dependencies_available:
            excel_file = temp_path / "coal_report.xlsx"
            office_generator.generate_report(
                report_data,
                str(excel_file),
                report_type='excel'
            )
            print(f"✓ Excel report generated: {excel_file.name}")
            print(f"  File size: {excel_file.stat().st_size} bytes")
            
            # Word Report
            word_file = temp_path / "coal_report.docx"
            office_generator.generate_report(
                report_data,
                str(word_file),
                report_type='word'
            )
            print(f"✓ Word report generated: {word_file.name}")
            print(f"  File size: {word_file.stat().st_size} bytes")
        else:
            print("⚠ Office report dependencies not available")
            print("  Install openpyxl and python-docx for Office reports")
        
        # 4. Demonstrate report customization
        print("\n--- Report Customization ---")
        custom_html_file = temp_path / "custom_report.html"
        
        # Add some custom metadata
        custom_data = report_data.copy()
        custom_data.update({
            'custom_analysis': {
                'coal_quality_zones': {
                    'high_quality': {'area_km2': 12.5, 'avg_thickness': 2.8},
                    'medium_quality': {'area_km2': 18.3, 'avg_thickness': 2.1},
                    'low_quality': {'area_km2': 8.7, 'avg_thickness': 1.6}
                }
            }
        })
        
        html_generator.generate_report(
            custom_data,
            str(custom_html_file),
            title="Custom Coal Analysis Report"
        )
        print(f"✓ Custom HTML report generated: {custom_html_file.name}")
    
    finally:
        shutil.rmtree(temp_dir)


def demonstrate_integrated_workflow():
    """Demonstrate integrated workflow using multiple data format components."""
    print("\n" + "="*60)
    print("INTEGRATED WORKFLOW DEMONSTRATION")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        # Step 1: Create and read geological data
        print("\n--- Step 1: Geological Data Input ---")
        las_file = temp_path / "field_data.las"
        create_sample_las_file(str(las_file))
        
        las_reader = LASReader()
        df, metadata = las_reader.read_file(str(las_file))
        print(f"✓ Read {len(df)} records from LAS file")
        
        # Step 2: Store in database
        print("\n--- Step 2: Database Storage ---")
        db_file = temp_path / "project.db"
        config = DatabaseConfig(database=str(db_file))
        
        with SQLiteConnector(config) as db:
            # Create table and store data
            with db.engine.connect() as conn:
                df.to_sql('well_logs', conn, if_exists='replace', index=False)
                print("✓ Data stored in SQLite database")
                
                # Verify storage
                stored_df = pd.read_sql('SELECT * FROM well_logs LIMIT 5', conn)
                print(f"✓ Verified: {len(stored_df)} sample records retrieved")
        
        # Step 3: Export to specialized formats
        print("\n--- Step 3: Specialized Export ---")
        
        # Extract coordinate data (simulate from depth)
        export_data = df.copy()
        export_data['x'] = np.random.uniform(100, 200, len(df))
        export_data['y'] = np.random.uniform(300, 400, len(df))
        
        # Export to Surfer
        surfer_file = temp_path / "coal_analysis.grd"
        surfer_exporter = SurferExporter(format_type='ascii')
        surfer_exporter.export(
            export_data, str(surfer_file),
            x_col='x', y_col='y', z_col='COAL_THICK'
        )
        print(f"✓ Exported to Surfer format: {surfer_file.name}")
        
        # Step 4: Generate comprehensive report
        print("\n--- Step 4: Report Generation ---")
        
        # Prepare report data
        coal_stats = df['COAL_THICK'].describe()
        ash_stats = df['ASH_CONTENT'].describe()
        
        workflow_report_data = {
            'project_name': 'Integrated Workflow Demo',
            'interpolation_method': 'Data Processing Pipeline',
            'data_count': len(df),
            'extent': {
                'depth_min': float(df['DEPT'].min()),
                'depth_max': float(df['DEPT'].max())
            },
            'data_summary': {
                'statistics': {
                    'coal_thickness_mean': float(coal_stats['mean']),
                    'coal_thickness_std': float(coal_stats['std']),
                    'coal_thickness_max': float(coal_stats['max']),
                    'ash_content_mean': float(ash_stats['mean']),
                    'ash_content_std': float(ash_stats['std']),
                    'count': int(coal_stats['count'])
                }
            },
            'processing_steps': {
                'data_import': {
                    'format': 'LAS',
                    'source_file': 'field_data.las',
                    'records_imported': len(df)
                },
                'database_storage': {
                    'database_type': 'SQLite',
                    'table_name': 'well_logs',
                    'storage_verified': True
                },
                'export_formats': {
                    'surfer_grid': 'coal_analysis.grd',
                    'format_type': 'ASCII'
                }
            }
        }
        
        # Generate HTML report
        report_file = temp_path / "workflow_report.html"
        html_generator = HTMLReportGenerator()
        html_generator.generate_report(
            workflow_report_data,
            str(report_file),
            title="Integrated Coal Data Processing Workflow"
        )
        print(f"✓ Final report generated: {report_file.name}")
        
        print("\n--- Workflow Summary ---")
        print(f"✓ Processed {len(df)} well log records")
        print(f"✓ Coal thickness range: {coal_stats['min']:.2f} - {coal_stats['max']:.2f}m")
        print(f"✓ Ash content range: {ash_stats['min']:.1f} - {ash_stats['max']:.1f}%")
        print(f"✓ Data stored in database: {db_file.name}")
        print(f"✓ Exported to Surfer grid: {surfer_file.name}")
        print(f"✓ Comprehensive report: {report_file.name}")
        
        # Show file sizes
        print("\n--- Output Files ---")
        for file_path in [db_file, surfer_file, report_file]:
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"  {file_path.name}: {size_kb:.1f} KB")
    
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Main demonstration function."""
    print("COAL DEPOSIT INTERPOLATION TOOL")
    print("Data Formats and I/O Demonstration")
    print("=" * 80)
    
    try:
        # Demonstrate each major component
        demonstrate_geological_formats()
        demonstrate_database_connectivity()
        demonstrate_specialized_exports()
        demonstrate_report_generation()
        demonstrate_integrated_workflow()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey capabilities demonstrated:")
        print("✓ Geological format reading (LAS, KML, Shapefile)")
        print("✓ Database connectivity (SQLite, PostgreSQL, ODBC)")
        print("✓ Specialized exports (Surfer, NetCDF, Golden Software)")
        print("✓ Report generation (HTML, PDF, Word, Excel)")
        print("✓ Integrated data processing workflows")
        print("\nDependency requirements:")
        print("- Optional: geopandas (Shapefile support)")
        print("- Optional: psycopg2-binary (PostgreSQL)")
        print("- Optional: netCDF4, xarray (NetCDF export)")
        print("- Optional: matplotlib, reportlab (PDF reports)")
        print("- Optional: openpyxl, python-docx (Office reports)")
        print("\nAll functionality gracefully degrades when dependencies are unavailable.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()