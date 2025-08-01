"""
Input/Output module for comprehensive data I/O operations.

This module provides:
- Standard readers and writers (CSV, Excel, etc.)
- Geological format support (LAS, Shapefile, KML)
- Database connectivity (PostgreSQL, SQLite, ODBC)
- Specialized exports (Surfer, NetCDF, Golden Software)
- Report generation (PDF, HTML, Word/Excel)
"""

from .readers import *
from .writers import *

# Geological formats
from .geological_formats import (
    LASReader, ShapefileReader, KMLReader,
    read_geological_file, create_geological_reader
)

# Database connectors
from .database_connectors import (
    PostgreSQLConnector, SQLiteConnector, ODBCConnector,
    create_database_connector, DatabaseConfig, get_database_info
)

# Specialized exports
from .specialized_exports import (
    SurferExporter, NetCDFExporter, GoldenSoftwareExporter,
    create_exporter, export_data
)

# Report generators
from .report_generators import (
    PDFReportGenerator, HTMLReportGenerator, OfficeReportGenerator,
    create_report_generator, generate_report
)

__all__ = [
    # Standard I/O (from existing modules)
    # Geological formats
    'LASReader', 'ShapefileReader', 'KMLReader',
    'read_geological_file', 'create_geological_reader',
    
    # Database connectors
    'PostgreSQLConnector', 'SQLiteConnector', 'ODBCConnector',
    'create_database_connector', 'DatabaseConfig', 'get_database_info',
    
    # Specialized exports
    'SurferExporter', 'NetCDFExporter', 'GoldenSoftwareExporter',
    'create_exporter', 'export_data',
    
    # Report generators
    'PDFReportGenerator', 'HTMLReportGenerator', 'OfficeReportGenerator',
    'create_report_generator', 'generate_report'
]