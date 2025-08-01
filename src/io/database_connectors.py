"""
Database connectivity module for geological data.

Provides connections to various database systems:
- PostgreSQL/PostGIS for spatial data
- SQLite/SpatiaLite for local databases
- ODBC connections for enterprise systems
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    host: Optional[str] = None
    port: Optional[int] = None
    database: str = ""
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None
    additional_params: Dict[str, Any] = None


@dataclass
class DatabaseInfo:
    """Information about database and tables."""
    database_type: str
    database_name: str
    tables: List[str]
    spatial_tables: List[str]
    connection_info: Dict[str, Any]


class BaseDatabaseConnector(ABC):
    """Base class for database connectors."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish database connection."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close database connection."""
        pass
    
    @abstractmethod
    def list_tables(self) -> List[str]:
        """List available tables."""
        pass
    
    @abstractmethod
    def read_table(self, table_name: str, 
                   columns: Optional[List[str]] = None,
                   where_clause: Optional[str] = None,
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Read data from table."""
        pass
    
    @abstractmethod
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about table structure."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class PostgreSQLConnector(BaseDatabaseConnector):
    """
    PostgreSQL/PostGIS database connector.
    
    Provides access to PostgreSQL databases with optional PostGIS spatial extension.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import psycopg2
            import sqlalchemy
            self.psycopg2 = psycopg2
            self.sqlalchemy = sqlalchemy
            self.dependencies_available = True
        except ImportError:
            self.psycopg2 = None
            self.sqlalchemy = None
            self.dependencies_available = False
            self.logger.warning("psycopg2 or sqlalchemy not available - PostgreSQL support limited")
    
    def connect(self) -> bool:
        """Establish PostgreSQL connection."""
        if not self.dependencies_available:
            raise ImportError("psycopg2 and sqlalchemy are required for PostgreSQL support")
        
        try:
            if self.config.connection_string:
                connection_string = self.config.connection_string
            else:
                # Build connection string
                host = self.config.host or 'localhost'
                port = self.config.port or 5432
                database = self.config.database
                username = self.config.username
                password = self.config.password
                
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
            # Create SQLAlchemy engine
            self.engine = self.sqlalchemy.create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(self.sqlalchemy.text("SELECT 1"))
            
            self.logger.info(f"Connected to PostgreSQL database: {self.config.database}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close PostgreSQL connection."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            self.logger.info("Disconnected from PostgreSQL")
    
    def list_tables(self) -> List[str]:
        """List available tables in PostgreSQL database."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(self.sqlalchemy.text(query))
            tables = [row[0] for row in result]
        
        return tables
    
    def list_spatial_tables(self) -> List[str]:
        """List tables with spatial columns (PostGIS)."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        query = """
            SELECT DISTINCT f_table_name
            FROM geometry_columns
            WHERE f_table_schema = 'public'
            ORDER BY f_table_name
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(self.sqlalchemy.text(query))
                spatial_tables = [row[0] for row in result]
            return spatial_tables
        except Exception:
            # PostGIS not available or no spatial tables
            return []
    
    def read_table(self, table_name: str,
                   columns: Optional[List[str]] = None,
                   where_clause: Optional[str] = None,
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Read data from PostgreSQL table."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        # Build query
        if columns:
            columns_str = ', '.join(columns)
        else:
            columns_str = '*'
        
        query = f"SELECT {columns_str} FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        self.logger.info(f"Executing query: {query}")
        
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            self.logger.error(f"Error reading table {table_name}: {e}")
            raise
    
    def read_spatial_data(self, table_name: str,
                         geometry_column: str = 'geom',
                         srid: Optional[int] = None,
                         **kwargs) -> pd.DataFrame:
        """Read spatial data with coordinate extraction."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        # Build spatial query
        columns = kwargs.get('columns', [])
        if columns:
            # Add geometry coordinates
            columns_list = columns.copy()
            columns_list.extend([
                f"ST_X({geometry_column}) as x",
                f"ST_Y({geometry_column}) as y",
                f"ST_Z({geometry_column}) as z"
            ])
            columns_str = ', '.join(columns_list)
        else:
            columns_str = f"*, ST_X({geometry_column}) as x, ST_Y({geometry_column}) as y, ST_Z({geometry_column}) as z"
        
        query = f"SELECT {columns_str} FROM {table_name}"
        
        if kwargs.get('where_clause'):
            query += f" WHERE {kwargs['where_clause']}"
        
        if kwargs.get('limit'):
            query += f" LIMIT {kwargs['limit']}"
        
        self.logger.info(f"Executing spatial query: {query}")
        
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            self.logger.error(f"Error reading spatial data from {table_name}: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get PostgreSQL table information."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        # Get column information
        query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(self.sqlalchemy.text(query), {'table_name': table_name})
            columns = [
                {
                    'name': row[0],
                    'type': row[1],
                    'nullable': row[2] == 'YES',
                    'default': row[3]
                }
                for row in result
            ]
        
        # Get row count
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        with self.engine.connect() as conn:
            result = conn.execute(self.sqlalchemy.text(count_query))
            row_count = result.scalar()
        
        # Check for spatial columns
        spatial_columns = []
        if table_name in self.list_spatial_tables():
            spatial_query = """
                SELECT f_geometry_column, coord_dimension, srid, type
                FROM geometry_columns
                WHERE f_table_name = %s AND f_table_schema = 'public'
            """
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(self.sqlalchemy.text(spatial_query), {'f_table_name': table_name})
                    spatial_columns = [
                        {
                            'column': row[0],
                            'dimensions': row[1],
                            'srid': row[2],
                            'geometry_type': row[3]
                        }
                        for row in result
                    ]
            except Exception:
                pass
        
        return {
            'table_name': table_name,
            'columns': columns,
            'row_count': row_count,
            'spatial_columns': spatial_columns,
            'is_spatial': len(spatial_columns) > 0
        }


class SQLiteConnector(BaseDatabaseConnector):
    """
    SQLite/SpatiaLite database connector.
    
    Provides access to SQLite databases with optional SpatiaLite spatial extension.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import sqlite3
            import sqlalchemy
            self.sqlite3 = sqlite3
            self.sqlalchemy = sqlalchemy
            self.dependencies_available = True
        except ImportError:
            self.sqlite3 = None
            self.sqlalchemy = None
            self.dependencies_available = False
            self.logger.warning("sqlite3 or sqlalchemy not available")
    
    def connect(self) -> bool:
        """Establish SQLite connection."""
        if not self.dependencies_available:
            raise ImportError("sqlite3 and sqlalchemy are required for SQLite support")
        
        try:
            database_path = self.config.database
            
            # Create database file if it doesn't exist
            if not Path(database_path).exists():
                self.logger.info(f"Creating new SQLite database: {database_path}")
            
            # Create SQLAlchemy engine
            connection_string = f"sqlite:///{database_path}"
            self.engine = self.sqlalchemy.create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(self.sqlalchemy.text("SELECT 1"))
            
            # Try to enable SpatiaLite
            self._enable_spatialite()
            
            self.logger.info(f"Connected to SQLite database: {database_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to SQLite: {e}")
            return False
    
    def _enable_spatialite(self):
        """Try to enable SpatiaLite extension."""
        try:
            with self.engine.connect() as conn:
                conn.execute(self.sqlalchemy.text("SELECT load_extension('mod_spatialite')"))
                self.spatialite_enabled = True
                self.logger.info("SpatiaLite extension enabled")
        except Exception:
            self.spatialite_enabled = False
            self.logger.info("SpatiaLite extension not available")
    
    def disconnect(self):
        """Close SQLite connection."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            self.logger.info("Disconnected from SQLite")
    
    def list_tables(self) -> List[str]:
        """List available tables in SQLite database."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(self.sqlalchemy.text(query))
            tables = [row[0] for row in result]
        
        return tables
    
    def read_table(self, table_name: str,
                   columns: Optional[List[str]] = None,
                   where_clause: Optional[str] = None,
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Read data from SQLite table."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        # Build query
        if columns:
            columns_str = ', '.join(columns)
        else:
            columns_str = '*'
        
        query = f"SELECT {columns_str} FROM [{table_name}]"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        self.logger.info(f"Executing query: {query}")
        
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            self.logger.error(f"Error reading table {table_name}: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get SQLite table information."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        # Get column information
        query = f"PRAGMA table_info([{table_name}])"
        
        with self.engine.connect() as conn:
            result = conn.execute(self.sqlalchemy.text(query))
            columns = [
                {
                    'name': row[1],
                    'type': row[2],
                    'nullable': not bool(row[3]),
                    'default': row[4],
                    'primary_key': bool(row[5])
                }
                for row in result
            ]
        
        # Get row count
        count_query = f"SELECT COUNT(*) FROM [{table_name}]"
        with self.engine.connect() as conn:
            result = conn.execute(self.sqlalchemy.text(count_query))
            row_count = result.scalar()
        
        return {
            'table_name': table_name,
            'columns': columns,
            'row_count': row_count,
            'spatialite_enabled': getattr(self, 'spatialite_enabled', False)
        }
    
    def create_spatial_table(self, table_name: str, columns: Dict[str, str],
                           geometry_column: str = 'geom',
                           geometry_type: str = 'POINT',
                           srid: int = 4326):
        """Create a spatial table with SpatiaLite."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        if not getattr(self, 'spatialite_enabled', False):
            raise RuntimeError("SpatiaLite extension not available")
        
        # Create table
        columns_sql = ', '.join([f"{name} {dtype}" for name, dtype in columns.items()])
        create_query = f"CREATE TABLE [{table_name}] ({columns_sql})"
        
        with self.engine.connect() as conn:
            conn.execute(self.sqlalchemy.text(create_query))
            
            # Add geometry column
            add_geom_query = f"""
                SELECT AddGeometryColumn('{table_name}', '{geometry_column}', 
                                       {srid}, '{geometry_type}', 'XY')
            """
            conn.execute(self.sqlalchemy.text(add_geom_query))
            conn.commit()
        
        self.logger.info(f"Created spatial table: {table_name}")


class ODBCConnector(BaseDatabaseConnector):
    """
    ODBC database connector for enterprise systems.
    
    Provides access to various databases through ODBC drivers.
    """
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import pyodbc
            import sqlalchemy
            self.pyodbc = pyodbc
            self.sqlalchemy = sqlalchemy
            self.dependencies_available = True
        except ImportError:
            self.pyodbc = None
            self.sqlalchemy = None
            self.dependencies_available = False
            self.logger.warning("pyodbc or sqlalchemy not available - ODBC support limited")
    
    def connect(self) -> bool:
        """Establish ODBC connection."""
        if not self.dependencies_available:
            raise ImportError("pyodbc and sqlalchemy are required for ODBC support")
        
        try:
            if self.config.connection_string:
                connection_string = f"mssql+pyodbc:///?odbc_connect={self.config.connection_string}"
            else:
                # Build basic connection string
                driver = self.config.additional_params.get('driver', 'SQL Server') if self.config.additional_params else 'SQL Server'
                server = self.config.host
                database = self.config.database
                
                odbc_string = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database}"
                
                if self.config.username:
                    odbc_string += f";UID={self.config.username};PWD={self.config.password}"
                else:
                    odbc_string += ";Trusted_Connection=yes"
                
                connection_string = f"mssql+pyodbc:///?odbc_connect={odbc_string}"
            
            # Create SQLAlchemy engine
            self.engine = self.sqlalchemy.create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(self.sqlalchemy.text("SELECT 1"))
            
            self.logger.info(f"Connected to ODBC database: {self.config.database}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect via ODBC: {e}")
            return False
    
    def disconnect(self):
        """Close ODBC connection."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            self.logger.info("Disconnected from ODBC database")
    
    def list_tables(self) -> List[str]:
        """List available tables via ODBC."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        query = """
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(self.sqlalchemy.text(query))
            tables = [row[0] for row in result]
        
        return tables
    
    def read_table(self, table_name: str,
                   columns: Optional[List[str]] = None,
                   where_clause: Optional[str] = None,
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Read data from table via ODBC."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        # Build query
        if columns:
            columns_str = ', '.join(columns)
        else:
            columns_str = '*'
        
        query = f"SELECT"
        
        if limit:
            query += f" TOP {limit}"
        
        query += f" {columns_str} FROM [{table_name}]"
        
        if where_clause:
            query += f " WHERE {where_clause}"
        
        self.logger.info(f"Executing ODBC query: {query}")
        
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            self.logger.error(f"Error reading table {table_name}: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table information via ODBC."""
        if not hasattr(self, 'engine'):
            raise RuntimeError("Not connected to database")
        
        # Get column information
        query = """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(self.sqlalchemy.text(query), {'table_name': table_name})
            columns = [
                {
                    'name': row[0],
                    'type': row[1],
                    'nullable': row[2] == 'YES',
                    'default': row[3]
                }
                for row in result
            ]
        
        # Get row count
        count_query = f"SELECT COUNT(*) FROM [{table_name}]"
        with self.engine.connect() as conn:
            result = conn.execute(self.sqlalchemy.text(count_query))
            row_count = result.scalar()
        
        return {
            'table_name': table_name,
            'columns': columns,
            'row_count': row_count
        }


# Factory function for creating database connectors
def create_database_connector(db_type: str, config: DatabaseConfig) -> BaseDatabaseConnector:
    """
    Create appropriate database connector based on type.
    
    Args:
        db_type: Database type ('postgresql', 'sqlite', 'odbc')
        config: Database configuration
        
    Returns:
        Appropriate connector instance
    """
    db_type = db_type.lower()
    
    if db_type in ['postgresql', 'postgres', 'postgis']:
        return PostgreSQLConnector(config)
    elif db_type in ['sqlite', 'spatialite']:
        return SQLiteConnector(config)
    elif db_type in ['odbc', 'sqlserver', 'mssql']:
        return ODBCConnector(config)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def get_database_info(connector: BaseDatabaseConnector) -> DatabaseInfo:
    """
    Get comprehensive database information.
    
    Args:
        connector: Connected database connector
        
    Returns:
        DatabaseInfo with database metadata
    """
    tables = connector.list_tables()
    
    # Get spatial tables if supported
    spatial_tables = []
    if hasattr(connector, 'list_spatial_tables'):
        try:
            spatial_tables = connector.list_spatial_tables()
        except Exception:
            pass
    
    return DatabaseInfo(
        database_type=connector.__class__.__name__,
        database_name=connector.config.database,
        tables=tables,
        spatial_tables=spatial_tables,
        connection_info={
            'host': connector.config.host,
            'port': connector.config.port,
            'database': connector.config.database
        }
    )