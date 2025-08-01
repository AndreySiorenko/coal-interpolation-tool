"""
Geological data format readers for specialized formats.

Provides support for:
- LAS files (Log ASCII Standard for well logs)
- Shapefile format (ESRI)
- KML/KMZ files (Google Earth)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
import logging
from pathlib import Path
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class GeologicalDataInfo:
    """Container for geological data metadata."""
    format_type: str
    coordinate_system: Optional[str]
    data_source: str
    columns_info: Dict[str, Any]
    quality_info: Dict[str, Any]
    processing_notes: List[str]


class BaseGeologicalReader(ABC):
    """Base class for geological format readers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def read_file(self, file_path: str) -> Tuple[pd.DataFrame, GeologicalDataInfo]:
        """Read geological data file and return DataFrame with metadata."""
        pass
    
    @abstractmethod
    def validate_format(self, file_path: str) -> bool:
        """Validate if file is in the expected format."""
        pass


class LASReader(BaseGeologicalReader):
    """
    Reader for LAS (Log ASCII Standard) files.
    
    LAS files are commonly used in the oil and gas industry for well log data.
    This reader supports LAS 2.0 and basic LAS 3.0 formats.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_versions = ['2.0', '3.0']
    
    def read_file(self, file_path: str) -> Tuple[pd.DataFrame, GeologicalDataInfo]:
        """
        Read LAS file and extract well log data.
        
        Args:
            file_path: Path to LAS file
            
        Returns:
            Tuple of (DataFrame with log data, GeologicalDataInfo metadata)
        """
        self.logger.info(f"Reading LAS file: {file_path}")
        
        if not self.validate_format(file_path):
            raise ValueError(f"Invalid LAS file format: {file_path}")
        
        # Parse LAS file sections
        header_info, curve_info, log_data = self._parse_las_file(file_path)
        
        # Create DataFrame
        column_names = [curve['mnemonic'] for curve in curve_info]
        
        # Handle null values
        null_value = float(header_info.get('NULL', -999.25))
        log_data[log_data == null_value] = np.nan
        
        df = pd.DataFrame(log_data, columns=column_names)
        
        # Create metadata
        metadata = GeologicalDataInfo(
            format_type='LAS',
            coordinate_system=header_info.get('coordinate_system'),
            data_source=file_path,
            columns_info={
                curve['mnemonic']: {
                    'description': curve.get('description', ''),
                    'unit': curve.get('unit', ''),
                    'api_code': curve.get('api_code', '')
                } for curve in curve_info
            },
            quality_info={
                'version': header_info.get('VERS', 'Unknown'),
                'well_name': header_info.get('WELL', 'Unknown'),
                'company': header_info.get('COMP', 'Unknown'),
                'field': header_info.get('FLD', 'Unknown'),
                'start_depth': header_info.get('STRT'),
                'stop_depth': header_info.get('STOP'),
                'step': header_info.get('STEP'),
                'null_value': null_value
            },
            processing_notes=[]
        )
        
        # Add coordinate information if available
        if 'DEPT' in df.columns or 'DEPTH' in df.columns:
            depth_col = 'DEPT' if 'DEPT' in df.columns else 'DEPTH'
            metadata.processing_notes.append(f"Depth column: {depth_col}")
        
        # Check for common coal-related curves
        coal_curves = self._identify_coal_curves(column_names)
        if coal_curves:
            metadata.processing_notes.append(f"Coal-related curves found: {coal_curves}")
        
        self.logger.info(f"Successfully read LAS file with {len(df)} rows and {len(df.columns)} columns")
        
        return df, metadata
    
    def validate_format(self, file_path: str) -> bool:
        """Validate LAS file format."""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                # LAS files should start with ~VERSION or similar
                if not first_line.startswith('~'):
                    return False
                
                # Look for version information
                for line in f:
                    if line.startswith('VERS'):
                        version = line.split()[1]
                        return any(v in version for v in self.supported_versions)
                
                return True  # Basic validation passed
                
        except Exception as e:
            self.logger.error(f"Error validating LAS file {file_path}: {e}")
            return False
    
    def _parse_las_file(self, file_path: str) -> Tuple[Dict, List[Dict], np.ndarray]:
        """Parse LAS file into header, curve info, and data sections."""
        header_info = {}
        curve_info = []
        log_data = []
        
        current_section = None
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Section headers
                if line.startswith('~'):
                    current_section = line[1:].strip()
                    continue
                
                # Parse different sections
                if current_section in ['VERSION', 'V']:
                    self._parse_header_line(line, header_info)
                elif current_section in ['WELL', 'W']:
                    self._parse_header_line(line, header_info)
                elif current_section in ['CURVES', 'C']:
                    curve_dict = self._parse_curve_line(line)
                    if curve_dict:
                        curve_info.append(curve_dict)
                elif current_section in ['PARAMETER', 'P']:
                    self._parse_header_line(line, header_info)
                elif current_section in ['ASCII', 'A']:
                    # Parse data lines
                    data_line = self._parse_data_line(line)
                    if data_line:
                        log_data.append(data_line)
        
        return header_info, curve_info, np.array(log_data)
    
    def _parse_header_line(self, line: str, header_info: Dict):
        """Parse header information line."""
        parts = re.split(r'\s+', line, maxsplit=3)
        if len(parts) >= 2:
            mnemonic = parts[0]
            value = parts[1]
            
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            
            header_info[mnemonic] = value
    
    def _parse_curve_line(self, line: str) -> Optional[Dict]:
        """Parse curve information line."""
        parts = re.split(r'\s+', line, maxsplit=3)
        if len(parts) >= 2:
            curve_dict = {
                'mnemonic': parts[0],
                'unit': parts[1] if len(parts) > 1 else '',
                'api_code': parts[2] if len(parts) > 2 else '',
                'description': parts[3] if len(parts) > 3 else ''
            }
            return curve_dict
        return None
    
    def _parse_data_line(self, line: str) -> Optional[List[float]]:
        """Parse data line into numeric values."""
        try:
            values = [float(x) for x in line.split()]
            return values
        except ValueError:
            return None
    
    def _identify_coal_curves(self, column_names: List[str]) -> List[str]:
        """Identify potential coal-related curves."""
        coal_indicators = [
            'coal', 'ash', 'sulfur', 'moisture', 'btu', 'calorific',
            'carbon', 'volatile', 'fixed', 'gc', 'density', 'resistivity'
        ]
        
        coal_curves = []
        for col in column_names:
            col_lower = col.lower()
            for indicator in coal_indicators:
                if indicator in col_lower:
                    coal_curves.append(col)
                    break
        
        return coal_curves


class ShapefileReader(BaseGeologicalReader):
    """
    Reader for ESRI Shapefile format.
    
    Supports point, line, and polygon shapefiles with attribute data.
    Requires optional dependencies: geopandas, shapely.
    """
    
    def __init__(self):
        super().__init__()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import geopandas as gpd
            import shapely
            self.gpd = gpd
            self.dependencies_available = True
        except ImportError:
            self.gpd = None
            self.dependencies_available = False
            self.logger.warning("Geopandas not available - Shapefile support limited")
    
    def read_file(self, file_path: str) -> Tuple[pd.DataFrame, GeologicalDataInfo]:
        """
        Read Shapefile and extract spatial data.
        
        Args:
            file_path: Path to shapefile (.shp)
            
        Returns:
            Tuple of (DataFrame with spatial data, GeologicalDataInfo metadata)
        """
        self.logger.info(f"Reading Shapefile: {file_path}")
        
        if not self.validate_format(file_path):
            raise ValueError(f"Invalid Shapefile format: {file_path}")
        
        if not self.dependencies_available:
            raise ImportError("Geopandas is required for Shapefile support")
        
        # Read shapefile
        gdf = self.gpd.read_file(file_path)
        
        # Extract coordinates based on geometry type
        df = self._extract_coordinates(gdf)
        
        # Create metadata
        metadata = GeologicalDataInfo(
            format_type='Shapefile',
            coordinate_system=str(gdf.crs) if gdf.crs else None,
            data_source=file_path,
            columns_info={
                col: {'description': f'Attribute column: {col}', 'type': str(dtype)}
                for col, dtype in df.dtypes.items()
                if col not in ['geometry', 'x', 'y', 'z']
            },
            quality_info={
                'geometry_type': str(gdf.geometry.geom_type.iloc[0]) if len(gdf) > 0 else 'Unknown',
                'feature_count': len(gdf),
                'bounds': gdf.total_bounds.tolist() if len(gdf) > 0 else None,
                'crs': str(gdf.crs) if gdf.crs else None
            },
            processing_notes=[]
        )
        
        # Add processing notes
        if 'x' in df.columns and 'y' in df.columns:
            metadata.processing_notes.append("Coordinates extracted to x, y columns")
        
        self.logger.info(f"Successfully read Shapefile with {len(df)} features")
        
        return df, metadata
    
    def validate_format(self, file_path: str) -> bool:
        """Validate Shapefile format."""
        path = Path(file_path)
        
        # Check if .shp file exists
        if not path.suffix.lower() == '.shp':
            return False
        
        if not path.exists():
            return False
        
        # Check for required companion files
        required_files = ['.shx', '.dbf']
        for ext in required_files:
            companion_file = path.with_suffix(ext)
            if not companion_file.exists():
                self.logger.warning(f"Missing companion file: {companion_file}")
                return False
        
        return True
    
    def _extract_coordinates(self, gdf) -> pd.DataFrame:
        """Extract coordinates from geometry column."""
        df = gdf.copy()
        
        # Convert to DataFrame (remove geometry dependencies)
        df = pd.DataFrame(df.drop(columns=['geometry']))
        
        # Extract coordinates based on geometry type
        geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else None
        
        if geom_type == 'Point':
            df['x'] = gdf.geometry.x
            df['y'] = gdf.geometry.y
            
            # Check for Z coordinate
            if gdf.geometry.has_z.any():
                df['z'] = [geom.z if hasattr(geom, 'z') else np.nan for geom in gdf.geometry]
        
        elif geom_type in ['LineString', 'Polygon']:
            # For lines and polygons, use centroid
            centroids = gdf.geometry.centroid
            df['x'] = centroids.x
            df['y'] = centroids.y
            
            # Add additional geometric properties
            if geom_type == 'LineString':
                df['length'] = gdf.geometry.length
            elif geom_type == 'Polygon':
                df['area'] = gdf.geometry.area
                df['perimeter'] = gdf.geometry.length
        
        return df


class KMLReader(BaseGeologicalReader):
    """
    Reader for KML/KMZ (Google Earth) format.
    
    Supports basic KML parsing for points, lines, and polygons.
    """
    
    def __init__(self):
        super().__init__()
    
    def read_file(self, file_path: str) -> Tuple[pd.DataFrame, GeologicalDataInfo]:
        """
        Read KML file and extract spatial data.
        
        Args:
            file_path: Path to KML file
            
        Returns:
            Tuple of (DataFrame with spatial data, GeologicalDataInfo metadata)
        """
        self.logger.info(f"Reading KML file: {file_path}")
        
        if not self.validate_format(file_path):
            raise ValueError(f"Invalid KML file format: {file_path}")
        
        # Parse KML file
        features = self._parse_kml(file_path)
        
        if not features:
            raise ValueError("No features found in KML file")
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Create metadata
        metadata = GeologicalDataInfo(
            format_type='KML',
            coordinate_system='WGS84',  # KML uses WGS84
            data_source=file_path,
            columns_info={
                col: {'description': f'KML attribute: {col}', 'type': str(dtype)}
                for col, dtype in df.dtypes.items()
            },
            quality_info={
                'feature_count': len(df),
                'coordinate_system': 'WGS84'
            },
            processing_notes=['Coordinates assumed to be in WGS84 (EPSG:4326)']
        )
        
        self.logger.info(f"Successfully read KML file with {len(df)} features")
        
        return df, metadata
    
    def validate_format(self, file_path: str) -> bool:
        """Validate KML file format."""
        path = Path(file_path)
        
        if path.suffix.lower() not in ['.kml', '.kmz']:
            return False
        
        if not path.exists():
            return False
        
        # Basic content validation
        try:
            if path.suffix.lower() == '.kmz':
                # KMZ is zipped KML
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as kmz:
                    kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
                    return len(kml_files) > 0
            else:
                # Check for KML content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1000 chars
                    return '<kml' in content.lower()
        
        except Exception:
            return False
    
    def _parse_kml(self, file_path: str) -> List[Dict]:
        """Parse KML file and extract features."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("xml.etree.ElementTree is required for KML parsing")
        
        features = []
        
        # Handle KMZ files
        if file_path.lower().endswith('.kmz'):
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as kmz:
                kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
                if not kml_files:
                    return features
                
                kml_content = kmz.read(kml_files[0])
                root = ET.fromstring(kml_content)
        else:
            tree = ET.parse(file_path)
            root = tree.getroot()
        
        # Define namespace
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # Find all placemarks
        placemarks = root.findall('.//kml:Placemark', ns)
        
        for placemark in placemarks:
            feature = {}
            
            # Extract name and description
            name_elem = placemark.find('kml:name', ns)
            if name_elem is not None:
                feature['name'] = name_elem.text
            
            description_elem = placemark.find('kml:description', ns)
            if description_elem is not None:
                feature['description'] = description_elem.text
            
            # Extract coordinates
            coords = self._extract_kml_coordinates(placemark, ns)
            if coords:
                if len(coords) == 1:
                    # Point
                    feature['x'] = coords[0][0]
                    feature['y'] = coords[0][1]
                    if len(coords[0]) > 2:
                        feature['z'] = coords[0][2]
                    feature['geometry_type'] = 'Point'
                else:
                    # Line or polygon - use first point or centroid
                    feature['x'] = coords[0][0]
                    feature['y'] = coords[0][1]
                    if len(coords[0]) > 2:
                        feature['z'] = coords[0][2]
                    feature['geometry_type'] = 'LineString' if len(coords) > 1 else 'Point'
                    feature['vertex_count'] = len(coords)
            
            if feature:
                features.append(feature)
        
        return features
    
    def _extract_kml_coordinates(self, placemark, ns) -> List[Tuple]:
        """Extract coordinates from KML placemark."""
        coordinates = []
        
        # Look for different geometry types
        for geom_type in ['Point', 'LineString', 'Polygon']:
            geom_elem = placemark.find(f'.//kml:{geom_type}', ns)
            if geom_elem is not None:
                coord_elem = geom_elem.find('kml:coordinates', ns)
                if coord_elem is not None:
                    coord_text = coord_elem.text.strip()
                    # Parse coordinates (lon,lat,alt format)
                    for coord_set in coord_text.split():
                        parts = coord_set.split(',')
                        if len(parts) >= 2:
                            try:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                alt = float(parts[2]) if len(parts) > 2 else 0
                                coordinates.append((lon, lat, alt))
                            except ValueError:
                                continue
                break
        
        return coordinates


# Factory function for creating readers
def create_geological_reader(file_path: str) -> BaseGeologicalReader:
    """
    Create appropriate geological reader based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Appropriate reader instance
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.las':
        return LASReader()
    elif extension == '.shp':
        return ShapefileReader()
    elif extension in ['.kml', '.kmz']:
        return KMLReader()
    else:
        raise ValueError(f"Unsupported geological format: {extension}")


def read_geological_file(file_path: str) -> Tuple[pd.DataFrame, GeologicalDataInfo]:
    """
    Convenience function to read any supported geological format.
    
    Args:
        file_path: Path to the geological data file
        
    Returns:
        Tuple of (DataFrame, metadata)
    """
    reader = create_geological_reader(file_path)
    return reader.read_file(file_path)