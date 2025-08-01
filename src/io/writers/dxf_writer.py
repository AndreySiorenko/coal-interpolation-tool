"""
DXF writer for exporting interpolation results to AutoCAD DXF format.
"""

from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import warnings
import pandas as pd

from .base import BaseWriter, ExportFormat, ExportOptions, GridData, PointData, ExportError

# Try to import ezdxf - it's optional
try:
    import ezdxf
    from ezdxf import units
    from ezdxf.math import Vec3
    EZDXF_AVAILABLE = True
except ImportError:
    ezdxf = None
    EZDXF_AVAILABLE = False


@dataclass
class DXFExportOptions(ExportOptions):
    """
    DXF-specific export options.
    
    Attributes:
        units: Drawing units ('mm', 'm', 'ft', 'in', etc.)
        layer_name: Layer name for exported entities
        point_style: Style for point entities ('POINT', 'CIRCLE', 'CROSS')
        point_size: Size of point symbols
        contour_lines: Whether to generate contour lines for grid data
        contour_intervals: Contour interval for contour lines
        text_height: Height of text labels
        include_labels: Whether to include value labels
        color_by_value: Whether to color entities by value
        line_type: Line type for contours ('CONTINUOUS', 'DASHED', 'DOTTED')
        export_3d: Export as 3D entities when applicable
    """
    units: str = 'm'
    layer_name: str = 'INTERPOLATION'
    point_style: str = 'CIRCLE'
    point_size: float = 1.0
    contour_lines: bool = True
    contour_intervals: Optional[float] = None
    text_height: float = 2.0
    include_labels: bool = False
    color_by_value: bool = True
    line_type: str = 'CONTINUOUS'
    export_3d: bool = False


class DXFWriter(BaseWriter):
    """
    Writer for exporting data to AutoCAD DXF format.
    
    DXF is a CAD data file format used for data interoperability between
    AutoCAD and other programs. This writer can export:
    - Point data as points, circles, or crosses
    - Grid data as contour lines
    - Text labels with values
    - Color-coded entities
    
    Requires ezdxf library for functionality.
    """
    
    def __init__(self, options: Optional[DXFExportOptions] = None):
        """
        Initialize DXF writer.
        
        Args:
            options: DXF-specific export options
            
        Raises:
            ImportError: If ezdxf is not available
        """
        if not EZDXF_AVAILABLE:
            raise ImportError(
                "ezdxf library is required for DXF export. "
                "Install it with: pip install ezdxf"
            )
        
        if options is None:
            options = DXFExportOptions()
        elif not isinstance(options, DXFExportOptions):
            # Convert base options to DXF options
            dxf_options = DXFExportOptions()
            for field in options.__dataclass_fields__:
                if hasattr(options, field):
                    setattr(dxf_options, field, getattr(options, field))
            options = dxf_options
            
        super().__init__(options)
    
    @property
    def supported_formats(self) -> List[ExportFormat]:
        """Return list of formats supported by this writer."""
        return [ExportFormat.DXF]
    
    @property
    def file_extensions(self) -> List[str]:
        """Return list of file extensions for this writer."""
        return ['.dxf']
    
    def write_grid(self, 
                   data: GridData, 
                   filepath: Union[str, Path],
                   **kwargs) -> None:
        """
        Write grid data to DXF file.
        
        Grid data can be exported as:
        - Point grid with value labels
        - Contour lines at specified intervals
        - Colored surface representation
        
        Args:
            data: Grid data to export
            filepath: Output file path
            **kwargs: Additional DXF options
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Validate inputs
            self.validate_grid_data(data)
            filepath = self.validate_filepath(filepath)
            
            # Update options with kwargs
            dxf_options = self._update_options(**kwargs)
            
            # Create DXF document
            doc = ezdxf.new('R2018')  # Use AutoCAD 2018 format
            doc.units = self._get_units_enum(dxf_options.units)
            
            # Get model space
            msp = doc.modelspace()
            
            # Create layer
            layer = doc.layers.new(dxf_options.layer_name)
            
            # Export grid as points with optional labels
            self._write_grid_points(msp, data, dxf_options)
            
            # Generate contour lines if requested
            if dxf_options.contour_lines:
                self._write_contour_lines(msp, data, dxf_options)
            
            # Add metadata as text
            if dxf_options.include_metadata:
                self._write_metadata(msp, data, dxf_options)
            
            # Save document
            doc.saveas(filepath)
            
        except Exception as e:
            raise ExportError(f"Failed to export grid data to DXF: {e}")
    
    def write_points(self, 
                     data: PointData, 
                     filepath: Union[str, Path],
                     **kwargs) -> None:
        """
        Write point data to DXF file.
        
        Point data is exported as individual point entities with optional
        value labels and color coding.
        
        Args:
            data: Point data to export
            filepath: Output file path
            **kwargs: Additional DXF options
            
        Raises:
            ExportError: If export fails
        """
        try:
            # Validate inputs
            self.validate_point_data(data)
            filepath = self.validate_filepath(filepath)
            
            # Update options with kwargs
            dxf_options = self._update_options(**kwargs)
            
            # Create DXF document
            doc = ezdxf.new('R2018')
            doc.units = self._get_units_enum(dxf_options.units)
            
            # Get model space
            msp = doc.modelspace()
            
            # Create layer
            layer = doc.layers.new(dxf_options.layer_name)
            
            # Export points
            self._write_point_data(msp, data, dxf_options)
            
            # Add metadata as text
            if dxf_options.include_metadata:
                self._write_point_metadata(msp, data, dxf_options)
            
            # Save document
            doc.saveas(filepath)
            
        except Exception as e:
            raise ExportError(f"Failed to export point data to DXF: {e}")
    
    def _write_grid_points(self, 
                          msp, 
                          data: GridData, 
                          options: DXFExportOptions) -> None:
        """
        Write grid data as individual points.
        
        Args:
            msp: Model space
            data: Grid data
            options: Export options
        """
        # Create color mapping if requested
        color_map = None
        if options.color_by_value:
            color_map = self._create_color_map(data.values.flatten())
        
        # Iterate through grid points
        for i, x in enumerate(data.x_coords):
            for j, y in enumerate(data.y_coords):
                value = data.values[j, i]  # Note: j,i indexing for numpy arrays
                
                if np.isnan(value):
                    continue
                
                # Determine point coordinates
                if options.export_3d and data.is_3d:
                    # Use value as Z coordinate for 3D export
                    point_coords = Vec3(x, y, value)
                else:
                    point_coords = Vec3(x, y, 0)
                
                # Determine color
                color = 7  # Default white
                if color_map is not None:
                    color = color_map.get(value, 7)
                
                # Create point entity based on style
                if options.point_style == 'POINT':
                    entity = msp.add_point(point_coords)
                elif options.point_style == 'CIRCLE':
                    entity = msp.add_circle(point_coords, options.point_size)
                elif options.point_style == 'CROSS':
                    # Create a cross using lines
                    half_size = options.point_size / 2
                    msp.add_line(
                        Vec3(x - half_size, y, 0),
                        Vec3(x + half_size, y, 0)
                    )
                    entity = msp.add_line(
                        Vec3(x, y - half_size, 0),
                        Vec3(x, y + half_size, 0)
                    )
                
                # Set attributes
                entity.dxf.layer = options.layer_name
                entity.dxf.color = color
                
                # Add value label if requested
                if options.include_labels:
                    label_text = f"{value:.{options.precision}f}"
                    text_pos = Vec3(x + options.point_size, y + options.point_size, 0)
                    text_entity = msp.add_text(
                        label_text,
                        height=options.text_height,
                        dxfattribs={
                            'layer': options.layer_name,
                            'color': color
                        }
                    )
                    text_entity.set_pos(text_pos)
    
    def _write_contour_lines(self, 
                           msp, 
                           data: GridData, 
                           options: DXFExportOptions) -> None:
        """
        Generate and write contour lines for grid data.
        
        Args:
            msp: Model space
            data: Grid data
            options: Export options
        """
        try:
            from scipy.ndimage import gaussian_filter
            import matplotlib.pyplot as plt
            from matplotlib.contour import QuadContourSet
            
            # Smooth the data slightly to reduce noise in contours
            smoothed_values = gaussian_filter(data.values, sigma=0.5)
            
            # Determine contour levels
            if options.contour_intervals:
                min_val = np.nanmin(smoothed_values)
                max_val = np.nanmax(smoothed_values)
                levels = np.arange(
                    np.ceil(min_val / options.contour_intervals) * options.contour_intervals,
                    max_val,
                    options.contour_intervals
                )
            else:
                # Auto-generate 10 contour levels
                levels = np.linspace(np.nanmin(smoothed_values), np.nanmax(smoothed_values), 10)
            
            # Create meshgrid for contouring
            X, Y = np.meshgrid(data.x_coords, data.y_coords)
            
            # Generate contours using matplotlib
            fig, ax = plt.subplots()
            cs = ax.contour(X, Y, smoothed_values, levels=levels)
            plt.close(fig)
            
            # Extract contour lines and add to DXF
            color_map = self._create_color_map(levels) if options.color_by_value else None
            
            for level_idx, level in enumerate(levels):
                color = color_map.get(level, 7) if color_map else 7
                
                # Get contour paths for this level
                for collection in cs.collections:
                    if len(collection.get_paths()) > 0:
                        for path in collection.get_paths():
                            vertices = path.vertices
                            if len(vertices) > 1:
                                # Create polyline
                                points = [Vec3(pt[0], pt[1], 0) for pt in vertices]
                                polyline = msp.add_lwpolyline(points)
                                polyline.dxf.layer = options.layer_name
                                polyline.dxf.color = color
                                polyline.dxf.linetype = options.line_type
                                
                                # Add contour label
                                if options.include_labels and len(vertices) > len(vertices) // 2:
                                    mid_pt = vertices[len(vertices) // 2]
                                    label_text = f"{level:.{options.precision}f}"
                                    text_entity = msp.add_text(
                                        label_text,
                                        height=options.text_height * 0.8,
                                        dxfattribs={
                                            'layer': options.layer_name,
                                            'color': color
                                        }
                                    )
                                    text_entity.set_pos(Vec3(mid_pt[0], mid_pt[1], 0))
                        
        except ImportError:
            warnings.warn("scipy and matplotlib required for contour generation, skipping contours")
        except Exception as e:
            warnings.warn(f"Failed to generate contour lines: {e}")
    
    def _write_point_data(self, 
                         msp, 
                         data: PointData, 
                         options: DXFExportOptions) -> None:
        """
        Write scattered point data.
        
        Args:
            msp: Model space
            data: Point data
            options: Export options
        """
        # Create color mapping if requested
        color_map = None
        if options.color_by_value:
            color_map = self._create_color_map(data.values)
        
        # Write each point
        for i, (coords, value) in enumerate(zip(data.coordinates, data.values)):
            if np.isnan(value):
                continue
            
            # Determine point coordinates
            if options.export_3d and data.is_3d:
                point_coords = Vec3(coords[0], coords[1], coords[2])
            else:
                point_coords = Vec3(coords[0], coords[1], 0)
            
            # Determine color
            color = 7  # Default white
            if color_map is not None:
                color = color_map.get(value, 7)
            
            # Create point entity
            if options.point_style == 'POINT':
                entity = msp.add_point(point_coords)
            elif options.point_style == 'CIRCLE':
                entity = msp.add_circle(point_coords, options.point_size)
            elif options.point_style == 'CROSS':
                # Create cross using lines
                half_size = options.point_size / 2
                x, y = coords[0], coords[1]
                msp.add_line(
                    Vec3(x - half_size, y, 0),
                    Vec3(x + half_size, y, 0)
                )
                entity = msp.add_line(
                    Vec3(x, y - half_size, 0),
                    Vec3(x, y + half_size, 0)
                )
            
            # Set attributes
            entity.dxf.layer = options.layer_name
            entity.dxf.color = color
            
            # Add point ID if available
            if data.point_ids is not None and options.include_labels:
                label_text = str(data.point_ids[i])
                text_pos = Vec3(coords[0] + options.point_size, coords[1] + options.point_size, 0)
                text_entity = msp.add_text(
                    label_text,
                    height=options.text_height,
                    dxfattribs={
                        'layer': options.layer_name,
                        'color': color
                    }
                )
                text_entity.set_pos(text_pos)
            
            # Add value label if requested
            elif options.include_labels:
                label_text = f"{value:.{options.precision}f}"
                text_pos = Vec3(coords[0] + options.point_size, coords[1] + options.point_size, 0)
                text_entity = msp.add_text(
                    label_text,
                    height=options.text_height,
                    dxfattribs={
                        'layer': options.layer_name,
                        'color': color
                    }
                )
                text_entity.set_pos(text_pos)
    
    def _write_metadata(self, 
                       msp, 
                       data: GridData, 
                       options: DXFExportOptions) -> None:
        """
        Write metadata as text annotations.
        
        Args:
            msp: Model space
            data: Grid data
            options: Export options
        """
        # Position metadata text at corner of data bounds
        bounds = data.bounds
        text_x = bounds[0]  # xmin
        text_y = bounds[3] + (bounds[3] - bounds[2]) * 0.1  # ymax + 10% of y range
        
        # Create metadata text
        metadata_lines = [
            f"Grid Size: {data.shape}",
            f"Cell Size: {data.cell_size}",
            f"Bounds: {bounds}",
            f"Data Range: {np.nanmin(data.values):.{options.precision}f} - {np.nanmax(data.values):.{options.precision}f}",
        ]
        
        if data.coordinate_system:
            metadata_lines.append(f"CRS: {data.coordinate_system}")
        
        # Add custom metadata
        if data.metadata:
            for key, value in data.metadata.items():
                metadata_lines.append(f"{key}: {value}")
        
        # Write each line
        for i, line in enumerate(metadata_lines):
            text_entity = msp.add_text(
                line,
                height=options.text_height,
                dxfattribs={
                    'layer': options.layer_name,
                    'color': 7
                }
            )
            text_entity.set_pos(Vec3(text_x, text_y - i * options.text_height * 1.5, 0))
    
    def _write_point_metadata(self, 
                            msp, 
                            data: PointData, 
                            options: DXFExportOptions) -> None:
        """
        Write point data metadata as text annotations.
        
        Args:
            msp: Model space
            data: Point data
            options: Export options
        """
        # Position metadata text at corner of data bounds
        bounds = data.bounds
        text_x = bounds[0]  # xmin
        text_y = bounds[3] + (bounds[3] - bounds[2]) * 0.1 if len(bounds) >= 4 else bounds[1] + 10  # ymax + buffer
        
        # Create metadata text
        metadata_lines = [
            f"Points: {data.n_points}",
            f"Bounds: {bounds}",
            f"Value Range: {np.nanmin(data.values):.{options.precision}f} - {np.nanmax(data.values):.{options.precision}f}",
        ]
        
        if data.coordinate_system:
            metadata_lines.append(f"CRS: {data.coordinate_system}")
        
        # Add custom metadata
        if data.metadata:
            for key, value in data.metadata.items():
                metadata_lines.append(f"{key}: {value}")
        
        # Write each line
        for i, line in enumerate(metadata_lines):
            text_entity = msp.add_text(
                line,
                height=options.text_height,
                dxfattribs={
                    'layer': options.layer_name,
                    'color': 7
                }
            )
            text_entity.set_pos(Vec3(text_x, text_y - i * options.text_height * 1.5, 0))
    
    def _create_color_map(self, values: np.ndarray) -> Dict[float, int]:
        """
        Create a color mapping for values.
        
        Args:
            values: Array of values to map
            
        Returns:
            Dictionary mapping values to AutoCAD color indices
        """
        # Remove NaN values
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            return {}
        
        # Define color range (AutoCAD color indices 1-255)
        colors = list(range(1, 256))  # Skip 0 (ByBlock) and use 1-255
        
        # Create mapping
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        
        if min_val == max_val:
            return {min_val: colors[len(colors)//2]}  # Use middle color for constant values
        
        color_map = {}
        for value in valid_values:
            # Normalize value to 0-1 range
            normalized = (value - min_val) / (max_val - min_val)
            # Map to color index
            color_idx = int(normalized * (len(colors) - 1))
            color_map[value] = colors[color_idx]
        
        return color_map
    
    def _get_units_enum(self, units_str: str):
        """
        Convert units string to ezdxf units enum.
        
        Args:
            units_str: Units string
            
        Returns:
            ezdxf units enum value
        """
        units_map = {
            'mm': units.MM,
            'm': units.M,
            'cm': units.CM,
            'km': units.KM,
            'in': units.IN,
            'ft': units.FT,
            'yd': units.YD,
            'mi': units.MI,
        }
        
        return units_map.get(units_str.lower(), units.M)  # Default to meters
    
    def _update_options(self, **kwargs) -> DXFExportOptions:
        """
        Update export options with provided kwargs.
        
        Args:
            **kwargs: Additional options
            
        Returns:
            Updated DXF export options
        """
        # Create a copy of current options
        options = DXFExportOptions()
        
        # Copy current values
        for field in self.options.__dataclass_fields__:
            if hasattr(self.options, field):
                setattr(options, field, getattr(self.options, field))
        
        # Update with kwargs
        for key, value in kwargs.items():
            if hasattr(options, key):
                setattr(options, key, value)
        
        return options
    
    def export_summary(self, 
                      data: Union[GridData, PointData], 
                      filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get summary information about the export.
        
        Args:
            data: Data to be exported
            filepath: Target file path
            
        Returns:
            Dictionary with export summary
        """
        filepath = Path(filepath)
        
        summary = {
            'format': 'DXF',
            'filepath': str(filepath),
            'data_type': 'grid' if isinstance(data, GridData) else 'points',
            'bounds': data.bounds,
            'units': self.options.units,
            'layer_name': self.options.layer_name,
            'point_style': self.options.point_style,
            'include_labels': self.options.include_labels,
            'color_by_value': self.options.color_by_value,
        }
        
        if isinstance(data, GridData):
            summary.update({
                'grid_shape': data.shape,
                'cell_size': data.cell_size,
                'n_points': data.n_points,
                'contour_lines': self.options.contour_lines,
            })
        else:
            summary.update({
                'n_points': data.n_points,
                'is_3d': data.is_3d,
            })
        
        if hasattr(data, 'coordinate_system') and data.coordinate_system:
            summary['coordinate_system'] = data.coordinate_system
        
        return summary


def create_dxf_writer(units: str = 'm',
                     layer_name: str = 'INTERPOLATION',
                     point_style: str = 'CIRCLE',
                     contour_lines: bool = True,
                     include_labels: bool = False) -> DXFWriter:
    """
    Factory function to create a DXF writer with common options.
    
    Args:
        units: Drawing units
        layer_name: Layer name for entities
        point_style: Style for point entities
        contour_lines: Whether to generate contour lines
        include_labels: Whether to include labels
        
    Returns:
        Configured DXF writer
    """
    if not EZDXF_AVAILABLE:
        raise ImportError("ezdxf library is required for DXF export")
    
    options = DXFExportOptions(
        units=units,
        layer_name=layer_name,
        point_style=point_style,
        contour_lines=contour_lines,
        include_labels=include_labels
    )
    
    return DXFWriter(options)