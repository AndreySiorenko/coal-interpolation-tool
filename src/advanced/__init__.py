"""
Advanced functions package for coal deposit interpolation.

This package provides advanced interpolation capabilities including:
- Data compositing methods
- Declustering techniques
- Advanced preprocessing
"""

from .data_compositor import DataCompositor, CompositingResult
from .declustering import CellDeclusterer, PolygonDeclusterer, DeclusteringResult

__all__ = [
    'DataCompositor',
    'CompositingResult',
    'CellDeclusterer', 
    'PolygonDeclusterer',
    'DeclusteringResult'
]