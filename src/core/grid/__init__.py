"""
Grid generation module for spatial interpolation.

This module provides classes and functions for generating regular grids
used in spatial interpolation operations.
"""

from .generator import GridGenerator, GridParameters, GridGenerationError

__all__ = [
    'GridGenerator',
    'GridParameters', 
    'GridGenerationError'
]