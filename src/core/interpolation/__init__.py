"""
Interpolation algorithms module.
"""

from .base import BaseInterpolator
from .idw import IDWInterpolator
from .rbf import RBFInterpolator
from .kriging import KrigingInterpolator