"""
Optimization tools and algorithms for SMPL-X parameter fitting.
"""

from .optimizers import ParameterOptimizer
from .objectives import LandmarkObjective, SilhouetteObjective
from .constraints import PoseConstraints, ShapeConstraints

__all__ = [
    "ParameterOptimizer",
    "LandmarkObjective",
    "SilhouetteObjective", 
    "PoseConstraints",
    "ShapeConstraints",
]
