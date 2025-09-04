"""
Core SMPL-X functionality and model handling.
"""

from .model import SMPLXModel
from .parameters import SMPLXParameters
from .mesh import SMPLXMesh

__all__ = [
    "SMPLXModel",
    "SMPLXParameters", 
    "SMPLXMesh",
]
