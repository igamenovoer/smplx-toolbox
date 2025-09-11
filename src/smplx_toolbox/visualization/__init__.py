"""SMPL visualization module.

This module provides visualization capabilities for SMPL family models using PyVista.
It includes a main visualizer class that can add SMPL-specific visualization elements
(meshes, joints, skeletons) to PyVista plotters.

Classes
-------
SMPLVisualizer : Main SMPL visualization class
VisualizationConfig : Configuration for default colors and styles

Functions
---------
get_smpl_bone_connections : Get SMPL skeleton bone connections
get_smplh_bone_connections : Get SMPL-H skeleton bone connections
get_smplx_bone_connections : Get SMPL-X skeleton bone connections
create_polydata_from_vertices_faces : Convert vertices and faces to PyVista PolyData
resolve_joint_selection : Resolve joint selection to indices

Examples
--------
Basic usage:

>>> from smplx_toolbox.core.unified_model import UnifiedSmplModel
>>> from smplx_toolbox.visualization import SMPLVisualizer
>>>
>>> # Load your SMPL model
>>> model = UnifiedSmplModel.from_file("path/to/model.pkl")
>>>
>>> # Create visualizer
>>> viz = SMPLVisualizer.from_model(model)
>>>
>>> # Generate output and add visualizations
>>> output = model.forward()
>>> viz.add_mesh(output, color='lightblue')
>>> viz.add_smpl_joints(output, joints=['body'], labels=True)
>>> viz.add_smpl_skeleton(output)
>>>
>>> # Display
>>> plotter = viz.get_plotter()
>>> plotter.show()

Using with existing plotter:

>>> import pyvista as pv
>>>
>>> # Create your own plotter
>>> plotter = pv.Plotter(window_size=(1200, 800))
>>> plotter.add_axes()
>>>
>>> # Use with visualizer
>>> viz = SMPLVisualizer.from_model(model, plotter=plotter)
>>> viz.add_mesh(output)
>>>
>>> # Add your own custom visualizations
>>> plotter.add_points(my_keypoints, color='yellow')
>>> plotter.show()
"""

from .config import VisualizationConfig
from .plotter import SMPLVisualizer
from .utils import (
    create_polydata_from_vertices_faces,
    add_axes,
    get_smpl_bone_connections,
    get_smplh_bone_connections,
    get_smplx_bone_connections,
    resolve_joint_selection,
)

__all__ = [
    # Main class
    "SMPLVisualizer",
    # Configuration
    "VisualizationConfig",
    # Utility functions
    "get_smpl_bone_connections",
    "get_smplh_bone_connections",
    "get_smplx_bone_connections",
    "add_axes",
    "create_polydata_from_vertices_faces",
    "resolve_joint_selection",
]

__version__ = "0.1.0"
