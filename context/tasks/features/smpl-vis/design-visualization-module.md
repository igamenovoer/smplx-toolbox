# SMPL Visualization Module Design Document

## Overview

The SMPL visualization module provides a focused interface to add SMPL model visualizations (meshes, joints, skeleton) to PyVista plotters. Users can provide their own plotter or let the module create one, then retrieve the plotter to call `show()` or add custom visualizations using PyVista's own API.

## Core Concept

The module centers around a single class that:
1. Takes a unified SMPL model as input
2. Adds SMPL-specific visualization objects to a PyVista plotter
3. Returns PyVista actors for user management
4. Provides the plotter for final display via `get_plotter()`

For generic visualization needs (points, lines, custom meshes), users should use PyVista's API directly on the plotter.

## Module Structure

```
src/smplx_toolbox/visualization/
├── __init__.py           # Module exports
├── plotter.py           # Main SMPLVisualizer class  
├── config.py            # Simple visualization configuration
└── utils.py             # Bone connections and utility functions
```

## Main Class: SMPLVisualizer

**Interface Design:**
```python
class SMPLVisualizer:
    """
    Add SMPL model visualizations to PyVista plotters.
    
    Takes a unified SMPL model and adds SMPL-specific visualization elements
    (mesh, selected joints, skeleton) to a PyVista plotter. The plotter can be 
    user-provided or created internally. Returns actors for user management.
    
    For generic visualization (arbitrary points, lines, meshes), use PyVista's
    API directly on the plotter obtained via get_plotter().
    """
    
    def __init__(self) -> None:
        """Initialize empty visualizer."""
        # Initialize member variables to None
        
    @property  
    def plotter(self) -> pv.Plotter | Any | None:
        """Get the PyVista plotter instance (regular or BackgroundPlotter)."""
        
    @property
    def model(self) -> UnifiedSmplModel | None:
        """Get the SMPL model."""
        
    def set_plotter(self, plotter: pv.Plotter | Any) -> None:
        """
        Set PyVista plotter to use for visualization.
        
        Parameters
        ----------
        plotter : pv.Plotter or pyvistaqt.BackgroundPlotter
            The plotter to add visualizations to
        """
        
    def set_model(self, model: UnifiedSmplModel) -> None:
        """
        Set the SMPL model to visualize.
        
        Parameters
        ----------
        model : UnifiedSmplModel
            The unified SMPL model
        """
        
    @classmethod
    def from_model(
        cls, 
        model: UnifiedSmplModel,
        plotter: pv.Plotter | Any | None = None,
        background: bool = False
    ):
        """
        Create visualizer from SMPL model.
        
        Parameters
        ----------
        model : UnifiedSmplModel
            The SMPL model to visualize
        plotter : pv.Plotter or pyvistaqt.BackgroundPlotter, optional
            Existing plotter to use. If None, creates one based on background parameter.
        background : bool, optional
            If plotter is None and background=True, creates BackgroundPlotter (non-blocking).
            If plotter is None and background=False, creates regular Plotter (blocking).
            Ignored if plotter is provided.
            
        Returns
        -------
        SMPLVisualizer
            Configured visualizer instance
        """
        
    def add_mesh(
        self,
        output: UnifiedSmplOutput | None = None,
        **kwargs
    ) -> Any:
        """
        Add SMPL mesh to the plotter.
        
        Parameters
        ----------
        output : UnifiedSmplOutput, optional
            Model output. If None, generates from current model with default pose.
        **kwargs
            Arguments passed to plotter.add_mesh() like:
            - style: 'surface', 'wireframe', 'points'
            - color: color name or RGB tuple
            - opacity: 0-1 float
            - show_edges: bool
            
        Returns
        -------
        actor
            PyVista actor for the mesh
        """
        
    def add_smpl_joints(
        self,
        output: UnifiedSmplOutput | None = None,
        joints: list[str] | list[int] | None = None,
        size: float = 0.02,
        color: Any | None = None,
        labels: bool = False,
        label_font_size: int = 12,
        **kwargs
    ) -> dict[str, Any]:
        """
        Add SMPL joints to the plotter with optional selection and labels.
        
        Parameters
        ----------
        output : UnifiedSmplOutput, optional
            Model output. If None, generates from current model.
        joints : list of str or list of int, optional
            Joints to visualize. Can be:
            - None: visualize all joints for the model type
            - List of joint names (from smplx_toolbox.core.constants enums)
            - List of joint indices (model-specific, see skeleton mapping)
            Special keywords as list elements: 'body' (core 22), 'hands' (fingers), 'face', 'all'
        size : float, optional
            Joint sphere size (default: 0.02)
        color : str or tuple, optional
            Joint color (default: red)
        labels : bool, optional
            Whether to add joint name labels (default: False)
        label_font_size : int, optional
            Font size for joint labels (default: 12). Only used when labels=True.
        **kwargs
            Additional arguments passed to plotter.add_points()
            
        Returns
        -------
        dict[str, actor]
            Dictionary with 'points' actor and optionally 'labels' actor
            
        Examples
        --------
        >>> # Visualize all joints
        >>> viz.add_smpl_joints(output)
        
        >>> # Visualize only core body joints (expand keyword to list)
        >>> viz.add_smpl_joints(output, joints=['body'])
        
        >>> # Visualize specific joints by name with labels
        >>> viz.add_smpl_joints(output, joints=['left_wrist', 'right_wrist'], 
        ...                     labels=True, label_font_size=16)
        
        >>> # Visualize joints by index (model-specific)
        >>> viz.add_smpl_joints(output, joints=[20, 21])  # wrists in all models
        
        Notes
        -----
        Joint names are defined in smplx_toolbox.core.constants:
        - CoreBodyJoint: body joints (indices 0-21)
        - HandFingerJoint: finger joints (vary by model)
        - FaceJoint: face joints (SMPL-X only)
        
        Joint indices depend on model type:
        - SMPL: 0-23 (body + hand end effectors)
        - SMPL-H: 0-51 (body + detailed fingers)
        - SMPL-X: 0-54 (body + face + detailed fingers)
        """
        
    def add_smpl_skeleton(
        self,
        output: UnifiedSmplOutput | None = None,
        connections: list[tuple[int, int]] | None = None,
        radius: float = 0.005,
        color: Any | None = None,
        as_lines: bool = False,
        **kwargs
    ) -> Any:
        """
        Add SMPL bone skeleton to the plotter.
        
        Parameters
        ----------
        output : UnifiedSmplOutput, optional
            Model output. If None, generates from current model.
        connections : list[tuple[int, int]], optional
            Bone connections as (parent_idx, child_idx) pairs.
            If None, uses default skeleton for the model type.
        radius : float, optional
            Bone cylinder radius (default: 0.005). Ignored if as_lines=True.
        color : str or tuple, optional
            Bone color (default: blue)
        as_lines : bool, optional
            If True, render as lines instead of cylinders (default: False)
        **kwargs
            Additional arguments passed to plotter.add_mesh()
            
        Returns
        -------
        actor
            PyVista actor for the skeleton
        
        Examples
        --------
        >>> # Default skeleton for model type
        >>> viz.add_smpl_skeleton(output)
        
        >>> # Custom connections
        >>> connections = [(0, 1), (0, 2), (0, 3)]  # pelvis to hips and spine
        >>> viz.add_smpl_skeleton(output, connections=connections)
        
        >>> # Fast line rendering
        >>> viz.add_smpl_skeleton(output, as_lines=True, line_width=3)
        """
        
    def get_plotter(self) -> pv.Plotter | Any:
        """
        Get the configured PyVista plotter for display or further customization.
        
        Users can add custom visualizations using PyVista's API:
        - plotter.add_points() for arbitrary points
        - plotter.add_mesh() for custom meshes or lines
        - plotter.add_text() for annotations
        - etc.
        
        Returns
        -------
        pv.Plotter or pyvistaqt.BackgroundPlotter
            The plotter with all added visualizations
            
        Examples
        --------
        >>> viz = SMPLVisualizer.from_model(model)
        >>> viz.add_mesh(output)
        >>> plotter = viz.get_plotter()
        >>> 
        >>> # Add custom visualizations using PyVista API
        >>> plotter.add_points(my_keypoints, color='yellow')
        >>> plotter.add_text("SMPL Analysis", position='upper_left')
        >>> 
        >>> plotter.show()  # Display the visualization
        """
```

## Configuration Class (Simplified)

**Interface Design:**
```python
class VisualizationConfig:
    """
    Simple configuration for default colors and styles.
    """
    
    def __init__(self) -> None:
        """Initialize with sensible defaults."""
        # Default colors as RGB tuples
        self.m_mesh_color: tuple[float, float, float] = (0.8, 0.8, 0.9)
        self.m_joint_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
        self.m_bone_color: tuple[float, float, float] = (0.0, 0.0, 1.0)
        
    @property
    def mesh_color(self) -> tuple[float, float, float]:
        """Get default mesh color."""
        
    @property
    def joint_color(self) -> tuple[float, float, float]:
        """Get default joint color."""
        
    @property
    def bone_color(self) -> tuple[float, float, float]:
        """Get default bone color."""
```

## Utility Functions and Data

**In utils.py:**
```python
# Bone connections for different SMPL variants
def get_smpl_bone_connections() -> list[tuple[int, int]]:
    """Get SMPL skeleton bone connections (22 body joints + 2 hand end effectors)."""
    
def get_smplh_bone_connections() -> list[tuple[int, int]]:
    """Get SMPL-H skeleton including detailed hand connections (52 joints)."""
    
def get_smplx_bone_connections() -> list[tuple[int, int]]:
    """Get SMPL-X skeleton including face and hand connections (55 joints)."""

# Helper function for mesh conversion
def create_polydata_from_vertices_faces(
    vertices: np.ndarray, 
    faces: np.ndarray
) -> pv.PolyData:
    """Convert SMPL vertices and faces to PyVista PolyData."""

# Joint selection helpers
def resolve_joint_selection(
    joints: list[str] | list[int] | None,
    model_type: ModelType
) -> list[int]:
    """
    Resolve joint selection to indices for the given model type.
    
    Parameters
    ----------
    joints : list[str] | list[int] | None
        Joint selection (names list, indices list, or None for all)
        Can include special keywords as elements: 'body', 'hands', 'face', 'all'
    model_type : ModelType
        The SMPL model type
        
    Returns
    -------
    list[int]
        List of joint indices for the model
    """
```

## Usage Examples

### Basic Usage
```python
from smplx_toolbox.core.unified_model import UnifiedSmplModel
from smplx_toolbox.visualization import SMPLVisualizer

# Create visualizer (creates internal plotter)
viz = SMPLVisualizer.from_model(model)

# Add SMPL mesh
output = model.forward(pose_params)
mesh_actor = viz.add_mesh(output, color='lightblue', opacity=0.9)

# Add all joints with labels
joint_actors = viz.add_smpl_joints(output, labels=True, label_font_size=14)

# Add skeleton
skeleton_actor = viz.add_smpl_skeleton(output, color='red')

# Get plotter and display
plotter = viz.get_plotter()
plotter.show()  # User calls show()
```

### Selective Joint Visualization
```python
from smplx_toolbox.core.constants import CoreBodyJoint

# Visualize only wrists and ankles by name
viz.add_smpl_joints(output, joints=[
    CoreBodyJoint.LEFT_WRIST,
    CoreBodyJoint.RIGHT_WRIST,
    CoreBodyJoint.LEFT_ANKLE,
    CoreBodyJoint.RIGHT_ANKLE
], size=0.05, color='green', labels=True, label_font_size=18)

# Visualize body joints only (keyword must be in list)
viz.add_smpl_joints(output, joints=['body'])

# Visualize specific joints by index
viz.add_smpl_joints(output, joints=[15, 20, 21])  # head and wrists
```

### Integration with Existing Plotter
```python
import pyvista as pv

# User has their own plotter with custom setup
my_plotter = pv.Plotter(window_size=(1200, 800))
my_plotter.add_text("SMPL Analysis", position='upper_left')
my_plotter.add_axes()

# Use visualizer with existing plotter
viz = SMPLVisualizer.from_model(model, plotter=my_plotter)
viz.add_mesh(output)
viz.add_smpl_joints(output, joints=['body'], labels=True, label_font_size=10)
viz.add_smpl_skeleton(output)

# User adds custom elements using PyVista API directly
my_plotter.add_floor('-z')
sphere = pv.Sphere(center=(1, 0, 0))
my_plotter.add_mesh(sphere, color='green')

# Add custom points (e.g., detected keypoints)
my_keypoints = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
my_plotter.add_points(my_keypoints, color='yellow', point_size=10)

# User controls display
my_plotter.camera_position = 'xy'
my_plotter.show()
```

### Background (Non-blocking) Mode
```python
# Create with background plotter
viz = SMPLVisualizer.from_model(model, background=True)
viz.add_mesh(output)
viz.add_smpl_skeleton(output)

# Get plotter - window opens in background
plotter = viz.get_plotter()

# Continue with other code while visualization window stays open
for i in range(100):
    # Do other work
    new_output = model.forward(new_pose)
    # Could update visualization if needed
```

## Key Design Points

1. **SMPL-Focused**: Only provides SMPL-specific visualization methods
2. **PyVista Integration**: Users use PyVista API directly for generic needs
3. **Joint Selection**: Flexible joint selection by name or index
4. **Model Awareness**: Handles different joint sets for SMPL/SMPL-H/SMPL-X
5. **Actor Returns**: All methods return actors for user customization
6. **User Control**: User gets plotter and has full control
7. **Clean Separation**: SMPL-specific vs generic visualization clearly separated

## Dependencies

### Required
- `pyvista >= 0.40.0` - Core 3D visualization
- `numpy >= 1.19.0` - Array operations  
- `attrs >= 22.0.0` - Class definitions

### Optional
- `pyvistaqt >= 0.9.0` - For background (non-blocking) mode

## Implementation Notes

### Joint Resolution
- Map joint names to indices using `smplx_toolbox.core.constants`
- Handle model-specific joint sets (SMPL: 24, SMPL-H: 52, SMPL-X: 55)
- Support keywords like 'body', 'hands', 'face' for common selections

### Plotter Type Detection
- Detect whether plotter is `pv.Plotter` or `pvqt.BackgroundPlotter`
- Adapt behavior accordingly

### Mesh Data Creation
- Convert SMPL vertices and faces to `pv.PolyData` efficiently
- Use PyVista's built-in capabilities for all rendering

### Skeleton Generation
- Default connections based on model type
- Support custom connections for flexibility
- Option for fast line rendering vs cylinders