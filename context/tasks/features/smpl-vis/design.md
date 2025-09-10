# SMPL Visualization Module Design Document

## Overview

The SMPL visualization module provides comprehensive 3D visualization capabilities for SMPL-family models using PyVista and PyVistaQt. The module is designed to be flexible and integrate seamlessly with existing plotting workflows while providing rich visualization options for meshes, joints, bones, and keypoints.

## Core Architecture

### Design Principles

1. **Non-invasive Integration**: Users can provide their own PyVista plotter or let the module create one internally
2. **Extensible Visualization**: Support for multiple visualization modes (mesh, wireframe, joints, bones, etc.)
3. **Interactive Control**: Leverage PyVistaQt for interactive manipulation and real-time updates
4. **Performance Optimized**: Efficient handling of mesh data and updates for real-time visualization
5. **Consistent API**: Follow project coding standards with `m_` prefixes, read-only properties, and explicit setters

### Module Structure

```
src/smplx_toolbox/visualization/
├── __init__.py                 # Module exports
├── core/
│   ├── __init__.py
│   ├── plotter.py             # Main SMPLPlotter class
│   ├── components.py          # Visualization components (mesh, joints, etc.)
│   └── config.py              # Visualization configuration and styles
├── interactive/
│   ├── __init__.py
│   ├── qt_plotter.py          # Interactive Qt-based plotter
│   └── controls.py            # Interactive control widgets
└── utils/
    ├── __init__.py
    ├── colors.py              # Color schemes and palettes
    ├── transforms.py          # 3D transformations and utilities
    └── export.py              # Export utilities for rendered scenes
```

## Core Classes

### 1. SMPLPlotter

The main visualization class that handles SMPL model visualization.

```python
from typing import Any, Optional, Dict, List, Type, TypeVar, Union, Tuple
import numpy as np
import torch
import pyvista as pv
from attrs import define, field

from smplx_toolbox.core.unified_model import UnifiedSmplModel, UnifiedSmplOutput
from smplx_toolbox.core.constants import ModelType

T = TypeVar('T', bound='SMPLPlotter')

@define(kw_only=True)
class SMPLPlotter:
    """
    Main SMPL visualization class using PyVista.
    
    Provides comprehensive 3D visualization capabilities for SMPL-family models
    including mesh rendering, joint visualization, bone structure display, and
    interactive manipulation.
    
    Attributes
    ----------
    plotter : pyvista.Plotter or None
        The PyVista plotter instance
    model : UnifiedSmplModel or None
        The SMPL model to visualize
    config : VisualizationConfig or None
        Visualization configuration settings
    """
    
    def __init__(self) -> None:
        """Initialize empty SMPL plotter."""
        self.m_plotter: Optional[pv.Plotter] = None
        self.m_model: Optional[UnifiedSmplModel] = None
        self.m_config: Optional[VisualizationConfig] = None
        self.m_mesh_actors: Dict[str, Any] = {}
        self.m_joint_actors: Dict[str, Any] = {}
        self.m_bone_actors: Dict[str, Any] = {}
        self.m_current_output: Optional[UnifiedSmplOutput] = None
        
    @property
    def plotter(self) -> Optional[pv.Plotter]:
        """Get the PyVista plotter instance."""
        return self.m_plotter
        
    @property
    def model(self) -> Optional[UnifiedSmplModel]:
        """Get the SMPL model."""
        return self.m_model
        
    @property
    def config(self) -> Optional[VisualizationConfig]:
        """Get the visualization configuration."""
        return self.m_config
    
    def set_plotter(self, plotter: pv.Plotter) -> None:
        """Set the PyVista plotter instance."""
        self.m_plotter = plotter
        
    def set_model(self, model: UnifiedSmplModel) -> None:
        """Set the SMPL model to visualize."""
        self.m_model = model
        
    def set_config(self, config: VisualizationConfig) -> None:
        """Set the visualization configuration."""
        self.m_config = config
    
    @classmethod
    def from_model(
        cls: Type[T], 
        model: UnifiedSmplModel,
        plotter: Optional[pv.Plotter] = None,
        config: Optional[VisualizationConfig] = None
    ) -> T:
        """
        Create plotter from SMPL model.
        
        Parameters
        ----------
        model : UnifiedSmplModel
            The SMPL model to visualize
        plotter : pv.Plotter, optional
            Existing PyVista plotter to use. If None, creates new plotter.
        config : VisualizationConfig, optional
            Visualization configuration. If None, uses default config.
            
        Returns
        -------
        SMPLPlotter
            Configured plotter instance
        """
        instance = cls()
        instance.set_model(model)
        
        if plotter is None:
            plotter = pv.Plotter()
        instance.set_plotter(plotter)
        
        if config is None:
            config = VisualizationConfig.default()
        instance.set_config(config)
        
        return instance
    
    def add_mesh(
        self,
        output: Optional[UnifiedSmplOutput] = None,
        style: str = "surface",
        color: Optional[Union[str, Tuple[float, float, float]]] = None,
        opacity: float = 1.0,
        show_edges: bool = False
    ) -> None:
        """
        Add SMPL mesh to the plotter.
        
        Parameters
        ----------
        output : UnifiedSmplOutput, optional
            Model output to visualize. If None, generates from current model.
        style : {'surface', 'wireframe', 'points'}, optional
            Mesh rendering style (default: 'surface')
        color : str or tuple, optional
            Mesh color (default: uses config)
        opacity : float, optional
            Mesh opacity between 0-1 (default: 1.0)
        show_edges : bool, optional
            Whether to show mesh edges (default: False)
        """
        
    def add_joints(
        self,
        output: Optional[UnifiedSmplOutput] = None,
        joint_size: float = 0.02,
        color: Optional[Union[str, Tuple[float, float, float]]] = None,
        show_labels: bool = False
    ) -> None:
        """
        Add joint points to the plotter.
        
        Parameters
        ----------
        output : UnifiedSmplOutput, optional
            Model output to visualize. If None, generates from current model.
        joint_size : float, optional
            Size of joint spheres (default: 0.02)
        color : str or tuple, optional
            Joint color (default: uses config)
        show_labels : bool, optional
            Whether to show joint name labels (default: False)
        """
        
    def add_bones(
        self,
        output: Optional[UnifiedSmplOutput] = None,
        bone_radius: float = 0.01,
        color: Optional[Union[str, Tuple[float, float, float]]] = None
    ) -> None:
        """
        Add bone connections to the plotter.
        
        Parameters
        ----------
        output : UnifiedSmplOutput, optional
            Model output to visualize. If None, generates from current model.
        bone_radius : float, optional
            Radius of bone cylinders (default: 0.01)
        color : str or tuple, optional
            Bone color (default: uses config)
        """
        
    def update_visualization(
        self,
        output: UnifiedSmplOutput,
        update_mesh: bool = True,
        update_joints: bool = True,
        update_bones: bool = True
    ) -> None:
        """
        Update existing visualization with new model output.
        
        Parameters
        ----------
        output : UnifiedSmplOutput
            New model output to display
        update_mesh : bool, optional
            Whether to update mesh (default: True)
        update_joints : bool, optional
            Whether to update joints (default: True)
        update_bones : bool, optional
            Whether to update bones (default: True)
        """
        
    def clear_visualization(self) -> None:
        """Clear all visualization elements from the plotter."""
        
    def get_plotter(self) -> pv.Plotter:
        """
        Get the configured PyVista plotter ready for display.
        
        Returns
        -------
        pv.Plotter
            The configured plotter instance
        """
```

### 2. VisualizationConfig

Configuration class for visualization settings and styles.

```python
@define(kw_only=True)
class VisualizationConfig:
    """
    Configuration settings for SMPL visualization.
    
    Attributes
    ----------
    mesh_color : tuple
        Default mesh color as RGB tuple
    joint_color : tuple
        Default joint color as RGB tuple
    bone_color : tuple
        Default bone color as RGB tuple
    background_color : tuple
        Background color as RGB tuple
    lighting : bool
        Whether to enable lighting
    camera_position : str
        Default camera position
    """
    
    def __init__(self) -> None:
        """Initialize default visualization configuration."""
        self.m_mesh_color: Tuple[float, float, float] = (0.8, 0.8, 0.9)
        self.m_joint_color: Tuple[float, float, float] = (1.0, 0.2, 0.2)
        self.m_bone_color: Tuple[float, float, float] = (0.2, 0.2, 1.0)
        self.m_background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
        self.m_lighting: bool = True
        self.m_camera_position: str = "xy"
        self.m_window_size: Tuple[int, int] = (800, 600)
        
    @property
    def mesh_color(self) -> Tuple[float, float, float]:
        """Get default mesh color."""
        return self.m_mesh_color
        
    @property
    def joint_color(self) -> Tuple[float, float, float]:
        """Get default joint color."""
        return self.m_joint_color
        
    # ... similar properties for other attributes
        
    def set_mesh_color(self, color: Tuple[float, float, float]) -> None:
        """Set default mesh color."""
        self.m_mesh_color = color
        
    def set_joint_color(self, color: Tuple[float, float, float]) -> None:
        """Set default joint color."""
        self.m_joint_color = color
        
    # ... similar setters for other attributes
    
    @classmethod
    def default(cls: Type[T]) -> T:
        """Create default configuration."""
        return cls()
        
    @classmethod
    def minimal(cls: Type[T]) -> T:
        """Create minimal configuration for performance."""
        instance = cls()
        instance.set_lighting(False)
        return instance
        
    @classmethod
    def presentation(cls: Type[T]) -> T:
        """Create configuration optimized for presentations."""
        instance = cls()
        instance.set_background_color((1.0, 1.0, 1.0))  # White background
        instance.set_mesh_color((0.2, 0.4, 0.8))  # Blue mesh
        return instance
```

### 3. SMPLInteractivePlotter

Interactive plotter using PyVistaQt for real-time manipulation.

```python
try:
    import pyvistaqt as pvqt
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

@define(kw_only=True)
class SMPLInteractivePlotter(SMPLPlotter):
    """
    Interactive SMPL plotter with Qt integration.
    
    Extends SMPLPlotter with interactive capabilities including parameter
    sliders, real-time updates, and advanced manipulation controls.
    """
    
    def __init__(self) -> None:
        """Initialize interactive plotter."""
        super().__init__()
        if not QT_AVAILABLE:
            raise ImportError("PyVistaQt not available. Install with: pip install pyvistaqt")
        
        self.m_qt_plotter: Optional[pvqt.BackgroundPlotter] = None
        self.m_control_widgets: Dict[str, Any] = {}
        self.m_parameter_sliders: Dict[str, Any] = {}
        
    @property
    def qt_plotter(self) -> Optional[pvqt.BackgroundPlotter]:
        """Get the Qt background plotter."""
        return self.m_qt_plotter
    
    def set_qt_plotter(self, qt_plotter: pvqt.BackgroundPlotter) -> None:
        """Set the Qt background plotter."""
        self.m_qt_plotter = qt_plotter
        self.set_plotter(qt_plotter)  # Also set base plotter
    
    @classmethod
    def from_model_interactive(
        cls: Type[T],
        model: UnifiedSmplModel,
        config: Optional[VisualizationConfig] = None,
        show: bool = True
    ) -> T:
        """
        Create interactive plotter from SMPL model.
        
        Parameters
        ----------
        model : UnifiedSmplModel
            The SMPL model to visualize
        config : VisualizationConfig, optional
            Visualization configuration
        show : bool, optional
            Whether to show the plotter immediately (default: True)
            
        Returns
        -------
        SMPLInteractivePlotter
            Configured interactive plotter
        """
        instance = cls()
        instance.set_model(model)
        
        if config is None:
            config = VisualizationConfig.default()
        instance.set_config(config)
        
        # Create Qt plotter
        qt_plotter = pvqt.BackgroundPlotter(
            window_size=config.window_size,
            show=show
        )
        qt_plotter.set_background(config.background_color)
        instance.set_qt_plotter(qt_plotter)
        
        return instance
    
    def add_parameter_controls(self) -> None:
        """Add interactive parameter control widgets."""
        
    def add_pose_controls(self) -> None:
        """Add pose manipulation controls."""
        
    def add_shape_controls(self) -> None:
        """Add shape parameter controls."""
        
    def enable_real_time_updates(self) -> None:
        """Enable real-time visualization updates."""
```

### 4. VisualizationComponents

Specialized components for different visualization elements.

```python
@define(kw_only=True)
class MeshComponent:
    """Component for mesh visualization."""
    
    def __init__(self) -> None:
        self.m_mesh_data: Optional[pv.PolyData] = None
        self.m_actor: Optional[Any] = None
        self.m_style: str = "surface"
        self.m_color: Tuple[float, float, float] = (0.8, 0.8, 0.9)
        self.m_opacity: float = 1.0
        
    def create_mesh_data(self, vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
        """Create PyVista mesh data from vertices and faces."""
        
    def update_mesh_data(self, vertices: np.ndarray) -> None:
        """Update mesh vertices while preserving topology."""

@define(kw_only=True)
class JointComponent:
    """Component for joint visualization."""
    
    def __init__(self) -> None:
        self.m_joint_spheres: Dict[str, pv.PolyData] = {}
        self.m_joint_actors: Dict[str, Any] = {}
        self.m_joint_size: float = 0.02
        self.m_color: Tuple[float, float, float] = (1.0, 0.2, 0.2)
        
    def create_joint_spheres(self, joint_positions: np.ndarray, joint_names: List[str]) -> None:
        """Create sphere representations for joints."""
        
    def update_joint_positions(self, joint_positions: np.ndarray) -> None:
        """Update joint positions."""

@define(kw_only=True)
class BoneComponent:
    """Component for bone/skeleton visualization."""
    
    def __init__(self) -> None:
        self.m_bone_cylinders: Dict[str, pv.PolyData] = {}
        self.m_bone_actors: Dict[str, Any] = {}
        self.m_bone_radius: float = 0.01
        self.m_color: Tuple[float, float, float] = (0.2, 0.2, 1.0)
        self.m_bone_connections: List[Tuple[int, int]] = []
        
    def create_bone_structure(self, joint_positions: np.ndarray) -> None:
        """Create cylindrical bone connections between joints."""
        
    def update_bone_structure(self, joint_positions: np.ndarray) -> None:
        """Update bone structure with new joint positions."""
```

## Data Structures

### Bone Connectivity

Define standard bone connections for SMPL skeleton:

```python
# Standard SMPL bone connections (parent -> child joint indices)
SMPL_BONE_CONNECTIONS = [
    # Torso
    (0, 1),   # pelvis -> left_hip
    (0, 2),   # pelvis -> right_hip
    (0, 3),   # pelvis -> spine1
    (1, 4),   # left_hip -> left_knee
    (2, 5),   # right_hip -> right_knee
    (3, 6),   # spine1 -> spine2
    (4, 7),   # left_knee -> left_ankle
    (5, 8),   # right_knee -> right_ankle
    (6, 9),   # spine2 -> spine3
    (7, 10),  # left_ankle -> left_foot
    (8, 11),  # right_ankle -> right_foot
    (9, 12),  # spine3 -> neck
    (12, 15), # neck -> head
    
    # Arms
    (9, 13),  # spine3 -> left_collar
    (9, 14),  # spine3 -> right_collar
    (13, 16), # left_collar -> left_shoulder
    (14, 17), # right_collar -> right_shoulder
    (16, 18), # left_shoulder -> left_elbow
    (17, 19), # right_shoulder -> right_elbow
    (18, 20), # left_elbow -> left_wrist
    (19, 21), # right_elbow -> right_wrist
]

# Extended connections for SMPL-H/SMPL-X (includes hands)
SMPLH_BONE_CONNECTIONS = SMPL_BONE_CONNECTIONS + [
    # Left hand finger connections (simplified)
    # Add connections from wrist to finger roots, then finger chains
]
```

### Color Schemes

Pre-defined color schemes for different visualization modes:

```python
class ColorSchemes:
    """Pre-defined color schemes for SMPL visualization."""
    
    ANATOMICAL = {
        'mesh': (0.96, 0.87, 0.70),      # Skin tone
        'joints': (0.8, 0.2, 0.2),       # Red joints
        'bones': (0.9, 0.9, 0.9),        # White bones
        'background': (0.2, 0.2, 0.25)   # Dark background
    }
    
    TECHNICAL = {
        'mesh': (0.7, 0.7, 0.8),         # Light blue-gray
        'joints': (1.0, 0.5, 0.0),       # Orange joints
        'bones': (0.2, 0.6, 1.0),        # Blue bones
        'background': (0.1, 0.1, 0.1)    # Black background
    }
    
    PRESENTATION = {
        'mesh': (0.2, 0.4, 0.8),         # Blue mesh
        'joints': (0.8, 0.2, 0.2),       # Red joints
        'bones': (0.4, 0.4, 0.4),        # Gray bones
        'background': (1.0, 1.0, 1.0)    # White background
    }
```

## API Usage Examples

### Basic Usage

```python
import torch
import smplx
from smplx_toolbox.core.unified_model import UnifiedSmplModel, PoseByKeypoints
from smplx_toolbox.visualization import SMPLPlotter, VisualizationConfig

# Load SMPL model
base_model = smplx.create(model_path='path/to/models', model_type='smplx')
unified_model = UnifiedSmplModel.from_smpl_model(base_model)

# Create visualization
config = VisualizationConfig.presentation()
plotter = SMPLPlotter.from_model(unified_model, config=config)

# Add visualization components
pose = PoseByKeypoints(left_shoulder=torch.tensor([[0.0, 0.0, -1.5]]))
output = unified_model(pose)

plotter.add_mesh(output)
plotter.add_joints(output, show_labels=True)
plotter.add_bones(output)

# Show visualization
plot = plotter.get_plotter()
plot.show()
```

### Interactive Usage

```python
from smplx_toolbox.visualization import SMPLInteractivePlotter

# Create interactive plotter
interactive_plotter = SMPLInteractivePlotter.from_model_interactive(
    unified_model, 
    show=True
)

# Add interactive controls
interactive_plotter.add_parameter_controls()
interactive_plotter.add_pose_controls()
interactive_plotter.enable_real_time_updates()

# The plotter window is shown automatically and remains interactive
```

### Integration with Existing Code

```python
import pyvista as pv

# User creates their own plotter
user_plotter = pv.Plotter(window_size=(1200, 800))
user_plotter.add_text("My Custom SMPL Analysis", position='upper_left')

# Add custom elements
user_plotter.add_axes()
user_plotter.add_floor('-z')

# Integrate SMPL visualization
smpl_vis = SMPLPlotter.from_model(unified_model, plotter=user_plotter)
smpl_vis.add_mesh(output)
smpl_vis.add_joints(output)

# User retains full control
user_plotter.camera_position = 'xz'
user_plotter.show()
```

## Performance Considerations

### Efficient Updates

- Use `pyvista.PolyData.points` property for vertex updates without recreating meshes
- Cache PyVista objects to avoid repeated creation
- Implement level-of-detail for complex scenes

### Memory Management

- Clear unused actors when switching between models
- Use weak references for callback functions
- Implement proper cleanup in `__del__` methods

## Dependencies

### Required

- `pyvista >= 0.40.0` - Core 3D visualization
- `numpy >= 1.19.0` - Array operations
- `attrs >= 22.0.0` - Class definitions

### Optional

- `pyvistaqt >= 0.9.0` - Qt integration for interactivity
- `vtk >= 9.0.0` - Advanced VTK features
- `matplotlib >= 3.5.0` - Colormaps and color utilities

## Testing Strategy

### Unit Tests

- Test component creation and updates
- Verify mesh data conversion
- Check color scheme applications

### Integration Tests

- Test with different SMPL model types
- Verify PyVista plotter integration
- Test interactive controls (if Qt available)

### Visual Tests

- Screenshot comparisons for regression testing
- Rendering validation across platforms

## Future Extensions

### Advanced Features

- Animation playback for temporal sequences
- Multi-model comparison visualization
- Texture mapping and appearance models
- Export capabilities (images, videos, 3D formats)

### Performance Optimizations

- GPU-accelerated mesh updates
- Level-of-detail rendering
- Streaming for large datasets

This design provides a comprehensive, extensible foundation for SMPL model visualization while maintaining the project's coding standards and architectural principles.