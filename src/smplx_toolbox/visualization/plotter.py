"""Main SMPL visualization class.

This module provides the SMPLVisualizer class for adding SMPL model visualizations
to PyVista plotters.

Classes
-------
SMPLVisualizer : Main SMPL visualization class
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

import numpy as np
import pyvista as pv
import torch

from smplx_toolbox.core.constants import ModelType
from smplx_toolbox.core.containers import UnifiedSmplInputs, UnifiedSmplOutput
from smplx_toolbox.core.unified_model import UnifiedSmplModel
from .config import VisualizationConfig
from .utils import (
    create_polydata_from_vertices_faces,
    get_smpl_bone_connections,
    get_smplh_bone_connections,
    get_smplx_bone_connections,
    resolve_joint_selection,
)

T = TypeVar("T", bound="SMPLVisualizer")


class SMPLVisualizer:
    """Add SMPL model visualizations to PyVista plotters.

    Takes a unified SMPL model and adds SMPL-specific visualization elements
    (mesh, selected joints, skeleton) to a PyVista plotter. The plotter can be
    user-provided or created internally. Returns actors for user management.

    For generic visualization (arbitrary points, lines, meshes), use PyVista's
    API directly on the plotter obtained via get_plotter().

    Attributes
    ----------
    plotter : pv.Plotter or BackgroundPlotter or None
        The PyVista plotter instance
    model : UnifiedSmplModel or None
        The SMPL model being visualized

    Examples
    --------
    >>> from smplx_toolbox.core.unified_model import UnifiedSmplModel
    >>> from smplx_toolbox.visualization import SMPLVisualizer
    >>>
    >>> # Create visualizer with model
    >>> viz = SMPLVisualizer.from_model(model)
    >>>
    >>> # Add visualizations
    >>> output = model.forward(pose_params)
    >>> mesh_actor = viz.add_mesh(output)
    >>> joint_actors = viz.add_smpl_joints(output, labels=True)
    >>> skeleton_actor = viz.add_smpl_skeleton(output)
    >>>
    >>> # Get plotter and display
    >>> plotter = viz.get_plotter()
    >>> plotter.show()
    """

    def __init__(self) -> None:
        """Initialize empty visualizer.

        All member variables are initialized to None and can be configured
        later using setter methods or factory methods.
        """
        self.m_plotter: pv.Plotter | Any | None = None
        self.m_model: UnifiedSmplModel | None = None
        self.m_config: VisualizationConfig | None = None

    @property
    def plotter(self) -> pv.Plotter | Any | None:
        """Get the PyVista plotter instance (regular or BackgroundPlotter).

        Returns
        -------
        pv.Plotter or BackgroundPlotter or None
            The plotter instance if set, None otherwise
        """
        return self.m_plotter

    @property
    def model(self) -> UnifiedSmplModel | None:
        """Get the SMPL model.

        Returns
        -------
        UnifiedSmplModel or None
            The SMPL model if set, None otherwise
        """
        return self.m_model

    def set_plotter(self, plotter: pv.Plotter | Any) -> None:
        """Set PyVista plotter to use for visualization.

        Parameters
        ----------
        plotter : pv.Plotter or pyvistaqt.BackgroundPlotter
            The plotter to add visualizations to
        """
        self.m_plotter = plotter

    def set_model(self, model: UnifiedSmplModel) -> None:
        """Set the SMPL model to visualize.

        Parameters
        ----------
        model : UnifiedSmplModel
            The unified SMPL model
        """
        self.m_model = model

    @classmethod
    def from_model(
        cls: Type[T],
        model: UnifiedSmplModel,
        plotter: pv.Plotter | Any | None = None,
        background: bool = False,
    ) -> T:
        """Create visualizer from SMPL model.

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

        Examples
        --------
        >>> # Create with regular plotter
        >>> viz = SMPLVisualizer.from_model(model)
        >>>
        >>> # Create with background plotter
        >>> viz = SMPLVisualizer.from_model(model, background=True)
        >>>
        >>> # Use existing plotter
        >>> my_plotter = pv.Plotter()
        >>> viz = SMPLVisualizer.from_model(model, plotter=my_plotter)
        """
        instance = cls()
        instance.set_model(model)
        instance.m_config = VisualizationConfig()

        if plotter is not None:
            instance.set_plotter(plotter)
        else:
            if background:
                # Try to import pyvistaqt for background plotter
                try:
                    import pyvistaqt as pvqt

                    instance.m_plotter = pvqt.BackgroundPlotter()
                except ImportError:
                    import warnings

                    warnings.warn(
                        "pyvistaqt not installed, falling back to regular plotter. "
                        "Install with: pip install pyvistaqt",
                        UserWarning,
                    )
                    instance.m_plotter = pv.Plotter()
            else:
                instance.m_plotter = pv.Plotter()

        return instance

    def add_mesh(self, output: UnifiedSmplOutput | None = None, **kwargs: Any) -> Any:
        """Add SMPL mesh to the plotter.

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

        Raises
        ------
        ValueError
            If neither output nor model is provided

        Examples
        --------
        >>> # Add mesh with default settings
        >>> actor = viz.add_mesh(output)
        >>>
        >>> # Add mesh with custom appearance
        >>> actor = viz.add_mesh(output, color='lightblue',
        ...                      opacity=0.9, show_edges=True)
        """
        if self.m_plotter is None:
            raise ValueError("Plotter not set. Use set_plotter() or from_model()")

        if output is None:
            if self.m_model is None:
                raise ValueError("Either output or model must be provided")
            # Generate default output with neutral pose
            default_inputs = UnifiedSmplInputs()
            output = self.m_model.forward(default_inputs)

        # Convert vertices and faces to numpy if they're tensors
        if isinstance(output.vertices, torch.Tensor):
            vertices = output.vertices[0].detach().cpu().numpy()  # Take first batch
        else:
            vertices = np.asarray(output.vertices)
            if vertices.ndim == 3:
                vertices = vertices[0]  # Take first batch

        if isinstance(output.faces, torch.Tensor):
            faces = output.faces.detach().cpu().numpy()
        else:
            faces = np.asarray(output.faces)

        # Create PyVista mesh
        mesh = create_polydata_from_vertices_faces(vertices, faces)

        # Set default color if not provided
        if "color" not in kwargs and self.m_config is not None:
            kwargs["color"] = self.m_config.mesh_color

        # Add to plotter
        actor = self.m_plotter.add_mesh(mesh, **kwargs)
        return actor

    def add_smpl_joints(
        self,
        output: UnifiedSmplOutput | None = None,
        joints: list[str] | list[int] | None = None,
        size: float = 0.02,
        color: Any | None = None,
        labels: bool = False,
        label_font_size: int = 12,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add SMPL joints to the plotter with optional selection and labels.

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

        Raises
        ------
        ValueError
            If neither output nor model is provided

        Examples
        --------
        >>> # Visualize all joints
        >>> actors = viz.add_smpl_joints(output)
        >>>
        >>> # Visualize only core body joints
        >>> actors = viz.add_smpl_joints(output, joints=['body'])
        >>>
        >>> # Visualize specific joints by name with labels
        >>> actors = viz.add_smpl_joints(output,
        ...                              joints=['left_wrist', 'right_wrist'],
        ...                              labels=True, label_font_size=16)
        >>>
        >>> # Visualize joints by index
        >>> actors = viz.add_smpl_joints(output, joints=[20, 21])
        """
        if self.m_plotter is None:
            raise ValueError("Plotter not set. Use set_plotter() or from_model()")

        if output is None:
            if self.m_model is None:
                raise ValueError("Either output or model must be provided")
            # Generate default output with neutral pose
            default_inputs = UnifiedSmplInputs()
            output = self.m_model.forward(default_inputs)

        if self.m_model is None:
            raise ValueError("Model must be set to resolve joint selection")

        # Get joint indices to visualize
        joint_indices = resolve_joint_selection(joints, self.m_model.model_type)

        # Extract joint positions
        if isinstance(output.joints, torch.Tensor):
            all_joints = output.joints[0].detach().cpu().numpy()  # Take first batch
        else:
            all_joints = np.asarray(output.joints)
            if all_joints.ndim == 3:
                all_joints = all_joints[0]  # Take first batch

        # Select requested joints
        selected_joints = all_joints[joint_indices]

        # Set default color
        if color is None and self.m_config is not None:
            color = self.m_config.joint_color

        # Add points
        kwargs["point_size"] = (
            size * 1000
        )  # PyVista expects point_size in different scale
        points_actor = self.m_plotter.add_points(selected_joints, color=color, **kwargs)

        result = {"points": points_actor}

        # Add labels if requested
        if labels:
            # Get joint names
            model_type = self.m_model.model_type
            if model_type == ModelType.SMPL:
                from smplx_toolbox.core.constants import get_smpl_joint_names

                all_joint_names = get_smpl_joint_names()
            elif model_type == ModelType.SMPLH:
                from smplx_toolbox.core.constants import get_smplh_joint_names

                all_joint_names = get_smplh_joint_names()
            else:  # SMPLX
                from smplx_toolbox.core.constants import get_smplx_joint_names

                all_joint_names = get_smplx_joint_names()

            # Get names for selected joints
            joint_labels: list[str | int] = [
                all_joint_names[idx] for idx in joint_indices
            ]

            # Add labels
            labels_actor = self.m_plotter.add_point_labels(
                selected_joints,
                joint_labels,
                font_size=label_font_size,
                point_size=0,  # Don't show points again
                shape_opacity=0,  # Hide label background
            )
            result["labels"] = labels_actor

        return result

    def add_smpl_skeleton(
        self,
        output: UnifiedSmplOutput | None = None,
        connections: list[tuple[int, int]] | None = None,
        radius: float = 0.005,
        color: Any | None = None,
        as_lines: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Add SMPL bone skeleton to the plotter.

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

        Raises
        ------
        ValueError
            If neither output nor model is provided

        Examples
        --------
        >>> # Default skeleton for model type
        >>> actor = viz.add_smpl_skeleton(output)
        >>>
        >>> # Custom connections
        >>> connections = [(0, 1), (0, 2), (0, 3)]  # pelvis to hips and spine
        >>> actor = viz.add_smpl_skeleton(output, connections=connections)
        >>>
        >>> # Fast line rendering
        >>> actor = viz.add_smpl_skeleton(output, as_lines=True, line_width=3)
        """
        if self.m_plotter is None:
            raise ValueError("Plotter not set. Use set_plotter() or from_model()")

        if output is None:
            if self.m_model is None:
                raise ValueError("Either output or model must be provided")
            # Generate default output with neutral pose
            default_inputs = UnifiedSmplInputs()
            output = self.m_model.forward(default_inputs)

        if self.m_model is None:
            raise ValueError("Model must be set to determine skeleton structure")

        # Get default connections if not provided
        if connections is None:
            model_type = self.m_model.model_type
            if model_type == ModelType.SMPL:
                connections = get_smpl_bone_connections()
            elif model_type == ModelType.SMPLH:
                connections = get_smplh_bone_connections()
            else:  # SMPLX
                connections = get_smplx_bone_connections()

        # Extract joint positions
        if isinstance(output.joints, torch.Tensor):
            joints = output.joints[0].detach().cpu().numpy()  # Take first batch
        else:
            joints = np.asarray(output.joints)
            if joints.ndim == 3:
                joints = joints[0]  # Take first batch

        # Set default color
        if color is None and self.m_config is not None:
            color = self.m_config.bone_color

        if as_lines:
            # Create lines for fast rendering
            lines = []
            for parent_idx, child_idx in connections:
                # Check if indices are valid for the model
                if parent_idx < len(joints) and child_idx < len(joints):
                    lines.append([2, parent_idx, child_idx])

            if lines:
                # Create line polydata
                lines_array = np.hstack(lines)
                skeleton_mesh = pv.PolyData(joints)
                skeleton_mesh.lines = lines_array

                # Add as lines
                if "line_width" not in kwargs:
                    kwargs["line_width"] = 3
                actor = self.m_plotter.add_mesh(skeleton_mesh, color=color, **kwargs)
            else:
                actor = None
        else:
            # Create cylinders for each bone
            combined_mesh = None

            for parent_idx, child_idx in connections:
                # Check if indices are valid
                if parent_idx < len(joints) and child_idx < len(joints):
                    start = joints[parent_idx]
                    end = joints[child_idx]

                    # Create cylinder between joints
                    cylinder = pv.Cylinder(
                        center=(start + end) / 2,
                        direction=end - start,
                        height=float(np.linalg.norm(end - start)),
                        radius=radius,
                    )

                    if combined_mesh is None:
                        combined_mesh = cylinder
                    else:
                        combined_mesh = combined_mesh.merge(cylinder)

            if combined_mesh is not None:
                actor = self.m_plotter.add_mesh(combined_mesh, color=color, **kwargs)
            else:
                actor = None

        return actor

    def get_plotter(self) -> pv.Plotter | Any:
        """Get the configured PyVista plotter for display or further customization.

        Users can add custom visualizations using PyVista's API:
        - plotter.add_points() for arbitrary points
        - plotter.add_mesh() for custom meshes or lines
        - plotter.add_text() for annotations
        - etc.

        Returns
        -------
        pv.Plotter or pyvistaqt.BackgroundPlotter
            The plotter with all added visualizations

        Raises
        ------
        ValueError
            If plotter has not been set

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
        if self.m_plotter is None:
            raise ValueError("Plotter not set. Use set_plotter() or from_model()")
        return self.m_plotter
