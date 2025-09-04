"""Minimal SMPL-X model adapter.

Design goals
------------
* Do NOT duplicate or re-expose official :mod:`smplx` API surface.
* Hold ONLY a reference to a user-constructed smplx model (created externally
    via :func:`smplx.create` or direct class instantiation).
* Provide a tiny convenience for turning an existing *SMPL output* (the
    object returned by calling the model) into a :class:`trimesh.Trimesh`.
* If no output is supplied, perform a neutral forward pass (empty call) to
    obtain the canonical / identity mesh.

Non-goals / avoided features
---------------------------
* No parameter sampling, loading helpers, or vertex extraction helpers.
* No argument forwarding duplication of smplx keyword parameters.
* No silent fallbacks; errors are explicit.

Usage
-----
        import smplx
        from smplx_toolbox.core.smplx_model import SMPLXModel

        base = smplx.create("./models", model_type="smplx", gender="neutral")
        wrapper = SMPLXModel.from_smplx(base)

        # Case 1: Use precomputed output (preferred when you already did a forward)
        output = base(return_verts=True)
        mesh = wrapper.to_mesh(output)

        # Case 2: Ask wrapper to run a neutral forward (no params) for you
        neutral_mesh = wrapper.to_mesh(None)

        mesh.show()  # if viewer deps installed
"""

from __future__ import annotations

from typing import Sequence, Optional, Type, TypeVar, Dict, Any, Tuple
import torch

import numpy as np
import trimesh
import smplx
from smplx.utils import SMPLXOutput

__all__ = ["SMPLXModel"]

# Type aliases (restricted to SMPLX only)
SmplModelType = smplx.SMPLX
SmplOutput = SMPLXOutput


T = TypeVar("T", bound="SMPLXModel")


class SMPLXModel:
    """Thin wrapper referencing a user-provided SMPL/SMPLH/SMPLX model.

    This class intentionally avoids re-implementing or proxying the official
    :mod:`smplx` API. It only supplies a focused helper to obtain a
    :class:`trimesh.Trimesh` from an already computed model output, or from a
    neutral (identity) forward pass when no output is supplied.

    Notes
    -----
    * Construction requires an existing SMPL-X model via :meth:`from_smplx`.
    * Member variables follow the project convention with an ``m_`` prefix.
    * Read-only property accessors expose underlying model and faces.
    * All validation errors raise explicit exceptions.

    Unsupported Models
    ------------------
    SMPL (body-only) and SMPLH (with hands) instances are NOT accepted here.
    Convert them to SMPL-X first using the official transfer scripts:
    https://github.com/vchoutas/smplx/tree/main/transfer_model
    (e.g., see configs like `smplh2smplx.yaml`).
    """

    def __init__(self) -> None:
        """Initialize empty wrapper with no base model reference yet."""
        self.m_base: Optional[SmplModelType] = None
        self.m_cached_neutral_output: Optional[SmplOutput] = None
        self.m_joint_names: Optional[list[str]] = None
        self.m_coordinate_system: str = "y_up"  # default SMPL-X coordinate system

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_smplx(cls: Type[T], model: SmplModelType) -> T:
        """Create wrapper from an existing SMPL-X model.

        Parameters
        ----------
        model : SMPLX
            A fully constructed SMPL-X model instance.

        Returns
        -------
        SMPLXModel
            New wrapper instance referencing the given model.
        """
        # Defensive type check: enforce SMPL-X only.
        if not isinstance(model, smplx.SMPLX):  # pragma: no cover - runtime guard
            raise ValueError(
                "Only SMPL-X models are supported. Please convert your model to SMPL-X "
                "using the official transfer scripts: "
                "https://github.com/vchoutas/smplx/tree/main/transfer_model"
            )
        instance = cls()
        instance.m_base = model
        return instance

    # ------------------------------------------------------------------
    # Properties (read-only)
    # ------------------------------------------------------------------
    @property
    def base(self) -> SmplModelType:
        """Underlying smplx model (read-only).

        Raises
        ------
        RuntimeError
            If the wrapper has not been initialized with a model.
        """
        if self.m_base is None:  # pragma: no cover - defensive
            raise RuntimeError("SMPLXModel has no base model reference")
        return self.m_base

    @property
    def faces(self) -> np.ndarray:
        """Triangle faces of the underlying model."""
        return self.base.faces

    @property
    def num_joints(self) -> int:
        """Number of joints in the model."""
        return self.base.get_num_joints()

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the model."""
        return self.base.get_num_verts()

    @property
    def joint_names(self) -> list[str]:
        """Names of all joints in hierarchical order."""
        if self.m_joint_names is None:
            # SMPL-X standard joint names
            self.m_joint_names = [
                "pelvis",
                "left_hip",
                "right_hip",
                "spine1",
                "left_knee",
                "right_knee",
                "spine2",
                "left_ankle",
                "right_ankle",
                "spine3",
                "left_foot",
                "right_foot",
                "neck",
                "left_collar",
                "right_collar",
                "head",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "jaw",
                "left_eye",
                "right_eye",
                # Hand joints would follow here (if using full hand model)
            ]
        return self.m_joint_names

    @property
    def coordinate_system(self) -> str:
        """Current coordinate system ('y_up' or 'z_up')."""
        return self.m_coordinate_system

    # ------------------------------------------------------------------
    # Setter methods
    # ------------------------------------------------------------------
    def set_coordinate_system(self, system: str) -> None:
        """Set the coordinate system for vertex/joint outputs.

        Parameters
        ----------
        system : str
            Either 'y_up' (SMPL-X default) or 'z_up' (Blender/Unity).
        """
        if system not in ("y_up", "z_up"):
            raise ValueError("Coordinate system must be 'y_up' or 'z_up'")
        self.m_coordinate_system = system

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        betas: torch.Tensor | np.ndarray | None = None,
        body_pose: torch.Tensor | np.ndarray | None = None,
        global_orient: torch.Tensor | np.ndarray | None = None,
        transl: torch.Tensor | np.ndarray | None = None,
        expression: torch.Tensor | np.ndarray | None = None,
        jaw_pose: torch.Tensor | np.ndarray | None = None,
        left_hand_pose: torch.Tensor | np.ndarray | None = None,
        right_hand_pose: torch.Tensor | np.ndarray | None = None,
        return_joints: bool = False,
        **kwargs: Any,
    ) -> SmplOutput:
        """Forward pass through the SMPL-X model with parameter control.

        Parameters
        ----------
        betas : tensor or array, optional
            Shape parameters (10 values).
        body_pose : tensor or array, optional
            Body pose parameters (21 joints × 3).
        global_orient : tensor or array, optional
            Global orientation (1 × 3).
        transl : tensor or array, optional
            Global translation (3 values).
        expression : tensor or array, optional
            Facial expression parameters (10 values).
        jaw_pose : tensor or array, optional
            Jaw pose (1 × 3).
        left_hand_pose : tensor or array, optional
            Left hand pose parameters.
        right_hand_pose : tensor or array, optional
            Right hand pose parameters.
        return_joints : bool, optional
            Whether to return joint positions in output.
        **kwargs : Any
            Additional arguments passed to the underlying model.

        Returns
        -------
        SMPLXOutput
            Model output with vertices, joints, and other data.
        """

        # Convert numpy arrays to torch tensors if needed
        def to_tensor(x: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            return x

        # Prepare parameters
        params = {
            "betas": to_tensor(betas),
            "body_pose": to_tensor(body_pose),
            "global_orient": to_tensor(global_orient),
            "transl": to_tensor(transl),
            "expression": to_tensor(expression),
            "jaw_pose": to_tensor(jaw_pose),
            "left_hand_pose": to_tensor(left_hand_pose),
            "right_hand_pose": to_tensor(right_hand_pose),
            "return_verts": True,
            "return_joints": return_joints,
        }

        # Filter out None values
        params = {
            k: v for k, v in params.items() if v is not None or k.startswith("return_")
        }
        params.update(kwargs)

        # Forward pass
        output = self.base(**params)

        # Apply coordinate transformation if needed
        if self.m_coordinate_system == "z_up":
            output = self._transform_to_z_up(output)

        return output

    def _transform_to_z_up(self, output: SmplOutput) -> SmplOutput:
        """Transform output from Y-up to Z-up coordinate system.

        Parameters
        ----------
        output : SMPLXOutput
            Original model output in Y-up coordinates.

        Returns
        -------
        SMPLXOutput
            Transformed output in Z-up coordinates.
        """

        # Y-up to Z-up: swap Y and Z, negate new Y
        # Transform matrix: [[1,0,0], [0,0,1], [0,-1,0]]
        def transform_coords(tensor: torch.Tensor) -> torch.Tensor:
            if tensor is None:
                return tensor
            x, y, z = tensor[..., 0], tensor[..., 1], tensor[..., 2]
            return torch.stack([x, -z, y], dim=-1)

        # Transform vertices and joints
        if hasattr(output, "vertices") and output.vertices is not None:
            output.vertices = transform_coords(output.vertices)
        if hasattr(output, "joints") and output.joints is not None:
            output.joints = transform_coords(output.joints)

        return output

    def get_neutral_pose(self, use_cache: bool = True) -> SmplOutput:
        """Get the neutral (T-pose) output of the model.

        Parameters
        ----------
        use_cache : bool, optional
            Whether to use cached result if available (default True).

        Returns
        -------
        SMPLXOutput
            Model output in neutral pose.
        """
        if use_cache and self.m_cached_neutral_output is not None:
            return self.m_cached_neutral_output

        output = self.forward(return_joints=True)

        if use_cache:
            self.m_cached_neutral_output = output

        return output

    def get_joint_positions(
        self,
        output: SmplOutput | None = None,
        batch_index: int = 0,
    ) -> np.ndarray:
        """Extract joint positions from model output.

        Parameters
        ----------
        output : SMPLXOutput, optional
            Model output. If None, uses neutral pose.
        batch_index : int, optional
            Batch element index (default 0).

        Returns
        -------
        np.ndarray
            Joint positions of shape (num_joints, 3).
        """
        if output is None:
            output = self.get_neutral_pose()

        if not hasattr(output, "joints") or output.joints is None:
            raise ValueError(
                "No joints in output. Call forward() with return_joints=True"
            )

        joints = output.joints.detach().cpu().numpy()

        if joints.ndim == 3:
            return joints[batch_index]
        return joints

    def to_mesh(
        self,
        output: SmplOutput | None,
        *,
        batch_index: int = 0,
        combine_batch: bool = False,
        process: bool = False,
        vertex_colors: Sequence[float] | None = None,
    ) -> trimesh.Trimesh:
        """Convert a SMPL output to a :class:`trimesh.Trimesh`.

        If *output* is ``None`` a neutral forward pass (identity pose/shape)
        is executed using the underlying model with ``return_verts=True``.

        Parameters
        ----------
        output : SMPLOutput | SMPLHOutput | SMPLXOutput | None
            The previously computed output from calling the smplx model. If
            ``None`` the wrapper will invoke the model with no parameters.
        batch_index : int, optional
            Batch element to convert when not combining the batch (default 0).
        combine_batch : bool, optional
            If True concatenate all batch elements into a single mesh
            (disconnected components). Default False.
        process : bool, optional
            Forwarded to ``trimesh.Trimesh`` (default False to avoid implicit
            geometry alterations).
        vertex_colors : sequence of float, optional
            Uniform RGB or RGBA values in [0,1] applied per vertex.

        Returns
        -------
        trimesh.Trimesh
            Constructed mesh.

        Raises
        ------
        ValueError
            If vertices are missing or have unexpected shape.
        IndexError
            If *batch_index* is out of range.
        RuntimeError
            If base model reference is missing.
        """
        if output is None:
            output = self.get_neutral_pose()
        verts_t = output.vertices  # type: ignore[attr-defined]
        if verts_t is None:  # pragma: no cover - defensive
            raise ValueError(
                "Vertices not present in output (ensure return_verts=True)"
            )
        v = verts_t.detach().cpu().numpy()
        if v.ndim != 3:
            raise ValueError(f"Expected (B,V,3) vertices, got {v.shape}")
        faces = self.faces
        if combine_batch:
            parts: list[np.ndarray] = []
            all_faces: list[np.ndarray] = []
            offset = 0
            for batch_vertices in v:
                arr = np.asarray(batch_vertices, dtype=float)
                parts.append(arr)
                all_faces.append(faces + offset)
                offset += arr.shape[0]
            verts = np.vstack(parts)
            faces_cat = np.vstack(all_faces)
            mesh = trimesh.Trimesh(verts, faces_cat, process=process)
        else:
            if not (0 <= batch_index < v.shape[0]):
                raise IndexError(
                    f"batch_index {batch_index} out of range (batch={v.shape[0]})"
                )
            mesh = trimesh.Trimesh(v[batch_index], faces, process=process)
        if vertex_colors is not None:
            color = np.asarray(vertex_colors, dtype=float)
            if color.size not in (3, 4):  # pragma: no cover
                raise ValueError("vertex_colors must have length 3 or 4")
            rgba = np.ones((mesh.vertices.shape[0], 4), dtype=np.uint8)
            fc = np.zeros(4, dtype=float)
            fc[: color.size] = np.clip(color, 0.0, 1.0)
            rgba[:] = (fc * 255).astype(np.uint8)
            if mesh.visual is not None:  # type: ignore[truthy-bool]
                mesh.visual.vertex_colors = rgba  # type: ignore[attr-defined]
        return mesh

    def __repr__(self) -> str:  # pragma: no cover
        cls_name = self.m_base.__class__.__name__ if self.m_base is not None else "None"
        return f"SMPLXModel(base={cls_name})"
