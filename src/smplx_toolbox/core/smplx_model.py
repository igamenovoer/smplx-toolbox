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

        # Create base model and wrapper
        base = smplx.create("./models", model_type="smplx", gender="neutral")
        wrapper = SMPLXModel.from_smplx(base)

        # Case 1: Use precomputed output (preferred when you already did a forward)
        output = wrapper.base_model(return_verts=True)
        mesh = wrapper.to_mesh(output)

        # Case 2: Ask wrapper to run a neutral forward (no params) for you
        neutral_mesh = wrapper.to_mesh(None)

        mesh.show()  # if viewer deps installed
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import smplx
import trimesh
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

    Coordinate System
    -----------------
    This wrapper preserves the standard SMPL-X coordinate system:
    * Y-axis: up (vertical)
    * Z-axis: forward (facing direction)
    * X-axis: right (from model's perspective)
    * Units: meters

    Notes
    -----
    * Construction requires an existing SMPL-X model via :meth:`from_smplx`.
    * Member variables follow the project convention with an ``m_`` prefix.
    * Read-only property accessors expose underlying model and faces.
    * All validation errors raise explicit exceptions.
    * For forward passes, use the base_model directly: wrapper.base_model(...)

    Unsupported Models
    ------------------
    SMPL (body-only) and SMPLH (with hands) instances are NOT accepted here.
    Convert them to SMPL-X first using the official transfer scripts:
    https://github.com/vchoutas/smplx/tree/main/transfer_model
    (e.g., see configs like `smplh2smplx.yaml`).
    """

    def __init__(self) -> None:
        """Initialize empty wrapper with no base model reference yet."""
        self.m_base: SmplModelType | None = None
        self.m_joint_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_smplx(cls: type[T], model: SmplModelType) -> T:
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
    def base_model(self) -> SmplModelType:
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
        return self.base_model.faces

    @property
    def num_joints(self) -> int:
        """Number of joints in the model."""
        # SMPL-X has 55 joints by default, plus additional keypoints
        # The actual number depends on keypoint usage
        if hasattr(self.base_model, "get_num_joints"):
            return self.base_model.get_num_joints()
        elif hasattr(self.base_model, "J_regressor"):
            return self.base_model.J_regressor.shape[0]
        else:
            return 55  # SMPL-X default

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the model."""
        if hasattr(self.base_model, "get_num_verts"):
            return self.base_model.get_num_verts()
        elif hasattr(self.base_model, "v_template"):
            return self.base_model.v_template.shape[0]
        else:
            return 10475  # SMPL-X default

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_joint_positions(
        self,
        output: SmplOutput,
        batch_index: int = 0,
    ) -> np.ndarray:
        """Extract joint positions from model output.

        Parameters
        ----------
        output : SMPLXOutput
            Model output from calling the base_model.
        batch_index : int, optional
            Batch element index (default 0).

        Returns
        -------
        np.ndarray
            Joint positions of shape (num_joints, 3).
        """
        if not hasattr(output, "joints") or output.joints is None:
            raise ValueError(
                "No joints in output. Call base_model with return_joints=True"
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
            # Run neutral forward pass
            output = self.base_model(return_verts=True)
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
