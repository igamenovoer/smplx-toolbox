"""Minimal SMPL-H (SMPL+Hands) model adapter.

Design goals
------------
* Do NOT duplicate or re-expose official :mod:`smplx` API surface.
* Hold ONLY a reference to a user-constructed smplx SMPL-H model (created externally
    via :func:`smplx.create` or direct class instantiation).
* Provide a tiny convenience for turning an existing *SMPLH output* (the
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
    from smplx_toolbox.core.smplh_model import SMPLHModel

    # Create base model and wrapper
    base = smplx.create("./models", model_type="smplh", gender="neutral")
    wrapper = SMPLHModel.from_smplh(base)

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
from smplx.utils import SMPLHOutput

__all__ = ["SMPLHModel"]

# Type aliases (restricted to SMPLH only)
SmplhModelType = smplx.SMPLH
SmplhOutput = SMPLHOutput


T = TypeVar("T", bound="SMPLHModel")


class SMPLHModel:
    """Thin wrapper referencing a user-provided SMPL-H (SMPL+Hands) model.

    This class intentionally avoids re-implementing or proxying the official
    :mod:`smplx` API. It only supplies a focused helper to obtain a
    :class:`trimesh.Trimesh` from an already computed model output, or from a
    neutral (identity) forward pass when no output is supplied.

    SMPL-H vs SMPL-X
    ----------------
    * SMPL-H: Body (23 joints) + Hands (30 joints) = 52 joints total
    * SMPL-X: Body + Hands + Face (jaw, eyes) + expressions = 55 joints
    * SMPL-H has no facial expression or jaw/eye pose parameters
    * SMPL-H shares the same hand model (MANO) with SMPL-X

    Coordinate System
    -----------------
    This wrapper preserves the standard SMPL-H coordinate system:
    * Y-axis: up (vertical)
    * Z-axis: forward (facing direction)
    * X-axis: right (from model's perspective)
    * Units: meters

    Notes
    -----
    * Construction requires an existing SMPL-H model via :meth:`from_smplh`.
    * Member variables follow the project convention with an ``m_`` prefix.
    * Read-only property accessors expose underlying model and faces.
    * All validation errors raise explicit exceptions.
    * For forward passes, use the base_model directly: wrapper.base_model(...)

    Conversion from SMPL
    --------------------
    To convert SMPL to SMPL-H, you can:
    1. Copy body parameters directly (betas, body_pose, global_orient, transl)
    2. Initialize hand poses to zeros or small curl for realism
    See context/hints/smplx-kb/howto-convert-smplh-to-smplx.md for details.
    """

    def __init__(self) -> None:
        """Initialize empty wrapper with no base model reference yet."""
        self.m_base: SmplhModelType | None = None
        self.m_joint_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_smplh(cls: type[T], model: SmplhModelType) -> T:
        """Create wrapper from an existing SMPL-H model.

        Parameters
        ----------
        model : SMPLH
            A fully constructed SMPL-H model instance.

        Returns
        -------
        SMPLHModel
            New wrapper instance referencing the given model.
        """
        # Defensive type check: enforce SMPL-H only.
        if not isinstance(model, smplx.SMPLH):  # pragma: no cover - runtime guard
            raise ValueError(
                "Only SMPL-H models are supported. Got type: "
                f"{type(model).__name__}. Please create an SMPL-H model using "
                "smplx.create(model_path, model_type='smplh', ...)"
            )
        instance = cls()
        instance.m_base = model
        return instance

    # ------------------------------------------------------------------
    # Properties (read-only)
    # ------------------------------------------------------------------
    @property
    def base_model(self) -> SmplhModelType:
        """Underlying smplx SMPL-H model (read-only).

        Raises
        ------
        RuntimeError
            If the wrapper has not been initialized with a model.
        """
        if self.m_base is None:  # pragma: no cover - defensive
            raise RuntimeError("SMPLHModel has no base model reference")
        return self.m_base

    @property
    def faces(self) -> np.ndarray:
        """Triangle faces of the underlying model."""
        return self.base_model.faces

    @property
    def num_joints(self) -> int:
        """Number of joints in the model.

        SMPL-H typically has:
        - 22 body joints (SMPL body minus pelvis as it's root)
        - 30 hand joints (15 per hand from MANO)
        - Total: 52 joints (including root)
        """
        # SMPL-H has different joint counts depending on configuration
        if hasattr(self.base_model, "get_num_joints"):
            return self.base_model.get_num_joints()
        elif hasattr(self.base_model, "J_regressor"):
            return self.base_model.J_regressor.shape[0]
        else:
            # Standard SMPL-H: 22 body + 30 hand joints
            return 52

    @property
    def num_body_joints(self) -> int:
        """Number of body joints (excluding hands)."""
        # SMPL-H body joints (typically 21 or 23 depending on whether root is counted)
        if hasattr(self.base_model, "NUM_BODY_JOINTS"):
            return self.base_model.NUM_BODY_JOINTS
        return 21  # SMPL body joints minus root

    @property
    def num_hand_joints(self) -> int:
        """Number of hand joints per hand."""
        # MANO has 15 joints per hand
        if hasattr(self.base_model, "NUM_HAND_JOINTS"):
            return self.base_model.NUM_HAND_JOINTS
        return 15  # MANO default

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the model."""
        if hasattr(self.base_model, "get_num_verts"):
            return self.base_model.get_num_verts()
        elif hasattr(self.base_model, "v_template"):
            return self.base_model.v_template.shape[0]
        else:
            return 6890  # SMPL-H default (same as SMPL)

    @property
    def joint_names(self) -> list[str]:
        """Names of all joints in hierarchical order.

        Returns body joints followed by left hand joints, then right hand joints.
        """
        if self.m_joint_names is None:
            # SMPL-H joint names: body + hands
            self.m_joint_names = [
                # Body joints (SMPL structure)
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
                # Left hand joints (MANO)
                "left_index1",
                "left_index2",
                "left_index3",
                "left_middle1",
                "left_middle2",
                "left_middle3",
                "left_pinky1",
                "left_pinky2",
                "left_pinky3",
                "left_ring1",
                "left_ring2",
                "left_ring3",
                "left_thumb1",
                "left_thumb2",
                "left_thumb3",
                # Right hand joints (MANO)
                "right_index1",
                "right_index2",
                "right_index3",
                "right_middle1",
                "right_middle2",
                "right_middle3",
                "right_pinky1",
                "right_pinky2",
                "right_pinky3",
                "right_ring1",
                "right_ring2",
                "right_ring3",
                "right_thumb1",
                "right_thumb2",
                "right_thumb3",
            ]
        return self.m_joint_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_joint_positions(
        self,
        output: SmplhOutput,
        batch_index: int = 0,
    ) -> np.ndarray:
        """Extract joint positions from model output.

        Parameters
        ----------
        output : SMPLHOutput
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

    def get_body_joint_positions(
        self,
        output: SmplhOutput,
        batch_index: int = 0,
    ) -> np.ndarray:
        """Extract only body joint positions (excluding hands).

        Parameters
        ----------
        output : SMPLHOutput
            Model output from calling the base_model.
        batch_index : int, optional
            Batch element index (default 0).

        Returns
        -------
        np.ndarray
            Body joint positions of shape (22, 3).
        """
        joints = self.get_joint_positions(output, batch_index)
        # Body joints are typically the first 22 joints
        return joints[:22]

    def get_hand_joint_positions(
        self,
        output: SmplhOutput,
        batch_index: int = 0,
        hand: str = "both",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Extract hand joint positions.

        Parameters
        ----------
        output : SMPLHOutput
            Model output from calling the base_model.
        batch_index : int, optional
            Batch element index (default 0).
        hand : str, optional
            Which hand(s) to return: "left", "right", or "both" (default).

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            If hand="left" or "right": Joint positions of shape (15, 3).
            If hand="both": Tuple of (left_joints, right_joints).
        """
        joints = self.get_joint_positions(output, batch_index)

        # Hand joints typically start after body joints (index 22)
        left_hand_start = 22
        left_hand_end = left_hand_start + 15
        right_hand_start = left_hand_end
        right_hand_end = right_hand_start + 15

        left_hand_joints = joints[left_hand_start:left_hand_end]
        right_hand_joints = joints[right_hand_start:right_hand_end]

        if hand == "left":
            return left_hand_joints
        elif hand == "right":
            return right_hand_joints
        elif hand == "both":
            return left_hand_joints, right_hand_joints
        else:
            raise ValueError(
                f"Invalid hand option: {hand}. Use 'left', 'right', or 'both'."
            )

    def to_mesh(
        self,
        output: SmplhOutput | None,
        *,
        batch_index: int = 0,
        combine_batch: bool = False,
        process: bool = False,
        vertex_colors: Sequence[float] | None = None,
    ) -> trimesh.Trimesh:
        """Convert a SMPL-H output to a :class:`trimesh.Trimesh`.

        If *output* is ``None`` a neutral forward pass (identity pose/shape)
        is executed using the underlying model with ``return_verts=True``.

        Parameters
        ----------
        output : SMPLHOutput | None
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
