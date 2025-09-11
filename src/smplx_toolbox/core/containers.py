"""Data containers for the Unified SMPL Model system.

This module contains the input, output, and pose specification containers used by
the unified SMPL family model implementation. These classes provide structured,
type-safe interfaces for model inputs and outputs.

Classes
-------
UnifiedSmplInputs
    Standardized input container for model forward pass
NamedPose
    Lightweight accessor around packed pose `(B, N, 3)` using ModelType
UnifiedSmplOutput
    Standardized output container from model forward pass
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch
from attrs import define, field, fields
from torch import Tensor

from .constants import (
    ModelType,
    get_smplh_joint_names,
    get_smplx_joint_names,
    CoreBodyJoint,
)


@define(kw_only=True)
class UnifiedSmplInputs:
    """Standardized input container for the unified SMPL model forward pass.

    Attributes
    ----------
    root_orient : torch.Tensor, optional
        Global orientation of the pelvis in axis-angle format (B, 3).
    pose_body : torch.Tensor, optional
        Pose of the 21 body joints in axis-angle format (B, 63).
    left_hand_pose : torch.Tensor, optional
        Pose of the 15 left hand finger joints in axis-angle format (B, 45).
    right_hand_pose : torch.Tensor, optional
        Pose of the 15 right hand finger joints in axis-angle format (B, 45).
    pose_jaw : torch.Tensor, optional
        Pose of the jaw joint in axis-angle format (B, 3). SMPL-X only.
    left_eye_pose : torch.Tensor, optional
        Pose of the left eyeball in axis-angle format (B, 3). SMPL-X only.
    right_eye_pose : torch.Tensor, optional
        Pose of the right eyeball in axis-angle format (B, 3). SMPL-X only.
    betas : torch.Tensor, optional
        Shape parameters of the model (B, n_betas).
    expression : torch.Tensor, optional
        Expression parameters for the face (B, n_expr). SMPL-X only.
    trans : torch.Tensor, optional
        Global translation of the model (B, 3).
    v_template : torch.Tensor, optional
        Custom template vertices to use instead of the model's default (B, V, 3).
    joints_override : torch.Tensor, optional
        Custom joint positions to override the model's computed joints (B, J*, 3).
    v_shaped : torch.Tensor, optional
        Pre-shaped vertices to bypass the shape-dependent part of the model (B, V, 3).

    Notes
    -----
    All tensor fields are optional and should have the batch dimension first (B, ...).
    Missing pose segments are automatically filled with zeros of the expected size for the
    model type during the forward pass.
    """

    # Pose parameters (axis-angle in radians)
    root_orient: Tensor | None = None  # (B, 3) - pelvis/global orientation
    pose_body: Tensor | None = None  # (B, 63) - 21 body joints * 3
    left_hand_pose: Tensor | None = None  # (B, 45) - 15 finger joints * 3
    right_hand_pose: Tensor | None = None  # (B, 45) - 15 finger joints * 3
    pose_jaw: Tensor | None = None  # (B, 3) - jaw joint (SMPL-X only)
    left_eye_pose: Tensor | None = None  # (B, 3) - left eyeball (SMPL-X only)
    right_eye_pose: Tensor | None = None  # (B, 3) - right eyeball (SMPL-X only)

    # Shape and expression
    betas: Tensor | None = None  # (B, n_betas) - shape parameters
    expression: Tensor | None = None  # (B, n_expr) - facial expression (SMPL-X only)
    hand_betas: Tensor | None = (
        None  # (B, H) - optional MANO hand shape (SMPL-H MANO variant)
    )
    use_hand_pca: bool | None = None  # Hint: whether hands are PCA in the base model
    num_hand_pca_comps: int | None = None  # Hint: number of PCA comps if PCA is used

    # Translation
    trans: Tensor | None = None  # (B, 3) - global translation

    # Advanced (may be ignored by some models)
    v_template: Tensor | None = None  # (B, V, 3) - custom template vertices
    joints_override: Tensor | None = None  # (B, J*, 3) - custom joint positions
    v_shaped: Tensor | None = None  # (B, V, 3) - shaped vertices

    @property
    def hand_pose(self) -> Tensor | None:
        """Concatenated left and right hand poses.

        Returns
        -------
        torch.Tensor or None
            A tensor of shape (B, 90) if both `left_hand_pose` and
            `right_hand_pose` are present, otherwise None.
        """
        if self.left_hand_pose is not None and self.right_hand_pose is not None:
            return torch.cat([self.left_hand_pose, self.right_hand_pose], dim=-1)
        return None

    @property
    def eyes_pose(self) -> Tensor | None:
        """Concatenated left and right eye poses.

        Returns
        -------
        torch.Tensor or None
            A tensor of shape (B, 6) if both `left_eye_pose` and
            `right_eye_pose` are present, otherwise None.
        """
        if self.left_eye_pose is not None and self.right_eye_pose is not None:
            return torch.cat([self.left_eye_pose, self.right_eye_pose], dim=-1)
        return None

    @classmethod
    def from_kwargs(cls, **kwargs) -> UnifiedSmplInputs:
        """Create an instance from keyword arguments."""
        return cls(**kwargs)

    def batch_size(self) -> int | None:
        """Infer the batch size from the first non-None tensor attribute.

        Returns
        -------
        int or None
            The batch size if it can be inferred, otherwise None.
        """
        for f in fields(UnifiedSmplInputs):
            value = getattr(self, f.name)
            if value is not None and isinstance(value, Tensor):
                return value.shape[0]
        return None

    def check_valid(
        self,
        model_type: ModelType | str,
        *,
        num_betas: int | None = None,
        num_expressions: int | None = None,
    ) -> None:
        """Verify that tensor presence and shapes are consistent with the model type.

        Parameters
        ----------
        model_type : ModelType
            The target model type to validate against ('smpl', 'smplh', 'smplx').
        num_betas : int, optional
            The number of shape parameters expected by the model.
        num_expressions : int, optional
            The number of expression parameters expected by the model (SMPL-X only).

        Raises
        ------
        ValueError
            If any of the inputs are incompatible with the specified model type.
        """
        batch_size = self.batch_size()

        # Common shape checks
        if self.root_orient is not None and self.root_orient.shape != (batch_size, 3):
            raise ValueError(
                f"root_orient must be (B, 3), got {self.root_orient.shape}"
            )

        if self.pose_body is not None and self.pose_body.shape != (batch_size, 63):
            raise ValueError(f"pose_body must be (B, 63), got {self.pose_body.shape}")

        if self.trans is not None and self.trans.shape != (batch_size, 3):
            raise ValueError(f"trans must be (B, 3), got {self.trans.shape}")

        # Validate betas shape if provided
        if self.betas is not None and num_betas is not None:
            if self.betas.shape[1] != num_betas:
                raise ValueError(
                    f"betas shape mismatch: got {self.betas.shape[1]} parameters, "
                    f"model expects {num_betas}"
                )

        # Model-specific validation
        if model_type == "smpl":
            # SMPL: no hands, face, or expression
            if self.left_hand_pose is not None or self.right_hand_pose is not None:
                raise ValueError("SMPL does not support hand poses")
            if self.pose_jaw is not None:
                raise ValueError("SMPL does not support jaw pose")
            if self.left_eye_pose is not None or self.right_eye_pose is not None:
                raise ValueError("SMPL does not support eye poses")
            if self.expression is not None:
                raise ValueError("SMPL does not support facial expressions")
            if self.hand_betas is not None:
                # SMPL has no separate hand shape space
                raise ValueError("SMPL does not support hand_betas")

        elif model_type == "smplh":
            # SMPL-H: requires both hands if any provided, no face
            has_left = self.left_hand_pose is not None
            has_right = self.right_hand_pose is not None

            if has_left != has_right:
                raise ValueError(
                    "SMPL-H requires both left and right hand poses or neither"
                )

            if self.left_hand_pose is not None and self.left_hand_pose.shape != (
                batch_size,
                45,
            ):
                raise ValueError(
                    f"left_hand_pose must be (B, 45), got {self.left_hand_pose.shape}"
                )
            if self.right_hand_pose is not None and self.right_hand_pose.shape != (
                batch_size,
                45,
            ):
                raise ValueError(
                    f"right_hand_pose must be (B, 45), got {self.right_hand_pose.shape}"
                )

            if self.pose_jaw is not None:
                raise ValueError("SMPL-H does not support jaw pose")
            if self.left_eye_pose is not None or self.right_eye_pose is not None:
                raise ValueError("SMPL-H does not support eye poses")
            if self.expression is not None:
                raise ValueError("SMPL-H does not support facial expressions")
            # hand_betas can be present for MANO-variant models; allow silently

        elif model_type == "smplx":
            # SMPL-X: if hands provided, both required; same for eyes
            has_left_hand = self.left_hand_pose is not None
            has_right_hand = self.right_hand_pose is not None
            has_left_eye = self.left_eye_pose is not None
            has_right_eye = self.right_eye_pose is not None

            if has_left_hand != has_right_hand:
                raise ValueError(
                    "SMPL-X requires both left and right hand poses or neither"
                )
            if has_left_eye != has_right_eye:
                raise ValueError(
                    "SMPL-X requires both left and right eye poses or neither"
                )

            if self.left_hand_pose is not None and self.left_hand_pose.shape != (
                batch_size,
                45,
            ):
                raise ValueError(
                    f"left_hand_pose must be (B, 45), got {self.left_hand_pose.shape}"
                )
            if self.right_hand_pose is not None and self.right_hand_pose.shape != (
                batch_size,
                45,
            ):
                raise ValueError(
                    f"right_hand_pose must be (B, 45), got {self.right_hand_pose.shape}"
                )

            if self.pose_jaw is not None and self.pose_jaw.shape != (batch_size, 3):
                raise ValueError(f"pose_jaw must be (B, 3), got {self.pose_jaw.shape}")

            if self.left_eye_pose is not None and self.left_eye_pose.shape != (
                batch_size,
                3,
            ):
                raise ValueError(
                    f"left_eye_pose must be (B, 3), got {self.left_eye_pose.shape}"
                )
            if self.right_eye_pose is not None and self.right_eye_pose.shape != (
                batch_size,
                3,
            ):
                raise ValueError(
                    f"right_eye_pose must be (B, 3), got {self.right_eye_pose.shape}"
                )

            # Validate expression shape if provided
            if self.expression is not None and num_expressions is not None:
                if self.expression.shape[1] != num_expressions:
                    raise ValueError(
                        f"expression shape mismatch: got {self.expression.shape[1]} parameters, "
                        f"model expects {num_expressions}"
                    )
            # hand_betas ignored in SMPL-X models


    # ------------------------------------------------------------------
    # Conversion methods (produce per-family input dicts in AA space)
    # The wrapper will finalize device/dtype, padding, PCA conversion, and
    # pad/truncate betas/expressions.
    # ------------------------------------------------------------------
    def to_smpl_inputs(self) -> dict[str, Tensor | bool]:
        """Convert to SMPL-friendly inputs (axis-angle, no hands/face).

        Returns
        -------
        dict[str, Tensor | bool]
            Dictionary with keys suitable for smplx.SMPL forward. Body pose is
            left as 63-DoF (21x3). The wrapper pads to 69 as needed.
        """
        out: dict[str, Tensor | bool] = {
            "global_orient": self.root_orient
            if self.root_orient is not None
            else torch.zeros((self.batch_size() or 1, 3)),
            "body_pose": self.pose_body
            if self.pose_body is not None
            else torch.zeros((self.batch_size() or 1, 63)),
            "return_verts": True,
        }
        if self.betas is not None:
            out["betas"] = self.betas
        if self.trans is not None:
            out["transl"] = self.trans
        return out

    def to_smplh_inputs(self, with_hand_shape: bool) -> dict[str, Tensor | bool]:
        """Convert to SMPL-H-friendly inputs (axis-angle hands).

        Parameters
        ----------
        with_hand_shape : bool
            If True and `hand_betas` is present and supported, include it.

        Returns
        -------
        dict[str, Tensor | bool]
            Dictionary with body pose (63-DoF) and hand AA(45) if present.
        """
        out: dict[str, Tensor | bool] = self.to_smpl_inputs()
        # Add hands if available
        if self.left_hand_pose is not None:
            out["left_hand_pose"] = self.left_hand_pose
        if self.right_hand_pose is not None:
            out["right_hand_pose"] = self.right_hand_pose
        # Optional MANO hand shape (wrapper will filter if unsupported)
        if with_hand_shape and self.hand_betas is not None:
            out["hand_betas"] = self.hand_betas  # type: ignore[assignment]
        return out

    def to_smplx_inputs(self) -> dict[str, Tensor | bool]:
        """Convert to SMPL-X-friendly inputs (axis-angle hands + face).

        Returns
        -------
        dict[str, Tensor | bool]
            Dictionary with body, hands, jaw, eyes, betas/expressions when present.
        """
        out: dict[str, Tensor | bool] = self.to_smplh_inputs(with_hand_shape=False)
        # Face/eyes
        if self.pose_jaw is not None:
            out["jaw_pose"] = self.pose_jaw
        if self.left_eye_pose is not None:
            out["leye_pose"] = self.left_eye_pose
        if self.right_eye_pose is not None:
            out["reye_pose"] = self.right_eye_pose
        if self.expression is not None:
            out["expression"] = self.expression
        return out


    # PoseByKeypoints has been removed. Use NamedPose for inspection/editing.


@define(kw_only=True)
class NamedPose:
    """Lightweight accessor around a packed axis-angle pose `(B, N, 3)`.

    This utility interprets a packed pose tensor using the model type's joint
    namespace and provides convenience getters and setters by joint name.

    Parameters
    ----------
    model_type : ModelType
        The SMPL family model type (`SMPL`, `SMPLH`, `SMPLX`).
    packed_pose : torch.Tensor, optional
        Packed axis-angle pose of shape `(B, N, 3)`. If omitted, a zero tensor
        is allocated with `batch_size` and the appropriate `N` for `model_type`.
    batch_size : int, optional
        The batch size used when allocating `packed_pose` if it is not
        provided. Defaults to 1.

    Notes
    -----
    - Getters return `None` for unknown joint names.
    - Setters raise `KeyError` for unknown joint names and `ValueError` for
      invalid shapes. `(B, 3)` inputs are reshaped to `(B, 1, 3)` automatically.
    - `repeat(n)` replicates the batch in-place to size `B * n`.
    """

    model_type: ModelType
    packed_pose: Tensor | None = None
    batch_size: int = 1

    # Internal mapping caches
    _name_to_index: dict[str, int] = field(init=False, factory=dict)
    _index_to_name: list[str] = field(init=False, factory=list)

    def __attrs_post_init__(self) -> None:
        # Build joint name order and mapping based on model_type
        if self.model_type == ModelType.SMPL:
            names = [j.value for j in CoreBodyJoint]
        elif self.model_type == ModelType.SMPLH:
            names = get_smplh_joint_names()
        elif self.model_type == ModelType.SMPLX:
            names = get_smplx_joint_names()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._index_to_name = list(names)
        self._name_to_index = {name: i for i, name in enumerate(self._index_to_name)}

        expected_n = len(self._index_to_name)

        if self.packed_pose is None:
            # Allocate zeros on CPU float32 by default
            self.packed_pose = torch.zeros(
                (self.batch_size, expected_n, 3), dtype=torch.float32
            )
        else:
            # Validate provided shape
            if self.packed_pose.ndim != 3 or self.packed_pose.shape[2] != 3:
                raise ValueError(
                    "packed_pose must have shape (B, N, 3); got "
                    f"{tuple(self.packed_pose.shape)}"
                )
            if self.packed_pose.shape[1] != expected_n:
                raise ValueError(
                    f"packed_pose N mismatch for {self.model_type}: expected {expected_n}, "
                    f"got {self.packed_pose.shape[1]}"
                )

    # -----------------
    # Convenience API
    # -----------------
    def get_joint_pose(self, name: str) -> Tensor | None:
        """Get a copy of the joint pose `(B, 1, 3)` for the given name.

        Parameters
        ----------
        name : str
            Joint name within the current model type's namespace.

        Returns
        -------
        torch.Tensor or None
            A detached clone of shape `(B, 1, 3)` if the joint exists, else None.
        """
        idx = self.get_joint_index(name)
        if idx is None:
            return None
        return self.packed_pose[:, idx : idx + 1, :].detach().clone()  # type: ignore[index]

    def set_joint_pose(self, name: str, pose: Tensor) -> bool:
        """Set the joint pose by name.

        Accepts `(B, 1, 3)` or `(B, 3)` and reshapes `(B, 3) -> (B, 1, 3)`
        automatically. Copies values under `torch.no_grad()`.

        Parameters
        ----------
        name : str
            Joint name within the current model type's namespace.
        pose : torch.Tensor
            Pose tensor of shape `(B, 1, 3)` or `(B, 3)`.

        Returns
        -------
        bool
            True if the joint was set.

        Raises
        ------
        KeyError
            If the joint name is not recognized for the current model type.
        ValueError
            If the provided pose shape is invalid.
        """
        idx = self.get_joint_index(name)
        if idx is None:
            raise KeyError(
                f"Unknown joint '{name}' for model type {self.model_type.value}"
            )

        B = self.packed_pose.shape[0]  # type: ignore[union-attr]
        target = pose
        if target.ndim == 2 and target.shape == (B, 3):
            target = target.view(B, 1, 3)
        elif target.ndim == 3 and target.shape == (B, 1, 3):
            pass
        else:
            raise ValueError(
                f"pose must be (B, 3) or (B, 1, 3); got {tuple(target.shape)}"
            )

        with torch.no_grad():
            self.packed_pose[:, idx : idx + 1, :].copy_(  # type: ignore[index]
                target.to(device=self.packed_pose.device, dtype=self.packed_pose.dtype)  # type: ignore[union-attr]
            )
        return True

    def to_dict(self) -> dict[str, Tensor]:
        """Get a mapping from joint name to a view of `(B, 1, 3)`.

        Notes
        -----
        The returned tensors are views into `packed_pose`. In-place edits will
        mutate `packed_pose` directly.
        """
        return {
            name: self.packed_pose[:, i : i + 1, :]  # type: ignore[index]
            for i, name in enumerate(self._index_to_name)
        }

    # -----------------
    # Name/index helpers
    # -----------------
    def get_joint_index(self, name: str) -> int | None:
        """Get the 0-based joint index for `name`, or None if not found."""
        return self._name_to_index.get(name)

    def get_joint_indices(self, names: list[str]) -> list[int | None]:
        """Vectorized variant of :meth:`get_joint_index`."""
        return [self.get_joint_index(n) for n in names]

    def get_joint_name(self, index: int) -> str:
        """Get the canonical joint name for the given index.

        Raises
        ------
        IndexError
            If the index is out of range for the current model type.
        """
        if index < 0 or index >= len(self._index_to_name):
            raise IndexError(
                f"Index {index} out of range for model type {self.model_type.value}"
            )
        return self._index_to_name[index]

    def get_joint_names(self, indices: list[int]) -> list[str]:
        """Vectorized variant of :meth:`get_joint_name`."""
        return [self.get_joint_name(i) for i in indices]

    # -----------------
    # Batch utilities
    # -----------------
    def repeat(self, n: int) -> None:
        """Repeat the batch in-place to size `B * n`.

        Parameters
        ----------
        n : int
            Replication factor; must be positive.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        self.packed_pose = self.packed_pose.repeat(n, 1, 1)  # type: ignore[union-attr]


@define(kw_only=True)
class UnifiedSmplOutput:
    """Standardized output container from the unified SMPL model's forward pass.

    This class holds the results of a forward pass, providing a consistent
    interface regardless of the underlying SMPL model type.

    Attributes
    ----------
    vertices : torch.Tensor
        The final mesh vertices of shape (B, V, 3).
    faces : torch.Tensor
        The mesh faces (connectivity) of shape (F, 3).
    joints : torch.Tensor
        The final joint positions in the unified 55-joint SMPL-X format,
        of shape (B, 55, 3).
    full_pose : torch.Tensor
        The flattened full pose vector that was used for Linear Blend Skinning (LBS),
        of shape (B, P). The size P depends on the model type.
    extras : dict[str, Any]
        A dictionary containing model-specific or intermediate outputs, such as
        raw (un-unified) joints, joint mappings, or pre-computed shaped vertices.
    """

    vertices: Tensor  # (B, V, 3) - mesh vertices
    faces: Tensor  # (F, 3) - face connectivity
    joints: Tensor  # (B, J, 3) - unified joint set
    full_pose: Tensor  # (B, P) - flattened pose used for LBS
    extras: dict[str, Any] = field(factory=dict)  # Model-specific extras

    @property
    def num_vertices(self) -> int:
        """Get the number of vertices in the mesh."""
        return self.vertices.shape[1]

    @property
    def num_joints(self) -> int:
        """Get the number of joints in the unified set (always 55)."""
        return self.joints.shape[1]

    @property
    def num_faces(self) -> int:
        """Get the number of faces in the mesh."""
        return self.faces.shape[0]

    @property
    def batch_size(self) -> int:
        """Get the batch size of the output."""
        return self.vertices.shape[0]

    @property
    def body_joints(self) -> Tensor:
        """Get the body joints from the unified set.

        Returns
        -------
        torch.Tensor
            A tensor of shape (B, 22, 3) containing the body joints.
        """
        return self.joints[:, :22]

    @property
    def hand_joints(self) -> Tensor:
        """Get the hand joints from the unified set.

        Returns
        -------
        torch.Tensor
            A tensor of shape (B, 30, 3) containing the 15 left and 15 right
            hand joints.
        """
        return self.joints[:, 22:52]

    @property
    def face_joints(self) -> Tensor:
        """Get the face joints from the unified set.

        Returns
        -------
        torch.Tensor
            A tensor of shape (B, 3, 3) containing the jaw, left eye, and
            right eye joints.
        """
        return self.joints[:, 52:55]
