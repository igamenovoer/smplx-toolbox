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

import numpy as np

import torch
from attrs import define, field, fields
from torch import Tensor

from .constants import (
    ModelType,
    get_smpl_joint_names,
    get_smplh_joint_names,
    get_smplx_joint_names,
    CoreBodyJoint,
    FaceJoint,
    HandFingerJoint,
)


@define(kw_only=True)
class UnifiedSmplInputs:
    """Standardized input container for the unified SMPL model forward pass.

    Attributes
    ----------
    named_pose : NamedPose, optional
        Preferred single source of intrinsic pose truth (excludes pelvis). Use
        ``named_pose.packed_pose`` to get/set joint AAs by name.
    global_orient : torch.Tensor, optional
        Global orientation `(B, 3)` to pass to SMPL/SMPL-X as `global_orient`.
        This is separate from `named_pose` (which encodes intrinsic pose only).
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
    - Prefer ``named_pose`` to manage all pose data (intrinsic joints only).
      Pelvis/global orientation must be provided via ``global_orient``.
    - All tensor fields are optional and should have the batch dimension first
      (B, ...). Missing pose segments are automatically filled with zeros of the
      expected size for the model type during the forward pass.
    """

    # Single source of pose truth (preferred) — intrinsic pose only (no pelvis)
    named_pose: "NamedPose | None" = None

    # Orientation (separate from NamedPose)
    global_orient: Tensor | None = None  # (B, 3) - pelvis/global orientation

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

    # -----------------
    # Properties
    # -----------------
    # global_orient is a plain member. No property indirection.

    # Note: Aggregate helpers have moved to NamedPose (hand_pose, eyes_pose).

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
        # Prefer named_pose when available
        if self.named_pose is not None and self.named_pose.packed_pose is not None:
            return int(self.named_pose.packed_pose.shape[0])

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
        # Validate global orientation shape if provided
        if self.global_orient is not None and self.global_orient.shape != (
            batch_size,
            3,
        ):
            raise ValueError(
                f"global_orient must be (B, 3), got {self.global_orient.shape}"
            )

        if self.trans is not None and self.trans.shape != (batch_size, 3):
            raise ValueError(f"trans must be (B, 3), got {self.trans.shape}")

        # Validate betas shape if provided
        if self.betas is not None and num_betas is not None:
            if self.betas.shape[1] != num_betas:
                raise ValueError(
                    f"betas shape mismatch: got {self.betas.shape[1]} parameters, "
                    f"model expects {num_betas}"
                )

        # Model-specific minimal validation
        if str(model_type) == "smplx":
            if self.expression is not None and num_expressions is not None:
                if self.expression.shape[1] != num_expressions:
                    raise ValueError(
                        f"expression shape mismatch: got {self.expression.shape[1]} parameters, "
                        f"model expects {num_expressions}"
                    )


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
        # Prefer named_pose if provided; otherwise fall back to segmented fields
        if self.named_pose is not None and self.named_pose.packed_pose is not None:
            B = int(self.named_pose.packed_pose.shape[0])
            body = self._npz_body_pose(self.named_pose)
            out: dict[str, Tensor | bool] = {
                "global_orient": self.global_orient
                if self.global_orient is not None
                else torch.zeros((B, 3)),
                "body_pose": body if body is not None else torch.zeros((B, 63)),
                "return_verts": True,
            }
        else:
            out = {
                "global_orient": self.global_orient
                if self.global_orient is not None
                else torch.zeros((self.batch_size() or 1, 3)),
                "body_pose": torch.zeros((self.batch_size() or 1, 63)),
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
        # Add hands if available (prefer from named_pose)
        if self.named_pose is not None and self.named_pose.packed_pose is not None:
            lh, rh = self._npz_hand_poses(self.named_pose)
            if lh is not None:
                out["left_hand_pose"] = lh
            if rh is not None:
                out["right_hand_pose"] = rh
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
        # Face/eyes (prefer from named_pose)
        if self.named_pose is not None and self.named_pose.packed_pose is not None:
            jaw, leye, reye = self._npz_face_poses(self.named_pose)
            if jaw is not None:
                out["jaw_pose"] = jaw
            if leye is not None:
                out["leye_pose"] = leye
            if reye is not None:
                out["reye_pose"] = reye
        if self.expression is not None:
            out["expression"] = self.expression
        return out

    # -----------------
    # NamedPose slicers
    # -----------------
    def _npz_body_pose(self, npz: "NamedPose") -> Tensor | None:
        if npz.packed_pose is None:
            return None
        B = int(npz.packed_pose.shape[0])
        parts: list[Tensor] = []
        for e in CoreBodyJoint:
            if e == CoreBodyJoint.PELVIS:
                # Exclude pelvis; body_pose is 21 joints (63 DoF)
                continue
            g = npz.get_joint_pose(e.value)
            if g is None:
                parts.append(torch.zeros((B, 1, 3), dtype=npz.packed_pose.dtype))
            else:
                parts.append(g)
        body = torch.cat(parts, dim=1)
        return body.reshape(B, 63)

    def _npz_hand_poses(self, npz: "NamedPose") -> tuple[Tensor | None, Tensor | None]:
        if npz.packed_pose is None:
            return None, None
        B = int(npz.packed_pose.shape[0])

        def collect(names: list[str]) -> Tensor:
            parts: list[Tensor] = []
            for name in names:
                g = npz.get_joint_pose(name)
                if g is None:
                    parts.append(
                        torch.zeros((B, 1, 3), dtype=npz.packed_pose.dtype)
                    )
                else:
                    parts.append(g)
            return torch.cat(parts, dim=1).reshape(B, 45)

        left_names = [e.value for e in HandFingerJoint if e.name.startswith("LEFT_")]
        right_names = [e.value for e in HandFingerJoint if e.name.startswith("RIGHT_")]

        lh = collect(left_names) if npz.get_joint_index(left_names[0]) is not None else None
        rh = (
            collect(right_names) if npz.get_joint_index(right_names[0]) is not None else None
        )
        return lh, rh

    def _npz_face_poses(self, npz: "NamedPose") -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if npz.packed_pose is None:
            return None, None, None
        B = int(npz.packed_pose.shape[0])
        jaw = npz.get_joint_pose(FaceJoint.JAW.value)
        le = npz.get_joint_pose(FaceJoint.LEFT_EYE_SMPLHF.value)
        re = npz.get_joint_pose(FaceJoint.RIGHT_EYE_SMPLHF.value)
        jaw_v = jaw.view(B, 3) if jaw is not None else None
        le_v = le.view(B, 3) if le is not None else None
        re_v = re.view(B, 3) if re is not None else None
        return jaw_v, le_v, re_v


    # PoseByKeypoints has been removed. Use NamedPose for inspection/editing.


@define(kw_only=True)
class NamedPose:
    """Lightweight accessor around intrinsic axis-angle pose `(B, N, 3)`.

    This utility interprets a packed pose tensor using the model type's joint
    namespace and provides convenience getters and setters by joint name.
    The packed pose EXCLUDES the pelvis (global orientation). Use
    :attr:`pelvis` as a zero AA convenience or pass `global_orient` separately
    via :class:`UnifiedSmplInputs`.

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
    - Getters return `None` for unknown joint names (except 'pelvis', which
      raises a `KeyError` to signal it is not part of intrinsic pose).
    - Setters raise `KeyError` for unknown joint names (or 'pelvis') and
      `ValueError` for invalid shapes. `(B, 3)` inputs are reshaped to `(B, 1, 3)`.
    """

    model_type: ModelType
    packed_pose: Tensor | None = None
    batch_size: int = 1

    # Internal mapping caches
    _name_to_index: dict[str, int] = field(init=False, factory=dict)
    _index_to_name: list[str] = field(init=False, factory=list)

    def __attrs_post_init__(self) -> None:
        # Build intrinsic joint name order and mapping (exclude pelvis)
        if self.model_type == ModelType.SMPL:
            names_full = [j.value for j in CoreBodyJoint]
        elif self.model_type == ModelType.SMPLH:
            names_full = get_smplh_joint_names()
        elif self.model_type == ModelType.SMPLX:
            names_full = get_smplx_joint_names()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._index_to_name = [n for n in names_full if n != CoreBodyJoint.PELVIS.value]
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
        if name == CoreBodyJoint.PELVIS.value:
            raise KeyError("'pelvis' is not stored in NamedPose. Use .pelvis or provide global_orient.")
        idx = self.get_joint_index(name)
        if idx is None:
            return None
        return self.packed_pose[:, idx : idx + 1, :].detach().clone()  # type: ignore[index]

    def set_joint_pose_value(self, name: str, pose: Tensor | np.ndarray) -> bool:
        """Set the joint pose value by name without affecting gradients.

        Accepts `(B, 1, 3)` or `(B, 3)` and reshapes `(B, 3) -> (B, 1, 3)`
        automatically. Copies values under `torch.no_grad()` directly into
        `packed_pose` so this operation does not participate in autograd.

        Parameters
        ----------
        name : str
            Joint name within the current model type's namespace.
        pose : torch.Tensor or numpy.ndarray
            Pose tensor/array of shape `(B, 1, 3)` or `(B, 3)`.

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
        if name == CoreBodyJoint.PELVIS.value:
            raise KeyError("Cannot set 'pelvis' in NamedPose. Provide global_orient separately.")
        idx = self.get_joint_index(name)
        if idx is None:
            raise KeyError(
                f"Unknown joint '{name}' for model type {self.model_type.value}"
            )

        # Convert numpy input to tensor on the right device/dtype
        if isinstance(pose, np.ndarray):
            target_t = torch.from_numpy(pose)
        else:
            target_t = pose

        B = self.packed_pose.shape[0]  # type: ignore[union-attr]
        if target_t.ndim == 2 and tuple(target_t.shape) == (B, 3):
            target_t = target_t.view(B, 1, 3)
        elif target_t.ndim == 3 and tuple(target_t.shape) == (B, 1, 3):
            pass
        else:
            raise ValueError(
                f"pose must be (B, 3) or (B, 1, 3); got {tuple(target_t.shape)}"
            )

        with torch.no_grad():
            self.packed_pose[:, idx : idx + 1, :].copy_(  # type: ignore[index]
                target_t.to(device=self.packed_pose.device, dtype=self.packed_pose.dtype)  # type: ignore[union-attr]
            )
        return True

    def to_dict(self, with_pelvis: bool = False) -> dict[str, Tensor]:
        """Get a mapping from joint name to a view of `(B, 1, 3)`.

        Notes
        -----
        The returned tensors are views into `packed_pose`. In-place edits will
        mutate `packed_pose` directly. Pelvis is excluded by default; set
        ``with_pelvis=True`` to include a zero AA pelvis entry.
        """
        out = {
            name: self.packed_pose[:, i : i + 1, :]  # type: ignore[index]
            for i, name in enumerate(self._index_to_name)
        }
        if with_pelvis:
            B = self.packed_pose.shape[0] if self.packed_pose is not None else self.batch_size
            out[CoreBodyJoint.PELVIS.value] = torch.zeros((int(B), 1, 3))
        return out

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

    @property
    def root_orient(self) -> Tensor:
        """View of the pelvis joint axis-angle `(B, 3)`.

        Returns a view into `packed_pose` for the pelvis joint, matching the
        global orientation expected by SMPL family models.
        """
        if self.packed_pose is None:
            # Initialize a zero-sized view based on batch_size if needed
            B = self.batch_size
            return torch.zeros((B, 3))
        idx = self.get_joint_index(CoreBodyJoint.PELVIS.value)
        if idx is None:
            # Should not happen for supported model types; return zeros of proper B
            B = int(self.packed_pose.shape[0])
            return torch.zeros((B, 3), dtype=self.packed_pose.dtype, device=self.packed_pose.device)
        B = int(self.packed_pose.shape[0])
        return self.packed_pose[:, idx, :].view(B, 3)

    def to_model_type(self, smpl_type: ModelType | str, copy: bool = False) -> "NamedPose":
        """Convert this NamedPose to another SMPL family model type.

        Parameters
        ----------
        smpl_type : ModelType or str
            Target model type ('smpl', 'smplh', or 'smplx').
        copy : bool, optional
            If False (default), construct the new `packed_pose` using views and
            concatenations of views to preserve autograd relationships so that
            gradients flowing through the returned instance will propagate back
            to this instance's `packed_pose` where joints overlap. If True,
            deep-copy values so the returned instance is independent.

        Returns
        -------
        NamedPose
            A new NamedPose instance in the requested model type.
        """
        if isinstance(smpl_type, str):
            target_type = ModelType(smpl_type)
        else:
            target_type = smpl_type

        # Fast-path: identical type
        if target_type == self.model_type:
            if copy:
                new_pose = (
                    self.packed_pose.clone().contiguous()
                    if self.packed_pose is not None
                    else None
                )
            else:
                new_pose = self.packed_pose  # share tensor so gradients and edits relate
            return NamedPose(model_type=target_type, packed_pose=new_pose, batch_size=self.batch_size)

        # Build target name order
        if target_type == ModelType.SMPL:
            target_names = get_smpl_joint_names()
        elif target_type == ModelType.SMPLH:
            target_names = get_smplh_joint_names()
        elif target_type == ModelType.SMPLX:
            target_names = get_smplx_joint_names()
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown target model type: {target_type}")

        if self.packed_pose is None:
            # Should not happen due to __attrs_post_init__, but guard defensively
            B = self.batch_size
            device = None
            dtype = None
        else:
            B = int(self.packed_pose.shape[0])
            device = self.packed_pose.device
            dtype = self.packed_pose.dtype

        parts: list[Tensor] = []
        for name in target_names:
            idx = self.get_joint_index(name)
            if idx is None or self.packed_pose is None:
                # Missing joint in source: neutral zero AA
                parts.append(torch.zeros((B, 1, 3), device=device, dtype=dtype))
            else:
                # Take a view to preserve gradient relationships
                parts.append(self.packed_pose[:, idx : idx + 1, :])

        # Concatenate into target shape; autograd keeps links to source slices
        new_packed = torch.cat(parts, dim=1) if parts else None
        if copy and new_packed is not None:
            # Ensure an independent, contiguous memory layout
            new_packed = new_packed.clone().contiguous()

        return NamedPose(model_type=target_type, packed_pose=new_packed, batch_size=B)

    # -----------------
    # Aggregate pose getters
    # -----------------
    def hand_pose(self) -> Tensor | None:
        """Concatenate left and right hand poses into a flat AA vector.

        Returns
        -------
        torch.Tensor or None
            Shape ``(B, 90)`` if both hands are present in this model type,
            otherwise None. Concatenation order is left hand (15x3) then right
            hand (15x3), matching SMPL-H/SMPL-X expectations.
        """
        if self.packed_pose is None:
            return None
        B = int(self.packed_pose.shape[0])

        left_names = [e.value for e in HandFingerJoint if e.name.startswith("LEFT_")]
        right_names = [e.value for e in HandFingerJoint if e.name.startswith("RIGHT_")]

        # Verify at least the first joint of each hand exists; if not, return None
        if self.get_joint_index(left_names[0]) is None or self.get_joint_index(
            right_names[0]
        ) is None:
            return None

        def collect(names: list[str]) -> Tensor:
            parts: list[Tensor] = []
            for name in names:
                g = self.get_joint_pose(name)
                if g is None:
                    parts.append(torch.zeros((B, 1, 3), dtype=self.packed_pose.dtype))
                else:
                    parts.append(g)
            return torch.cat(parts, dim=1).reshape(B, 45)

        lh = collect(left_names)
        rh = collect(right_names)
        return torch.cat([lh, rh], dim=-1)

    def eyes_pose(self) -> Tensor | None:
        """Concatenate left and right eye poses into a flat AA vector.

        Returns
        -------
        torch.Tensor or None
            Shape ``(B, 6)`` if both eyes are present (SMPL-X), otherwise None.
        """
        if self.packed_pose is None:
            return None
        B = int(self.packed_pose.shape[0])
        le = self.get_joint_pose(FaceJoint.LEFT_EYE_SMPLHF.value)
        re = self.get_joint_pose(FaceJoint.RIGHT_EYE_SMPLHF.value)
        if le is None or re is None:
            return None
        return torch.cat([le.view(B, 3), re.view(B, 3)], dim=-1)

    @property
    def pelvis(self) -> Tensor:
        """Zero AA for pelvis `(B, 3)` — intrinsic poses exclude pelvis by design.

        Provided as a convenience when constructing full poses for LBS.
        """
        B = self.packed_pose.shape[0] if self.packed_pose is not None else self.batch_size
        return torch.zeros((int(B), 3))

    # (No batch utilities; handle batching outside this class.)


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
