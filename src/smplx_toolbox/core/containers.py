"""Data containers for the Unified SMPL Model system.

This module contains the input, output, and pose specification containers used by
the unified SMPL family model implementation. These classes provide structured,
type-safe interfaces for model inputs and outputs.

Classes
-------
UnifiedSmplInputs
    Standardized input container for model forward pass
PoseByKeypoints
    User-friendly per-joint pose specification
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

from .constants import ModelType


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
        """Create an instance from keyword arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments corresponding to the attributes of the class.

        Returns
        -------
        UnifiedSmplInputs
            An instance of the class.
        """
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

    @classmethod
    def from_keypoint_pose(
        cls, kpts: PoseByKeypoints, *, model_type: ModelType | str
    ) -> UnifiedSmplInputs:
        """Convert a per-joint keypoint pose specification to segmented inputs.

        This method takes a `PoseByKeypoints` object, which allows specifying poses
        for individual joints by name, and converts it into the standard segmented
        `UnifiedSmplInputs` format required by the model.

        Parameters
        ----------
        kpts : PoseByKeypoints
            A `PoseByKeypoints` object containing the per-joint pose specifications.
        model_type : ModelType
            The target model type ('smpl', 'smplh', 'smplx') for which to
            generate the inputs. This determines which pose segments (e.g., hands, face)
            are created.

        Returns
        -------
        UnifiedSmplInputs
            An instance with the appropriate pose segments filled from the keypoint data.
        """
        batch_size = kpts.batch_size()
        if batch_size is None:
            batch_size = 1

        device = None
        dtype = torch.float32

        # Find device and dtype from first non-None tensor
        for f in fields(PoseByKeypoints):
            value = getattr(kpts, f.name)
            if value is not None and isinstance(value, Tensor):
                device = value.device
                dtype = value.dtype
                break

        def get_or_zeros(field_name: str) -> Tensor:
            """Get field value or return zeros."""
            value = getattr(kpts, field_name, None)
            if value is not None:
                return value
            return torch.zeros((batch_size, 3), device=device, dtype=dtype)

        # Root orientation
        root_orient = getattr(kpts, "root", None)

        # Body pose: 21 joints in order
        body_joints = [
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
        ]

        body_pose_parts = []
        for joint in body_joints:
            body_pose_parts.append(get_or_zeros(joint))
        pose_body = torch.cat(body_pose_parts, dim=-1)  # (B, 63)

        # Hands
        left_hand_joints = [
            "left_thumb1",
            "left_thumb2",
            "left_thumb3",
            "left_index1",
            "left_index2",
            "left_index3",
            "left_middle1",
            "left_middle2",
            "left_middle3",
            "left_ring1",
            "left_ring2",
            "left_ring3",
            "left_pinky1",
            "left_pinky2",
            "left_pinky3",
        ]

        right_hand_joints = [
            "right_thumb1",
            "right_thumb2",
            "right_thumb3",
            "right_index1",
            "right_index2",
            "right_index3",
            "right_middle1",
            "right_middle2",
            "right_middle3",
            "right_ring1",
            "right_ring2",
            "right_ring3",
            "right_pinky1",
            "right_pinky2",
            "right_pinky3",
        ]

        # Only include hands if model supports them
        left_hand_pose = None
        right_hand_pose = None
        if model_type in ["smplh", "smplx"]:
            left_parts = [get_or_zeros(j) for j in left_hand_joints]
            right_parts = [get_or_zeros(j) for j in right_hand_joints]
            left_hand_pose = torch.cat(left_parts, dim=-1)  # (B, 45)
            right_hand_pose = torch.cat(right_parts, dim=-1)  # (B, 45)

        # Face/eyes (SMPL-X only)
        pose_jaw = None
        left_eye_pose = None
        right_eye_pose = None
        if model_type == "smplx":
            pose_jaw = getattr(kpts, "jaw", None)
            # Handle eye aliases
            left_eye_pose = getattr(kpts, "left_eye", None)
            if left_eye_pose is None:
                left_eye_pose = getattr(kpts, "left_eyeball", None)

            right_eye_pose = getattr(kpts, "right_eye", None)
            if right_eye_pose is None:
                right_eye_pose = getattr(kpts, "right_eyeball", None)

        return cls(
            root_orient=root_orient,
            pose_body=pose_body,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            pose_jaw=pose_jaw,
            left_eye_pose=left_eye_pose,
            right_eye_pose=right_eye_pose,
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


@define(kw_only=True)
class PoseByKeypoints:
    """User-friendly per-joint axis-angle pose specification.

    This class provides a convenient way to define a pose by specifying the
    axis-angle rotation for individual joints by name. All unspecified joints
    are assumed to have a zero rotation.

    Attributes
    ----------
    root : torch.Tensor, optional
        Rotation for the corresponding joint. All attributes are optional Tensors
        of shape (B, 3) in axis-angle format. See the class definition for a
        full list of supported joint names.

    Notes
    -----
    All joint fields are optional (B, 3) tensors in axis-angle format.
    `None` means the joint rotation is not specified and will be treated as zeros.
    The class includes aliases like `left_eyeball` for `left_eye`.
    """

    # Root and body trunk
    root: Tensor | None = None
    left_hip: Tensor | None = None
    right_hip: Tensor | None = None
    spine1: Tensor | None = None
    left_knee: Tensor | None = None
    right_knee: Tensor | None = None
    spine2: Tensor | None = None
    left_ankle: Tensor | None = None
    right_ankle: Tensor | None = None
    spine3: Tensor | None = None
    left_foot: Tensor | None = None
    right_foot: Tensor | None = None
    neck: Tensor | None = None
    left_collar: Tensor | None = None
    right_collar: Tensor | None = None
    head: Tensor | None = None
    left_shoulder: Tensor | None = None
    right_shoulder: Tensor | None = None
    left_elbow: Tensor | None = None
    right_elbow: Tensor | None = None
    left_wrist: Tensor | None = None
    right_wrist: Tensor | None = None

    # Face/eyes (SMPL-X only)
    jaw: Tensor | None = None
    left_eye: Tensor | None = None
    right_eye: Tensor | None = None
    left_eyeball: Tensor | None = None  # Alias for left_eye
    right_eyeball: Tensor | None = None  # Alias for right_eye

    # Left hand (15 joints)
    left_thumb1: Tensor | None = None
    left_thumb2: Tensor | None = None
    left_thumb3: Tensor | None = None
    left_index1: Tensor | None = None
    left_index2: Tensor | None = None
    left_index3: Tensor | None = None
    left_middle1: Tensor | None = None
    left_middle2: Tensor | None = None
    left_middle3: Tensor | None = None
    left_ring1: Tensor | None = None
    left_ring2: Tensor | None = None
    left_ring3: Tensor | None = None
    left_pinky1: Tensor | None = None
    left_pinky2: Tensor | None = None
    left_pinky3: Tensor | None = None

    # Right hand (15 joints)
    right_thumb1: Tensor | None = None
    right_thumb2: Tensor | None = None
    right_thumb3: Tensor | None = None
    right_index1: Tensor | None = None
    right_index2: Tensor | None = None
    right_index3: Tensor | None = None
    right_middle1: Tensor | None = None
    right_middle2: Tensor | None = None
    right_middle3: Tensor | None = None
    right_ring1: Tensor | None = None
    right_ring2: Tensor | None = None
    right_ring3: Tensor | None = None
    right_pinky1: Tensor | None = None
    right_pinky2: Tensor | None = None
    right_pinky3: Tensor | None = None

    @classmethod
    def from_kwargs(cls, **kwargs) -> PoseByKeypoints:
        """Create an instance from keyword arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments corresponding to the attributes of the class.

        Returns
        -------
        PoseByKeypoints
            An instance of the class.
        """
        return cls(**kwargs)

    def batch_size(self) -> int | None:
        """Infer the batch size from the first non-None tensor attribute.

        Returns
        -------
        int or None
            The batch size if it can be inferred, otherwise None.
        """
        for f in fields(PoseByKeypoints):
            value = getattr(self, f.name)
            if value is not None and isinstance(value, Tensor):
                return value.shape[0]
        return None

    def check_valid_by_keypoints(
        self,
        model_type: ModelType | str,
        strict: bool = False,
        warn_fn: Callable[[str], None] | None = None,
    ) -> None:
        """Validate keypoint inputs against model capabilities.

        This method checks for common issues, such as providing hand joint data
        for a model that doesn't support it (e.g., SMPL), or providing
        inconsistent batch sizes across joints.

        Parameters
        ----------
        model_type : ModelType
            The target model type ('smpl', 'smplh', 'smplx').
        strict : bool, optional
            If True, raise a `ValueError` for validation failures. If False,
            issue a warning instead. Defaults to False.
        warn_fn : Callable[[str], None], optional
            A custom function to handle warnings. Defaults to `warnings.warn`.

        Raises
        ------
        ValueError
            If `strict` is True and a validation check fails.
        """
        batch_size = self.batch_size()
        warn = warn_fn or (lambda msg: warnings.warn(msg, stacklevel=3))

        # Check batch consistency
        for f in fields(PoseByKeypoints):
            value = getattr(self, f.name)
            if value is not None and isinstance(value, Tensor):
                if value.shape[0] != batch_size:
                    raise ValueError(f"Inconsistent batch size in {f.name}")
                if value.shape != (batch_size, 3):
                    raise ValueError(f"{f.name} must be (B, 3), got {value.shape}")

        # Model-specific checks
        if model_type == "smpl":
            # SMPL: no hands or face
            has_hands = any(
                getattr(self, f.name, None) is not None
                for f in fields(PoseByKeypoints)
                if "thumb" in f.name
                or "index" in f.name
                or "middle" in f.name
                or "ring" in f.name
                or "pinky" in f.name
            )
            if has_hands:
                msg = "SMPL does not support hand joints - they will be ignored"
                if strict:
                    raise ValueError(msg)
                warn(msg)

            if (
                self.jaw is not None
                or self.left_eye is not None
                or self.right_eye is not None
            ):
                msg = "SMPL does not support face joints - they will be ignored"
                if strict:
                    raise ValueError(msg)
                warn(msg)

        elif model_type == "smplh":
            # Check for partial hand specification
            left_hand_joints = [
                "left_thumb1",
                "left_thumb2",
                "left_thumb3",
                "left_index1",
                "left_index2",
                "left_index3",
                "left_middle1",
                "left_middle2",
                "left_middle3",
                "left_ring1",
                "left_ring2",
                "left_ring3",
                "left_pinky1",
                "left_pinky2",
                "left_pinky3",
            ]
            right_hand_joints = [
                "right_thumb1",
                "right_thumb2",
                "right_thumb3",
                "right_index1",
                "right_index2",
                "right_index3",
                "right_middle1",
                "right_middle2",
                "right_middle3",
                "right_ring1",
                "right_ring2",
                "right_ring3",
                "right_pinky1",
                "right_pinky2",
                "right_pinky3",
            ]

            left_provided = [
                j for j in left_hand_joints if getattr(self, j, None) is not None
            ]
            right_provided = [
                j for j in right_hand_joints if getattr(self, j, None) is not None
            ]

            if left_provided and len(left_provided) < len(left_hand_joints):
                msg = f"Partial left hand specification ({len(left_provided)}/15 joints) - missing joints will be zero-filled"
                if strict:
                    raise ValueError(msg)
                warn(msg)

            if right_provided and len(right_provided) < len(right_hand_joints):
                msg = f"Partial right hand specification ({len(right_provided)}/15 joints) - missing joints will be zero-filled"
                if strict:
                    raise ValueError(msg)
                warn(msg)

            # SMPL-H: face not supported
            if (
                self.jaw is not None
                or self.left_eye is not None
                or self.right_eye is not None
            ):
                msg = "SMPL-H does not support face joints - they will be ignored"
                if strict:
                    raise ValueError(msg)
                warn(msg)

        elif model_type == "smplx":
            # Check for single eye specification
            has_left_eye = self.left_eye is not None or self.left_eyeball is not None
            has_right_eye = self.right_eye is not None or self.right_eyeball is not None

            if has_left_eye and not has_right_eye:
                msg = "Only left eye specified - right eye will be zero-filled"
                warn(msg)
            elif has_right_eye and not has_left_eye:
                msg = "Only right eye specified - left eye will be zero-filled"
                warn(msg)


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
