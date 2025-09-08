"""Unified SMPL Family Model - provides a unified API for SMPL, SMPL-H and SMPL-X models.

This module provides a single Python class that abstracts model differences (SMPL vs SMPL-H vs SMPL-X)
while exposing a consistent interface for posing and retrieving outputs.

Key Components:
    - UnifiedSmplModel: Main adapter class wrapping smplx models
    - UnifiedSmplInputs: Standardized input container
    - PoseByKeypoints: User-friendly per-joint pose specification
    - UnifiedSmplOutput: Standardized output container
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import numpy as np
import torch
import torch.nn as nn
from attrs import define, field, fields
from torch import Tensor

__all__ = [
    "UnifiedSmplModel",
    "UnifiedSmplInputs",
    "PoseByKeypoints",
    "UnifiedSmplOutput"
]

# Type aliases
DeviceLike = str | torch.device
ModelType = Literal["smpl", "smplh", "smplx"]
T = TypeVar("T", bound="UnifiedSmplModel")


@define(kw_only=True)
class UnifiedSmplInputs:
    """Standardized input container for unified SMPL model forward pass.

    All tensor fields are optional and should have batch dimension first (B, ...).
    Missing segments are auto-filled with zeros of the expected size for the model type.
    """

    # Pose parameters (axis-angle in radians)
    root_orient: Tensor | None = None  # (B, 3) - pelvis/global orientation
    pose_body: Tensor | None = None    # (B, 63) - 21 body joints * 3
    left_hand_pose: Tensor | None = None   # (B, 45) - 15 finger joints * 3
    right_hand_pose: Tensor | None = None  # (B, 45) - 15 finger joints * 3
    pose_jaw: Tensor | None = None     # (B, 3) - jaw joint (SMPL-X only)
    left_eye_pose: Tensor | None = None   # (B, 3) - left eyeball (SMPL-X only)
    right_eye_pose: Tensor | None = None  # (B, 3) - right eyeball (SMPL-X only)

    # Shape and expression
    betas: Tensor | None = None        # (B, n_betas) - shape parameters
    expression: Tensor | None = None   # (B, n_expr) - facial expression (SMPL-X only)

    # Translation
    trans: Tensor | None = None        # (B, 3) - global translation

    # Advanced (may be ignored by some models)
    v_template: Tensor | None = None   # (B, V, 3) - custom template vertices
    joints_override: Tensor | None = None  # (B, J*, 3) - custom joint positions
    v_shaped: Tensor | None = None     # (B, V, 3) - shaped vertices

    @property
    def hand_pose(self) -> Tensor | None:
        """Concatenation of left and right hand poses -> (B, 90) when both present."""
        if self.left_hand_pose is not None and self.right_hand_pose is not None:
            return torch.cat([self.left_hand_pose, self.right_hand_pose], dim=-1)
        return None

    @property
    def eyes_pose(self) -> Tensor | None:
        """Concatenation of left and right eye poses -> (B, 6) when both present."""
        if self.left_eye_pose is not None and self.right_eye_pose is not None:
            return torch.cat([self.left_eye_pose, self.right_eye_pose], dim=-1)
        return None

    @classmethod
    def from_kwargs(cls, **kwargs) -> UnifiedSmplInputs:
        """Convenience constructor from keyword arguments."""
        return cls(**kwargs)

    def batch_size(self) -> int | None:
        """Infer batch size from first non-None field."""
        for f in fields(UnifiedSmplInputs):
            value = getattr(self, f.name)
            if value is not None and isinstance(value, Tensor):
                return value.shape[0]
        return None

    def check_valid(self, model_type: ModelType) -> None:
        """Verify tensor presence and shapes are consistent with the model type.

        Args:
            model_type: The target model type to validate against

        Raises:
            ValueError: If inputs are incompatible with the model type
        """
        batch_size = self.batch_size()

        # Common shape checks
        if self.root_orient is not None and self.root_orient.shape != (batch_size, 3):
            raise ValueError(f"root_orient must be (B, 3), got {self.root_orient.shape}")

        if self.pose_body is not None and self.pose_body.shape != (batch_size, 63):
            raise ValueError(f"pose_body must be (B, 63), got {self.pose_body.shape}")

        if self.trans is not None and self.trans.shape != (batch_size, 3):
            raise ValueError(f"trans must be (B, 3), got {self.trans.shape}")

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

        elif model_type == "smplh":
            # SMPL-H: requires both hands if any provided, no face
            has_left = self.left_hand_pose is not None
            has_right = self.right_hand_pose is not None

            if has_left != has_right:
                raise ValueError("SMPL-H requires both left and right hand poses or neither")

            if has_left and self.left_hand_pose.shape != (batch_size, 45):
                raise ValueError(f"left_hand_pose must be (B, 45), got {self.left_hand_pose.shape}")
            if has_right and self.right_hand_pose.shape != (batch_size, 45):
                raise ValueError(f"right_hand_pose must be (B, 45), got {self.right_hand_pose.shape}")

            if self.pose_jaw is not None:
                raise ValueError("SMPL-H does not support jaw pose")
            if self.left_eye_pose is not None or self.right_eye_pose is not None:
                raise ValueError("SMPL-H does not support eye poses")
            if self.expression is not None:
                raise ValueError("SMPL-H does not support facial expressions")

        elif model_type == "smplx":
            # SMPL-X: if hands provided, both required; same for eyes
            has_left_hand = self.left_hand_pose is not None
            has_right_hand = self.right_hand_pose is not None
            has_left_eye = self.left_eye_pose is not None
            has_right_eye = self.right_eye_pose is not None

            if has_left_hand != has_right_hand:
                raise ValueError("SMPL-X requires both left and right hand poses or neither")
            if has_left_eye != has_right_eye:
                raise ValueError("SMPL-X requires both left and right eye poses or neither")

            if has_left_hand and self.left_hand_pose.shape != (batch_size, 45):
                raise ValueError(f"left_hand_pose must be (B, 45), got {self.left_hand_pose.shape}")
            if has_right_hand and self.right_hand_pose.shape != (batch_size, 45):
                raise ValueError(f"right_hand_pose must be (B, 45), got {self.right_hand_pose.shape}")

            if self.pose_jaw is not None and self.pose_jaw.shape != (batch_size, 3):
                raise ValueError(f"pose_jaw must be (B, 3), got {self.pose_jaw.shape}")

            if has_left_eye and self.left_eye_pose.shape != (batch_size, 3):
                raise ValueError(f"left_eye_pose must be (B, 3), got {self.left_eye_pose.shape}")
            if has_right_eye and self.right_eye_pose.shape != (batch_size, 3):
                raise ValueError(f"right_eye_pose must be (B, 3), got {self.right_eye_pose.shape}")

    @classmethod
    def from_keypoint_pose(cls, kpts: PoseByKeypoints, *, model_type: ModelType) -> UnifiedSmplInputs:
        """Convert per-joint keypoint pose to segmented inputs.

        Args:
            kpts: Per-joint pose specification
            model_type: Target model type for conversion

        Returns:
            UnifiedSmplInputs with appropriate segments filled
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
            "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
            "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
            "neck", "left_collar", "right_collar", "head",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist"
        ]

        body_pose_parts = []
        for joint in body_joints:
            body_pose_parts.append(get_or_zeros(joint))
        pose_body = torch.cat(body_pose_parts, dim=-1)  # (B, 63)

        # Hands
        left_hand_joints = [
            "left_thumb1", "left_thumb2", "left_thumb3",
            "left_index1", "left_index2", "left_index3",
            "left_middle1", "left_middle2", "left_middle3",
            "left_ring1", "left_ring2", "left_ring3",
            "left_pinky1", "left_pinky2", "left_pinky3"
        ]

        right_hand_joints = [
            "right_thumb1", "right_thumb2", "right_thumb3",
            "right_index1", "right_index2", "right_index3",
            "right_middle1", "right_middle2", "right_middle3",
            "right_ring1", "right_ring2", "right_ring3",
            "right_pinky1", "right_pinky2", "right_pinky3"
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
            left_eye_pose = getattr(kpts, "left_eye", None) or getattr(kpts, "left_eyeball", None)
            right_eye_pose = getattr(kpts, "right_eye", None) or getattr(kpts, "right_eyeball", None)

        return cls(
            root_orient=root_orient,
            pose_body=pose_body,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            pose_jaw=pose_jaw,
            left_eye_pose=left_eye_pose,
            right_eye_pose=right_eye_pose
        )


@define(kw_only=True)
class PoseByKeypoints:
    """User-friendly per-joint axis-angle pose specification.

    All joint fields are optional (B, 3) tensors in axis-angle format.
    None means "not specified -> use zeros".
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
        """Convenience constructor from keyword arguments."""
        return cls(**kwargs)

    def batch_size(self) -> int | None:
        """Infer batch size from first non-None tensor."""
        for f in fields(PoseByKeypoints):
            value = getattr(self, f.name)
            if value is not None and isinstance(value, Tensor):
                return value.shape[0]
        return None

    def check_valid_by_keypoints(self, model_type: ModelType, strict: bool = False) -> None:
        """Validate keypoint inputs for the target model type.

        Args:
            model_type: Target model type
            strict: If True, reject incompatible fields; if False, warn and ignore

        Raises:
            ValueError: If validation fails
        """
        batch_size = self.batch_size()

        # Check all provided tensors have correct shape
        for f in fields(PoseByKeypoints):
            value = getattr(self, f.name)
            if value is not None and isinstance(value, Tensor) and value.shape != (batch_size, 3):
                raise ValueError(f"{f.name} must be (B, 3), got {value.shape}")

        # Model-specific validation
        face_joints = ["jaw", "left_eye", "right_eye", "left_eyeball", "right_eyeball"]
        hand_joints = [f.name for f in fields(PoseByKeypoints)
                      if "thumb" in f.name or "index" in f.name or "middle" in f.name
                      or "ring" in f.name or "pinky" in f.name]

        if model_type == "smpl":
            # SMPL: no face or hands
            for joint in face_joints + hand_joints:
                if getattr(self, joint, None) is not None:
                    if strict:
                        raise ValueError(f"SMPL does not support {joint}")
                    else:
                        warnings.warn(f"SMPL does not support {joint}, ignoring", stacklevel=2)

        elif model_type == "smplh":
            # SMPL-H: no face
            for joint in face_joints:
                if getattr(self, joint, None) is not None:
                    if strict:
                        raise ValueError(f"SMPL-H does not support {joint}")
                    else:
                        warnings.warn(f"SMPL-H does not support {joint}, ignoring", stacklevel=2)

        # SMPL-X accepts everything


@define(kw_only=True)
class UnifiedSmplOutput:
    """Standardized output container from unified SMPL model forward pass."""

    vertices: Tensor  # (B, V, 3) - mesh vertices
    faces: Tensor     # (F, 3) - face connectivity
    joints: Tensor    # (B, J, 3) - unified joint set
    full_pose: Tensor # (B, P) - flattened pose used for LBS
    extras: dict[str, Any] = field(factory=dict)  # Model-specific extras

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return self.vertices.shape[1]

    @property
    def num_joints(self) -> int:
        """Number of joints in the unified set."""
        return self.joints.shape[1]

    @property
    def num_faces(self) -> int:
        """Number of faces in the mesh."""
        return self.faces.shape[0]

    @property
    def batch_size(self) -> int:
        """Batch size of the output."""
        return self.vertices.shape[0]

    @property
    def body_joints(self) -> Tensor:
        """Body joints (first 22) from unified set."""
        return self.joints[:, :22]

    @property
    def hand_joints(self) -> Tensor:
        """Hand joints (30 total: 15 per hand) from unified set."""
        return self.joints[:, 22:52]

    @property
    def face_joints(self) -> Tensor:
        """Face joints (jaw + 2 eyes) from unified set."""
        return self.joints[:, 52:55]


class UnifiedSmplModel:
    """Unified adapter for SMPL family models (SMPL, SMPL-H, SMPL-X).

    This class provides a consistent interface for working with different SMPL model variants,
    abstracting away their differences while exposing common functionality.

    Key features:
        - Auto-detection of model type from provided instance
        - Normalized inputs and outputs across model variants
        - Joint set unification to SMPL-X scheme
        - Support for per-keypoint pose specification
    """

    def __init__(self) -> None:
        """Initialize empty wrapper with no base model reference yet."""
        self.m_deformable_model: nn.Module | None = None
        self.m_missing_joint_fill: str | None = None
        self.m_warn_fn: Callable[[str], None] | None = None
        # Optional auxiliary mapping tensors (lazy initialized)
        self.m_joint_mapping: dict[int, int] | None = None

    @classmethod
    def from_smpl_model(
        cls: type[T],
        deformable_model: nn.Module,
        *,
        missing_joint_fill: Literal["nan", "zero"] = "nan",
        warn_fn: Callable[[str], None] | None = None
    ) -> T:
        """Create unified model wrapper from existing SMPL family model.

        Args:
            deformable_model: Pre-loaded model instance (from smplx.create)
            missing_joint_fill: How to fill missing joints ("nan" or "zero")
            warn_fn: Optional warning callback

        Returns:
            Configured UnifiedSmplModel instance
        """
        instance = cls()
        instance.m_deformable_model = deformable_model
        instance.m_missing_joint_fill = missing_joint_fill
        instance.m_warn_fn = warn_fn or warnings.warn

        return instance

    def _detect_model_type(self) -> str:
        """Auto-detect model type from wrapped instance."""
        if self.m_deformable_model is None:
            raise ValueError("No model loaded")

        model = self.m_deformable_model
        type_name = type(model).__name__.lower()

        # Try type name first
        if type_name in ["smpl", "smplh", "smplx"]:
            return type_name

        # Heuristics based on attributes
        if hasattr(model, "jaw_pose") or hasattr(model, "leye_pose") or hasattr(model, "reye_pose"):
            return "smplx"
        elif hasattr(model, "left_hand_pose") and hasattr(model, "right_hand_pose"):
            return "smplh"
        else:
            return "smpl"

    @property
    def model_type(self) -> str:
        """Get detected model type."""
        return self._detect_model_type()

    @property
    def num_betas(self) -> int:
        """Get number of shape parameters."""
        model = self.m_deformable_model
        if hasattr(model, "num_betas"):
            return model.num_betas
        elif hasattr(model, "shapedirs"):
            return model.shapedirs.shape[-1]
        elif hasattr(model, "betas") and model.betas is not None:
            return model.betas.shape[-1]
        return 10  # Default

    @property
    def num_expressions(self) -> int:
        """Get number of expression parameters (0 for non-SMPL-X)."""
        if self.model_type != "smplx":
            return 0
        model = self.m_deformable_model
        if hasattr(model, "num_expression_coeffs"):
            return model.num_expression_coeffs
        elif hasattr(model, "expression") and model.expression is not None:
            return model.expression.shape[-1]
        return 10  # Default for SMPL-X

    @property
    def dtype(self) -> torch.dtype:
        """Get model data type."""
        model = self.m_deformable_model
        if hasattr(model, "v_template"):
            return model.v_template.dtype
        elif hasattr(model, "shapedirs"):
            return model.shapedirs.dtype
        return torch.float32

    @property
    def device(self) -> torch.device:
        """Get model device."""
        model = self.m_deformable_model
        # Try parameters first
        try:
            param = next(model.parameters())
            return param.device
        except StopIteration:
            pass
        # Try buffers
        try:
            buffer = next(model.buffers())
            return buffer.device
        except StopIteration:
            pass
        return torch.device("cpu")

    @property
    def faces(self) -> Tensor:
        """Get face connectivity."""
        model = self.m_deformable_model
        if hasattr(model, "faces_tensor"):
            return model.faces_tensor
        elif hasattr(model, "faces"):
            faces = model.faces
            if isinstance(faces, np.ndarray):
                return torch.from_numpy(faces).long()
            return faces
        raise AttributeError("Model has no faces")

    def _normalize_inputs(self, inputs: UnifiedSmplInputs | PoseByKeypoints) -> dict[str, Tensor]:
        """Normalize inputs for the wrapped model.

        Args:
            inputs: Input specification

        Returns:
            Dictionary of normalized parameters for the model
        """
        model_type = self.model_type

        # Convert keypoints if needed
        if isinstance(inputs, PoseByKeypoints):
            inputs.check_valid_by_keypoints(model_type, strict=False)
            inputs = UnifiedSmplInputs.from_keypoint_pose(inputs, model_type=model_type)

        # Validate inputs
        inputs.check_valid(model_type)

        batch_size = inputs.batch_size() or 1
        device = self.device
        dtype = self.dtype

        # Prepare normalized inputs
        normalized = {}

        # Helper to ensure tensor or create zeros
        def ensure_tensor(value: Tensor | None, shape: tuple[int, ...]) -> Tensor:
            if value is not None:
                return value.to(device=device, dtype=dtype)
            return torch.zeros(shape, device=device, dtype=dtype)

        # Common parameters
        if model_type in ["smpl", "smplh", "smplx"]:
            normalized["global_orient"] = ensure_tensor(inputs.root_orient, (batch_size, 3))
            normalized["body_pose"] = ensure_tensor(inputs.pose_body, (batch_size, 63))

            if inputs.betas is not None:
                normalized["betas"] = inputs.betas.to(device=device, dtype=dtype)
            if inputs.trans is not None:
                normalized["transl"] = inputs.trans.to(device=device, dtype=dtype)

        # SMPL-H specific
        if model_type in ["smplh", "smplx"]:
            normalized["left_hand_pose"] = ensure_tensor(inputs.left_hand_pose, (batch_size, 45))
            normalized["right_hand_pose"] = ensure_tensor(inputs.right_hand_pose, (batch_size, 45))

        # SMPL-X specific
        if model_type == "smplx":
            # Always provide jaw and eye poses for SMPL-X (zeros if not specified)
            normalized["jaw_pose"] = ensure_tensor(inputs.pose_jaw, (batch_size, 3))
            normalized["leye_pose"] = ensure_tensor(inputs.left_eye_pose, (batch_size, 3))
            normalized["reye_pose"] = ensure_tensor(inputs.right_eye_pose, (batch_size, 3))

            # Always provide expression for SMPL-X (zeros if not specified)
            num_expr = self.num_expressions
            if num_expr > 0:
                normalized["expression"] = ensure_tensor(inputs.expression, (batch_size, num_expr))

        # Always request vertices and joints
        normalized["return_verts"] = True
        normalized["return_joints"] = True

        return normalized

    def _unify_joints(self, joints_raw: Tensor, model_type: str) -> tuple[Tensor, dict[str, Any]]:
        """Convert raw joints to unified SMPL-X joint set.

        Args:
            joints_raw: Raw joint output from model
            model_type: Model type

        Returns:
            Unified joints and extras dict with mapping info
        """
        batch_size = joints_raw.shape[0]
        device = joints_raw.device
        dtype = joints_raw.dtype

        extras = {"joints_raw": joints_raw}

        if model_type == "smplx":
            # SMPL-X already has the target joint set (55 joints)
            # Just take the first 55 joints (excluding extra keypoints)
            unified = joints_raw[:, :55]

        elif model_type == "smplh":
            # SMPL-H has 52 joints (body + hands, no face)
            # Need to add placeholders for jaw and eyes (3 joints)
            unified = torch.zeros((batch_size, 55, 3), device=device, dtype=dtype)

            # Copy body and hand joints (first 52)
            unified[:, :52] = joints_raw[:, :52]

            # Fill missing face joints (jaw=22, left_eye=23, right_eye=24 in SMPL-X)
            # These indices need adjustment - in SMPL-X they're at different positions
            # For now, use NaN or zero for missing joints
            if self.m_missing_joint_fill == "nan":
                unified[:, 52:55] = float("nan")
            else:
                unified[:, 52:55] = 0.0

            extras["missing_joints"] = list(range(52, 55))

        else:  # smpl
            # SMPL has 23 or 24 joints (body only, no hands or face)
            # Need to add placeholders for hands and face
            unified = torch.zeros((batch_size, 55, 3), device=device, dtype=dtype)

            # Copy body joints
            num_body = min(joints_raw.shape[1], 23)
            unified[:, :num_body] = joints_raw[:, :num_body]

            # Fill missing joints
            if self.m_missing_joint_fill == "nan":
                unified[:, num_body:] = float("nan")
            else:
                unified[:, num_body:] = 0.0

            extras["missing_joints"] = list(range(num_body, 55))

        return unified, extras

    def _compute_full_pose(self, normalized_inputs: dict[str, Tensor]) -> Tensor:
        """Compute the full flattened pose vector used for LBS.

        Args:
            normalized_inputs: Normalized model inputs

        Returns:
            Full pose vector (B, P)
        """
        model_type = self.model_type
        pose_parts = []

        # Always start with root
        if "global_orient" in normalized_inputs:
            pose_parts.append(normalized_inputs["global_orient"])

        # Body pose
        if "body_pose" in normalized_inputs:
            pose_parts.append(normalized_inputs["body_pose"])

        # Model-specific parts
        if model_type == "smplx":
            # SMPL-X: add jaw and eyes before hands
            if "jaw_pose" in normalized_inputs:
                pose_parts.append(normalized_inputs["jaw_pose"])
            if "leye_pose" in normalized_inputs:
                pose_parts.append(normalized_inputs["leye_pose"])
            if "reye_pose" in normalized_inputs:
                pose_parts.append(normalized_inputs["reye_pose"])

        # Hands (SMPL-H and SMPL-X)
        if model_type in ["smplh", "smplx"]:
            if "left_hand_pose" in normalized_inputs:
                pose_parts.append(normalized_inputs["left_hand_pose"])
            if "right_hand_pose" in normalized_inputs:
                pose_parts.append(normalized_inputs["right_hand_pose"])

        if pose_parts:
            return torch.cat(pose_parts, dim=-1)
        else:
            # Fallback: return empty pose
            batch_size = 1
            if "global_orient" in normalized_inputs:
                batch_size = normalized_inputs["global_orient"].shape[0]
            return torch.zeros((batch_size, 3), device=self.device, dtype=self.dtype)

    def __call__(self, inputs: UnifiedSmplInputs | PoseByKeypoints) -> UnifiedSmplOutput:
        """Make the model callable."""
        return self.forward(inputs)

    def forward(self, inputs: UnifiedSmplInputs | PoseByKeypoints) -> UnifiedSmplOutput:
        """Run forward pass with normalized inputs.

        Args:
            inputs: Input specification (unified or per-keypoint)

        Returns:
            Unified output with vertices, faces, joints, and full pose
        """
        if self.m_deformable_model is None:
            raise ValueError("No model loaded")

        # Normalize inputs
        normalized = self._normalize_inputs(inputs)

        # Call wrapped model
        output = self.m_deformable_model(**normalized)

        # Extract outputs
        vertices = output.vertices
        joints_raw = output.joints

        # Unify joints
        joints_unified, extras = self._unify_joints(joints_raw, self.model_type)

        # Compute full pose
        full_pose = self._compute_full_pose(normalized)

        # Add any extra outputs
        if hasattr(output, "v_shaped"):
            extras["v_shaped"] = output.v_shaped

        return UnifiedSmplOutput(
            vertices=vertices,
            faces=self.faces,
            joints=joints_unified,
            full_pose=full_pose,
            extras=extras
        )

    def to(self, device: DeviceLike) -> UnifiedSmplModel:  # noqa: ARG002
        """Move adapter's auxiliary tensors to device.

        Note: Users must handle moving the wrapped model separately via
        deformable_model.to(device).

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        # Currently no auxiliary tensors to move
        # Future: move any cached mapping tensors
        if self.m_warn_fn:
            self.m_warn_fn(
                "UnifiedSmplModel.to() only moves adapter tensors. "
                "Move the wrapped model with model.m_deformable_model.to(device)"
            )
        return self

    def eval(self) -> UnifiedSmplModel:
        """Set wrapped model to eval mode."""
        if self.m_deformable_model is not None:
            self.m_deformable_model.eval()
        return self

    def train(self, mode: bool = True) -> UnifiedSmplModel:
        """Set wrapped model to train mode."""
        if self.m_deformable_model is not None:
            self.m_deformable_model.train(mode)
        return self

    def get_joint_names(self, unified: bool = True) -> list[str]:
        """Get joint names for the model.

        Args:
            unified: If True, return unified SMPL-X joint names

        Returns:
            List of joint names
        """
        if unified:
            # Return standard SMPL-X 55 joint names
            # This would come from a predefined list
            return [f"joint_{i}" for i in range(55)]  # Placeholder
        else:
            # Return model-specific names if available
            model = self.m_deformable_model
            if hasattr(model, "joint_names"):
                return model.joint_names
            return [f"joint_{i}" for i in range(self._get_raw_joint_count())]

    def _get_raw_joint_count(self) -> int:
        """Get the raw joint count for the model type."""
        model_type = self.model_type
        if model_type == "smplx":
            return 55
        elif model_type == "smplh":
            return 52
        else:  # smpl
            return 24

    def select_joints(
        self,
        joints: Tensor,
        indices: list[int] | Tensor | None = None,
        names: list[str] | None = None
    ) -> Tensor:
        """Select subset of joints by indices or names.

        Args:
            joints: Joint tensor (B, J, 3)
            indices: Joint indices to select
            names: Joint names to select (uses get_joint_names to map)

        Returns:
            Selected joints (B, n, 3)
        """
        if indices is not None and names is not None:
            raise ValueError("Provide either indices or names, not both")

        if names is not None:
            # Convert names to indices
            joint_names = self.get_joint_names(unified=True)
            indices = []
            for name in names:
                if name in joint_names:
                    indices.append(joint_names.index(name))
                else:
                    if self.m_warn_fn:
                        self.m_warn_fn(f"Joint name '{name}' not found")

        if indices is not None:
            if isinstance(indices, list):
                indices = torch.tensor(indices, dtype=torch.long, device=joints.device)
            return joints[:, indices]

        return joints
