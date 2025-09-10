"""Constants and type aliases for the Unified SMPL Model system.

This module contains joint names, type definitions, and other constants used
throughout the unified SMPL family model implementation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeVar

import torch
from attrs import define, field

__all__ = [
    # Enums
    "CoreBodyJoint",
    "HandFingerJoint",
    "FaceJoint",
    "SMPLSpecialJoint",
    "ModelType",
    "MissingJointFill",
    # Functions
    "get_smpl_joint_names",
    "get_smplh_joint_names",
    "get_smplx_joint_names",
    "get_joint_index",
    # Explicit mappings
    "SMPL_JOINT_NAME_TO_INDEX",
    "SMPLH_JOINT_NAME_TO_INDEX",
    "SMPLX_JOINT_NAME_TO_INDEX",
    # Keypoint containers
    "SMPLKeypoints",
    "SMPLHKeypoints",
    "SMPLXKeypoints",
    # Legacy compatibility
    "SMPL_JOINT_NAMES",
    "SMPLH_JOINT_NAMES",
    "SMPLX_JOINT_NAMES",
    # Type aliases
    "DeviceLike",
    "T",
]


class CoreBodyJoint(StrEnum):
    """Core body joints shared by all SMPL family models (indices 0-21).

    These 22 joints form the basic skeleton present in SMPL, SMPL-H, and SMPL-X.
    The string values match the traditional joint names used in SMPL models.
    """

    PELVIS = "pelvis"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    SPINE1 = "spine1"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    SPINE2 = "spine2"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    SPINE3 = "spine3"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    NECK = "neck"
    LEFT_COLLAR = "left_collar"
    RIGHT_COLLAR = "right_collar"
    HEAD = "head"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"


class HandFingerJoint(StrEnum):
    """Explicit left and right finger joint names.

    Each finger joint is explicitly named with LEFT_ or RIGHT_ prefix.
    The ordering follows the canonical skeleton mapping documentation.
    """

    # Left hand joints (indices 22-36 in SMPL-H, 25-39 in SMPL-X)
    LEFT_INDEX1 = "left_index1"
    LEFT_INDEX2 = "left_index2"
    LEFT_INDEX3 = "left_index3"
    LEFT_MIDDLE1 = "left_middle1"
    LEFT_MIDDLE2 = "left_middle2"
    LEFT_MIDDLE3 = "left_middle3"
    LEFT_PINKY1 = "left_pinky1"
    LEFT_PINKY2 = "left_pinky2"
    LEFT_PINKY3 = "left_pinky3"
    LEFT_RING1 = "left_ring1"
    LEFT_RING2 = "left_ring2"
    LEFT_RING3 = "left_ring3"
    LEFT_THUMB1 = "left_thumb1"
    LEFT_THUMB2 = "left_thumb2"
    LEFT_THUMB3 = "left_thumb3"

    # Right hand joints (indices 37-51 in SMPL-H, 40-54 in SMPL-X)
    RIGHT_INDEX1 = "right_index1"
    RIGHT_INDEX2 = "right_index2"
    RIGHT_INDEX3 = "right_index3"
    RIGHT_MIDDLE1 = "right_middle1"
    RIGHT_MIDDLE2 = "right_middle2"
    RIGHT_MIDDLE3 = "right_middle3"
    RIGHT_PINKY1 = "right_pinky1"
    RIGHT_PINKY2 = "right_pinky2"
    RIGHT_PINKY3 = "right_pinky3"
    RIGHT_RING1 = "right_ring1"
    RIGHT_RING2 = "right_ring2"
    RIGHT_RING3 = "right_ring3"
    RIGHT_THUMB1 = "right_thumb1"
    RIGHT_THUMB2 = "right_thumb2"
    RIGHT_THUMB3 = "right_thumb3"


class FaceJoint(StrEnum):
    """Face joints specific to SMPL-X models.

    These joints are only present in SMPL-X, not in SMPL or SMPL-H.
    """

    JAW = "jaw"
    LEFT_EYE_SMPLHF = "left_eye_smplhf"
    RIGHT_EYE_SMPLHF = "right_eye_smplhf"


class SMPLSpecialJoint(StrEnum):
    """Special end-effector joints specific to base SMPL.

    These replace the detailed hand joints in the basic SMPL model.
    """

    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


# Left hand joints in canonical skeleton mapping order
_LEFT_HAND_JOINTS = [
    HandFingerJoint.LEFT_INDEX1, HandFingerJoint.LEFT_INDEX2, HandFingerJoint.LEFT_INDEX3,
    HandFingerJoint.LEFT_MIDDLE1, HandFingerJoint.LEFT_MIDDLE2, HandFingerJoint.LEFT_MIDDLE3,
    HandFingerJoint.LEFT_PINKY1, HandFingerJoint.LEFT_PINKY2, HandFingerJoint.LEFT_PINKY3,
    HandFingerJoint.LEFT_RING1, HandFingerJoint.LEFT_RING2, HandFingerJoint.LEFT_RING3,
    HandFingerJoint.LEFT_THUMB1, HandFingerJoint.LEFT_THUMB2, HandFingerJoint.LEFT_THUMB3,
]

# Right hand joints in canonical skeleton mapping order
_RIGHT_HAND_JOINTS = [
    HandFingerJoint.RIGHT_INDEX1, HandFingerJoint.RIGHT_INDEX2, HandFingerJoint.RIGHT_INDEX3,
    HandFingerJoint.RIGHT_MIDDLE1, HandFingerJoint.RIGHT_MIDDLE2, HandFingerJoint.RIGHT_MIDDLE3,
    HandFingerJoint.RIGHT_PINKY1, HandFingerJoint.RIGHT_PINKY2, HandFingerJoint.RIGHT_PINKY3,
    HandFingerJoint.RIGHT_RING1, HandFingerJoint.RIGHT_RING2, HandFingerJoint.RIGHT_RING3,
    HandFingerJoint.RIGHT_THUMB1, HandFingerJoint.RIGHT_THUMB2, HandFingerJoint.RIGHT_THUMB3,
]


def get_smpl_joint_names() -> list[str]:
    """Get the official SMPL joint names (24 joints).

    Uses the explicit SMPL_JOINT_NAME_TO_INDEX mapping to return joint names
    in the correct order as specified in the skeleton mapping documentation.

    Returns
    -------
    list[str]
        List of 24 SMPL joint names in skeleton mapping order.
    """
    # Sort by index to ensure correct ordering
    sorted_items = sorted(SMPL_JOINT_NAME_TO_INDEX.items(), key=lambda x: x[1])
    return [name for name, index in sorted_items]


def get_smplh_joint_names() -> list[str]:
    """Get the official SMPL-H joint names (52 joints).

    Uses the explicit SMPLH_JOINT_NAME_TO_INDEX mapping to return joint names
    in the correct order as specified in the skeleton mapping documentation.

    Returns
    -------
    list[str]
        List of 52 SMPL-H joint names in skeleton mapping order.
    """
    # Sort by index to ensure correct ordering
    sorted_items = sorted(SMPLH_JOINT_NAME_TO_INDEX.items(), key=lambda x: x[1])
    return [name for name, index in sorted_items]


def get_smplx_joint_names() -> list[str]:
    """Get the official SMPL-X joint names (first 55 joints).

    Uses the explicit SMPLX_JOINT_NAME_TO_INDEX mapping to return joint names
    in the correct order as specified in the skeleton mapping documentation.

    Returns
    -------
    list[str]
        List of 55 SMPL-X joint names in skeleton mapping order.
    """
    # Sort by index to ensure correct ordering
    sorted_items = sorted(SMPLX_JOINT_NAME_TO_INDEX.items(), key=lambda x: x[1])
    return [name for name, index in sorted_items]


def get_joint_index(joint_name: str, model_type: ModelType | str) -> int:
    """Get the index of a joint name for a specific model type.

    Parameters
    ----------
    joint_name : str
        The name of the joint to look up.
    model_type : ModelType | str
        The SMPL model type ('smpl', 'smplh', or 'smplx').

    Returns
    -------
    int
        The index of the joint in the model's joint array.

    Raises
    ------
    ValueError
        If the joint name is not found in the specified model type.
    KeyError
        If the joint name doesn't exist in the model's joint set.
    """
    if isinstance(model_type, str):
        model_type = ModelType(model_type)

    if model_type == ModelType.SMPL:
        mapping = SMPL_JOINT_NAME_TO_INDEX
    elif model_type == ModelType.SMPLH:
        mapping = SMPLH_JOINT_NAME_TO_INDEX
    elif model_type == ModelType.SMPLX:
        mapping = SMPLX_JOINT_NAME_TO_INDEX
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if joint_name not in mapping:
        available_joints = list(mapping.keys())
        raise KeyError(f"Joint '{joint_name}' not found in {model_type} model. "
                      f"Available joints: {available_joints}")

    return mapping[joint_name]


# Explicit joint name-to-index mappings
# These make the joint ordering completely explicit and self-documenting

def _build_smpl_joint_mapping() -> dict[str, int]:
    """Build SMPL joint name-to-index mapping using enum values."""
    mapping = {}
    index = 0

    # Core body joints (0-21) - same across all SMPL variants
    for joint in CoreBodyJoint:
        mapping[joint.value] = index
        index += 1

    # SMPL-specific end effectors (22-23)
    for joint in SMPLSpecialJoint:
        mapping[joint.value] = index
        index += 1

    return mapping

SMPL_JOINT_NAME_TO_INDEX: dict[str, int] = _build_smpl_joint_mapping()

def _build_smplh_joint_mapping() -> dict[str, int]:
    """Build SMPL-H joint name-to-index mapping using enum values."""
    mapping = {}
    index = 0

    # Core body joints (0-21) - same across all SMPL variants
    for joint in CoreBodyJoint:
        mapping[joint.value] = index
        index += 1

    # Left hand joints (22-36) - 15 joints, using explicit enum values
    for finger_joint in _LEFT_HAND_JOINTS:
        mapping[finger_joint.value] = index
        index += 1

    # Right hand joints (37-51) - 15 joints, using explicit enum values
    for finger_joint in _RIGHT_HAND_JOINTS:
        mapping[finger_joint.value] = index
        index += 1

    return mapping

SMPLH_JOINT_NAME_TO_INDEX: dict[str, int] = _build_smplh_joint_mapping()

def _build_smplx_joint_mapping() -> dict[str, int]:
    """Build SMPL-X joint name-to-index mapping using enum values."""
    mapping = {}
    index = 0

    # Core body joints (0-21) - same across all SMPL variants
    for joint in CoreBodyJoint:
        mapping[joint.value] = index
        index += 1

    # Face joints (22-24) - SMPL-X only
    for joint in FaceJoint:
        mapping[joint.value] = index
        index += 1

    # Left hand joints (25-39) - 15 joints, using explicit enum values
    for finger_joint in _LEFT_HAND_JOINTS:
        mapping[finger_joint.value] = index
        index += 1

    # Right hand joints (40-54) - 15 joints, using explicit enum values
    for finger_joint in _RIGHT_HAND_JOINTS:
        mapping[finger_joint.value] = index
        index += 1

    return mapping

SMPLX_JOINT_NAME_TO_INDEX: dict[str, int] = _build_smplx_joint_mapping()

# Legacy compatibility - these will be replaced with function calls
SMPL_JOINT_NAMES = get_smpl_joint_names()
SMPLH_JOINT_NAMES = get_smplh_joint_names()
SMPLX_JOINT_NAMES = get_smplx_joint_names()


class ModelType(StrEnum):
    """Supported SMPL family model types.

    Values are strings and compare equal to their string representation, e.g.:
    ``ModelType.SMPLX == "smplx"`` is True.

    Use :meth:`values` to list all options.
    """

    SMPL = "smpl"
    SMPLH = "smplh"
    SMPLX = "smplx"

    @classmethod
    def values(cls) -> list[str]:
        """Return all enum values as a list of strings."""
        return [e.value for e in cls]

    def get_joint_names(self) -> list[str]:
        """Get the joint names for this model type.

        Returns
        -------
        list[str]
            List of joint names appropriate for this model type.
        """
        if self == ModelType.SMPL:
            return get_smpl_joint_names()
        elif self == ModelType.SMPLH:
            return get_smplh_joint_names()
        elif self == ModelType.SMPLX:
            return get_smplx_joint_names()
        else:
            raise ValueError(f"Unknown model type: {self}")


class MissingJointFill(StrEnum):
    """Strategy for filling joints not present in the base model.

    - ``NAN``: fill with NaNs
    - ``ZERO``: fill with zeros
    """

    NAN = "nan"
    ZERO = "zero"

    @classmethod
    def values(cls) -> list[str]:
        """Return all enum values as a list of strings."""
        return [e.value for e in cls]


# Type aliases
DeviceLike = str | torch.device
"""Type alias for device specifications (string or torch.device)."""

# Forward reference for type variable
T = TypeVar("T")
"""Type variable for generic classes."""


# Model-specific keypoint data containers
@define(kw_only=True)
class SMPLKeypoints:
    """Keypoint data container for SMPL models (24 joints).

    Stores keypoint data for the 24 joints present in base SMPL models,
    following the exact order from the skeleton mapping documentation.

    Attributes
    ----------
    keypoints : Any, optional
        Keypoint position data (typically tensor or array of shape [..., 24, 3])
    confidences : Any, optional
        Keypoint confidence scores (typically tensor or array of shape [..., 24])
    """

    keypoints: Any | None = field(default=None)
    confidences: Any | None = field(default=None)

    @classmethod
    def get_ordered_names(cls) -> list[str]:
        """Get SMPL joint names in skeleton mapping order.

        Returns the 24 SMPL joint names in the exact order specified
        in the skeleton mapping documentation (indices 0-23).

        Returns
        -------
        list[str]
            List of 24 SMPL joint names in mapping order.
        """
        return get_smpl_joint_names()


@define(kw_only=True)
class SMPLHKeypoints:
    """Keypoint data container for SMPL-H models (52 joints).

    Stores keypoint data for the 52 core joints present in SMPL-H models,
    following the exact order from the skeleton mapping documentation.

    Attributes
    ----------
    keypoints : Any, optional
        Keypoint position data (typically tensor or array of shape [..., 52, 3])
    confidences : Any, optional
        Keypoint confidence scores (typically tensor or array of shape [..., 52])
    """

    keypoints: Any | None = field(default=None)
    confidences: Any | None = field(default=None)

    @classmethod
    def get_ordered_names(cls) -> list[str]:
        """Get SMPL-H joint names in skeleton mapping order.

        Returns the 52 SMPL-H joint names in the exact order specified
        in the skeleton mapping documentation (indices 0-51).

        Returns
        -------
        list[str]
            List of 52 SMPL-H joint names in mapping order.
        """
        return get_smplh_joint_names()


@define(kw_only=True)
class SMPLXKeypoints:
    """Keypoint data container for SMPL-X models (55 joints).

    Stores keypoint data for the first 55 joints present in SMPL-X models,
    following the exact order from the skeleton mapping documentation.
    This forms the unified joint set used across the toolbox.

    Attributes
    ----------
    keypoints : Any, optional
        Keypoint position data (typically tensor or array of shape [..., 55, 3])
    confidences : Any, optional
        Keypoint confidence scores (typically tensor or array of shape [..., 55])
    """

    keypoints: Any | None = field(default=None)
    confidences: Any | None = field(default=None)

    @classmethod
    def get_ordered_names(cls) -> list[str]:
        """Get SMPL-X joint names in skeleton mapping order.

        Returns the first 55 SMPL-X joint names in the exact order specified
        in the skeleton mapping documentation (indices 0-54). This forms the
        unified joint set used throughout the toolbox.

        Returns
        -------
        list[str]
            List of 55 SMPL-X joint names in mapping order.
        """
        return get_smplx_joint_names()
