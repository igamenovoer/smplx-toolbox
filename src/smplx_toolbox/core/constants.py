"""Constants and type aliases for the Unified SMPL Model system.

This module contains joint names, type definitions, and other constants used
throughout the unified SMPL family model implementation.
"""

from __future__ import annotations

from typing import TypeVar
from enum import StrEnum

import torch


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
    """Generic finger joint names (without left/right prefix).
    
    These are the base names for the 15 joints per hand.
    In practice, they're prefixed with 'left_' or 'right_'.
    """
    
    THUMB1 = "thumb1"
    THUMB2 = "thumb2"
    THUMB3 = "thumb3"
    INDEX1 = "index1"
    INDEX2 = "index2"
    INDEX3 = "index3"
    MIDDLE1 = "middle1"
    MIDDLE2 = "middle2"
    MIDDLE3 = "middle3"
    RING1 = "ring1"
    RING2 = "ring2"
    RING3 = "ring3"
    PINKY1 = "pinky1"
    PINKY2 = "pinky2"
    PINKY3 = "pinky3"


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


def get_hand_joints(side: str) -> list[str]:
    """Get all hand joint names for a specific side.
    
    Parameters
    ----------
    side : str
        Either 'left' or 'right'.
        
    Returns
    -------
    list[str]
        List of 15 hand joint names with the appropriate prefix.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side}")
    
    # Order matters - must match SMPL-H/SMPL-X joint ordering
    finger_order = [
        HandFingerJoint.INDEX1, HandFingerJoint.INDEX2, HandFingerJoint.INDEX3,
        HandFingerJoint.MIDDLE1, HandFingerJoint.MIDDLE2, HandFingerJoint.MIDDLE3,
        HandFingerJoint.PINKY1, HandFingerJoint.PINKY2, HandFingerJoint.PINKY3,
        HandFingerJoint.RING1, HandFingerJoint.RING2, HandFingerJoint.RING3,
        HandFingerJoint.THUMB1, HandFingerJoint.THUMB2, HandFingerJoint.THUMB3,
    ]
    
    return [f"{side}_{joint.value}" for joint in finger_order]


def get_smpl_joint_names() -> list[str]:
    """Get the official SMPL joint names (24 joints).
    
    Returns
    -------
    list[str]
        List of 24 SMPL joint names.
    """
    joints: list[str] = [joint.value for joint in CoreBodyJoint]  # First 22 joints
    joints.extend([SMPLSpecialJoint.LEFT_HAND.value, SMPLSpecialJoint.RIGHT_HAND.value])
    return joints


def get_smplh_joint_names() -> list[str]:
    """Get the official SMPL-H joint names (52 joints).
    
    Returns
    -------
    list[str]
        List of 52 SMPL-H joint names (22 body + 30 hand joints).
    """
    joints: list[str] = [joint.value for joint in CoreBodyJoint]  # First 22 joints
    joints.extend(get_hand_joints("left"))  # 15 left hand joints
    joints.extend(get_hand_joints("right"))  # 15 right hand joints
    return joints


def get_smplx_joint_names() -> list[str]:
    """Get the official SMPL-X joint names (first 55 joints).
    
    Returns
    -------
    list[str]
        List of 55 SMPL-X joint names forming the unified set.
    """
    joints: list[str] = [joint.value for joint in CoreBodyJoint]  # First 22 joints
    joints.extend([joint.value for joint in FaceJoint])  # 3 face joints (jaw, eyes)
    joints.extend(get_hand_joints("left"))  # 15 left hand joints
    joints.extend(get_hand_joints("right"))  # 15 right hand joints
    return joints


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