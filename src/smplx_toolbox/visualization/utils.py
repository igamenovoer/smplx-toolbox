"""Utility functions for SMPL visualization.

This module provides helper functions for visualization including bone connections
for different SMPL model variants and joint selection utilities.

Functions
---------
get_smpl_bone_connections : Get SMPL skeleton bone connections
get_smplh_bone_connections : Get SMPL-H skeleton bone connections
get_smplx_bone_connections : Get SMPL-X skeleton bone connections
create_polydata_from_vertices_faces : Convert vertices and faces to PyVista PolyData
resolve_joint_selection : Resolve joint selection to indices
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

from smplx_toolbox.core.constants import (
    CoreBodyJoint,
    FaceJoint,
    HandFingerJoint,
    ModelType,
    SMPLSpecialJoint,
    SMPL_JOINT_NAME_TO_INDEX,
    SMPLH_JOINT_NAME_TO_INDEX,
    SMPLX_JOINT_NAME_TO_INDEX,
)


def get_smpl_bone_connections() -> list[tuple[int, int]]:
    """Get SMPL skeleton bone connections (22 body joints + 2 hand end effectors).

    Returns the parent-child bone connections for the SMPL model.
    The connections form the skeleton structure of the model.

    Returns
    -------
    list[tuple[int, int]]
        List of (parent_index, child_index) pairs representing bones

    Notes
    -----
    SMPL has 24 joints total:
    - 22 core body joints (indices 0-21)
    - 2 hand end effectors (indices 22-23)
    """
    # Core body skeleton connections
    connections = [
        # Pelvis to hips and spine
        (0, 1),  # pelvis -> left_hip
        (0, 2),  # pelvis -> right_hip
        (0, 3),  # pelvis -> spine1
        # Left leg chain
        (1, 4),  # left_hip -> left_knee
        (4, 7),  # left_knee -> left_ankle
        (7, 10),  # left_ankle -> left_foot
        # Right leg chain
        (2, 5),  # right_hip -> right_knee
        (5, 8),  # right_knee -> right_ankle
        (8, 11),  # right_ankle -> right_foot
        # Spine chain
        (3, 6),  # spine1 -> spine2
        (6, 9),  # spine2 -> spine3
        (9, 12),  # spine3 -> neck
        (12, 15),  # neck -> head
        # Left arm chain
        (9, 13),  # spine3 -> left_collar
        (13, 16),  # left_collar -> left_shoulder
        (16, 18),  # left_shoulder -> left_elbow
        (18, 20),  # left_elbow -> left_wrist
        # Right arm chain
        (9, 14),  # spine3 -> right_collar
        (14, 17),  # right_collar -> right_shoulder
        (17, 19),  # right_shoulder -> right_elbow
        (19, 21),  # right_elbow -> right_wrist
        # Hand end effectors (SMPL-specific)
        (20, 22),  # left_wrist -> left_hand
        (21, 23),  # right_wrist -> right_hand
    ]

    return connections


def get_smplh_bone_connections() -> list[tuple[int, int]]:
    """Get SMPL-H skeleton including detailed hand connections (52 joints).

    Returns the parent-child bone connections for the SMPL-H model,
    including detailed finger joints.

    Returns
    -------
    list[tuple[int, int]]
        List of (parent_index, child_index) pairs representing bones

    Notes
    -----
    SMPL-H has 52 joints:
    - 22 core body joints (indices 0-21)
    - 15 left hand joints (indices 22-36)
    - 15 right hand joints (indices 37-51)
    """
    # Start with core body connections (same as SMPL but without end effectors)
    connections = [
        # Pelvis to hips and spine
        (0, 1),  # pelvis -> left_hip
        (0, 2),  # pelvis -> right_hip
        (0, 3),  # pelvis -> spine1
        # Left leg chain
        (1, 4),  # left_hip -> left_knee
        (4, 7),  # left_knee -> left_ankle
        (7, 10),  # left_ankle -> left_foot
        # Right leg chain
        (2, 5),  # right_hip -> right_knee
        (5, 8),  # right_knee -> right_ankle
        (8, 11),  # right_ankle -> right_foot
        # Spine chain
        (3, 6),  # spine1 -> spine2
        (6, 9),  # spine2 -> spine3
        (9, 12),  # spine3 -> neck
        (12, 15),  # neck -> head
        # Left arm chain
        (9, 13),  # spine3 -> left_collar
        (13, 16),  # left_collar -> left_shoulder
        (16, 18),  # left_shoulder -> left_elbow
        (18, 20),  # left_elbow -> left_wrist
        # Right arm chain
        (9, 14),  # spine3 -> right_collar
        (14, 17),  # right_collar -> right_shoulder
        (17, 19),  # right_shoulder -> right_elbow
        (19, 21),  # right_elbow -> right_wrist
    ]

    # Left hand finger connections (22-36)
    # Connect wrist to finger bases
    connections.extend(
        [
            (20, 22),  # left_wrist -> left_index1
            (20, 25),  # left_wrist -> left_middle1
            (20, 28),  # left_wrist -> left_pinky1
            (20, 31),  # left_wrist -> left_ring1
            (20, 34),  # left_wrist -> left_thumb1
        ]
    )

    # Left finger chains
    connections.extend(
        [
            (22, 23),
            (23, 24),  # left_index chain
            (25, 26),
            (26, 27),  # left_middle chain
            (28, 29),
            (29, 30),  # left_pinky chain
            (31, 32),
            (32, 33),  # left_ring chain
            (34, 35),
            (35, 36),  # left_thumb chain
        ]
    )

    # Right hand finger connections (37-51)
    # Connect wrist to finger bases
    connections.extend(
        [
            (21, 37),  # right_wrist -> right_index1
            (21, 40),  # right_wrist -> right_middle1
            (21, 43),  # right_wrist -> right_pinky1
            (21, 46),  # right_wrist -> right_ring1
            (21, 49),  # right_wrist -> right_thumb1
        ]
    )

    # Right finger chains
    connections.extend(
        [
            (37, 38),
            (38, 39),  # right_index chain
            (40, 41),
            (41, 42),  # right_middle chain
            (43, 44),
            (44, 45),  # right_pinky chain
            (46, 47),
            (47, 48),  # right_ring chain
            (49, 50),
            (50, 51),  # right_thumb chain
        ]
    )

    return connections


def get_smplx_bone_connections() -> list[tuple[int, int]]:
    """Get SMPL-X skeleton including face and hand connections (55 joints).

    Returns the parent-child bone connections for the SMPL-X model,
    including face joints and detailed finger joints.

    Returns
    -------
    list[tuple[int, int]]
        List of (parent_index, child_index) pairs representing bones

    Notes
    -----
    SMPL-X has 55 joints:
    - 22 core body joints (indices 0-21)
    - 3 face joints (indices 22-24)
    - 15 left hand joints (indices 25-39)
    - 15 right hand joints (indices 40-54)
    """
    # Start with core body connections
    connections = [
        # Pelvis to hips and spine
        (0, 1),  # pelvis -> left_hip
        (0, 2),  # pelvis -> right_hip
        (0, 3),  # pelvis -> spine1
        # Left leg chain
        (1, 4),  # left_hip -> left_knee
        (4, 7),  # left_knee -> left_ankle
        (7, 10),  # left_ankle -> left_foot
        # Right leg chain
        (2, 5),  # right_hip -> right_knee
        (5, 8),  # right_knee -> right_ankle
        (8, 11),  # right_ankle -> right_foot
        # Spine chain
        (3, 6),  # spine1 -> spine2
        (6, 9),  # spine2 -> spine3
        (9, 12),  # spine3 -> neck
        (12, 15),  # neck -> head
        # Left arm chain
        (9, 13),  # spine3 -> left_collar
        (13, 16),  # left_collar -> left_shoulder
        (16, 18),  # left_shoulder -> left_elbow
        (18, 20),  # left_elbow -> left_wrist
        # Right arm chain
        (9, 14),  # spine3 -> right_collar
        (14, 17),  # right_collar -> right_shoulder
        (17, 19),  # right_shoulder -> right_elbow
        (19, 21),  # right_elbow -> right_wrist
    ]

    # Face connections
    connections.extend(
        [
            (15, 22),  # head -> jaw
            (15, 23),  # head -> left_eye_smplhf
            (15, 24),  # head -> right_eye_smplhf
        ]
    )

    # Left hand finger connections (25-39)
    # Connect wrist to finger bases
    connections.extend(
        [
            (20, 25),  # left_wrist -> left_index1
            (20, 28),  # left_wrist -> left_middle1
            (20, 31),  # left_wrist -> left_pinky1
            (20, 34),  # left_wrist -> left_ring1
            (20, 37),  # left_wrist -> left_thumb1
        ]
    )

    # Left finger chains
    connections.extend(
        [
            (25, 26),
            (26, 27),  # left_index chain
            (28, 29),
            (29, 30),  # left_middle chain
            (31, 32),
            (32, 33),  # left_pinky chain
            (34, 35),
            (35, 36),  # left_ring chain
            (37, 38),
            (38, 39),  # left_thumb chain
        ]
    )

    # Right hand finger connections (40-54)
    # Connect wrist to finger bases
    connections.extend(
        [
            (21, 40),  # right_wrist -> right_index1
            (21, 43),  # right_wrist -> right_middle1
            (21, 46),  # right_wrist -> right_pinky1
            (21, 49),  # right_wrist -> right_ring1
            (21, 52),  # right_wrist -> right_thumb1
        ]
    )

    # Right finger chains
    connections.extend(
        [
            (40, 41),
            (41, 42),  # right_index chain
            (43, 44),
            (44, 45),  # right_middle chain
            (46, 47),
            (47, 48),  # right_pinky chain
            (49, 50),
            (50, 51),  # right_ring chain
            (52, 53),
            (53, 54),  # right_thumb chain
        ]
    )

    return connections


def create_polydata_from_vertices_faces(
    vertices: np.ndarray, faces: np.ndarray
) -> pv.PolyData:
    """Convert SMPL vertices and faces to PyVista PolyData.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions array of shape (N, 3)
    faces : np.ndarray
        Face connectivity array of shape (F, 3) or (F, 4)

    Returns
    -------
    pv.PolyData
        PyVista mesh object

    Notes
    -----
    The faces array is expected to contain vertex indices.
    PyVista expects faces in a flat format with the number of vertices
    per face as the first element.
    """
    # Ensure vertices are float32 for efficiency
    vertices = np.asarray(vertices, dtype=np.float32)

    # Convert faces to PyVista format
    # PyVista expects: [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
    faces = np.asarray(faces, dtype=np.int32)

    if faces.ndim == 2:
        n_faces, n_verts_per_face = faces.shape
        # Add the count column
        pv_faces = np.empty((n_faces, n_verts_per_face + 1), dtype=np.int32)
        pv_faces[:, 0] = n_verts_per_face
        pv_faces[:, 1:] = faces
        # Flatten for PyVista
        pv_faces = pv_faces.ravel()
    else:
        # Assume already in PyVista format
        pv_faces = faces

    # Create PolyData
    mesh = pv.PolyData(vertices, pv_faces)
    return mesh


def resolve_joint_selection(
    joints: list[str] | list[int] | None, model_type: ModelType
) -> list[int]:
    """Resolve joint selection to indices for the given model type.

    Parameters
    ----------
    joints : list[str] | list[int] | None
        Joint selection. Can be:
        - None: return all joints for the model type
        - List of joint names (from constants enums)
        - List of joint indices (model-specific)
        Special keywords as elements: 'body', 'hands', 'face', 'all'
    model_type : ModelType
        The SMPL model type

    Returns
    -------
    list[int]
        List of joint indices for the model

    Raises
    ------
    ValueError
        If joint name is not found in the model
    KeyError
        If keyword is not recognized
    """
    # Get the appropriate mapping for this model type
    if model_type == ModelType.SMPL:
        name_to_index = SMPL_JOINT_NAME_TO_INDEX
        max_joints = 24
    elif model_type == ModelType.SMPLH:
        name_to_index = SMPLH_JOINT_NAME_TO_INDEX
        max_joints = 52
    elif model_type == ModelType.SMPLX:
        name_to_index = SMPLX_JOINT_NAME_TO_INDEX
        max_joints = 55
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # If None, return all joints
    if joints is None:
        return list(range(max_joints))

    # Process the joint selection
    indices: list[int] = []

    for joint in joints:
        if isinstance(joint, str):
            # Check for special keywords
            if joint == "body":
                # Core 22 body joints (0-21)
                indices.extend(range(22))
            elif joint == "hands":
                # Hand joints vary by model
                if model_type == ModelType.SMPL:
                    # SMPL has hand end effectors at 22, 23
                    indices.extend([22, 23])
                elif model_type == ModelType.SMPLH:
                    # SMPL-H has detailed fingers at 22-51
                    indices.extend(range(22, 52))
                elif model_type == ModelType.SMPLX:
                    # SMPL-X has detailed fingers at 25-54
                    indices.extend(range(25, 55))
            elif joint == "face":
                # Face joints only in SMPL-X
                if model_type == ModelType.SMPLX:
                    indices.extend([22, 23, 24])  # jaw, left_eye, right_eye
            elif joint == "all":
                indices.extend(range(max_joints))
            else:
                # Look up joint name
                if joint in name_to_index:
                    indices.append(name_to_index[joint])
                else:
                    # Try looking up as enum value
                    found = False
                    # Check CoreBodyJoint
                    for core_joint in CoreBodyJoint:
                        if joint == core_joint or joint == core_joint.value:
                            if core_joint.value in name_to_index:
                                indices.append(name_to_index[core_joint.value])
                                found = True
                                break

                    if not found and model_type in [ModelType.SMPLH, ModelType.SMPLX]:
                        # Check HandFingerJoint
                        for hand_joint in HandFingerJoint:
                            if joint == hand_joint or joint == hand_joint.value:
                                if hand_joint.value in name_to_index:
                                    indices.append(name_to_index[hand_joint.value])
                                    found = True
                                    break

                    if not found and model_type == ModelType.SMPLX:
                        # Check FaceJoint
                        for face_joint in FaceJoint:
                            if joint == face_joint or joint == face_joint.value:
                                if face_joint.value in name_to_index:
                                    indices.append(name_to_index[face_joint.value])
                                    found = True
                                    break

                    if not found and model_type == ModelType.SMPL:
                        # Check SMPLSpecialJoint
                        for special_joint in SMPLSpecialJoint:
                            if joint == special_joint or joint == special_joint.value:
                                if special_joint.value in name_to_index:
                                    indices.append(name_to_index[special_joint.value])
                                    found = True
                                    break

                    if not found:
                        raise ValueError(
                            f"Joint '{joint}' not found in {model_type} model"
                        )

        elif isinstance(joint, int):
            # Direct index
            if 0 <= joint < max_joints:
                indices.append(joint)
            else:
                raise ValueError(
                    f"Joint index {joint} out of range for {model_type} (0-{max_joints - 1})"
                )
        else:
            raise TypeError(f"Joint must be str or int, got {type(joint)}")

    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    return unique_indices
