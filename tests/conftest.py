"""
Test configuration and fixtures for SMPL-X Toolbox.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_parameters():
    """Sample SMPL-X parameters for testing."""
    return {
        "betas": np.random.randn(10),  # Shape parameters
        "global_orient": np.random.randn(3),  # Global orientation
        "body_pose": np.random.randn(63),  # Body pose parameters
        "left_hand_pose": np.random.randn(45),  # Left hand pose
        "right_hand_pose": np.random.randn(45),  # Right hand pose
        "jaw_pose": np.random.randn(3),  # Jaw pose
        "leye_pose": np.random.randn(3),  # Left eye pose
        "reye_pose": np.random.randn(3),  # Right eye pose
        "expression": np.random.randn(50),  # Facial expression
        "transl": np.random.randn(3),  # Global translation
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_mesh_data():
    """Sample mesh data for testing."""
    vertices = np.random.randn(10475, 3)  # SMPL-X vertex count
    faces = np.random.randint(0, 10475, (20906, 3))  # Approximate face count
    return {"vertices": vertices, "faces": faces}


@pytest.fixture
def sample_landmarks():
    """Sample landmark data for testing."""
    return {
        "landmarks_2d": np.random.randn(68, 2),  # 2D facial landmarks
        "landmarks_3d": np.random.randn(68, 3),  # 3D facial landmarks
        "body_joints": np.random.randn(25, 3),  # SMPL-X body joints
    }
