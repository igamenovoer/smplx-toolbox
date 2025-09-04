"""
Test suite for SMPL-X Toolbox core functionality.
"""

import pytest
import numpy as np
from smplx_toolbox import __version__


def test_version():
    """Test that version is properly defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_package_imports():
    """Test that main package imports work correctly."""
    try:
        import smplx_toolbox
        import smplx_toolbox.core
        import smplx_toolbox.optimization
        import smplx_toolbox.visualization
        import smplx_toolbox.utils
    except ImportError as e:
        pytest.fail(f"Failed to import package modules: {e}")


class TestSampleParameters:
    """Test parameter validation and manipulation."""
    
    def test_parameter_structure(self, sample_parameters):
        """Test that sample parameters have correct structure."""
        required_keys = [
            "betas", "global_orient", "body_pose", 
            "left_hand_pose", "right_hand_pose",
            "jaw_pose", "leye_pose", "reye_pose",
            "expression", "transl"
        ]
        
        for key in required_keys:
            assert key in sample_parameters
            assert isinstance(sample_parameters[key], np.ndarray)
    
    def test_parameter_shapes(self, sample_parameters):
        """Test that parameters have expected shapes."""
        expected_shapes = {
            "betas": (10,),
            "global_orient": (3,),
            "body_pose": (63,),
            "left_hand_pose": (45,),
            "right_hand_pose": (45,),
            "jaw_pose": (3,),
            "leye_pose": (3,),
            "reye_pose": (3,),
            "expression": (50,),
            "transl": (3,),
        }
        
        for key, expected_shape in expected_shapes.items():
            assert sample_parameters[key].shape == expected_shape


class TestSampleMeshData:
    """Test mesh data structure and validation."""
    
    def test_mesh_structure(self, sample_mesh_data):
        """Test that mesh data has correct structure."""
        assert "vertices" in sample_mesh_data
        assert "faces" in sample_mesh_data
        assert isinstance(sample_mesh_data["vertices"], np.ndarray)
        assert isinstance(sample_mesh_data["faces"], np.ndarray)
    
    def test_mesh_shapes(self, sample_mesh_data):
        """Test that mesh has expected dimensions."""
        vertices = sample_mesh_data["vertices"]
        faces = sample_mesh_data["faces"]
        
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3  # x, y, z coordinates
        assert faces.ndim == 2
        assert faces.shape[1] == 3  # triangular faces
        
        # Verify face indices are valid
        assert np.all(faces >= 0)
        assert np.all(faces < len(vertices))


class TestSampleLandmarks:
    """Test landmark data structure and validation."""
    
    def test_landmark_structure(self, sample_landmarks):
        """Test that landmark data has correct structure."""
        required_keys = ["landmarks_2d", "landmarks_3d", "body_joints"]
        
        for key in required_keys:
            assert key in sample_landmarks
            assert isinstance(sample_landmarks[key], np.ndarray)
    
    def test_landmark_dimensions(self, sample_landmarks):
        """Test that landmarks have correct dimensions."""
        assert sample_landmarks["landmarks_2d"].shape[1] == 2
        assert sample_landmarks["landmarks_3d"].shape[1] == 3
        assert sample_landmarks["body_joints"].shape[1] == 3
