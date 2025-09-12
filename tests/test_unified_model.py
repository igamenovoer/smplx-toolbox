#!/usr/bin/env python
"""Test UnifiedSmplModel implementation."""

import torch
import torch.nn as nn

from smplx_toolbox.core import (
    UnifiedSmplInputs,
    UnifiedSmplModel,
    UnifiedSmplOutput,
    NamedPose,
)
from smplx_toolbox.core.constants import ModelType


class MockSMPLModel(nn.Module):
    """Mock SMPL model for testing without actual model files."""

    def __init__(self, model_type="smplx"):
        super().__init__()
        self.model_type = model_type
        self.batch_size = 1

        # Mock parameters for detection
        if model_type == "smplx":
            self.num_joints = 55
            self.num_verts = 10475
            # Add SMPL-X specific attributes for detection
            self.jaw_pose = nn.Parameter(torch.zeros(1, 3))
            self.leye_pose = nn.Parameter(torch.zeros(1, 3))
            self.reye_pose = nn.Parameter(torch.zeros(1, 3))
            self.left_hand_pose = nn.Parameter(torch.zeros(1, 45))
            self.right_hand_pose = nn.Parameter(torch.zeros(1, 45))
        elif model_type == "smplh":
            self.num_joints = 52
            self.num_verts = 6890
            # Add SMPL-H specific attributes for detection
            self.left_hand_pose = nn.Parameter(torch.zeros(1, 45))
            self.right_hand_pose = nn.Parameter(torch.zeros(1, 45))
        else:  # smpl
            self.num_joints = 24
            self.num_verts = 6890
            # SMPL has no hand or face parameters

        # Mock faces
        self.faces = torch.randint(0, self.num_verts, (13776, 3))

        # Mock joint names
        self.joint_names = [f"joint_{i}" for i in range(self.num_joints)]

    def forward(self, **kwargs):
        """Mock forward pass."""
        batch_size = kwargs.get("betas", torch.zeros(1, 10)).shape[0]

        # Create mock output
        output = type("Output", (), {})()
        output.vertices = torch.randn(batch_size, self.num_verts, 3)
        output.joints = torch.randn(batch_size, self.num_joints, 3)
        output.full_pose = torch.zeros(batch_size, self.num_joints, 3, 3)

        # Add identity matrices for rotation
        for i in range(self.num_joints):
            output.full_pose[:, i] = torch.eye(3)

        return output


def test_unified_model_creation():
    """Test UnifiedSmplModel creation."""
    print("\n" + "=" * 60)
    print("  Testing UnifiedSmplModel Creation")
    print("=" * 60)

    # Test with SMPL-X mock
    print("\n[TEST 1] Creating UnifiedSmplModel from SMPL-X...")
    mock_smplx = MockSMPLModel("smplx")
    unified = UnifiedSmplModel.from_smpl_model(mock_smplx)
    print(f"  [OK] Created, model_type={unified.model_type}")
    assert unified.model_type == "smplx"

    # Test with SMPL-H mock
    print("\n[TEST 2] Creating UnifiedSmplModel from SMPL-H...")
    mock_smplh = MockSMPLModel("smplh")
    unified = UnifiedSmplModel.from_smpl_model(mock_smplh)
    print(f"  [OK] Created, model_type={unified.model_type}")
    assert unified.model_type == "smplh"

    # Test with SMPL mock
    print("\n[TEST 3] Creating UnifiedSmplModel from SMPL...")
    mock_smpl = MockSMPLModel("smpl")
    unified = UnifiedSmplModel.from_smpl_model(mock_smpl)
    print(f"  [OK] Created, model_type={unified.model_type}")
    assert unified.model_type == "smpl"

    print("\n[SUCCESS] All creation tests passed!")


def test_unified_model_forward():
    """Test UnifiedSmplModel forward pass."""
    print("\n" + "=" * 60)
    print("  Testing UnifiedSmplModel Forward Pass")
    print("=" * 60)

    mock_model = MockSMPLModel("smplx")
    unified = UnifiedSmplModel.from_smpl_model(mock_model)

    # Create inputs
    print("\n[TEST 1] Forward pass with UnifiedSmplInputs...")
    npz = NamedPose(model_type=ModelType.SMPLX, batch_size=2)
    inputs = UnifiedSmplInputs(
        betas=torch.randn(2, 10),
        named_pose=npz,
    )

    output = unified(inputs)
    print(
        f"  [OK] Output shape: vertices={output.vertices.shape}, joints={output.joints.shape}"
    )
    assert output.vertices.shape == (2, 10475, 3)
    assert output.joints.shape == (2, 55, 3)

    print("\n[SUCCESS] Forward pass test passed!")


def test_unified_output():
    """Test UnifiedSmplOutput functionality."""
    print("\n" + "=" * 60)
    print("  Testing UnifiedSmplOutput")
    print("=" * 60)

    # Create mock output
    output = UnifiedSmplOutput(
        vertices=torch.randn(1, 10475, 3),
        joints=torch.randn(1, 55, 3),
        faces=torch.randint(0, 10475, (13776, 3)),
        full_pose=torch.randn(1, 165),  # SMPL-X pose size
    )

    print("\n[TEST 1] Output properties...")
    print(f"  - num_vertices: {output.num_vertices}")
    print(f"  - num_joints: {output.num_joints}")
    print(f"  - num_faces: {output.num_faces}")
    print(f"  - batch_size: {output.batch_size}")

    assert output.num_vertices == 10475
    assert output.num_joints == 55
    assert output.num_faces == 13776
    assert output.batch_size == 1

    print("\n[TEST 2] Body parts access...")
    print(f"  - body_joints shape: {output.body_joints.shape}")
    print(f"  - hand_joints shape: {output.hand_joints.shape}")
    print(f"  - face_joints shape: {output.face_joints.shape}")

    assert output.body_joints.shape == (1, 22, 3)
    assert output.hand_joints.shape == (1, 30, 3)  # 15 per hand
    assert output.face_joints.shape == (1, 3, 3)  # jaw + 2 eyes

    print("\n[SUCCESS] All output tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("  UnifiedSmplModel Test Suite")
    print("=" * 60)

    try:
        test_unified_model_creation()
        test_unified_model_forward()
        test_unified_output()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
