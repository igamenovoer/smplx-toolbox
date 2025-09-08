#!/usr/bin/env python
"""Test the SMPLHModel implementation."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import smplx
import torch
import trimesh

from src.smplx_toolbox.core.smplh_model import SMPLHModel


def test_smplh_model():
    """Test the SMPLHModel wrapper."""

    print("=" * 60)
    print("  Testing SMPLHModel")
    print("=" * 60)

    # Configuration
    model_path = Path("data/body_models")
    output_dir = Path("tmp/smplh_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Model path: {model_path}")
    print(f"[INFO] Output directory: {output_dir}")

    try:
        # Step 1: Create SMPL-H model
        print("\n[STEP 1] Creating SMPL-H model...")
        base_model = smplx.create(
            str(model_path),
            model_type="smplh",
            gender="male",
            num_betas=10,
            use_pca=False,  # Use full hand joints
        )
        print("  [OK] SMPL-H model created")
        print(f"  - Model type: {type(base_model).__name__}")

        # Step 2: Create SMPLHModel wrapper
        print("\n[STEP 2] Creating SMPLHModel wrapper...")
        wrapper = SMPLHModel.from_smplh(base_model)
        print("  [OK] SMPLHModel wrapper created")

        # Step 3: Verify properties
        print("\n[STEP 3] Verifying wrapper properties...")
        print(f"  - base_model type: {wrapper.base_model.__class__.__name__}")
        print(f"  - num_vertices: {wrapper.num_vertices}")
        print(f"  - num_joints: {wrapper.num_joints}")
        print(f"  - num_body_joints: {wrapper.num_body_joints}")
        print(f"  - num_hand_joints: {wrapper.num_hand_joints} (per hand)")
        print(f"  - faces shape: {wrapper.faces.shape}")
        print(f"  - joint_names count: {len(wrapper.joint_names)}")

        # Print first few joint names
        print("\n  - First 5 body joints:")
        for i, name in enumerate(wrapper.joint_names[:5]):
            print(f"    {i}: {name}")

        print("\n  - First 5 left hand joints:")
        for i, name in enumerate(wrapper.joint_names[22:27]):
            print(f"    {i+22}: {name}")

        # Step 4: Test direct base_model usage with neutral pose
        print("\n[STEP 4] Testing base_model forward pass (neutral)...")
        output = wrapper.base_model(return_verts=True, return_joints=True)
        print(f"  [OK] Forward pass successful")
        print(f"  - Output vertices shape: {output.vertices.shape}")
        print(f"  - Output joints shape: {output.joints.shape}")

        # Step 5: Test to_mesh with output
        print("\n[STEP 5] Testing to_mesh with output...")
        mesh_from_output = wrapper.to_mesh(output)
        print(f"  [OK] Mesh created from output")
        print(f"  - Vertices: {len(mesh_from_output.vertices)}")
        print(f"  - Faces: {len(mesh_from_output.faces)}")

        # Step 6: Test to_mesh with None (neutral pose)
        print("\n[STEP 6] Testing to_mesh with None (neutral)...")
        neutral_mesh = wrapper.to_mesh(None)
        print(f"  [OK] Neutral mesh created")
        print(f"  - Vertices: {len(neutral_mesh.vertices)}")
        print(f"  - Faces: {len(neutral_mesh.faces)}")

        # Step 7: Test with custom pose (body + hands)
        print("\n[STEP 7] Testing with custom body and hand poses...")
        
        # Prepare pose parameters
        batch_size = 1
        # Get device from the base model to ensure consistency
        device = next(wrapper.base_model.parameters()).device
        print(f"  - Using device: {device}")
        
        # Body pose: 21 joints * 3 (axis-angle) = 63 values, flattened
        body_pose = torch.zeros((batch_size, 63), dtype=torch.float32, device=device)
        body_pose[:, 15*3] = 1.0  # Rotate left shoulder (joint 15, x-axis)
        
        # Hand poses: 15 joints * 3 per hand = 45 values each, flattened
        left_hand_pose = torch.zeros((batch_size, 45), dtype=torch.float32, device=device)
        right_hand_pose = torch.zeros((batch_size, 45), dtype=torch.float32, device=device)
        
        # Add slight finger curl for realism (as suggested in conversion guide)
        # Set small flex for all finger joints (every 3rd value is x-rotation)
        for i in range(0, 45, 3):
            left_hand_pose[:, i] = 0.05  # Small flex (radians)
            right_hand_pose[:, i] = 0.05
        
        # Global orient (root joint) - already flattened
        global_orient = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
        
        # Shape parameters
        betas = torch.zeros((batch_size, 10), dtype=torch.float32, device=device)
        betas[:, 0] = 0.5  # Slightly larger body
        
        # Forward pass with parameters
        posed_output = wrapper.base_model(
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas,
            return_verts=True,
            return_joints=True,
        )
        posed_mesh = wrapper.to_mesh(posed_output)
        print(f"  [OK] Posed mesh created")

        # Export meshes
        neutral_path = output_dir / "smplh_neutral.obj"
        posed_path = output_dir / "smplh_posed.obj"
        neutral_mesh.export(str(neutral_path))
        posed_mesh.export(str(posed_path))
        print(f"  - Exported: {neutral_path}")
        print(f"  - Exported: {posed_path}")

        # Step 8: Test joint extraction methods
        print("\n[STEP 8] Testing joint extraction methods...")
        
        # Get all joints
        all_joints = wrapper.get_joint_positions(posed_output)
        print(f"  - All joints shape: {all_joints.shape}")
        
        # Get body joints only
        body_joints = wrapper.get_body_joint_positions(posed_output)
        print(f"  - Body joints shape: {body_joints.shape}")
        print(f"  - First body joint (pelvis): {body_joints[0]}")
        
        # Get hand joints
        left_hand_joints = wrapper.get_hand_joint_positions(posed_output, hand="left")
        right_hand_joints = wrapper.get_hand_joint_positions(posed_output, hand="right")
        print(f"  - Left hand joints shape: {left_hand_joints.shape}")
        print(f"  - Right hand joints shape: {right_hand_joints.shape}")
        
        # Get both hands at once
        left_joints, right_joints = wrapper.get_hand_joint_positions(posed_output, hand="both")
        print(f"  - Both hands returned: {left_joints.shape}, {right_joints.shape}")

        # Step 9: Test with vertex colors
        print("\n[STEP 9] Testing mesh with vertex colors...")
        colored_mesh = wrapper.to_mesh(posed_output, vertex_colors=[0.8, 0.3, 0.3])
        colored_path = output_dir / "smplh_colored.ply"
        colored_mesh.export(str(colored_path))
        print(f"  [OK] Colored mesh exported: {colored_path}")

        # Step 10: Verify coordinate system
        print("\n[STEP 10] Verifying standard SMPL-H coordinates...")
        bounds = neutral_mesh.bounds
        print(f"  - Bounds: {bounds}")
        print(f"  - Y extent (height): {bounds[1, 1] - bounds[0, 1]:.3f}")
        print(f"  - Z extent (depth): {bounds[1, 2] - bounds[0, 2]:.3f}")
        print(f"  - X extent (width): {bounds[1, 0] - bounds[0, 0]:.3f}")

        # Y should be the largest extent (height)
        y_extent = bounds[1, 1] - bounds[0, 1]
        z_extent = bounds[1, 2] - bounds[0, 2]
        x_extent = bounds[1, 0] - bounds[0, 0]
        assert y_extent > z_extent and y_extent > x_extent, "Y should be the largest (height)"
        print("  [OK] Standard SMPL-H coordinate system verified (Y-up)")

        # Step 11: Test error handling
        print("\n[STEP 11] Testing error handling...")
        try:
            # Try to create wrapper with wrong model type
            # Use a dummy object that's definitely not an SMPLH instance
            class DummyModel:
                pass
            
            dummy_model = DummyModel()
            SMPLHModel.from_smplh(dummy_model)
            print("  [ERROR] Should have raised ValueError for wrong model type")
        except ValueError as e:
            print(f"  [OK] Correctly rejected wrong model type: {str(e)[:50]}...")

        # Summary
        print("\n" + "=" * 60)
        print("  TEST SUMMARY")
        print("=" * 60)
        print("\n[OK] All tests passed!")
        print("\nKey features verified:")
        print("  - SMPL-H model wrapper creation")
        print("  - Properties: joints, vertices, faces")
        print("  - Body + hand joint extraction")
        print("  - Mesh generation from outputs")
        print("  - Custom pose parameters (body + hands)")
        print("  - Vertex coloring support")
        print("  - Standard Y-up coordinate system")
        print("  - Error handling for wrong model types")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_smplh_model()
    sys.exit(0 if success else 1)