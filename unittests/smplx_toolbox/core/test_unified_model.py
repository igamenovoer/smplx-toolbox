#!/usr/bin/env python
"""
Unit tests for UnifiedSmplModel and related data containers using real SMPL family models.

This suite:
- Loads official SMPL/SMPL-H/SMPL-X models from data/body_models via the local smplx implementation
- Verifies UnifiedSmplModel API, normalization, unified joint set, and full_pose composition
- Tests input containers and keypoint conversion behaviors independent of model
- Confirms faces dtype/shape contract and adapter utility methods

Notes
- Requires model files under:
    data/body_models/smpl/
    data/body_models/smplh/
    data/body_models/smplx/
- Skips tests gracefully if models or smplx source not found.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

# Make local smplx implementation importable (context/refcode/smplx/smplx/*)
_SMPXLIB_ROOT = Path("context/refcode/smplx")
if _SMPXLIB_ROOT.exists():
    sys.path.insert(0, str(_SMPXLIB_ROOT.resolve()))

try:
    # Official API entrypoint from local copy
    from smplx import create as smplx_create  # type: ignore
    # Also import class to support file-based NPZ in nested folders
    from smplx import SMPLH as _SMPLHClass  # type: ignore
except Exception as _imp_err:  # pragma: no cover
    pytest.skip(f"Local smplx source not importable: {_imp_err}", allow_module_level=True)

from smplx_toolbox.core import (
    UnifiedSmplModel,
    UnifiedSmplInputs,
    PoseByKeypoints,
    UnifiedSmplOutput,
)

# ------------------------------------------------------------------------
# Environment checks
# ------------------------------------------------------------------------

_MODEL_ROOT = Path("data/body_models").resolve()

def _has_model_dir(model_type: str) -> bool:
    return (_MODEL_ROOT / model_type).exists()


def _smplh_npz_has_hand_keys(path: Path) -> bool:
    """Check whether an SMPL-H npz contains hand keys expected by smplx.

    Required keys include hands_componentsl/r and hands_meanl/r.
    """
    try:
        data = np.load(str(path), allow_pickle=True)
        keys = set(data.files)
        return {
            "hands_componentsl",
            "hands_componentsr",
            "hands_meanl",
            "hands_meanr",
        }.issubset(keys)
    except Exception:
        return False


# ------------------------------------------------------------------------
# Fixtures: Real model loaders
# ------------------------------------------------------------------------

@pytest.fixture(scope="session")
def smpl_model() -> Any:
    if not _has_model_dir("smpl"):
        pytest.skip("SMPL model directory missing under data/body_models/smpl")
    return smplx_create(model_path=str(_MODEL_ROOT), model_type="smpl")


@pytest.fixture(scope="session")
def smplh_model() -> Any:
    if not _has_model_dir("smplh"):
        pytest.skip("SMPL-H model directory missing under data/body_models/smplh")
    smplh_dir = _MODEL_ROOT / "smplh"
    # Prefer 16-beta NPZ models if present; support nested gender folders
    for gender in ("neutral", "male", "female"):
        npz = smplh_dir / f"SMPLH_{gender.upper()}.npz"
        if npz.exists() and _smplh_npz_has_hand_keys(npz):
            return smplx_create(
                model_path=str(_MODEL_ROOT),
                model_type="smplh",
                gender=gender,
                ext="npz",
                num_betas=16,
            )
        # Nested folder case: data/body_models/smplh/{gender}/model.npz
        nested_npz = smplh_dir / gender / "model.npz"
        if nested_npz.exists() and _smplh_npz_has_hand_keys(nested_npz):
            # Direct class construction supports file path + ext
            return _SMPLHClass(
                model_path=str(nested_npz),
                gender=gender,
                ext="npz",
                num_betas=16,
                use_pca=False,
            )
    # Fallback to PKL if NPZ not available
    for gender in ("neutral", "male", "female"):
        pkl = smplh_dir / f"SMPLH_{gender.upper()}.pkl"
        if pkl.exists():
            return smplx_create(
                model_path=str(_MODEL_ROOT),
                model_type="smplh",
                gender=gender,
            )
    pytest.skip("SMPL-H model file not available (no NPZ or PKL found)")


@pytest.fixture(scope="session")
def smplx_model() -> Any:
    if not _has_model_dir("smplx"):
        pytest.skip("SMPL-X model directory missing under data/body_models/smplx")
    # For SMPL-X, npz expected; the create() handles defaults
    return smplx_create(model_path=str(_MODEL_ROOT), model_type="smplx")


@pytest.fixture
def batch2_inputs() -> UnifiedSmplInputs:
    # Common minimal inputs (B=2)
    return UnifiedSmplInputs(
        betas=torch.randn(2, 10),
        root_orient=torch.randn(2, 3),
        pose_body=torch.randn(2, 63),
    )


# ------------------------------------------------------------------------
# Tests: Factory and detection (real models)
# ------------------------------------------------------------------------

def test_factory_and_detection_real(smpl_model: Any, smplh_model: Any, smplx_model: Any) -> None:
    uni = UnifiedSmplModel.from_smpl_model(smplx_model)
    assert uni.model_type == "smplx"

    uni = UnifiedSmplModel.from_smpl_model(smplh_model)
    assert uni.model_type == "smplh"

    uni = UnifiedSmplModel.from_smpl_model(smpl_model)
    assert uni.model_type == "smpl"


# ------------------------------------------------------------------------
# Tests: UnifiedSmplInputs basic props and validation (no model needed)
# ------------------------------------------------------------------------

def test_inputs_computed_properties() -> None:
    B = 3
    inputs = UnifiedSmplInputs(
        left_hand_pose=torch.randn(B, 45),
        right_hand_pose=torch.randn(B, 45),
        left_eye_pose=torch.randn(B, 3),
        right_eye_pose=torch.randn(B, 3),
    )
    assert inputs.hand_pose is not None
    assert inputs.hand_pose.shape == (B, 90)
    assert inputs.eyes_pose is not None
    assert inputs.eyes_pose.shape == (B, 6)


def test_inputs_validation_rules_smpl_disallows_face_and_hands() -> None:
    B = 1
    inputs = UnifiedSmplInputs(
        root_orient=torch.zeros(B, 3),
        pose_body=torch.zeros(B, 63),
        left_hand_pose=torch.zeros(B, 45),
        right_hand_pose=torch.zeros(B, 45),
        expression=torch.zeros(B, 10),
        pose_jaw=torch.zeros(B, 3),
        left_eye_pose=torch.zeros(B, 3),
        right_eye_pose=torch.zeros(B, 3),
    )
    with pytest.raises(ValueError):
        inputs.check_valid("smpl")


def test_inputs_validation_rules_smplh_requires_both_hands() -> None:
    B = 2
    inputs = UnifiedSmplInputs(
        root_orient=torch.zeros(B, 3),
        pose_body=torch.zeros(B, 63),
        left_hand_pose=torch.zeros(B, 45),
    )
    with pytest.raises(ValueError):
        inputs.check_valid("smplh")


def test_inputs_validation_rules_smplx_pairwise_hands_and_eyes() -> None:
    B = 2
    # Only one eye provided
    inputs = UnifiedSmplInputs(
        root_orient=torch.zeros(B, 3),
        pose_body=torch.zeros(B, 63),
        left_eye_pose=torch.zeros(B, 3),
    )
    with pytest.raises(ValueError):
        inputs.check_valid("smplx")

    # Only one hand provided
    inputs = UnifiedSmplInputs(
        root_orient=torch.zeros(B, 3),
        pose_body=torch.zeros(B, 63),
        right_hand_pose=torch.zeros(B, 45),
    )
    with pytest.raises(ValueError):
        inputs.check_valid("smplx")


# ------------------------------------------------------------------------
# Tests: PoseByKeypoints conversion and validation (no model needed)
# ------------------------------------------------------------------------

def test_pose_by_keypoints_to_inputs_smplx_minimal() -> None:
    B = 2
    kpts = PoseByKeypoints(
        root=torch.randn(B, 3),
        left_elbow=torch.randn(B, 3),
        right_elbow=torch.randn(B, 3),
        jaw=torch.randn(B, 3),
        left_eyeball=torch.randn(B, 3),
        right_eye=torch.randn(B, 3),
    )
    # Validate (should accept subset and zero-fill)
    kpts.check_valid_by_keypoints("smplx", strict=False)

    # Convert
    inputs = UnifiedSmplInputs.from_keypoint_pose(kpts, model_type="smplx")
    assert inputs.root_orient is not None
    assert inputs.pose_body is not None
    assert inputs.pose_body.shape == (B, 63)
    assert inputs.pose_jaw is not None and inputs.pose_jaw.shape == (B, 3)
    assert inputs.left_eye_pose is not None and inputs.right_eye_pose is not None
    assert inputs.left_eye_pose.shape == (B, 3) and inputs.right_eye_pose.shape == (B, 3)


def test_pose_by_keypoints_to_inputs_smplh_drops_face() -> None:
    B = 1
    kpts = PoseByKeypoints(
        root=torch.randn(B, 3),
        jaw=torch.randn(B, 3),
        left_eye=torch.randn(B, 3),
        right_eye=torch.randn(B, 3),
        left_index1=torch.randn(B, 3),
    )
    kpts.check_valid_by_keypoints("smplh", strict=False)
    inputs = UnifiedSmplInputs.from_keypoint_pose(kpts, model_type="smplh")
    assert inputs.pose_jaw is None
    assert inputs.left_eye_pose is None
    assert inputs.right_eye_pose is None
    assert inputs.left_hand_pose is not None and inputs.right_hand_pose is not None
    assert inputs.left_hand_pose.shape == (B, 45)
    assert inputs.right_hand_pose.shape == (B, 45)


# ------------------------------------------------------------------------
# Tests: Forward pass and outputs (real models)
# ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_type,expected_V,expected_unified_J,expected_P",
    [
        ("smplx", 10475, 55, 165),  # root(3)+body(63)+jaw(3)+eyes(6)+hands(90)
        ("smplh", 6890, 55, 156),   # root(3)+body(63)+hands(90)
        ("smpl", 6890, 55, 66),     # root(3)+body(63)
    ],
)
def test_forward_shapes_and_unification_real(
    model_type: str,
    expected_V: int,
    expected_unified_J: int,
    expected_P: int,
    batch2_inputs: UnifiedSmplInputs,
) -> None:
    if model_type == "smplx":
        if not _has_model_dir("smplx"):
            pytest.skip("SMPL-X model directory missing")
        model = smplx_create(model_path=str(_MODEL_ROOT), model_type="smplx")
    elif model_type == "smplh":
        if not _has_model_dir("smplh"):
            pytest.skip("SMPL-H model directory missing")
        smplh_dir = _MODEL_ROOT / "smplh"
        # Prefer NPZ variants (flat or nested), then fallback to PKL
        chosen = None
        for gender in ("neutral", "male", "female"):
            flat_npz = smplh_dir / f"SMPLH_{gender.upper()}.npz"
            if flat_npz.exists() and _smplh_npz_has_hand_keys(flat_npz):
                chosen = (gender, "npz")
                break
            nested_npz = smplh_dir / gender / "model.npz"
            if nested_npz.exists() and _smplh_npz_has_hand_keys(nested_npz):
                # Mark as nested npz
                chosen = (gender, "npz_nested")
                break
        if chosen is None:
            for gender in ("neutral", "male", "female"):
                if (smplh_dir / f"SMPLH_{gender.upper()}.pkl").exists():
                    chosen = (gender, "pkl")
                    break
        if chosen is None:
            pytest.skip("SMPL-H model file not available (no NPZ or PKL found)")
        gender, ext = chosen
        if ext == "npz":
            model = smplx_create(
                model_path=str(_MODEL_ROOT), model_type="smplh", gender=gender, ext="npz", num_betas=16
            )
        elif ext == "npz_nested":
            nested_npz = smplh_dir / gender / "model.npz"
            model = _SMPLHClass(
                model_path=str(nested_npz), gender=gender, ext="npz", num_betas=16, use_pca=False
            )
        else:
            model = smplx_create(model_path=str(_MODEL_ROOT), model_type="smplh", gender=gender)
    else:
        if not _has_model_dir("smpl"):
            pytest.skip("SMPL model directory missing")
        model = smplx_create(model_path=str(_MODEL_ROOT), model_type="smpl")

    uni = UnifiedSmplModel.from_smpl_model(model)

    out = uni(batch2_inputs)
    assert isinstance(out, UnifiedSmplOutput)

    # Vertices/joints shape checks
    assert out.vertices.shape[0] == 2
    assert out.vertices.shape[1] == expected_V
    assert out.vertices.shape[2] == 3

    assert out.joints.shape[0] == 2
    assert out.joints.shape[1] == expected_unified_J
    assert out.joints.shape[2] == 3

    # faces contract
    assert out.faces.ndim == 2 and out.faces.shape[1] == 3
    assert out.faces.dtype == torch.long

    # full_pose size
    assert out.full_pose.shape == (2, expected_P)

    # partitions
    assert out.body_joints.shape == (2, 22, 3)
    assert out.hand_joints.shape == (2, 30, 3)
    assert out.face_joints.shape == (2, 3, 3)

    # extras
    assert "joints_raw" in out.extras
    if uni.model_type == "smplh":
        assert "missing_joints" in out.extras
        # SMPL-H is missing face joints (jaw=22, left_eye=23, right_eye=24)
        assert out.extras["missing_joints"] == [22, 23, 24]
    if uni.model_type == "smpl":
        assert "missing_joints" in out.extras
        # SMPL is missing face and hand joints (indices 22-54 in unified set)
        # It should have at least these missing
        missing = out.extras["missing_joints"]
        assert 22 in missing  # jaw
        assert 23 in missing  # left_eye_smplhf
        assert 24 in missing  # right_eye_smplhf
        # Hand joints should also be missing (25-54)


# ------------------------------------------------------------------------
# Tests: Joint names & selection
# ------------------------------------------------------------------------

def test_get_joint_names_and_selection() -> None:
    if not _has_model_dir("smplx"):
        pytest.skip("SMPL-X model directory missing")

    uni = UnifiedSmplModel.from_smpl_model(
        smplx_create(model_path=str(_MODEL_ROOT), model_type="smplx")
    )
    names = uni.get_joint_names(unified=True)
    assert len(names) == 55
    # Verify we get actual joint names, not placeholders
    assert names[0] == "pelvis"
    assert names[22] == "jaw"
    assert names[54] == "right_thumb3"

    # select by indices and by names should agree
    B = 1
    J = 55
    joints = torch.randn(B, J, 3)
    indices = [0, 10, 22, 54]
    subset_by_idx = uni.select_joints(joints, indices=indices)
    subset_by_names = uni.select_joints(joints, names=[names[i] for i in indices])
    assert torch.allclose(subset_by_idx, subset_by_names)


# ------------------------------------------------------------------------
# Tests: Faces property (real models)
# ------------------------------------------------------------------------

def test_faces_property_dtype_and_shape_real(smplx_model: Any, smplh_model: Any) -> None:
    uni_x = UnifiedSmplModel.from_smpl_model(smplx_model)
    faces_x = uni_x.faces
    assert isinstance(faces_x, torch.Tensor)
    assert faces_x.dtype == torch.long
    assert faces_x.shape[1] == 3

    uni_h = UnifiedSmplModel.from_smpl_model(smplh_model)
    faces_h = uni_h.faces
    assert isinstance(faces_h, torch.Tensor)
    assert faces_h.dtype == torch.long
    assert faces_h.shape[1] == 3


# ------------------------------------------------------------------------
# Tests: Adapter utility methods (real models)
# ------------------------------------------------------------------------

def test_to_eval_train_do_not_crash_real() -> None:
    if not _has_model_dir("smplx"):
        pytest.skip("SMPL-X model directory missing")
    model = smplx_create(model_path=str(_MODEL_ROOT), model_type="smplx")
    uni = UnifiedSmplModel.from_smpl_model(model)
    # to() should not move wrapped model; ensure no exception and chain returned
    assert uni.to("cpu") is uni

    # eval/train proxy into wrapped model (no assertion on model state; just smoke)
    uni.eval()
    uni.train(False)
