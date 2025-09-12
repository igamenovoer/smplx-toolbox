#!/usr/bin/env python
"""
Unit tests for UnifiedSmplModel and related data containers using real SMPL family models.

This suite:
- Loads official SMPL/SMPL-H/SMPL-X models from data/body_models via the installed smplx package
- Verifies UnifiedSmplModel API, normalization, unified joint set, and full_pose composition
- Tests input containers and keypoint conversion behaviors independent of model
- Confirms faces dtype/shape contract and adapter utility methods

Notes
- Requires model files under:
    data/body_models/smpl/
    data/body_models/smplh/
    data/body_models/smplx/
- Skips tests gracefully if models or the smplx package are not found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

# Import the installed smplx package directly (no local context/refcode usage)
try:
    import smplx  # noqa: F401
    from smplx import create as smplx_create
except Exception as _imp_err:  # pragma: no cover - environment dependent
    pytest.skip(f"smplx not importable: {_imp_err}", allow_module_level=True)

from smplx_toolbox.core import UnifiedSmplModel, UnifiedSmplInputs, UnifiedSmplOutput, NamedPose
from smplx_toolbox.core.constants import ModelType

# ------------------------------------------------------------------------
# Environment checks
# ------------------------------------------------------------------------

_MODEL_ROOT = Path("data/body_models").resolve()

def _has_model_dir(model_type: str) -> bool:
    return (_MODEL_ROOT / model_type).exists()



# ------------------------------------------------------------------------
# Fixtures: Real model loaders
# ------------------------------------------------------------------------

@pytest.fixture(scope="session")
def smpl_model() -> Any:
    if not _has_model_dir("smpl"):
        pytest.skip("SMPL model directory missing under data/body_models/smpl")
    try:
        return smplx_create(model_path=str(_MODEL_ROOT), model_type="smpl")
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"SMPL model not loadable: {e}")


@pytest.fixture(scope="session")
def smplh_model() -> Any:
    if not _has_model_dir("smplh"):
        pytest.skip("SMPL-H model directory missing under data/body_models/smplh")
    smplh_dir = _MODEL_ROOT / "smplh"
    # Use PKL models for SMPL-H
    for gender in ("neutral", "male", "female"):
        pkl = smplh_dir / f"SMPLH_{gender.upper()}.pkl"
        if pkl.exists():
            try:
                return smplx_create(
                    model_path=str(_MODEL_ROOT),
                    model_type="smplh",
                    gender=gender,
                )
            except Exception as e:  # pragma: no cover
                pytest.skip(f"SMPL-H model not loadable: {e}")
    pytest.skip("SMPL-H PKL model file not available (no NEUTRAL/MALE/FEMALE found)")


@pytest.fixture(scope="session")
def smplx_model() -> Any:
    if not _has_model_dir("smplx"):
        pytest.skip("SMPL-X model directory missing under data/body_models/smplx")
    # For SMPL-X, npz expected; the create() handles defaults
    try:
        return smplx_create(model_path=str(_MODEL_ROOT), model_type="smplx")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"SMPL-X model not loadable: {e}")


@pytest.fixture
def batch2_inputs() -> UnifiedSmplInputs:
    # Common minimal inputs (B=2) using NamedPose (preferred)
    B = 2
    npz = NamedPose(model_type=ModelType.SMPLX, batch_size=B)
    return UnifiedSmplInputs(named_pose=npz, betas=torch.randn(B, 10))


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

def test_named_pose_aggregate_properties() -> None:
    B = 3
    npz = NamedPose(model_type=ModelType.SMPLX, batch_size=B)
    hands = npz.hand_pose()
    eyes = npz.eyes_pose()
    assert hands is not None and hands.shape == (B, 90)
    assert eyes is not None and eyes.shape == (B, 6)


def test_inputs_validation_basic_no_error() -> None:
    # Minimal check: validation runs without error for any model type when using NamedPose
    B = 1
    npz = NamedPose(model_type=ModelType.SMPLX, batch_size=B)
    inputs = UnifiedSmplInputs(named_pose=npz)
    inputs.check_valid("smpl")
    inputs.check_valid("smplh")
    inputs.check_valid("smplx")


# ------------------------------------------------------------------------
# PoseByKeypoints has been removed; tests now focus on UnifiedSmplInputs and forward paths.


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
        # Use PKL models for SMPL-H
        model = None
        for gender in ("neutral", "male", "female"):
            if (smplh_dir / f"SMPLH_{gender.upper()}.pkl").exists():
                model = smplx_create(model_path=str(_MODEL_ROOT), model_type="smplh", gender=gender)
                break
        if model is None:
            pytest.skip("SMPL-H PKL model file not available (no NEUTRAL/MALE/FEMALE found)")
    else:
        if not _has_model_dir("smpl"):
            pytest.skip("SMPL model directory missing")
        try:
            model = smplx_create(model_path=str(_MODEL_ROOT), model_type="smpl")
        except Exception as e:
            pytest.skip(f"SMPL model not loadable: {e}")

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
