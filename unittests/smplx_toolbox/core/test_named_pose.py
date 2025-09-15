#!/usr/bin/env python
"""
Unit tests for NamedPose utility.

Covers initialization, name/index mapping, get/set semantics, view behavior,
and batch repeat for all model types.
"""

from __future__ import annotations

import pytest
import torch

from smplx_toolbox.core.constants import ModelType
from smplx_toolbox.core import NamedPose


@pytest.mark.parametrize(
    "mt,expected_n",
    [
        (ModelType.SMPL, 21),   # intrinsic pose excludes pelvis
        (ModelType.SMPLH, 51),  # 52 - pelvis
        (ModelType.SMPLX, 54),  # 55 - pelvis
    ],
)
def test_named_pose_init_and_shapes(mt: ModelType, expected_n: int) -> None:
    npz = NamedPose(model_type=mt, batch_size=2)
    assert npz.intrinsic_pose is not None
    assert npz.intrinsic_pose.shape == (2, expected_n, 3)


def test_named_pose_get_set_smplx() -> None:
    B = 2
    npz = NamedPose(model_type=ModelType.SMPLX, batch_size=B)
    # Set with (B,3)
    val = torch.randn(B, 3)
    assert npz.get_joint_index("left_eye_smplhf") is not None
    assert npz.get_joint_index("left_eye") is None  # unknown name in this namespace

    ok = npz.set_joint_pose_value("left_eye_smplhf", val)
    assert ok is True

    got = npz.get_joint_pose("left_eye_smplhf")
    assert got is not None and got.shape == (B, 1, 3)
    assert torch.allclose(got.view(B, 3), val)

    # Unknown names: getters -> None, setters -> KeyError
    assert npz.get_joint_pose("left_eye") is None
    with pytest.raises(KeyError):
        _ = npz.set_joint_pose_value("left_eye", val)


def test_named_pose_to_dict_view_semantics_smplh() -> None:
    B = 1
    npz = NamedPose(model_type=ModelType.SMPLH, batch_size=B)
    d = npz.to_dict()
    # Pick a joint and mutate via view
    name = "left_index1"
    assert name in d
    before = npz.get_joint_pose(name)
    assert before is not None
    d[name].add_(1.0)
    after = npz.get_joint_pose(name)
    assert after is not None
    assert torch.allclose(after - before, torch.ones_like(after))


def test_named_pose_name_index_helpers_and_errors() -> None:
    npz = NamedPose(model_type=ModelType.SMPL)
    # Name/index round trip
    # get_joint_name now follows SMPL indexing (0=pelvis)
    assert npz.get_joint_name(0) == "pelvis"
    # For any intrinsic joint name, index == get_joint_index(name) + 1
    example_name = npz.get_joint_names([1])[0]  # name at SMPL index 1
    intrinsic_idx = npz.get_joint_index(example_name)
    assert intrinsic_idx is not None and intrinsic_idx + 1 == 1
    # Index error
    with pytest.raises(IndexError):
        _ = npz.get_joint_name(999)
    # Pelvis setter is allowed; should set root_pose
    B = 2
    npz2 = NamedPose(model_type=ModelType.SMPL, batch_size=B)
    root = torch.randn(B, 3)
    ok = npz2.set_joint_pose_value("pelvis", root)
    assert ok is True
    assert npz2.pelvis.shape == (B, 3)
    assert torch.allclose(npz2.pelvis, root)
