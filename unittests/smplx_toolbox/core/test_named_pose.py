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
        (ModelType.SMPL, 22),
        (ModelType.SMPLH, 52),
        (ModelType.SMPLX, 55),
    ],
)
def test_named_pose_init_and_shapes(mt: ModelType, expected_n: int) -> None:
    npz = NamedPose(model_type=mt, batch_size=2)
    assert npz.packed_pose is not None
    assert npz.packed_pose.shape == (2, expected_n, 3)


def test_named_pose_get_set_and_repeat_smplx() -> None:
    B = 2
    npz = NamedPose(model_type=ModelType.SMPLX, batch_size=B)
    # Set with (B,3)
    val = torch.randn(B, 3)
    assert npz.get_joint_index("left_eye_smplhf") is not None
    assert npz.get_joint_index("left_eye") is None  # unknown name in this namespace

    ok = npz.set_joint_pose("left_eye_smplhf", val)
    assert ok is True

    got = npz.get_joint_pose("left_eye_smplhf")
    assert got is not None and got.shape == (B, 1, 3)
    assert torch.allclose(got.view(B, 3), val)

    # Unknown names: getters -> None, setters -> KeyError
    assert npz.get_joint_pose("left_eye") is None
    with pytest.raises(KeyError):
        _ = npz.set_joint_pose("left_eye", val)

    # Repeat batch
    npz.repeat(3)
    assert npz.packed_pose.shape[0] == B * 3
    # Values at repeated slots should match
    assert torch.allclose(
        npz.packed_pose[:B, npz.get_joint_index("left_eye_smplhf") or 0, :],  # type: ignore[index]
        npz.packed_pose[B : 2 * B, npz.get_joint_index("left_eye_smplhf") or 0, :],  # type: ignore[index]
    )


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
    idxs = list(range(npz.packed_pose.shape[1]))  # type: ignore[union-attr]
    names = npz.get_joint_names(idxs)
    assert len(names) == len(idxs)
    r = npz.get_joint_indices(names)
    assert r == idxs
    # Index error
    with pytest.raises(IndexError):
        _ = npz.get_joint_name(999)
    # Shape error
    with pytest.raises(ValueError):
        _ = npz.set_joint_pose("pelvis", torch.randn(2, 2))

