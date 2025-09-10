from __future__ import annotations

import os

import pytest
import torch

from smplx_toolbox.vposer import load_vposer


@pytest.mark.unit
def test_load_vposer_and_decode_encode() -> None:
    ckpt_path = os.path.join("data", "vposer", "vposer-v2.ckpt")
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found at {ckpt_path}")

    model = load_vposer(ckpt_path, map_location="cpu")
    assert model.latent_dim > 0 and model.num_joints == 21

    with torch.no_grad():
        z = torch.zeros(2, model.latent_dim)
        out = model.decode(z)
        assert out["pose_body"].shape == (2, 21, 3)
        # Encode zeros AA
        aa = torch.zeros(2, 21, 3)
        qz = model.encode(aa)
        assert hasattr(qz, "mean") and hasattr(qz, "scale")

